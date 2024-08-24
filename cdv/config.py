from collections.abc import Sequence
from json import JSONDecodeError
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import e3nn_jax
import jax
import jax.numpy as jnp
import optax
import pyrallis
from flax import linen as nn
from flax.struct import dataclass
from pyrallis.fields import field

from cdv import layers
from cdv.layers import Identity, LazyInMLP
from cdv.mace import MaceModel
from cdv.vae import VAE, Decoder, Encoder, LatticeVAE, PropertyPredictor

pyrallis.set_config_type('toml')


@dataclass
class LogConfig:
    log_dir: Path = Path('logs/')

    exp_name: Optional[str] = None

    # How many times to make a log each epoch.
    # 208 = 2^4 * 13 steps per epoch with batch of 1: evenly dividing this is nice.
    logs_per_epoch: int = 8

    # Checkpoint every n epochs:
    epochs_per_ckpt: int = 2

    # Test every n epochs:
    epochs_per_valid: int = 2


@dataclass
class DataConfig:
    # The name of the dataset to use.
    dataset_name: str = 'mptrj'

    # Folder of raw data files.
    raw_data_folder: Path = Path('data/')

    # Folder of processed data files.
    data_folder: Path = Path('precomputed/')

    # Seed for dataset shuffling. Controls the order batches are given to train.
    shuffle_seed: int = 1618

    # Train split.
    train_split: int = 30
    # Test split.
    test_split: int = 3
    # Valid split.
    valid_split: int = 3

    # Batches per group to take, 0 means everything. Should only be used for testing.
    batches_per_group: int = 0

    # Number of nodes in each batch to pad to.
    batch_n_nodes: int = 1024
    # Number of neighbors per node.
    k: int = 16
    # Number of graphs in each batch to pad to.
    batch_n_graphs: int = 32

    @property
    def graph_shape(self) -> tuple[int, int, int]:
        return (self.batch_n_nodes, self.k, self.batch_n_graphs)

    @property
    def metadata(self) -> Mapping[str, Any] | None:
        import json

        path = self.dataset_folder / 'metadata.json'

        if not path.exists():
            return None

        try:
            with open(path, 'r') as metadata:
                metadata = json.load(metadata)
                return metadata
        except JSONDecodeError:
            return None

    def __post_init__(self):
        pass
        # num_splits = self.train_split + self.test_split + self.valid_split
        # num_batches = self.metadata['data_size'] // self.metadata['batch_size']
        # if num_batches % num_splits != 0:
        #     msg = f'Data is split {num_splits} ways, which does not divide {num_batches}'
        #     raise ValueError(msg)

    @property
    def dataset_folder(self) -> Path:
        """Folder where dataset-specific files are stored."""
        return self.data_folder / self.dataset_name

    @property
    def num_species(self) -> int:
        """Number of unique elements."""
        return len(self.metadata['elements'])


class LoggingLevel(Enum):
    """The logging level."""

    debug = logging.DEBUG
    info = logging.INFO
    warning = logging.WARNING
    error = logging.ERROR
    critical = logging.CRITICAL


@dataclass
class RegressionLossConfig:
    """Config defining the loss function."""

    # delta for smoother_l1_loss: the switch point between the smooth version and pure L1 loss.
    loss_delta: float = 1 / 64

    # Whether to use RMSE loss instead.
    use_rmse: bool = False

    def smoother_l1_loss(self, preds, targets):
        a = -5 / 2 / 8
        b = 63 / 8 / 6
        c = -35 / 4 / 4
        d = 35 / 8 / 2
        x = (preds - targets) / self.loss_delta
        x2 = x * x
        x_abs = jnp.abs(x)
        y = jnp.where(x_abs < 1, x2 * d + (x2 * c + (x2 * b + (x2 * a))), x_abs)
        return y * self.loss_delta

    def regression_loss(self, preds, targets, mask):
        if preds.shape != targets.shape:
            msg = f'Incorrect input shapes: {preds.shape} != {targets.shape}'
            raise ValueError(msg)
        if preds.ndim == 2:
            mask = mask[:, None]
        if self.use_rmse:
            return jnp.sqrt(optax.losses.squared_error(preds, targets).mean(where=mask))
        else:
            return self.smoother_l1_loss(preds, targets).mean(where=mask)


@dataclass
class CLIConfig:
    # Verbosity of output.
    verbosity: LoggingLevel = LoggingLevel.info
    # Whether to show progress bars.
    show_progress: bool = True

    def set_up_logging(self):
        from rich.logging import RichHandler
        from rich.pretty import install as pretty_install
        from rich.traceback import install as traceback_install

        pretty_install(crop=True, max_string=100, max_length=10)
        traceback_install(show_locals=False)

        import flax.traceback_util as ftu

        ftu.hide_flax_in_tracebacks()

        logging.basicConfig(
            level=self.verbosity.value,
            format='%(message)s',
            datefmt='[%X]',
            handlers=[
                RichHandler(
                    rich_tracebacks=False,
                    show_time=False,
                    show_level=False,
                    show_path=False,
                )
            ],
        )


@dataclass
class DeviceConfig:
    # Either 'cpu', 'gpu', or 'tpu'
    device: str = 'gpu'

    # Limits the number of GPUs used. 0 means no limit.
    max_gpus: int = 0

    # IDs of GPUs to use.
    gpu_ids: list[int] = field(default_factory=list)

    def devices(self):
        devs = jax.devices(self.device)
        if self.device == 'gpu' and self.max_gpus != 0:
            idx = list(range(len(devs)))
            order = [x for x in self.gpu_ids if x in idx] + [
                x for x in idx if x not in self.gpu_ids
            ]
            devs = [devs[i] for i in order[: self.max_gpus]]

        return devs

    def jax_device(self):
        devs = self.devices()

        if len(devs) > 1:
            from jax.experimental import mesh_utils
            from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

            jax.config.update('jax_threefry_partitionable', True)

            d = len(devs)
            mesh = Mesh(mesh_utils.create_device_mesh((d,), devices=jax.devices()[:d]), 'batch')
            sharding = NamedSharding(mesh, P('batch'))
            # replicated_sharding = NamedSharding(mesh, P())
            return sharding
        else:
            return devs[0]

    def __post_init__(self):
        import os

        os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

        # import jax
        # if self.max_gpus == 1:
        #     jax.config.update('jax_default_device', self.jax_device())


@dataclass
class Layer:
    """Serializable layer representation. Works for any named layer in layers.py or flax.nn."""

    # The name of the layer.
    name: str

    def build(self) -> Callable:
        """Makes a new layer with the given values, or returns the function if it's a function."""
        if self.name == 'Identity':
            return Identity()

        for module in (nn, layers):
            if hasattr(module, self.name):
                layer = getattr(module, self.name)
                if isinstance(layer, nn.Module):
                    return getattr(module, self.name)()
                else:
                    # something like relu
                    return layer

        msg = f'Could not find {self.name} in flax.linen or avid.layers'
        raise ValueError(msg)


@dataclass
class MLPConfig:
    """Settings for MLP configuration."""

    # Inner dimensions for the MLP.
    inner_dims: list[int] = field(default_factory=list)

    # Inner activation.
    activation: str = 'swish'

    # Final activation.
    final_activation: str = 'Identity'

    # Output dimension. None means the same size as the input.
    out_dim: Optional[int] = None

    # Dropout.
    dropout: float = 0.0

    # Whether to add residual.
    residual: bool = False

    # Number of heads, for equivariant layer.
    num_heads: int = 1

    def build(self) -> LazyInMLP:
        """Builds the head from the config."""
        return LazyInMLP(
            inner_dims=self.inner_dims,
            out_dim=self.out_dim,
            inner_act=Layer(self.activation).build(),
            final_act=Layer(self.final_activation).build(),
            dropout_rate=self.dropout,
            residual=self.residual,
        )


@dataclass
class MACEConfig:
    max_ell: int = 3
    num_interactions: int = 2
    hidden_irreps: str = '256x0e + 256x1o'
    # hidden_irreps = '16x0e + 16x1o'
    correlation: int = 3  # 4 is better but 5x slower
    readout_mlp_irreps: str = '16x0e'
    interaction_reduction: str = 'last'
    node_reduction: str = 'mean'
    gate: str = 'silu'

    def build(
        self,
        num_species: int,
        elem_indices: Sequence[int],
        output_graph_irreps: str,
        output_node_irreps: str | None = None,
        scalar_mean: float = 0.0,
        scalar_std: float = 1.0,
    ) -> MaceModel:
        return MaceModel(
            max_ell=self.max_ell,
            elem_indices=jnp.array(elem_indices),
            num_interactions=self.num_interactions,
            hidden_irreps=str(e3nn_jax.Irreps(self.hidden_irreps)),
            readout_mlp_irreps=str(e3nn_jax.Irreps(self.readout_mlp_irreps)),
            output_graph_irreps=str(e3nn_jax.Irreps(output_graph_irreps)),
            output_node_irreps=str(e3nn_jax.Irreps(output_node_irreps))
            if output_node_irreps
            else None,
            num_species=num_species,
            correlation=self.correlation,
            interaction_reduction=self.interaction_reduction,
            node_reduction=self.node_reduction,
            scalar_mean=scalar_mean,
            scalar_std=scalar_std,
        )


@dataclass
class LossConfig:
    """Config defining the loss function."""

    reg_loss: RegressionLossConfig = field(default_factory=RegressionLossConfig)

    def regression_loss(self, preds, targets, mask):
        return self.reg_loss.regression_loss(preds, targets, mask)


@dataclass
class TrainingConfig:
    """Training/optimizer parameters."""

    # Loss function.
    loss: LossConfig = field(default_factory=LossConfig)

    # Learning rate schedule: 'cosine' for warmup+cosine annealing
    lr_schedule_kind: str = 'cosine'

    # Initial learning rate, as a fraction of the base LR.
    start_lr_frac: float = 0.1

    # Base learning rate.
    base_lr: float = 4e-3

    # Final learning rate, as a fraction of the base LR.
    end_lr_frac: float = 0.04

    # Weight decay. AdamW interpretation, so multiplied by the learning rate.
    weight_decay: float = 0.03

    # Beta 1 for Adam.
    beta_1: float = 0.9

    # Beta 2 for Adam.
    beta_2: float = 0.999

    # Nestorov momentum.
    nestorov: bool = True

    # Gradient norm clipping.
    max_grad_norm: float = 1.0

    # Schedule-free.
    schedule_free: bool = False

    # Prodigy.
    prodigy: bool = False

    def lr_schedule(self, num_epochs: int, steps_in_epoch: int):
        if self.lr_schedule_kind == 'cosine':
            base_lr = self.base_lr
            if self.prodigy:
                base_lr = 1
            warmup_steps = steps_in_epoch * min(5, num_epochs // 2)
            return optax.warmup_cosine_decay_schedule(
                init_value=base_lr * self.start_lr_frac,
                peak_value=base_lr,
                warmup_steps=warmup_steps,
                decay_steps=num_epochs * steps_in_epoch,
                end_value=base_lr * self.end_lr_frac,
            )
        else:
            raise ValueError('Other learning rate schedules not implemented yet')

    def optimizer(self, learning_rate):
        if self.prodigy:
            tx = optax.contrib.prodigy(
                learning_rate,
                betas=(self.beta_1, self.beta_2),
                weight_decay=self.weight_decay,
                estim_lr_coef=self.base_lr / 4e-3,
            )
        else:
            tx = optax.adamw(
                learning_rate,
                b1=self.beta_1,
                b2=self.beta_2,
                weight_decay=self.weight_decay,
                nesterov=self.nestorov,
            )
        return optax.chain(tx, optax.clip_by_global_norm(self.max_grad_norm))


@dataclass
class MainConfig:
    # The batch size. Should be a multiple of data_batch_size to make data loading simple.
    batch_size: int = 32 * 1

    # Number of stacked batches to process at once, if not given by the number of devices. Useful
    # for mocking multi-GPU training batches with a single GPU.
    stack_size: int = 1

    # Use profiling.
    do_profile: bool = False

    # Number of epochs.
    num_epochs: int = 20

    # Folder to initialize all parameters from, if the folder exists.
    restart_from: Optional[Path] = None

    # Folder to initialize the encoders and downsampling.
    encoder_start_from: Optional[Path] = None

    # Display kind for training runs: One of 'dashboard', 'progress', or 'quiet'.
    display: str = 'dashboard'

    data: DataConfig = field(default_factory=DataConfig)
    cli: CLIConfig = field(default_factory=CLIConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    log: LogConfig = field(default_factory=LogConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    mace: MACEConfig = field(default_factory=MACEConfig)

    regressor: str = 'mace'

    task: str = 'e_form'

    def __post_init__(self):
        if (
            self.data.metadata is not None
            and self.batch_size % self.data.metadata['batch_size'] != 0
        ):
            raise ValueError(
                'Training batch size should be multiple of data batch size: {} does not divide {}'.format(
                    self.batch_size, self.data.metadata['batch_size']
                )
            )

        self.cli.set_up_logging()
        import warnings

        warnings.filterwarnings(message='Explicitly requested dtype', action='ignore')
        if not self.log.log_dir.exists():
            raise ValueError(f'Log directory {self.log.log_dir} does not exist!')

        from jax.experimental.compilation_cache.compilation_cache import set_cache_dir

        set_cache_dir('/tmp/jax_comp_cache')

    @property
    def train_batch_multiple(self) -> int:
        """How many files should be loaded per training step."""
        if self.data.metadata is None:
            return 1
        else:
            return self.batch_size // self.data.metadata['batch_size']

    # def build_diffusion(self) -> DiffusionModel:
    #     diffuser = MLPMixerDiffuser(
    #         embed_dims=self.diffusion.embed_dim,
    #         embed_max_freq=self.diffusion.unet.embed_max_freq,
    #         embed_min_freq=self.diffusion.unet.embed_min_freq,
    #         mixer=self.build_mlp().mixer,
    #     )
    #     return self.diffusion.diffusion.build(diffuser)

    def build_vae(self):
        # output irreps gets changed
        return VAE(
            Encoder(
                self.mace.build(self.data.num_species, '0e', None),
                latent_dim=256,
                latent_space=LatticeVAE(),
            ),
            PropertyPredictor(LazyInMLP([256], dropout_rate=0.5)),
            Decoder(self.mace.build(self.data.num_species, '0e', None)),
            prop_reg_loss=self.train.loss.regression_loss,
        )

    def build_regressor(self):
        if self.regressor == 'mace':
            return self.mace.build(
                self.data.num_species,
                self.data.metadata['element_indices'],
                '0e',
                scalar_mean=self.data.metadata['e_form']['mean'],
                scalar_std=self.data.metadata['e_form']['mean'],
            )
        else:
            raise ValueError(f'{self.regressor} not supported')


if __name__ == '__main__':
    from pathlib import Path

    from rich.prompt import Confirm

    if Confirm.ask('Generate configs/defaults.toml and configs/minimal.toml?'):
        default_path = Path('configs') / 'defaults.toml'
        minimal_path = Path('configs') / 'minimal.toml'

        default = MainConfig()

        with open(default_path, 'w') as outfile:
            pyrallis.cfgparsing.dump(default, outfile)

        with open(minimal_path, 'w') as outfile:
            pyrallis.cfgparsing.dump(default, outfile, omit_defaults=True)

        with default_path.open('r') as conf:
            pyrallis.cfgparsing.load(MainConfig, conf)
