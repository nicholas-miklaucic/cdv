import logging
from enum import Enum
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike
import numpy as np
import optax
import pyrallis
from pyrallis.fields import field as _field

from facet.config.common import dataclass
from facet.config.mace import MACEConfig
from facet.config.utils import Const
from facet.data.metadata import DatasetMetadata
from facet.optim import ema_params
from facet.regression import EFSLoss, EFSWrapper
from facet.utils import load_pytree

pyrallis.set_config_type('toml')

# field = lambda *args, **kwargs: _field(*args, **kwargs, is_mutable=True)
field = _field


@dataclass
class LogConfig:
    log_dir: Path = Path('logs/')

    exp_name: Optional[str] = None

    # How many times to make a log each epoch.
    logs_per_epoch: int = 16

    # Checkpoint every n epochs:
    epochs_per_ckpt: int = 5

    # Test every n epochs:
    epochs_per_valid: float = 0.5

    # Neptune tags.
    tags: list[str] = field(default_factory=list)

    # Parameters to log, if they exist. / indicates nesting.
    log_params: list[str] = field(
        default_factory=lambda: [
            'edge_embedding/basis/mu',
            'edge_embedding/basis/sigma',
            'edge_embedding/basis/freq',
            'edge_embedding/rmax',
        ]
    )


@dataclass
class DataConfig:
    # The name of the dataset to use.
    dataset_name: str = 'mp2022'

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
    batch_n_nodes: Optional[int] = None
    # Number of neighbors per node.
    k: Optional[int] = None
    # Number of graphs in each batch to pad to.
    batch_n_graphs: Optional[int] = None

    @property
    def graph_shape(self) -> tuple[int, int, int]:
        return (self.batch_n_nodes, self.k, self.batch_n_graphs)

    @property
    def metadata(self) -> DatasetMetadata | None:
        try:
            return DatasetMetadata(**load_pytree(self.dataset_folder / 'metadata.mpk'))
        except FileNotFoundError:
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
        return self.metadata.atomic_numbers.size

    def avg_num_neighbors(self, cutoff: ArrayLike):
        """Estimates average number of neighbors, given a certain cutoff."""
        return self.metadata.avg_num_neighbors(cutoff)


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
        """
        Computes f(|y - yhat|), where f is a "smoother L1 loss" function with these properties:

         - f(e) = |e| for e > self.loss_delta
         - f(0) = 0
         - e^2 <= f(e) <= |e| for all e
         - f and its first three derivatives are continuous everywhere

        Importantly, unlike Huber loss, there's no scaling factor to consider: f(e) is always
        between L2 and L1 loss, with loss_delta only controlling how sharp the higher-order
        derivatives are and how quickly the bound becomes tight.

        If loss_delta = 0, this is simply L1 loss. If loss_delta = inf, this is simply RMSE loss. Both
        are special-cased to avoid numerical issues, so differentiating them w.r.t loss_delta won't work.

        https://www.desmos.com/calculator/ntiznoeea8
        """
        if self.loss_delta == 0:
            return jnp.abs(preds - targets)
        elif self.loss_delta == np.inf:
            return jnp.square(preds - targets)
        else:
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
        """
        Computes a "smoother L1 loss" function with these properties:

         - f(e) = MAE(e) for e > self.loss_delta
         - f(0) = 0
         - MSE(e) <= f(e) <= MAE(e) for all e
         - f and its first three derivatives are continuous everywhere

        Importantly, unlike Huber loss, there's no scaling factor to consider: f(e) is always
        between L2 and L1 loss, with loss_delta only controlling how sharp the higher-order
        derivatives are and how quickly the bound becomes tight.

        If loss_delta = 0, this is simply L1 loss. If loss_delta = inf, this is simply L2 loss. Both
        are special-cased to avoid numerical issues, so differentiating them w.r.t loss_delta won't work.

        https://www.desmos.com/calculator/ntiznoeea8
        """
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
            from jax.sharding import Mesh, NamedSharding
            from jax.sharding import PartitionSpec as P

            jax.config.update('jax_threefry_partitionable', True)

            d = len(devs)
            mesh = Mesh(mesh_utils.create_device_mesh((d,), devices=jax.devices()[:d]), 'batch')
            sharding = NamedSharding(mesh, P('batch'))
            return sharding
        else:
            return devs[0]


@dataclass
class LossConfig:
    """Config defining the loss function."""

    reg_loss: RegressionLossConfig = field(default_factory=RegressionLossConfig)
    energy_weight: float = 1
    force_weight: float = 0
    stress_weight: float = 0

    def regression_loss(self, preds, targets, mask):
        return self.reg_loss.regression_loss(preds, targets, mask)

    @property
    def efs_wrapper(self) -> EFSWrapper:
        return EFSWrapper(
            compute_forces=self.force_weight != 0, compute_stress=self.stress_weight != 0
        )

    @property
    def efs_loss(self) -> EFSLoss:
        return EFSLoss(
            loss_fn=self.reg_loss.regression_loss,
            energy_weight=self.energy_weight,
            force_weight=self.force_weight,
            stress_weight=self.stress_weight,
        )


@dataclass
class LRScheduleConfig:
    """
    Learning rate schedule.

    Independent of scale: always has 1 as the base.
    """

    def build(self, num_epochs: int, steps_in_epoch: int) -> optax.Schedule:
        raise NotImplementedError


@dataclass
class WarmupCosine(LRScheduleConfig):
    """Cosine annealing with warmup."""

    kind: Const('cosine') = 'cosine'

    # Warmup, as a proportion of total training time.
    warmup_frac: float = 0.1

    # Starting LR, with a base of 1.
    start_lr: float = 0.01

    # Ending LR, with a base of 1.
    end_lr: float = 0.01

    def build(self, num_epochs: int, steps_in_epoch: int) -> optax.Schedule:
        num_steps = num_epochs * steps_in_epoch
        warmup_steps = round(num_steps * self.warmup_frac)
        return optax.warmup_cosine_decay_schedule(
            init_value=self.start_lr,
            peak_value=1,
            warmup_steps=warmup_steps,
            decay_steps=num_steps,
            end_value=self.end_lr,
        )


@dataclass
class WarmupPolynomial(LRScheduleConfig):
    """Polynomial schedule with warmup."""

    kind: Const('polynomial') = 'polynomial'

    # Warmup, as a proportion of total training time.
    warmup_frac: float = 0.1

    # Starting LR, with a base of 1.
    start_lr: float = 0.01

    # Ending LR, with a base of 1.
    end_lr: float = 0.01

    # Polynomial exponent.
    power: float = 1.0

    def build(self, num_epochs: int, steps_in_epoch: int) -> optax.Schedule:
        num_steps = num_epochs * steps_in_epoch
        warmup_steps = round(num_steps * self.warmup_frac)

        other_steps = num_steps - warmup_steps
        sched = optax.join_schedules(
            [
                optax.polynomial_schedule(
                    init_value=self.start_lr,
                    end_value=1,
                    power=1,
                    transition_steps=warmup_steps,
                ),
                optax.polynomial_schedule(
                    init_value=1,
                    end_value=self.end_lr,
                    power=1,
                    transition_steps=other_steps,
                ),
            ],
            boundaries=[warmup_steps],
        )

        return sched


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    # Base learning rate.
    base_lr: float = 1.0

    def build(self, learning_rate: optax.ScalarOrSchedule) -> optax.GradientTransformation:
        raise NotImplementedError


@dataclass
class AdamWConfig(OptimizerConfig):
    """
    Needs no introduction.

    One interesting tweak: supports the use of Mechanic for automatic LR tuning.
    """

    kind: Const('adamw') = 'adamw'

    # Weight decay. AdamW interpretation, so multiplied by the learning rate.
    weight_decay: float = 1e-3

    # Beta 1 for Adam.
    beta_1: float = 0.9

    # Beta 2 for Adam.
    beta_2: float = 0.999

    # Nesterov momentum.
    nesterov: bool = False

    # Use Mechanize to automatically tune the learning rate.
    mechanize: bool = False

    # Use schedule-free learning.
    schedule_free: bool = False

    def build(self, learning_rate: optax.ScalarOrSchedule) -> optax.GradientTransformation:
        tx = optax.adamw(
            learning_rate,
            b1=self.beta_1,
            b2=self.beta_2,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
            eps=1e-6,
        )
        if self.mechanize:
            tx = optax.contrib.mechanize(tx)

        if self.schedule_free:
            tx = optax.contrib.schedule_free_adamw(
                self.base_lr,
                b1=self.beta_1,
                b2=self.beta_2,
                weight_decay=self.weight_decay,
            )

        return tx


@dataclass
class ProdigyConfig(OptimizerConfig):
    """
    A learning-rate-free optimizer similar to AdamW.

    From "Prodigy: An Expeditiously Adaptive Parameter-Free Learner":
    https://arxiv.org/abs/2306.06101
    """

    kind: Const('prodigy') = 'prodigy'

    # Weight decay. AdamW interpretation, so multiplied by the learning rate.
    weight_decay: float = 1e-3

    # Beta 1 for Adam.
    beta_1: float = 0.9

    # Beta 2 for Adam.
    beta_2: float = 0.999

    def build(self, learning_rate: optax.ScalarOrSchedule) -> optax.GradientTransformation:
        tx = optax.contrib.prodigy(
            learning_rate,
            betas=(self.beta_1, self.beta_2),
            eps=1e-4,
            weight_decay=self.weight_decay,
            estim_lr_coef=self.base_lr,
            safeguard_warmup=True,
        )

        return tx


@dataclass
class TrainingConfig:
    """Training/optimizer parameters."""

    # Loss function.
    loss: LossConfig = field(default_factory=LossConfig)

    # Base learning rate.
    base_lr: float = 1.0

    # Learning rate schedule.
    lr_schedule: Union[WarmupCosine, WarmupPolynomial] = field(default_factory=WarmupPolynomial)

    # Optimizer.
    optimizer: Union[AdamWConfig, ProdigyConfig] = field(default_factory=ProdigyConfig)

    # Gradient norm clipping.
    max_grad_norm: float = 3.0

    # EMA decay rate.
    ema_gamma: float = 0.99

    # Steps between EMA updates.
    steps_between_ema: int = 16

    def __post_init__(self):
        if self.optimizer.kind == 'prodigy' and self.base_lr < 1e-2:
            logging.warning(
                f'Prodigy should normally be used with a base LR of 1. Are you sure you want {self.base_lr}?'
            )

    def build_lr_schedule(self, num_epochs: int, steps_in_epoch: int) -> optax.Schedule:
        return self.lr_schedule.build(num_epochs, steps_in_epoch)

    def build_optimizer(self, learning_rate):
        tx = self.optimizer.build(learning_rate)

        return optax.chain(
            tx,
            optax.clip_by_global_norm(self.max_grad_norm),
            ema_params(self.ema_gamma, self.steps_between_ema),
        )


@dataclass
class MainConfig:
    # The batch size. Should be a multiple of data_batch_size to make data loading simple.
    batch_size: int = 32 * 1

    # Number of stacked batches to process at once, if not given by the number of devices. Useful
    # for mocking multi-GPU training batches with a single GPU.
    stack_size: int = 1

    # Number of epochs.
    num_epochs: int = 20

    # Folder to initialize all parameters from, if the folder exists.
    restart_from: Optional[Path] = None

    # Checkpoint to use for parameters.
    checkpoint_params: Optional[Path] = None

    # Precision: 'f32' or 'bf16'.
    precision: str = 'f32'

    # Debug mode: turns off mid-run checkpointing and Neptune tracking.
    debug_mode: bool = False

    # Use profiling.
    do_profile: bool = False

    # Display kind for training runs: One of 'dashboard', 'progress', or 'quiet'.
    display: str = 'dashboard'

    data: DataConfig = field(default_factory=DataConfig)
    cli: CLIConfig = field(default_factory=CLIConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    log: LogConfig = field(default_factory=LogConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    model: MACEConfig = field(default_factory=MACEConfig)

    def __post_init__(self):
        if (
            self.data.metadata is not None
            and self.batch_size % self.data.metadata.batch_num_graphs != 0
        ):
            raise ValueError(
                'Training batch size should be multiple of data batch size: {} does not divide {}'.format(
                    self.batch_size, self.data.metadata.batch_num_graphs
                )
            )

        self.cli.set_up_logging()
        import warnings

        warnings.filterwarnings(message='Explicitly requested dtype', action='ignore')
        # if not self.log.log_dir.exists():
        #     raise ValueError(f'Log directory {self.log.log_dir} does not exist!')

        from jax.experimental.compilation_cache.compilation_cache import set_cache_dir

        set_cache_dir('/tmp/jax_comp_cache')
        # jax.config.update('jax_explain_cache_misses', True)

    @property
    def train_batch_multiple(self) -> int:
        """How many files should be loaded per training step."""
        if self.data.metadata is None:
            # if None, we're trying to load data so that we can compute the metadata. 1 is fine.
            return 1
        else:
            return self.batch_size // self.data.metadata.batch_num_graphs

    def build_regressor(self):
        return self.model.build(self.data.metadata, self.precision)

    def as_dict(self) -> dict:
        """Serializes the relevant values into a dictionary suitable for e.g., Neptune logging."""
        cfg: dict = pyrallis.encode(self)
        for key in ('do_profile', 'cli', 'log', 'debug_mode', 'display'):
            cfg.pop(key)

        for key in ('raw_data_folder', 'data_folder', 'batch_n_nodes', 'batch_n_graphs'):
            cfg['data'].pop(key)

        def convert_leaves(leaf):
            if isinstance(leaf, dict):
                return {k: convert_leaves(v) for k, v in leaf.items()}
            elif leaf is None:
                return 'None'
            elif isinstance(leaf, (tuple, list)):
                return {i: convert_leaves(v) for i, v in enumerate(leaf)}
            else:
                return leaf

        cfg = convert_leaves(cfg)

        return cfg


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
            cfg = pyrallis.cfgparsing.load(MainConfig, conf)

        print(cfg.as_dict())
