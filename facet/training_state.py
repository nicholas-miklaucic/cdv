import functools as ft
import logging
import random
import shutil
import time
from collections import defaultdict
from dataclasses import field
from datetime import datetime, timedelta
from os import PathLike
from pathlib import Path
from shutil import copytree, make_archive
from typing import Any, Literal, Mapping, Sequence, Union

import chex
import jax
import jax.numpy as jnp
import neptune  # type: ignore
import numpy as np
import optax  # type: ignore
import orbax.checkpoint as ocp  # type: ignore
import pandas as pd
import pyrallis
from flax import linen as nn
from flax import struct
from flax.training import train_state
from neptune.types import File  # type: ignore

from facet.checkpointing import best_ckpt
from facet.config import LossConfig, MainConfig
from facet.data.dataset import CrystalGraphs, dataloader
from facet.layers import Context
from facet.model_summary import model_summary
from facet.utils import get_nested_path, item_if_arr, load_pytree


@struct.dataclass
class Metrics:
    totals: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)

    def update(self, **kwargs):
        totals = self.totals.copy()
        counts = self.counts.copy()
        for k, v in kwargs.items():
            totals[k] = totals.get(k, 0) + v
            counts[k] = counts.get(k, 0) + 1

        return Metrics(totals, counts)

    def items(self):
        return {k: item_if_arr(self.totals[k] / self.counts[k]) for k in self.totals}.items()


class TrainState(train_state.TrainState):
    metrics: Metrics
    last_grad_norm: jax.Array

    def replicate_params(self, devices: set[jax.Device]):
        return jax.device_put_replicated(self.params, tuple(devices))


def create_train_state(module: nn.Module, optimizer, rng, batch: CrystalGraphs):
    b1 = jax.tree_map(lambda x: x[0], batch)
    # debug_structure(b1=b1)
    # jax.debug.visualize_array_sharding(b1.e_form)
    loss, params = module.init_with_output(
        {k: jax.random.key(v) for k, v in rng.items()}, b1, ctx=Context(training=False)
    )
    # jax.debug.visualize_array_sharding(loss)
    tx = optimizer
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        metrics=Metrics(),
        last_grad_norm=jnp.array(0.0, dtype=jnp.float32),
    )


@struct.dataclass
class Checkpoint:
    state: TrainState
    seed: int
    metrics_history: Mapping[str, Sequence[float]]
    curr_epoch: float


def eval_state(state: TrainState) -> TrainState:
    return state.replace(params=state.opt_state[-1].ema)  # type: ignore


def update_metrics(state: TrainState, metric_updates) -> TrainState:
    return state.replace(metrics=state.metrics.update(**metric_updates))


class TrainingRun:
    def __init__(self, config: MainConfig):
        self.seed = random.randint(100, 1000)
        self.rng = {'params': self.seed}
        logging.info(f'Seed: {self.seed}')
        self.config = config

        self.metrics_history: Mapping[str, list[Any]] = defaultdict(list)
        self.num_epochs = config.num_epochs
        self.steps_in_epoch, self.dl = dataloader(config, split='train', infinite=True)
        self.steps_in_test_epoch, self.test_dl = dataloader(config, split='valid', infinite=True)
        self.num_steps = self.steps_in_epoch * self.num_epochs
        self.steps = range(self.num_steps)
        self.curr_step = 0
        self.start_time = time.monotonic()
        self.scheduler = self.config.train.build_lr_schedule(
            num_epochs=self.num_epochs, steps_in_epoch=self.steps_in_epoch
        )
        self.optimizer = self.config.train.build_optimizer(self.scheduler)
        self.model = self.make_model()
        self.metrics = Metrics()

        opts = ocp.CheckpointManagerOptions(
            save_interval_steps=1,
            best_fn=lambda metrics: metrics['te_loss'],
            best_mode='min',
            max_to_keep=4,
            enable_async_checkpointing=True,
            keep_time_interval=timedelta(minutes=30),
        )
        self.mngr = ocp.CheckpointManager(
            ocp.test_utils.erase_and_create_empty('/tmp/jax_ckpt'),
            options=opts,
        )

        self.run = self.init_run()

        self.test_loss = 1000

    def init_run(self) -> neptune.Run:
        run = neptune.init_run(
            mode='debug' if self.config.debug_mode else 'async',
            tags=self.config.log.tags,
            capture_stderr=False,
            capture_stdout=False,
            source_files='facet/**/*.py',
        )

        run['parameters'] = self.config.as_dict()
        run['full_config'] = File.from_content(
            pyrallis.cfgparsing.dump(self.config, omit_defaults=False), extension='toml'
        )

        if not self.config.debug_mode:
            summary = model_summary(self.config)
            run['model_summary'] = File.from_content(summary['html'], extension='html')
            run['gflops'] = summary['gflops']

        run['dataset_metadata'].track_files(str(self.config.data.dataset_folder / 'metadata.json'))
        run['seed'] = self.seed
        return run

    @staticmethod
    @ft.partial(jax.jit, static_argnames=('config',))
    @chex.assert_max_traces(4)
    def compute_metrics(
        config: LossConfig,
        state: TrainState,
        batch: CrystalGraphs,
        preds,
        rng=None,
    ):
        losses: dict[str, Union[float, jax.Array]] = {k: jnp.mean(v) for k, v in preds.items()}
        losses['grad_norm'] = state.last_grad_norm
        metric_updates = dict(**losses)

        return metric_updates

    @staticmethod
    @ft.partial(jax.jit, static_argnames=('config',))
    @chex.assert_max_traces(2)
    def test_preds(config: LossConfig, state: TrainState, params, batch: CrystalGraphs, rng):
        """Evaluate metrics for a single batch."""

        from jax.experimental import mesh_utils
        from jax.experimental.shard_map import shard_map
        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec as P

        devs = jax.local_devices()
        mesh = Mesh(mesh_utils.create_device_mesh([len(devs)], devices=devs), 'batch')

        @ft.partial(jax.vmap, in_axes=(None, 0))
        def loss_fn(params, batch):
            preds = config.efs_wrapper(
                state.apply_fn, params, batch, ctx=Context(training=False), rngs=rng['params']
            )
            loss = config.efs_loss(batch, preds)
            return loss

        @ft.partial(
            shard_map,
            mesh=mesh,
            in_specs=(P(None), P('batch')),
            out_specs=P('batch'),
            check_rep=False,
        )
        def ploss_fn(params, batch):
            loss = loss_fn(params, batch)
            loss = jax.lax.pmean(loss, axis_name='batch')
            return loss

        loss = ploss_fn(params, batch)
        return loss

    @staticmethod
    @ft.partial(jax.jit, static_argnames=('config',))
    @chex.assert_max_traces(2)
    def train_grads(config: LossConfig, state: TrainState, params, batch: CrystalGraphs, rng):
        """Train for a single step."""
        from jax.experimental import mesh_utils
        from jax.experimental.shard_map import shard_map
        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec as P

        devs = jax.local_devices()
        mesh = Mesh(mesh_utils.create_device_mesh([len(devs)], devices=devs), 'batch')

        def loss_fn(params, batch):
            preds = config.efs_wrapper(
                state.apply_fn, params, batch, ctx=Context(training=True), rngs=rng['params']
            )
            loss = config.efs_loss(batch, preds)
            return loss['loss'].mean(), loss

        @ft.partial(jax.vmap, in_axes=(None, 0))
        def vgrad_fn(params, batch):
            grad_loss_fn = jax.grad(loss_fn, has_aux=True)
            grads, preds = grad_loss_fn(params, batch)
            return grads, preds

        def agg_grads(grads):
            return jax.tree_map(lambda x: jnp.mean(x, axis=0), grads)

        @ft.partial(
            shard_map,
            mesh=mesh,
            in_specs=(P(None), P('batch')),
            out_specs=(P(), P('batch')),
            check_rep=False,
        )
        def pgrad_fn(params, batch):
            grads, preds = vgrad_fn(params, batch)
            grads = agg_grads(grads)
            grads = jax.lax.pmean(grads, axis_name='batch')
            return grads, preds

        grads, preds = pgrad_fn(params, batch)
        return grads, preds

    @staticmethod
    @ft.partial(jax.jit)
    @chex.assert_max_traces(2)
    def update_train_state(state: TrainState, grads) -> TrainState:
        # grads = jax.tree_map(lambda x: x.mean(axis=0), grads)
        grad_norm = optax.global_norm(grads)
        state = state.apply_gradients(grads=grads, last_grad_norm=grad_norm)
        return state

    @staticmethod
    def train_step(config: LossConfig, state: TrainState, batch: CrystalGraphs, rng):
        """Train for a single step."""
        # debug_structure(state.params)
        # debug_structure(batch)
        rng = {k: jax.random.key((v + state.step) % 2**16) for k, v in rng.items()}
        grads, preds = TrainingRun.train_grads(config, state, state.params, batch, rng)
        # debug_structure(grads=grads, preds=preds)
        # print('params')
        # jax.debug.visualize_array_sharding(jax.tree_leaves(state.params)[0].reshape(-1))
        # print('preds')
        # jax.debug.visualize_array_sharding(preds[..., 0])
        state = TrainingRun.update_train_state(state, grads)
        return state, preds

    def make_model(self):
        return self.config.build_regressor()

    def next_step(self):
        return self.step(self.curr_step + 1, next(self.dl))

    def log_metric(
        self, metric_name: str, metric_value, split: Literal['train', 'valid', 'eval', None]
    ):
        if split == 'train':
            self.metrics_history[f'tr_{metric_name}'].append(metric_value)
            self.run[f'train/{metric_name}'].append(value=metric_value, step=self.curr_epoch)
        elif split == 'valid':
            self.metrics_history[f'te_{metric_name}'].append(metric_value)
            self.run[f'valid/{metric_name}'].append(value=metric_value, step=self.curr_epoch)
        elif split == 'eval':
            self.metrics_history[f'ev_{metric_name}'].append(metric_value)
            self.run[f'eval/{metric_name}'].append(value=metric_value, step=self.curr_epoch)
        elif split is None:
            self.metrics_history[f'{metric_name}'].append(metric_value)
            self.run[f'{metric_name}'].append(value=metric_value, step=self.curr_epoch)
        else:
            raise ValueError(f'Split invalid: {split}')

    @property
    def eval_state(self) -> TrainState:
        estate = eval_state(self.state)
        if getattr(self.config.train.optimizer, 'schedule_free', False):
            # debug_structure(estate.opt_state)
            return estate.replace(
                params=optax.contrib.schedule_free_eval_params(estate.opt_state[0], estate.params)
            )
        else:
            return estate

    def step(self, step: int, batch: CrystalGraphs):
        self.curr_step = step
        if step == 0:
            # initialize model
            self.state = create_train_state(
                module=self.model,
                optimizer=self.optimizer,
                rng=self.rng,
                batch=batch,
            )

            if self.config.restart_from is not None:
                if self.config.checkpoint_params is not None:
                    raise ValueError('Cannot give both restart_from and checkpoint_params')
                state = best_ckpt(self.config.restart_from)['state']
                self.state = self.state.replace(params=state['params'])
            elif self.config.checkpoint_params is not None:
                ckpt_params = load_pytree(self.config.checkpoint_params)

                # print(tuple(ckpt_params.keys()))
                if tuple(ckpt_params.keys()) != ('params',):
                    ckpt_params = {'params': ckpt_params}
                flattened, _ = jax.tree_util.tree_flatten_with_path(ckpt_params)
                # print(jax.tree_util.tree_flatten_with_path(self.state.params)[0])

                def replace(kp, x):
                    kpstr = jax.tree_util.keystr(kp)
                    for kp2, new_x in flattened:
                        if kpstr == jax.tree_util.keystr(kp2):
                            return new_x
                    logging.warning(f'{kpstr} not in checkpoint')
                    return x

                self.state = self.state.replace(
                    params=jax.tree_util.tree_map_with_path(replace, self.state.params)
                )

            # log number of parameters
            self.run['params'] = int(
                jax.tree.reduce(
                    lambda x, y: x + y, jax.tree.map(lambda x: x.size, self.state.params)
                )
            )

            for path in self.config.log.log_params:
                param = get_nested_path(self.state.params['params'], path)
                if param is None:
                    logging.warning(f'Could not find parameter {path}')
                    # debug_structure(params=self.state.params['params'])
        elif step >= self.num_steps:
            return None

        kwargs = dict(
            config=self.config.train.loss,
            batch=batch,
            rng=self.rng,
        )

        self.state, preds = self.train_step(state=self.state, **kwargs)  # type: ignore
        # jax.debug.visualize_array_sharding(preds[..., 0])

        # instead of unpacking kwargs, we specify order of kwargs so that jax doesn't compile this twice
        metric_updates = self.compute_metrics(
            self.config.train.loss, self.eval_state, batch, preds, None
        )
        self.state = update_metrics(self.state, metric_updates)

        if self.should_log or self.should_ckpt or self.should_validate:
            for metric, value in self.state.metrics.items():  # compute metrics
                if metric == 'grad_norm':
                    self.log_metric('grad_norm', value, None)
                    continue
                self.log_metric(metric, value, 'train')

                if not self.should_validate:
                    if f'te_{metric}' in self.metrics_history:
                        test_value = self.metrics_history[f'te_{metric}'][-1]
                    else:
                        test_value = 0
                    self.log_metric(metric, test_value, 'valid')

                    if f'ev_{metric}' in self.metrics_history:
                        test_value = self.metrics_history[f'ev_{metric}'][-1]
                    else:
                        test_value = 0
                    self.log_metric(metric, test_value, 'eval')

            self.state = self.state.replace(
                metrics=Metrics()
            )  # reset train_metrics for next training epoch

            self.log_metric('lr', self.lr, None)
            self.log_metric('step', self.curr_step, None)
            self.log_metric('epoch', self.curr_epoch, None)
            self.log_metric('rel_mins', (time.monotonic() - self.start_time) / 60, None)
            if max(self.metrics_history['epoch'], default=0) < 1:
                self.log_metric(
                    'throughput',
                    self.curr_step
                    * self.config.batch_size
                    * self.config.stack_size
                    / self.metrics_history['rel_mins'][-1],
                    None,
                )
            else:
                prev, curr = self.metrics_history['rel_mins'][-2:]
                min_delta = curr - prev

                prev, curr = self.metrics_history['step'][-2:]
                size = (curr - prev) * self.config.batch_size * self.config.stack_size
                self.log_metric('throughput', size / min_delta, None)

                # log specific parameters
                for path in self.config.log.log_params:
                    param = get_nested_path(self.state.params['params'], path)
                    if param is not None:
                        if param.size == 1:
                            self.run[f'model_params/{path}'].append(param.item())
                        elif param.size <= 64:  # nothing too crazy!
                            for i, val in enumerate(param.flatten()):
                                self.run[f'model_params/{path}/{i}'].append(val)
                        else:
                            # we don't want to accidentally upload a million values
                            continue

        # debug_structure(self.state)
        # print(self.metrics_history)

        def compute_test_metrics(test_state: TrainState):
            for i, test_batch in zip(range(self.steps_in_test_epoch), self.test_dl):
                test_preds = TrainingRun.test_preds(
                    self.config.train.loss,
                    test_state,
                    test_state.params,
                    test_batch,
                    {
                        k: jax.random.key((v + i + test_state.step) % 2**16)
                        for k, v in self.rng.items()
                    },
                )

                # order of updates matters here: JAX recompiles if they differ
                metric_updates = self.compute_metrics(
                    self.config.train.loss, test_state, test_batch, test_preds, None
                )

                test_state = update_metrics(test_state, metric_updates)

            return test_state

        if self.should_validate:
            self.test_state = self.eval_state.replace(metrics=Metrics())
            self.test_state = compute_test_metrics(self.test_state)

            for metric, value in self.test_state.metrics.items():
                if metric == 'grad_norm':
                    continue
                self.log_metric(metric, value, 'eval')
                if f'{metric}' == 'loss':
                    self.test_loss = value

            # compute values with current training: useful to debug difference between
            # e.g., EMA and normal training
            test_state_eval = self.state.replace(metrics=Metrics())
            test_state_eval = compute_test_metrics(test_state_eval)
            for metric, value in test_state_eval.metrics.items():
                if metric == 'grad_norm':
                    continue
                self.log_metric(metric, value, 'valid')

        if self.should_ckpt:
            # print(self.test_loss)
            self.mngr.save(
                self.curr_step,
                args=ocp.args.StandardSave(self.ckpt()),  # type: ignore
                metrics={'te_loss': self.test_loss},
            )

        return self

    def step_until_done(self):
        for step, batch in zip(self.steps, self.dl):
            yield self.step(step, batch)

    def run_to_completion(self):
        for step, batch in zip(self.steps, self.dl):
            self.step(step, batch)

    @property
    def curr_epoch(self) -> float:
        return self.curr_step / self.steps_in_epoch

    @property
    def is_last_step(self):
        return self.curr_step + 1 == self.num_steps

    @property
    def i_in_epoch(self):
        return (self.curr_step + 1) % self.steps_in_epoch

    def equispaced_steps(self, step, epoch_frac):
        """Selects steps that are equispaced and
        appear the correct number of times per epoch.

        For example, epoch_frac=4 selects every 4th epoch's last step, and epoch_frac=1/16 selects
        the steps after 1/16, 2/16, 3/16, ..., 16/16 of an epoch has completed.
        """
        repeat_after = int(np.ceil(epoch_frac)) * self.steps_in_epoch
        num_steps = round(repeat_after / (epoch_frac * self.steps_in_epoch))
        valid_steps = [round(x) for x in np.linspace(0, repeat_after, num_steps, endpoint=False)]
        return ((step + 1) % repeat_after) in valid_steps

    @property
    def should_log(self):
        return (
            self.equispaced_steps(self.curr_step, 1 / self.config.log.logs_per_epoch)
            or self.is_last_step
        ) or self.is_last_step

    @property
    def should_ckpt(self):
        return (
            self.equispaced_steps(self.curr_step, self.config.log.epochs_per_ckpt)
            or self.is_last_step
        ) and not self.config.debug_mode

    @property
    def should_validate(self):
        return (
            self.equispaced_steps(self.curr_step, self.config.log.epochs_per_valid)
            or self.is_last_step
        )

    @property
    def lr(self):
        return item_if_arr(self.scheduler(self.curr_step))  # type: ignore

    def ckpt(self):
        """Checkpoint PyTree."""
        return Checkpoint(
            self.state, self.seed, dict(self.metrics_history), self.curr_step / self.steps_in_epoch
        )

    def save_final(self, out_dir: str | PathLike):
        """Save final model to directory."""
        if not self.config.debug_mode:
            self.mngr.wait_until_finished()
            copytree(self.mngr.directory, Path(out_dir) / 'ckpts/')

    def finish(self):
        if self.config.debug_mode:
            return Path('/dev/null')

        now = datetime.now()
        if self.config.log.exp_name is None:
            exp_name = now.strftime('%m-%d-%H-%M')
        else:
            exp_name = self.config.log.exp_name

        folder = Path('logs/') / f'{exp_name}_{self.seed}'
        folder.mkdir(exist_ok=True)

        pd.DataFrame(self.metrics_history).to_feather(folder / 'metrics.feather')

        with open(folder / 'time_stopped.txt', 'w') as f:
            f.write(now.isoformat())

        with open(folder / 'config.toml', 'w') as outfile:
            pyrallis.cfgparsing.dump(self.config, outfile)

        self.save_final(folder / 'final_ckpt')

        self.run['exp_name'] = exp_name

        zipname = make_archive(exp_name, 'zip', root_dir=folder)
        new_path = Path('logs') / f'{exp_name}.zip'
        shutil.move(zipname, new_path)
        self.run['checkpoint'].upload(str(new_path))

        self.run.stop()

        return folder
