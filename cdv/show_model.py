import subprocess
from inspect import signature

import jax
import jax.numpy as jnp
import pyrallis
from jax.lib import xla_client
from flax import linen as nn

from cdv.config import MainConfig
from cdv.dataset import dataloader, load_file
from cdv.layers import Context
from cdv.utils import debug_stat, debug_structure, flax_summary, intercept_stat
from cdv.vae import prop_loss


# https://bnikolic.co.uk/blog/python/jax/2022/02/22/jax-outputgraph-rev.html
def to_dot_graph(x):
    return xla_client._xla.hlo_module_to_dot_graph(xla_client._xla.hlo_module_from_text(x))


@pyrallis.argparsing.wrap()
def show_model(config: MainConfig, make_hlo_dot=False, do_profile=False):
    kwargs = dict(ctx=Context(training=True))
    num_batches, dl = dataloader(config, split='train', infinite=True)
    for i, b in zip(range(2), dl):
        batch = b

    debug_structure(batch=batch)

    if config.task == 'e_form':
        mod = config.build_regressor()
        rngs = {}
    elif config.task == 'vae':
        mod = config.build_vae()
        rngs = {'noise': jax.random.key(123)}
    elif config.task == 'diled':
        mod = config.build_diled()
        rngs = {'noise': jax.random.key(123), 'time': jax.random.key(234)}

    rngs['params'] = jax.random.key(0)
    rngs['dropout'] = jax.random.key(1)
    b1 = jax.tree_map(lambda x: x[0], batch)
    # for k, v in rngs.items():
    #     rngs[k] = jax.device_put(v, )

    params = mod.init(rngs, b1, **kwargs)
    params = jax.device_put_replicated(params, config.device.devices())

    with jax.debug_nans():
        out = jax.vmap(lambda p, b: mod.apply(p, b, rngs=rngs, **kwargs))(params, batch)

    # kwargs['cg'] = b1
    # print(params['params']['edge_proj']['kernel'].devices())
    debug_structure(module=mod, out=out)
    debug_stat(input=batch)
    rngs.pop('params')
    flax_summary(mod, rngs=rngs, cg=b1, **kwargs)

    debug_stat(out=out)
    rot_batch = jax.pmap(lambda x: x.rotate(123)[0])(batch)

    if True:
        rot_out = jax.pmap(lambda p, b: mod.apply(p, b, rngs=rngs, **kwargs))(params, rot_batch)
    else:
        with nn.intercept_methods(intercept_stat):
            rot_out = jax.pmap(lambda p, b: mod.apply(p, b, rngs=rngs, **kwargs))(params, rot_batch)

    if config.task == 'e_form':
        debug_stat(equiv_error=jnp.abs(rot_out - out))
    elif config.task == 'vae':
        debug_stat(equiv_error=jax.tree.map(lambda x, y: jnp.abs(x - y), rot_out, out))

    def loss(params):
        preds = jax.pmap(lambda p, b: mod.apply(p, b, rngs=rngs, ctx=Context(training=True)))(
            params, batch
        )
        if config.task == 'e_form':
            return {
                'loss': jax.pmap(config.train.loss.regression_loss)(
                    preds, batch.e_form[..., None], batch.padding_mask
                )
            }
        else:
            return preds

    if do_profile:
        with jax.profiler.trace('/tmp/jax-trace', create_perfetto_trace=True):
            val, grad = jax.value_and_grad(lambda x: jnp.mean(loss(x)['loss']))(params)
            jax.block_until_ready(grad)
    else:
        val, grad = jax.value_and_grad(lambda x: jnp.mean(loss(x)['loss']))(params)
    debug_stat(val=val, grad=grad)

    if not make_hlo_dot:
        return

    grad_loss = jax.xla_computation(jax.value_and_grad(loss))(params)
    with open('model.hlo', 'w') as f:
        f.write(grad_loss.as_hlo_text())
    with open('model.dot', 'w') as f:
        f.write(grad_loss.as_hlo_dot_graph())

    grad_loss_opt = jax.jit(jax.value_and_grad(loss)).lower(params).compile()
    with open('model_opt.hlo', 'w') as f:
        f.write(grad_loss_opt.as_text())
    with open('model_opt.dot', 'w') as f:
        f.write(to_dot_graph(grad_loss_opt.as_text()))

    # debug_structure(grad_loss_opt.cost_analysis())

    for f in ('model.dot', 'model_opt.dot'):
        subprocess.run(['sfdp', f, '-Tsvg', '-O', '-x'])


if __name__ == '__main__':
    show_model()
