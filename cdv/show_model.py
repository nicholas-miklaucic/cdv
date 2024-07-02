import subprocess

import jax
import jax.numpy as jnp
import pyrallis
from jax.lib import xla_client

from cdv.config import MainConfig
from cdv.dataset import dataloader, load_file
from cdv.layers import Context
from cdv.utils import debug_stat, debug_structure, flax_summary


# https://bnikolic.co.uk/blog/python/jax/2022/02/22/jax-outputgraph-rev.html
def to_dot_graph(x):
    return xla_client._xla.hlo_module_to_dot_graph(xla_client._xla.hlo_module_from_text(x))


@pyrallis.argparsing.wrap()
def show_model(config: MainConfig, make_hlo_dot=False):
    kwargs = dict(ctx=Context(training=True))
    num_batches, dl = dataloader(config, split='train')
    for i, b in zip(range(3), dl):
        batch = b

    if config.task == 'e_form':
        mod = config.build_regressor()
        enc_batch = {'cg': batch}
        rngs = {}
    elif config.task == 'diled':
        mod = config.build_diled()
        enc_batch = {
            'cg': batch,
        }
        rngs = {'noise': jax.random.key(123), 'time': jax.random.key(234)}
    kwargs.update(enc_batch)
    out, params = mod.init_with_output(dict(params=jax.random.key(0), **rngs), **kwargs)
    debug_structure(module=mod, out=out)
    debug_stat(input=batch, out=out)
    flax_summary(mod, rngs=rngs, **kwargs)

    def loss(params):
        preds = mod.apply(params, batch, rngs=rngs, ctx=Context(training=True))
        if config.task == 'e_form':
            return {'loss': config.train.loss.regression_loss(preds, batch.graph_data.e_form.reshape(-1, 1), batch.padding_mask)}
        else:
            return preds

    debug_stat(grad=jax.grad(lambda x: jnp.mean(loss(x)['loss']))(params))

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
    # with jax.log_compiles():
    show_model()
