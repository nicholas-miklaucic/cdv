"""Layers useful in different contexts."""

from typing import Callable, Literal, Optional, Sequence

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import jax.random as jr
from eins import EinsOp
from flax import linen as nn
from flax import struct
from jaxtyping import Array, Float

from facet.utils import tcheck

E3Irreps = e3nn.Irreps
E3IrrepsArray = e3nn.IrrepsArray


import yaml
from yaml import Node, SafeDumper


def represent_irreps(d: SafeDumper, e: E3Irreps) -> Node:
    return d.represent_str(str(e))


def represent_irrep_array(d: SafeDumper, e: E3IrrepsArray) -> Node:
    return d.represent_dict({'irreps': e.irreps, 'array': e.array})


yaml.SafeDumper.add_representer(E3Irreps, represent_irreps)
yaml.SafeDumper.add_representer(E3IrrepsArray, represent_irrep_array)


def edge_vecs(cg):
    """CG -> nodes k xyz"""
    # print(cg)
    send_pos = cg.nodes.cart[:, None, :]  # nodes 1 3
    # print(cg.graph_data.lat[cg.nodes.graph_i].shape, cg.edges.to_jimage.shape)
    offsets = EinsOp('nodes abc xyz, nodes k abc -> nodes k xyz')(
        cg.graph_data.lat[cg.nodes.graph_i], cg.edges.to_jimage
    )
    recv_pos = cg.nodes.cart[cg.receivers] + offsets  # nodes k 3
    vecs = recv_pos - send_pos
    return vecs


def dists(cg):
    """CG -> nodes k"""
    return jnp.linalg.norm(edge_vecs(cg), axis=-1)


class Context(struct.PyTreeNode):
    training: bool


SegmentReductionKind = Literal['max', 'min', 'prod', 'sum', 'mean']


def segment_mean(data, segment_ids, **kwargs):
    return jax.ops.segment_sum(data, segment_ids, **kwargs) / (
        1e-6 + jax.ops.segment_sum(jnp.ones_like(data), segment_ids, **kwargs)
    )


def segment_reduce(reduction: SegmentReductionKind, data, segment_ids, **kwargs):
    try:
        fn = getattr(jax.ops, f'segment_{reduction}')
    except AttributeError:
        if reduction == 'mean':
            fn = segment_mean
        else:
            raise ValueError('Cannot find reduction')

    return fn(data, segment_ids, **kwargs)


class SegmentReduction(nn.Module):
    reduction: SegmentReductionKind = 'sum'

    def __call__(self, data, segments, num_segments, ctx):
        return segment_reduce(self.reduction, data, segments, num_segments=num_segments)


class Identity(nn.Module):
    """Module that does nothing, used so model summaries accurately communicate the purpose."""

    @nn.compact
    def __call__(self, x):
        return x


class LazyInMLP(nn.Module):
    """Customizable MLP with input dimension inferred at runtime."""

    inner_dims: Sequence[int]
    out_dim: Optional[int] = None
    residual: bool = False
    inner_act: Callable = nn.silu
    final_act: Callable = Identity()
    dropout_rate: float = 0.0
    kernel_init: Callable = nn.initializers.glorot_normal()
    bias_init: Callable = nn.initializers.truncated_normal()
    normalization: Literal['layer', 'weight', 'dytanh', 'none'] = 'layer'
    use_bias: bool = True

    @tcheck
    @nn.compact
    def __call__(self, x: Float[Array, ' n_in'], ctx: Context):
        if isinstance(x, e3nn.IrrepsArray):
            if not x.irreps.is_scalar():
                raise ValueError('MLP only works on scalar (0e) input.')
            x = x.array
            output_irrepsarray = True
        else:
            output_irrepsarray = False

        orig_x = x
        _curr_dim = x.shape[-1]
        if self.out_dim is None:
            out_dim = _curr_dim
        else:
            out_dim = self.out_dim

        if self.normalization == 'weight':
            Dense = lambda *args, **kwargs: nn.WeightNorm(
                nn.Dense(
                    *args,
                    use_bias=self.use_bias,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    dtype=x.dtype,
                    **kwargs,
                )
            )
        else:
            Dense = lambda *args, **kwargs: nn.Dense(
                *args,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                dtype=x.dtype,
                **kwargs,
            )

        for next_dim in self.inner_dims:
            x = Dense(next_dim)(x)
            x = self.inner_act(x)
            x = nn.Dropout(self.dropout_rate, deterministic=not ctx.training)(x)
            if self.normalization == 'layer':
                x = nn.LayerNorm(dtype=x.dtype, use_bias=self.use_bias)(x)
            elif self.normalization == 'dytanh':
                x = DyTanh()(x)
            _curr_dim = next_dim

        x = Dense(out_dim)(x)
        if self.residual:
            if orig_x.shape[-1] != out_dim:
                x_resid = Dense(out_dim)(orig_x)
            else:
                x_resid = orig_x

            x = x + x_resid

        x = self.final_act(x)

        if output_irrepsarray:
            x = e3nn.IrrepsArray(e3nn.Irreps(f'{x.shape[-1]}x0e'), x)

        return x


class DyTanh(nn.Module):
    """Dynamic tanh layer used in "Transformers without Normalization":
    https://arxiv.org/abs/2503.10622"""

    alpha_init: float = 0.5
    scale_init: Callable = nn.initializers.ones

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Applies tanh and then scales/shifts."""
        alpha = self.param('alpha', nn.initializers.constant(self.alpha_init), (1,))
        gamma = self.param('gamma', self.scale_init, x.shape[-1:])
        beta = self.param('beta', nn.initializers.zeros, x.shape[-1:])

        return gamma * jnp.tanh(jnp.clip(alpha * x, -5, 5)) + beta


class E3NormNorm(nn.Module):
    eps: float = 1e-6

    def __call__(self, x: 'E3IrrepsArray') -> 'E3IrrepsArray':
        """
        Normalizes the norm of each irrep to be mean 1.
        """
        return x
        normed = []

        for chunk in x.chunks:
            if chunk is None:
                normed.append(None)
            else:
                norm = jnp.sum(jnp.conj(chunk) * chunk, axis=-1, keepdims=True)
                norm = jnp.nanmean(
                    jnp.where(norm < self.eps**2, jnp.nan, jnp.sqrt(norm + self.eps)), keepdims=True
                )
                normed.append(chunk / (norm + self.eps))

        return e3nn.from_chunks(
            x.irreps,
            normed,
            x.shape[:-1],
            x.dtype,
        )


class GaussianDropout(nn.Module):
    """A Gaussian dropout layer."""

    sd: float
    deterministic: Optional[bool] = None
    rng_collection: str = 'dropout'

    @nn.compact
    def __call__(
        self,
        inputs,
        deterministic: Optional[bool] = None,
        rng: Optional[jr.PRNGKey] = None,
    ):
        """Applies a random dropout mask to the input.

        Args:
          inputs: the inputs that should be randomly masked.
          deterministic: if false the inputs are scaled by ``1 / (1 - rate)`` and
            masked, whereas if true, no mask is applied and the inputs are returned
            as is.
          rng: an optional PRNGKey used as the random key, if not specified, one
            will be generated using ``make_rng`` with the ``rng_collection`` name.

        Returns:
          The masked inputs reweighted to preserve mean.
        """
        deterministic = nn.merge_param('deterministic', self.deterministic, deterministic)

        if (self.sd == 0.0) or deterministic:
            return inputs

        if rng is None:
            rng = self.make_rng(self.rng_collection)
        broadcast_shape = list(inputs.shape)
        eps = jr.normal(rng, shape=broadcast_shape, dtype=inputs.dtype) * self.sd
        return inputs + eps


def shifted_softplus(x):
    """Shifted softplus activation used for SevenNet weight neural network.
    Goes through the origin."""
    return jax.nn.softplus(x) - jnp.log(2.0)


def normalized_shifted_softplus(x):
    """Normalized version of shifted_softplus, such that the mean squared output
    for a random normal input is 1."""
    return e3nn.normalize_function(shifted_softplus)(x)


def normalized_silu(x):
    """Normalized version of silu, such that the mean squared output
    for a random normal input is 1."""
    return e3nn.normalize_function(jax.nn.silu)(x)


# def soft_envelope(length, max_length):
#     return e3nn.soft_envelope(
#         length,
#         max_length,
#         arg_multiplicator=envelope_arg_multiplicator,
#         value_at_origin=envelope_value_at_origin,
#     )

# def polynomial_envelope(length, max_length, degree0: int, degree1: int):
#     return e3nn.poly_envelope(degree0, degree1, max_length)(length)

# def u_envelope(length, max_length, p: int):
#     return e3nn.poly_envelope(p - 1, 2, max_length)(length)


class MixerBlock(nn.Module):
    """
    A Flax Module to act as the mixer block layer for the MLP-Mixer Architecture.

    Attributes:
        tokens_mlp_dim: MLP Block 1
        channels_mlp_dim: MLP Block 2
    """

    tokens_mlp: nn.Module
    channels_mlp: LazyInMLP
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self, inputs: Float[Array, 'batch seq chan'], abys: Float[Array, '6 chan'], ctx: Context
    ) -> Float[Array, 'batch seq chan']:
        training = ctx.training
        a1, b1, y1, a2, b2, y2 = abys
        x = nn.LayerNorm(scale_init=nn.zeros, dtype=inputs.dtype)(inputs)
        x = x * y1 + b1

        x = self.tokens_mlp(x, training=training)
        x = nn.Dropout(rate=self.attention_dropout_rate)(x, deterministic=not training)
        x = x * a1
        x = x + inputs

        y = nn.LayerNorm(dtype=x.dtype, scale_init=nn.zeros)(x)
        y = y * y2 + b2
        y = self.channels_mlp(y, training=training)
        y = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

        y = y * a2

        return x + y


class MLPMixer(nn.Module):
    """
    Flax Module for the MLP-Mixer Architecture.

    Attributes:
        patches: Patch configuration
        num_classes: No of classes for the output head
        num_blocks: No of Blocks of Mixers to use
        hidden_dim: No of Hidden Dimension for the Patch-Wise Convolution Layer
        tokens_mlp_dim: No of dimensions for the MLP Block 1
        channels_mlp_dim: No of dimensions for the MLP Block 2
        approximate: If True, uses the approximate formulation of GELU in each MLP Block
        dtype: the dtype of the computation (default: float32)
    """

    num_layers: int
    tokens_mlp: nn.Module
    channels_mlp: LazyInMLP
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, abys, *, ctx: Context) -> Array:
        x = inputs
        # Num Blocks x Mixer Blocks
        for _ in range(self.num_layers):
            x = MixerBlock(
                tokens_mlp=self.tokens_mlp.copy(),
                channels_mlp=self.channels_mlp.copy(),
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
            )(x, abys, ctx=ctx)
        # Output Head
        x = nn.LayerNorm(dtype=x.dtype, name='pre_head_layer_norm')(x)
        return x


class DeepSetEncoder(nn.Module):
    """Deep Sets with several types of pooling. Permutation-invariant encoder."""

    phi: nn.Module

    @nn.compact
    def __call__(
        self, x: Float[Array, 'batch token chan'], training: bool
    ) -> Float[Array, 'batch out_dim']:
        phi_x = self.phi(x, training=training)
        phi_x = EinsOp('batch token out_dim -> batch out_dim token')(phi_x)
        op = 'batch out_dim token -> batch out_dim'
        phi_x_mean = EinsOp(op, reduce='mean')(phi_x)
        phi_x_std = EinsOp(op, reduce='std')(phi_x)
        phi_x = jnp.concatenate([phi_x_mean, phi_x_std], axis=-1)
        normed = nn.LayerNorm(dtype=x.dtype)(phi_x)
        return normed


class PermInvariantEncoder(nn.Module):
    """A la Deep Sets, constructs a permutation-invariant representation of the inputs based on aggregations.
    Uses mean, std, and differentiable quantiles."""

    @nn.compact
    def __call__(
        self, x: Float[Array, 'batch token chan'], axis=-1, keepdims=True
    ) -> Float[Array, 'batch out_dim']:
        x = EinsOp('batch token chan -> batch chan token')(x)
        x_mean = jnp.mean(x, axis=axis, keepdims=keepdims)
        # x_std = jnp.std(x, axis=axis, keepdims=keepdims)

        # x_whiten = (x - x_mean) / (x_std + 1e-8)

        # x_quants = []
        # for power in jnp.linspace(1, 3, 6):
        #     x_quants.append(
        #         jnp.mean(
        #             jnp.sign(x_whiten) * jnp.abs(x_whiten**power),
        #             axis=-1,
        #             keepdims=True,
        #         )
        #         ** (1 / power),
        #     )

        # This has serious numerical stability problems in the backward pass. Instead, I'll use something else.
        # eps = 0.02
        # quants = jnp.linspace(eps, 1 - eps, 14, dtype=jnp.bfloat16)
        # from ott.tools.soft_sort import quantile
        # x_quants = quantile(x, quants, axis=-1, weight=10 / x.shape[-1])
        return jnp.concat([x_mean], axis=-1)
