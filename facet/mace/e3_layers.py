"""Generic layers useful for dealing with irreps of E(3)."""

from collections import defaultdict
from typing import Callable, Literal

import e3nn_jax as e3nn
import jax.numpy as jnp
from flax import linen as nn

from facet.layers import Context, DyTanh, E3Irreps, E3IrrepsArray


def Linear(*args, **kwargs):
    # return nn.WeightNorm(e3nn.flax.Linear(*args, **kwargs))
    return e3nn.flax.Linear(*args, **kwargs)


class IrrepsModule(nn.Module):
    """Module that outputs irreps."""

    irreps_out: E3Irreps

    @property
    def ir_out(self) -> E3Irreps:
        return E3Irreps(self.irreps_out)


class IrrepsAdapter(IrrepsModule):
    """Generic module that can convert between irreps."""


class LinearAdapter(IrrepsAdapter):
    """Simple linear layer."""

    @nn.compact
    def __call__(self, x: E3IrrepsArray, ctx: Context) -> E3IrrepsArray:
        """batch irreps -> batch irreps_out"""
        # x = [n_nodes, irreps]
        return Linear(self.ir_out)(x)


class ResidualAdapter(IrrepsAdapter):
    """Simple residual adapter: pads with zeros and truncates instead of using any trained weights."""

    @nn.compact
    def __call__(self, x: E3IrrepsArray, ctx: Context) -> E3IrrepsArray:
        """batch irreps -> batch irreps_out"""
        x = x.simplify()
        out_chunks = []
        for out_mul, out_ir in self.ir_out:
            added = False
            for (in_mul, in_ir), chunk in zip(x.irreps, x.chunks):
                if added:
                    continue

                if in_ir == out_ir:
                    added = True
                    if in_mul == out_mul:  # no change needed
                        out_chunks.append(chunk)
                    elif in_mul < out_mul:  # pad with zeros
                        pad_len = out_mul - in_mul
                        pad_shape = list(chunk.shape)
                        pad_shape[-2] = pad_len
                        pad = jnp.zeros(pad_shape, dtype=chunk.dtype)
                        out_chunks.append(jnp.concat((chunk, pad), axis=-2))
                    else:  # truncate
                        out_chunks.append(chunk[..., :out_mul, :])

            if not added:
                out_chunks.append(None)
        return e3nn.from_chunks(self.ir_out, out_chunks, leading_shape=x.shape[:-1])


class ResidualLinearAdapter(IrrepsAdapter):
    """Linear layers between irreps of the same type and different multiplicities."""

    @nn.compact
    def __call__(self, x: E3IrrepsArray, ctx: Context) -> E3IrrepsArray:
        """batch irreps -> batch irreps_out"""
        x = x.simplify()
        out_chunks = []
        for out_mul, out_ir in self.ir_out:
            added = False
            for (in_mul, in_ir), chunk in zip(x.irreps, x.chunks):
                if added:
                    continue

                if in_ir == out_ir:
                    added = True
                    if in_mul == out_mul:
                        out_chunks.append(chunk)
                    else:
                        # -1 is irreps axis, which isn't changing
                        # -2 is channel axis
                        lin = nn.DenseGeneral(
                            out_mul, axis=-2, name=f'{in_mul}->{out_mul}_{out_ir}', use_bias=False
                        )
                        # debug_structure(chunk=chunk, lin=jnp.swapaxes(lin(chunk), -1, -2))
                        out_chunks.append(jnp.swapaxes(lin(chunk), -1, -2))

            if not added:
                out_chunks.append(None)
        return e3nn.from_chunks(self.ir_out, out_chunks, leading_shape=x.shape[:-1])


class NonlinearAdapter(IrrepsAdapter):
    """Nonlinear readout block: linear layer, gated nonlinearity, then a second linear layer."""

    hidden_irreps: E3Irreps
    activation: Callable = nn.silu
    gate: Callable = nn.sigmoid

    @nn.compact
    def __call__(self, x: E3IrrepsArray, ctx: Context) -> E3IrrepsArray:
        """batch irreps -> batch irreps_out"""
        # x = [n_nodes, irreps]
        hidden_irreps = E3Irreps(self.hidden_irreps)
        num_vectors = hidden_irreps.filter(
            drop=['0e', '0o']
        ).num_irreps  # Multiplicity of (l > 0) irreps
        # print(x.irreps)
        x = Linear(
            (hidden_irreps + E3Irreps(f'{num_vectors}x0e')).simplify(),
        )(x)
        # print((hidden_irreps + E3Irreps(f'{num_vectors}x0e')).simplify())
        # print(x.irreps)
        x = e3nn.gate(x, even_act=self.activation, even_gate_act=self.gate)
        return Linear(self.ir_out)(x)

    # [n_nodes, output_irreps]


class E3LayerNorm(nn.Module):
    """
    Layer norm modeled after Equiformer v3.

    https://arxiv.org/pdf/2306.12059

    Can separate scalars or everything.
    """

    separation: Literal['scalars', 'all-separate']
    scale_init: nn.initializers.Initializer = nn.ones
    learned_scale: bool = True
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: E3IrrepsArray, ctx: Context) -> E3IrrepsArray:
        groups = defaultdict(list)

        def get_group(ir: e3nn.Irrep):
            if self.separation == 'scalars':
                return 0 if ir.l == 0 else 1
            elif self.separation == 'all-separate':
                return ir.l

        for (mul, ir), chunk in zip(x.irreps, x.chunks):
            groups[get_group(ir)].append(chunk)

        data = {
            k: jnp.concat([v.reshape(*x.shape[:-1], -1) for v in vs], axis=-1)
            for k, vs in groups.items()
        }
        scales = {}
        shifts = {}

        # Broadcasting happens over both channel and irrep component.
        for group, values in data.items():
            # debug_structure(values=values)
            if group == 0:
                shifts[group] = values.mean(axis=-1)
            else:
                shifts[group] = jnp.array(0.0, dtype=x.dtype)

            scales[group] = jnp.sqrt(
                jnp.square(values - shifts[group][..., None]).mean(axis=-1) + self.eps
            )

        if self.learned_scale:
            learned_scale = self.param('ln_scale', self.scale_init, (x.irreps.num_irreps,))
        else:
            learned_scale = jnp.array(1.0, dtype=x.dtype)

        out_chunks = []
        for (mul, ir), chunk in zip(x.irreps, x.chunks):
            group = get_group(ir)
            # debug_structure(chunk=chunk, shift=shifts[group], scale=scales[group])
            out_chunks.append(
                (chunk - shifts[group][..., None, None])
                / (scales[group][..., None, None] + self.eps)
            )

        out = e3nn.from_chunks(x.irreps, out_chunks, leading_shape=x.shape[:-1])
        return out * learned_scale


class E3DyTanh(nn.Module):
    """E3-equivariant version of DyTanh layer, operating on norms."""

    inner: DyTanh

    @nn.compact
    def __call__(self, x: E3IrrepsArray, ctx: Context):
        norm = e3nn.norm(x).array
        new_norm = self.inner(norm)
        return x * (new_norm / (norm + 1e-6))


class E3SoftNorm(nn.Module):
    @nn.compact
    def __call__(self, x: E3IrrepsArray, ctx: Context):
        norm = e3nn.norm(x).array
        return x * (jnp.log1p(norm) / (norm + 1e-6))
