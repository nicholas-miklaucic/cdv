"""Self-connection blocks for MACE."""

from typing import Set, Union

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array, Float

from facet.e3.activations import S2Activation
from facet.layers import Context, DyTanh, E3Irreps, E3IrrepsArray, LazyInMLP
from facet.mace.e3_layers import IrrepsModule, Linear


def safe_norm(x: jnp.ndarray, axis: int | None = None, keepdims=False) -> jnp.ndarray:
    """nan-safe norm."""
    x2 = jnp.sum(x**2, axis=axis, keepdims=keepdims)
    return jnp.where(x2 == 0.0, 0.0, jnp.where(x2 == 0, 1.0, x2) ** 0.5)


A025582 = [0, 1, 3, 7, 12, 20, 30, 44, 65, 80, 96, 122, 147, 181, 203, 251, 289]


reduced_symmetric_tensor_product_basis = e3nn.reduced_symmetric_tensor_product_basis
reduced_tensor_product_basis = e3nn.reduced_tensor_product_basis
# e3nn.reduced_symmetric_tensor_product_basis


class SelfConnectionBlock(IrrepsModule):
    """Block for node updates, combining species and environment information."""

    @nn.nowrap
    def irreps_in(self) -> E3Irreps:
        """Gets the irreps that should be input in order to get the right output."""
        return self.ir_out

    @nn.compact
    def __call__(
        self,
        node_feats: E3IrrepsArray,  # [n_nodes, feature * irreps]
        node_specie: jnp.ndarray,  # [n_nodes, ] int
        species_embed: Float[Array, 'num_species embed_dim'],
        ctx: Context,
    ):
        raise NotImplementedError


class SymmetricContraction(nn.Module):
    correlation: int
    keep_irrep_out: Union[Set[e3nn.Irrep], str]
    num_species: int
    symmetric_tensor_product_basis: bool = True
    off_diagonal: bool = False

    @nn.compact
    def __call__(
        self,
        input: E3IrrepsArray,  # n_nodes, feats, irreps
        index: jnp.ndarray,
        species_embed: Float[Array, 'num_species embed_dim'],
        ctx: Context,
    ) -> E3IrrepsArray:
        if isinstance(self.keep_irrep_out, str):
            keep_irrep_out = E3Irreps(self.keep_irrep_out)
            assert all(mul == 1 for mul, _ in keep_irrep_out)
        else:
            keep_irrep_out = self.keep_irrep_out

        keep_irrep_out_set = {e3nn.Irrep(ir) for ir in keep_irrep_out}

        W = {}

        # Treat batch indices using vmap
        shape = jnp.broadcast_shapes(input.shape[:-2], index.shape)
        input = input.broadcast_to(shape + input.shape[-2:])
        index = jnp.broadcast_to(index, shape)

        num_rbf = 32

        use_half = True
        dtype = jnp.bfloat16 if use_half else jnp.float32

        if species_embed is None:
            species_embed_mod = nn.Embed(
                self.num_species, num_rbf, name='species_embed', param_dtype=dtype
            )
            # species_ind = index
            species_ind = species_embed_mod(index)
        else:
            species_embed_mlp = LazyInMLP(
                [], out_dim=num_rbf, name='species_radial_mlp', normalization='layer'
            )
            species_embed = species_embed_mlp(
                species_embed.astype(jnp.bfloat16) if use_half else species_embed, ctx=ctx
            )
            species_ind = species_embed[index]

        # print(input.shape, index.shape)

        for order in range(self.correlation, 0, -1):  # correlation, ..., 1
            if self.symmetric_tensor_product_basis:
                U = reduced_symmetric_tensor_product_basis(
                    input.irreps, order, keep_ir=keep_irrep_out_set
                )
            else:
                U = reduced_tensor_product_basis([input.irreps] * order, keep_ir=keep_irrep_out_set)
            # U = U / order  # normalization TODO(mario): put back after testing
            # NOTE(mario): The normalization constants (/order and /mul**0.5)
            # has been numerically checked to be correct.

            # TODO(mario) implement norm_p

            # ((w3 x + w2) x + w1) x
            #  \-----------/
            #       out

            for (mul, ir_out), u in zip(U.irreps, U.list):
                name = f'w{order}_{ir_out}'
                W[name] = self.param(
                    name,
                    nn.initializers.normal(
                        stddev=1, dtype=jnp.bfloat16 if use_half else jnp.float32
                    ),
                    (num_rbf, mul, input.shape[-2]),
                )

        # - This operation is parallel on the feature dimension (but each feature has its own parameters)
        # This operation is an efficient implementation of
        # vmap(lambda w, x: FunctionalLinear(irreps_out)(w, concatenate([x, tensor_product(x, x), tensor_product(x, x, x), ...])))(w, x)
        # up to x power self.correlation
        # assert input.ndim == 2  # [num_features, irreps_x.dim]
        # assert index.ndim == 0  # int

        out = dict()

        for order in range(self.correlation, 0, -1):  # correlation, ..., 1
            if self.off_diagonal:
                roll = lambda arr: jnp.roll(arr, A025582[order - 1])
                x_ = jax.vmap(roll)(input.array)
            else:
                x_ = input.array

            if self.symmetric_tensor_product_basis:
                U = reduced_symmetric_tensor_product_basis(
                    input.irreps, order, keep_ir=keep_irrep_out_set
                )
            else:
                U = reduced_tensor_product_basis([input.irreps] * order, keep_ir=keep_irrep_out_set)
            U = U / order  # normalization

            # ((w3 x + w2) x + w1) x
            #  \-----------/
            #       out

            einsum_kwargs = {
                'precision': jax.lax.Precision.DEFAULT,
                'preferred_element_type': jnp.bfloat16 if use_half else jnp.float32,
            }
            if use_half:
                x_ = x_.astype(jnp.bfloat16)

            for (mul, ir_out), u in zip(U.irreps, U.list):
                u = u.astype(x_.dtype)
                # u: ndarray [(irreps_x.dim)^order, multiplicity, ir_out.dim]
                # print(self)
                name = f'w{order}_{ir_out}'
                # this doesn't actually fix the problem: the distribution is still heavy-tailed,
                # it's just that now everything is 0 instead of having a few really large values.

                # W_normed = W[name] / jnp.sum(jnp.ones_like(W[name][0]))
                W_normed = W[name]
                w = jnp.einsum('be,e...->b...', species_ind, W_normed, **einsum_kwargs)

                # w = W[name][species]

                # [multiplicity, num_features]

                if ir_out not in out:
                    # debug_structure(u=u, w=w, x=x_)
                    out[ir_out] = (
                        'special',
                        jnp.einsum('...jki,bkc,bcj->bc...i', u, w, x_, **einsum_kwargs),
                    )  # [num_features, (irreps_x.dim)^(oder-1), ir_out.dim]
                else:
                    out[ir_out] += jnp.einsum(
                        '...ki,bkc->bc...i', u, w, **einsum_kwargs
                    )  # [num_features, (irreps_x.dim)^order, ir_out.dim]

            # ((w3 x + w2) x + w1) x
            #  \----------------/
            #         out (in the normal case)

            for ir_out in out:
                if isinstance(out[ir_out], tuple):
                    out[ir_out] = out[ir_out][1]
                    continue  # already done (special case optimization above)

                out[ir_out] = jnp.einsum(
                    'bc...ji,bcj->bc...i', out[ir_out], x_, **einsum_kwargs
                )  # [num_features, (irreps_x.dim)^(oder-1), ir_out.dim]

            # ((w3 x + w2) x + w1) x
            #  \-------------------/
            #           out

        # out[irrep_out] : [num_features, ir_out.dim]
        irreps_out = E3Irreps(sorted(out.keys()))
        # for k, v in out.items():
        #     debug_structure(**{str(k): v})
        return E3IrrepsArray.from_list(
            irreps_out,
            [out[ir][..., None, :] for (_, ir) in irreps_out],
            input.shape[:-1],
        )


class EquivariantProductBasisBlock(SelfConnectionBlock):
    correlation: int
    num_species: int
    symmetric_tensor_product_basis: bool = True
    off_diagonal: bool = False

    @nn.compact
    def __call__(
        self,
        node_feats: E3IrrepsArray,  # [n_nodes, feature * irreps]
        node_specie: jnp.ndarray,  # [n_nodes, ] int
        species_embed: Float[Array, 'num_species embed_dim'],
        ctx: Context,
    ) -> E3IrrepsArray:
        node_feats = node_feats.mul_to_axis().remove_nones()
        node_feats = SymmetricContraction(
            keep_irrep_out={ir for _, ir in self.ir_out},
            correlation=self.correlation,
            num_species=self.num_species,
            symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
            off_diagonal=self.off_diagonal,
        )(node_feats, node_specie, ctx=ctx, species_embed=species_embed)
        node_feats = node_feats.axis_to_mul()
        return Linear(self.irreps_out, name='proj_out')(node_feats)


class LinearSelfConnection(SelfConnectionBlock):
    """Linear layer update, as opposed to the tensor product formulation. From SevenNet."""

    @nn.compact
    def __call__(
        self,
        node_feats: E3IrrepsArray,  # [n_nodes, feature * irreps]
        node_specie: jnp.ndarray,  # [n_nodes, ] int
        species_embed: Float[Array, 'num_species embed_dim'],
        ctx: Context,
    ):
        linear_out = Linear(self.irreps_out)
        return linear_out(node_feats)


class GateSelfConnection(SelfConnectionBlock):
    @nn.nowrap
    def irreps_in(self) -> E3Irreps:
        non_scalars = self.ir_out.filter(drop=['0e', '0o']).num_irreps
        return (self.ir_out + E3Irreps(f'{non_scalars}x0e')).sort().irreps.simplify()

    @nn.compact
    def __call__(
        self,
        node_feats: E3IrrepsArray,  # [n_nodes, feature * irreps]
        node_specie: jnp.ndarray,  # [n_nodes, ] int
        species_embed: Float[Array, 'num_species embed_dim'],
        ctx: Context,
    ):
        return e3nn.gate(
            node_feats,
            even_act=jax.nn.silu,
            even_gate_act=jax.nn.silu,
            odd_act=jnp.tanh,
            odd_gate_act=jnp.tanh,
            normalize_act=True,
        )


class S2SelfConnection(SelfConnectionBlock):
    """
    Combines inputs as functions on the sphere, applies a pointwise nonlinearity, and then
    transforms back. Uses a linear layer on either side to mix channels.
    """

    act: S2Activation
    mlp: LazyInMLP
    num_heads: int
    dytanh_prenorm: bool

    @nn.compact
    def __call__(
        self,
        node_feats: E3IrrepsArray,  # [n_nodes, feature * irreps]
        node_specie: jnp.ndarray,  # [n_nodes, ] int
        species_embed: Float[Array, 'num_species embed_dim'],
        ctx: Context,
    ):
        # we can only project S2 activations to a single output irrep in each dimension. We need to
        # project so we have that many distinct irreps to do this with.
        if node_feats.irreps.lmax == 0:
            # return self.mlp.copy(inner_dims=[node_feats.shape[-1]])(node_feats, ctx=ctx)
            return self.mlp.copy(inner_dims=[node_feats.shape[-1]] * 2)(node_feats, ctx=ctx)

        up_mul = max([mul for mul, _ir in node_feats.filter(drop=['0e', '0o']).irreps])
        up_ir = E3Irreps([(up_mul, ir) for _mul, ir in node_feats.irreps])

        node_up = Linear(up_ir, name='proj_up')(node_feats).mul_to_axis()
        act = self.act.copy(activation=None, name='mix')
        signal = act.input_signal(node_up, ctx=ctx)
        # act: *batches up beta alpha
        # reshape so up is channel axis and get number of heads
        vals = signal.grid_values
        *batch, up, beta, alpha = vals.shape
        vals = jnp.moveaxis(vals, -3, -1)
        vals = vals.reshape(*batch, beta, alpha, self.num_heads, -1)

        if self.dytanh_prenorm:
            vals = DyTanh()(vals)

        # now apply MLP
        vals = self.mlp(vals, ctx=ctx)

        # end with activation for same reason
        # keep this one unbounded
        # vals = vals * jax.nn.tanh(vals)

        # now move back
        vals = vals.reshape(*batch, beta, alpha, up)
        vals = jnp.moveaxis(vals, -1, -3)

        # get back to spherical signal
        signal = signal.replace_values(vals)

        mix_out = act.output_irreps(signal, node_up).axis_to_mul()

        mix_out = e3nn.concatenate([node_feats.filter('0e'), mix_out], axis=-1)

        node_down = Linear(self.ir_out, name='proj_down')(mix_out)

        return node_down


class MLPSelfGate(SelfConnectionBlock):
    """
    Calculates all potential scalars from the tensor square,
    applies an MLP, and then computes the tensor product.
    """

    mlp_templ: LazyInMLP
    num_hidden_layers: int = 1

    @nn.compact
    def __call__(
        self,
        node_feats: E3IrrepsArray,  # [n_nodes, feature * irreps]
        node_specie: jnp.ndarray,  # [n_nodes, ] int
        species_embed: Float[Array, 'num_species embed_dim'],
        ctx: Context,
    ):
        if node_feats.irreps.lmax == 0:
            # all scalars: no need to gate, just straight MLP
            size = node_feats.irreps.num_irreps
            mlp = self.mlp_templ.copy(
                inner_dims=[size] * self.num_hidden_layers, out_dim=size, name='self_mlp'
            )
            pre_out = mlp(node_feats.array, ctx=ctx)
        else:
            scalars = e3nn.tensor_product(node_feats, node_feats, filter_ir_out=['0e'])
            x = jnp.concat((scalars.array, species_embed), axis=-1)
            out_dim = node_feats.filter(drop='0e').irreps.num_irreps
            mlp = self.mlp_templ.copy(
                inner_dims=[out_dim] * self.num_hidden_layers, out_dim=out_dim, name='self_gate'
            )
            y = E3IrrepsArray(f'{out_dim}x0e', mlp(x, ctx=ctx))
            z = e3nn.concatenate((node_feats, y))
            z = e3nn.gate(z)
            pre_out = z

        return Linear(self.ir_out)(pre_out)
