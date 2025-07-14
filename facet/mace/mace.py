"""
MACE network code. Adapted from https://github.com/ACEsuit/mace-jax.
"""

from collections.abc import Sequence
from typing import Callable, Literal

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from eins import EinsOp
from flax import linen as nn
from jaxtyping import Array, Float

from facet.data.databatch import CrystalGraphs
from facet.data.metadata import DatasetMetadata
from facet.layers import (
    Context,
    DyTanh,
    E3Irreps,
    E3IrrepsArray,
    LazyInMLP,
    SegmentReduction,
    edge_vecs,
)
from facet.mace.e3_layers import (
    E3DyTanh,
    E3LayerNorm,
    IrrepsModule,
    ResidualAdapter,
)
from facet.mace.edge_embedding import RadialEmbeddingBlock
from facet.mace.message_passing import ResidualInteraction, SimpleInteraction
from facet.mace.node_embedding import NodeEmbedding
from facet.mace.self_connection import (
    SelfConnectionBlock,
)
from facet.utils import get_or_init


def safe_norm(x: jnp.ndarray, axis: int | None = None, keepdims=False) -> jnp.ndarray:
    """nan-safe norm."""
    x2 = jnp.sum(x**2, axis=axis, keepdims=keepdims)
    return jnp.where(x2 == 0.0, 0.0, jnp.where(x2 == 0, 1.0, x2) ** 0.5)


class SevenNetRescale(nn.Module):
    metadata: DatasetMetadata

    def setup(self):
        self.scale = self.param('scale', nn.ones, (self.metadata.num_elements,))
        self.shift = self.param('shift', nn.zeros, (self.metadata.num_elements,))
        self.reduction = SegmentReduction('mean')

    def __call__(self, cg: CrystalGraphs, energies: Float[Array, ' nodes'], ctx: Context):
        # debug_structure(
        #     energies=energies, species=node_species, scale=self.metadata.atomwise_scale_energy
        # )
        # print(jnp.unique(cg.nodes.species, return_counts=True))
        energies = energies * self.scale[cg.nodes.species] + self.shift[cg.nodes.species]
        graph_energies = self.reduction(energies, cg.nodes.graph_i, cg.n_total_graphs, ctx=ctx)
        # debug_stat(scale=scale, shift=shift, energies=graph_energies)
        scaled_outs = graph_energies

        return scaled_outs[..., None]


class SpeciesWiseRescale(nn.Module):
    metadata: DatasetMetadata
    scale_trainable: bool
    shift_trainable: bool
    global_scale_trainable: bool
    global_shift_trainable: bool

    def setup(self):
        self.scale = get_or_init(
            self,
            'scale',
            jnp.array(self.metadata.atomwise_scale_energy),
            self.scale_trainable,
        )
        self.shift = get_or_init(
            self,
            'shift',
            jnp.array(self.metadata.atomwise_shift_energy),
            self.shift_trainable,
        )
        self.global_scale = get_or_init(
            self,
            'global_scale',
            jnp.array([self.metadata.scale_energy]),
            self.global_scale_trainable,
        )
        self.global_shift = get_or_init(
            self,
            'global_shift',
            jnp.array([self.metadata.shift_energy]),
            self.global_shift_trainable,
        )
        self.reduction = SegmentReduction('mean')

    def __call__(self, cg: CrystalGraphs, energies: Float[Array, ' nodes'], ctx: Context):
        # debug_structure(
        #     energies=energies, species=node_species, scale=self.metadata.atomwise_scale_energy
        # )
        # print(jnp.unique(cg.nodes.species, return_counts=True))
        num_atoms = jnp.clip(cg.n_node, 1.0, None)
        scale = jax.ops.segment_sum(
            jnp.take(self.scale, cg.nodes.species), cg.nodes.graph_i, cg.n_total_graphs
        )
        softplus_intercept = jnp.log(jnp.e - 1)
        transform = lambda x: jax.nn.softplus(x + softplus_intercept)
        scale = transform(scale) / num_atoms * transform(self.global_scale)
        shift = jax.ops.segment_sum(
            jnp.take(self.shift, cg.nodes.species), cg.nodes.graph_i, cg.n_total_graphs
        )
        shift = shift / num_atoms + self.global_shift

        graph_energies = self.reduction(energies, cg.nodes.graph_i, cg.n_total_graphs, ctx=ctx)
        # debug_stat(scale=scale, shift=shift, energies=graph_energies)
        scaled_outs = graph_energies * scale + shift

        return scaled_outs[..., None]


class MACELayer(nn.Module):
    interaction: SimpleInteraction
    self_connection: SelfConnectionBlock
    readout: IrrepsModule | None
    residual: bool
    resid_init: Callable
    resid_norm: str
    norm: nn.Module | None

    @nn.compact
    def __call__(
        self,
        vectors: E3IrrepsArray,  # [n_edges, 3]
        node_feats: E3IrrepsArray,  # [n_nodes, irreps]
        node_species: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        radial_embedding: jnp.ndarray,  # [n_edges, radial_embedding_dim]
        receivers: jnp.ndarray,  # [n_edges]
        species_embed: Float[Array, 'num_species num_embed'] | None,
        avg_num_neighbors: Float[Array, '1'],
        ctx: Context,
    ):
        """-> (n_nodes output_irreps, n_nodes features*hidden_irreps)"""
        x = self.interaction(
            vectors, node_feats, radial_embedding, receivers, avg_num_neighbors, ctx=ctx
        )

        x = self.self_connection(x, node_species, species_embed, ctx)

        if self.residual:
            if self.resid_norm == 'layer':
                x = E3LayerNorm(
                    separation='scalars',
                    scale_init=self.resid_init,
                    learned_scale=True,
                    name='resid_ln',
                )(x, ctx)
            elif self.resid_norm == 'dytanh':
                x = E3DyTanh(DyTanh(scale_init=self.resid_init, name='resid_dytanh'))(x, ctx)
            elif self.resid_norm == 'identity':
                pass
            else:
                raise ValueError(f'Unknown residual norm kind {self.resid_norm}')
            # resid = ResidualLinearAdapter(x.irreps)(node_feats, ctx=ctx)
            resid = ResidualAdapter(x.irreps)(node_feats, ctx=ctx)
            x = x + resid
            # x = E3DyTanh(DyTanh(scale_init=nn.initializers.ones, name='resid_dytanh'))(x, ctx)

        if self.norm is not None:
            # norm_mod = E3LayerNorm(
            #     separation='scalars', scale_init=nn.initializers.ones, name='layer_norm'
            # )
            norm_mod = self.norm.copy()
            x = norm_mod(x, ctx=ctx)

        if self.readout is not None:
            readout = self.readout(x, ctx=ctx)
            return x, readout
        else:
            return x, None


class MACE(IrrepsModule):
    hidden_irreps: Sequence[E3Irreps]
    node_embedding: NodeEmbedding
    radial_embedding: RadialEmbeddingBlock
    interaction_templ: SimpleInteraction
    self_connection_templ: SelfConnectionBlock
    readout_templ: IrrepsModule
    only_last_readout: bool
    share_species_embed: bool
    residual: bool
    resid_norm: str
    resid_init: Callable
    dataset_metadata: DatasetMetadata
    norm: nn.Module | None

    def setup(self):
        layers = []
        for i, ir_out in enumerate(list(self.hidden_irreps) + [self.ir_out]):
            ir_out = E3Irreps(ir_out)
            last = i == len(self.hidden_irreps)

            self_connection = self.self_connection_templ.copy(irreps_out=ir_out)

            # jax.debug.print(
            #     'irreps = {}, true_irreps = {}',
            #     self_connection.irreps_in(),
            #     self_connection.ir_out,
            # )
            interaction = self.interaction_templ.copy(irreps_out=self_connection.irreps_in())

            if last or not self.only_last_readout:
                readout = self.readout_templ.copy(irreps_out=self.ir_out)
            else:
                readout = None

            layers.append(
                MACELayer(
                    interaction=interaction,
                    self_connection=self_connection,
                    readout=readout,
                    residual=self.residual,
                    resid_init=self.resid_init,
                    resid_norm=self.resid_norm,
                    name=f'layer_{i}',
                    norm=self.norm,
                )
            )

        self.layers = layers

    def __call__(
        self,
        vectors: E3IrrepsArray,  # [n_nodes, k, 3]
        node_species: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        receivers: jnp.ndarray,  # [n_nodes, k]
        ctx: Context,
    ) -> E3IrrepsArray:
        # Embeddings
        node_feats = self.node_embedding(node_species, ctx=ctx).astype(
            vectors.dtype
        )  # [n_nodes, feature * irreps]
        species_embs = node_feats

        if not (hasattr(vectors, 'irreps') and hasattr(vectors, 'array')):
            vectors = E3IrrepsArray('1e', -vectors)

        radial_embedding = self.radial_embedding(safe_norm(vectors.array, axis=-1), ctx=ctx)
        radial_embedding = radial_embedding.astype(vectors.dtype)
        # debug_structure(radial_embedding=radial_embedding)
        avg_num_neighbors = self.radial_embedding.avg_num_neighbors(self.dataset_metadata)

        # vector_shs = e3nn.spherical_harmonics(self.interaction_templ.conv.max_ell, vectors, True)
        if isinstance(self.interaction_templ, ResidualInteraction):
            interact = self.interaction_templ.interaction
        else:
            interact = self.interaction_templ
        vector_shs = e3nn.spherical_harmonics(
            e3nn.Irreps(' + '.join([f'{ell}e' for ell in range(0, interact.conv.max_ell + 1)])),
            vectors,
            normalize=True,
            normalization='component',
        ).transform_by_matrix(jnp.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]))

        outputs = []
        for i, layer in enumerate(self.layers):
            node_feats, readout = layer(
                vector_shs,
                node_feats,
                node_species,
                radial_embedding,
                receivers,
                species_embed=species_embs.array if self.share_species_embed else None,
                avg_num_neighbors=avg_num_neighbors,
                ctx=ctx,
            )
            if readout is not None:
                outputs.append(readout)

        # print([k.shape for k in outputs])
        return e3nn.stack(outputs, axis=-2)  # [n_nodes, num_interactions, output_irreps]


class MaceModel(nn.Module):
    """Graph network that wraps MACE for invariant regression."""

    hidden_irreps: Sequence[str]

    node_embedding: NodeEmbedding
    edge_embedding: RadialEmbeddingBlock
    interaction: SimpleInteraction
    self_connection: SelfConnectionBlock
    readout: IrrepsModule
    head_templ: LazyInMLP
    head_adapter_templ: LazyInMLP | None
    residual: bool
    resid_norm: str
    resid_init: Callable
    rescale: nn.Module
    dataset_metadata: DatasetMetadata
    norm: nn.Module | None

    precision: Literal['f32', 'bf16']
    outs_per_node: int
    share_species_embed: bool = True

    # How to combine the outputs of different interaction blocks.
    # 'last' is special: it means the last block.
    block_reduction: str = 'last'

    def setup(self):
        self.mace = MACE(
            irreps_out=f'{self.outs_per_node}x0e',
            hidden_irreps=self.hidden_irreps,
            node_embedding=self.node_embedding,
            radial_embedding=self.edge_embedding,
            interaction_templ=self.interaction,
            self_connection_templ=self.self_connection,
            readout_templ=self.readout,
            only_last_readout=self.block_reduction == 'last',
            share_species_embed=self.share_species_embed,
            residual=self.residual,
            resid_init=self.resid_init,
            dataset_metadata=self.dataset_metadata,
            norm=self.norm,
            resid_norm=self.resid_norm,
        )

        self.head = self.head_templ.copy(out_dim=1, name='head')
        if self.head_adapter_templ is None:
            self.head_adapter = None
        else:
            self.head_adapter = self.head_adapter_templ.copy(
                out_dim=1, name='head_adapter', kernel_init=nn.initializers.normal(stddev=1e-3)
            )

        self.dtype = jnp.float32 if self.precision == 'f32' else jnp.bfloat16
        self.head_dropout = nn.Dropout(self.head_templ.dropout_rate)

    def head_outputs(self, cg: CrystalGraphs, ctx: Context) -> Float[Array, ' graphs head']:
        vecs = edge_vecs(cg).astype(self.dtype)

        # shape [n_nodes, n_interactions, output_irreps]
        mace_out = self.mace(
            vecs,
            cg.nodes.species,
            cg.receivers,
            ctx=ctx,
        ).array

        if self.block_reduction == 'last':
            out = mace_out[..., -1, :]
        else:
            out = EinsOp('nodes blocks mul -> nodes mul', reduce=self.block_reduction)(mace_out)

        return out

    def energy_head(self, out, cg: CrystalGraphs, ctx: Context):
        out = self.head_dropout(out, deterministic=not ctx.training)
        mlp_out = self.head(out, ctx=ctx)[..., 0]
        if self.head_adapter is None:
            mlp_out = mlp_out
        else:
            mlp_out = mlp_out + self.head_adapter(out, ctx=ctx)[..., 0]
        return self.rescale(cg, mlp_out, ctx=ctx)

    def __call__(
        self,
        cg: CrystalGraphs,
        ctx: Context,
    ) -> Float[Array, ' graphs 1']:
        out = self.head_outputs(cg, ctx)

        return self.energy_head(out, cg, ctx)
