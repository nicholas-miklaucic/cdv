"""
Modules that perform node embedding: given species information, constructing embeddings.
Global data would also be incorporated here, but the two are mostly orthogonal.
"""

from flax import linen as nn
from facet.data.metadata import DatasetMetadata
from facet.mace.e3_layers import E3IrrepsArray, IrrepsModule
from facet.layers import Context
from jaxtyping import Int, Array
import json
import numpy as np
import jax
import jax.numpy as jnp


class NodeEmbedding(IrrepsModule):
    """Initializes node embeddings."""

    def __call__(self, species: Int[Array, ' nodes'], ctx: Context) -> E3IrrepsArray:
        """nodes -> nodes irreps_out"""
        raise NotImplementedError


class LinearNodeEmbedding(NodeEmbedding):
    """Standard embedding layer."""

    num_species: int

    def setup(self):
        if self.ir_out.lmax > 0:
            raise ValueError(f'Irreps {self.ir_out} should just be scalars for node embedding.')
        self.out_dim = self.ir_out.dim
        self.embed = nn.Embed(self.num_species, self.out_dim)

    def __call__(self, node_species: Int[Array, ' nodes'], ctx: Context) -> E3IrrepsArray:
        return E3IrrepsArray(self.ir_out, self.embed(node_species))


class CTUAEEmbedding(NodeEmbedding):
    """Embedding locked to the CT-UAE formation energy results (DOI: 10.5281/zenodo.14557908)."""

    metadata: DatasetMetadata

    def setup(self):
        if self.ir_out.lmax > 0:
            raise ValueError(
                f'Irreps {self.ir_out.regroup()} should just be scalars for node embedding.'
            )
        if self.ir_out.num_irreps != 128:
            raise ValueError('CT-UAE embeddings are 128-dimensional, other sizes do not work')

        # data is atomic numbers 1-100
        # our atomic numbers start with a 0, so pad
        with jax.ensure_compile_time_eval():
            Z = jnp.load('precomputed/ct-uae-form.npy').astype(jnp.float32).T
            Z = jnp.vstack([Z[:1] * 0, Z])
            self.emb = Z[self.metadata.atomic_numbers]

    def __call__(self, node_species: Int[Array, ' nodes'], ctx: Context) -> E3IrrepsArray:
        return E3IrrepsArray(self.ir_out, self.emb[node_species])
