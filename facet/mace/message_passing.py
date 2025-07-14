import functools as ft
import operator

import e3nn_jax as e3nn
import jax
import jax.experimental
import jax.numpy as jnp
from e3nn_jax.legacy import FunctionalTensorProduct
from flax import linen as nn

from facet.layers import Context, DyTanh, LazyInMLP
from facet.mace.e3_layers import (
    E3DyTanh,
    E3Irreps,
    E3IrrepsArray,
    IrrepsModule,
    Linear,
)


class MPConv(IrrepsModule):
    avg_num_neighbors: float | None
    max_ell: int

    def __call__(
        self,
        vectors: E3IrrepsArray,  # [n_nodes, k, 3]
        node_feats: E3IrrepsArray,  # [n_nodes, irreps]
        radial_embedding: E3IrrepsArray,  # [n_nodes, k, radial_embedding_dim]
        receivers: jnp.ndarray,  # [n_nodes, k]
        avg_num_neighbors: jnp.ndarray,  # 1
        ctx: Context,
    ) -> E3IrrepsArray:
        raise NotImplementedError


class SevenNetConv(MPConv):
    """
    SevenNet convolution: like NequIP, but without the per-element weights.

    Code 'based on' Deepmind's implementation:
    https://github.com/google-deepmind/materials_discovery/blob/7372c3a2f90d480e62d61ec386451a321045cea7/model/nequip.py#L77

    Their sources, in their words:

    Implementation follows the original paper by Batzner et al.

    nature.com/articles/s41467-022-29939-5 and partially
    https://github.com/mir-group/nequip.
    """

    radial_weight: LazyInMLP
    radial_power: float = 1
    # @nn.compact
    # def __call__(
    #     self,
    #     node_features: IrrepsArray,
    #     node_attributes: IrrepsArray,
    #     edge_sh: Array,
    #     edge_src: Array,
    #     edge_dst: Array,
    #     edge_embedded: Array,
    # ) -> IrrepsArray:

    @nn.compact
    def __call__(
        self,
        vectors: E3IrrepsArray,  # [n_nodes, k, 3]
        node_feats: E3IrrepsArray,  # [n_nodes, irreps]
        radial_embedding: E3IrrepsArray,  # [n_nodes, k, radial_embedding_dim]
        receivers: jnp.ndarray,  # [n_nodes, k]
        avg_num_neighbors: jnp.ndarray,  # 1
        ctx: Context,
    ) -> E3IrrepsArray:
        avg_num_neighbors = (
            jnp.array([self.avg_num_neighbors])
            if self.avg_num_neighbors is not None
            else avg_num_neighbors
        )
        # Nequip outline, just the parts we're doing here:
        # TP + aggregate
        # divide by average number of neighbors
        # Concatenation

        # edge_sh is not given in this interface: perhaps that should be cached? is it memory-bound
        # or FLOP-bound?
        # edge_sh = e3nn.spherical_harmonics(
        #     e3nn.Irreps.spherical_harmonics(self.max_ell, p=1), vectors, normalize=True
        # )
        edge_sh = vectors

        # map node features onto edges for tp
        edge_features = node_feats  # [n_nodes, irreps]

        # we gather the instructions for the tp as well as the tp output irreps
        mode = 'uvu'
        trainable = True
        irreps_after_tp = []
        instructions = []

        # iterate over both arguments, i.e. node irreps and edge irreps
        # if they have a valid TP path for any of the target irreps,
        # add to instructions and put in appropriate position
        # we use uvu mode (where v is a single-element sum) and weights will
        # be provide externally by the scalar MLP
        # this triple for loop is similar to the one used in e3nn and nequip
        for i, (mul_in1, irreps_in1) in enumerate(node_feats.irreps):
            for j, (_, irreps_in2) in enumerate(edge_sh.irreps):
                for curr_irreps_out in irreps_in1 * irreps_in2:
                    if curr_irreps_out in self.ir_out:
                        k = len(irreps_after_tp)
                        irreps_after_tp += [(mul_in1, curr_irreps_out)]
                        instructions += [(i, j, k, mode, trainable)]

        # we will likely have constructed irreps in a non-l-increasing order
        # so we sort them to be in a l-increasing order
        irreps_after_tp, p, _inv = E3Irreps(irreps_after_tp).sort()

        # if we sort the target irreps, we will have to sort the instructions
        # acoordingly, using the permutation indices
        sorted_instructions = []

        for irreps_in1, irreps_in2, irreps_out, mode, trainable in instructions:
            sorted_instructions += [
                (
                    irreps_in1,
                    irreps_in2,
                    p[irreps_out],
                    mode,
                    trainable,
                )
            ]

        # print(sorted_instructions)

        # TP between spherical harmonics embedding of the edge vector
        # Y_ij(\hat{r}) and neighboring node h_j, weighted on a per-element basis
        # by the radial network R(r_ij)
        tp = FunctionalTensorProduct(
            irreps_in1=edge_features.irreps,
            irreps_in2=edge_sh.irreps,
            irreps_out=irreps_after_tp,
            instructions=sorted_instructions,
        )

        # scalar radial network, number of output neurons is the total number of
        # tensor product paths, nonlinearity must have f(0)=0 and MLP must not
        # have biases
        n_tp_weights = 0

        # get output dim of radial MLP / number of TP weights
        for ins in tp.instructions:
            if ins.has_weight:
                n_tp_weights += ft.reduce(operator.mul, ins.path_shape, 1)

        # build radial MLP R(r) that maps from interatomic distances to TP weights
        # must not use bias to that R(0)=0
        # fc = nn_util.MLP(
        #     (self.radial_net_n_hidden,) * self.radial_net_n_layers + (n_tp_weights,),
        #     self.radial_net_nonlinearity,
        #     use_bias=False,
        #     scalar_mlp_std=self.scalar_mlp_std,
        # )

        fc: LazyInMLP = self.radial_weight.copy(out_dim=n_tp_weights)

        # the TP weights (v dimension) are given by the FC
        weight = fc(radial_embedding, ctx=ctx)

        # debug_structure(weight=weight, edge_features=edge_features, sh=edge_sh)

        # tp between node features that have been mapped onto edges and edge RSH
        # weighted by FC weight, we vmap over the dimension of the edges

        edge_features = e3nn.vmap(
            e3nn.vmap(
                ft.partial(
                    tp.left_right,
                    custom_einsum_jvp=True,
                    fused=False,
                    sparse=False,
                ),
                in_axes=(0, None, 0),
            )
        )(weight.array, edge_features, edge_sh)
        # TODO: It's not great that e3nn_jax automatically upcasts internally,
        # but this would need to be fixed at the e3nn level.
        edge_features = jax.tree.map(lambda x: x.astype(node_feats.dtype), edge_features)

        h = node_feats
        # aggregate edges onto nodes after tp using e3nn-jax's index_add
        h_type = h.dtype

        e = edge_features.remove_zero_chunks().simplify()
        h = e3nn.index_add(receivers, e, out_dim=h.shape[0])
        h = h.astype(h_type)

        # normalize by the average (not local) number of neighbors
        h = h / (avg_num_neighbors**self.radial_power)

        return h


class NodeFeatureMLPWeightedConv(MPConv):
    """
    A middle ground between SevenNet and NequIP convolutions. Element information is integrated, but
    not through indexed weights. Instead, the source/target node features are used to generate
    weights in the radial basis. This continues to ensure the cutoff behaves properly, but allows
    for integration of node information.
    """

    node_feature_mlp: LazyInMLP

    @nn.compact
    def __call__(
        self,
        vectors: E3IrrepsArray,  # [n_nodes, k, 3]
        node_feats: E3IrrepsArray,  # [n_nodes, irreps]
        radial_embedding: E3IrrepsArray,  # [n_nodes, k, radial_embedding_dim]
        receivers: jnp.ndarray,  # [n_nodes, k]
        avg_num_neighbors: jnp.ndarray,  # 1
        ctx: Context,
    ) -> E3IrrepsArray:
        avg_num_neighbors = (
            jnp.array([self.avg_num_neighbors])
            if self.avg_num_neighbors is not None
            else avg_num_neighbors
        )
        edge_sh = vectors

        # map node features onto edges for tp
        edge_features = node_feats  # [n_nodes, irreps]

        # we gather the instructions for the tp as well as the tp output irreps
        mode = 'uvu'
        trainable = True
        irreps_after_tp = []
        instructions = []

        # iterate over both arguments, i.e. node irreps and edge irreps
        # if they have a valid TP path for any of the target irreps,
        # add to instructions and put in appropriate position
        # we use uvu mode (where v is a single-element sum) and weights will
        # be provide externally by the scalar MLP
        # this triple for loop is similar to the one used in e3nn and nequip
        for i, (mul_in1, irreps_in1) in enumerate(node_feats.irreps):
            for j, (_, irreps_in2) in enumerate(edge_sh.irreps):
                for curr_irreps_out in irreps_in1 * irreps_in2:
                    if curr_irreps_out in self.ir_out:
                        k = len(irreps_after_tp)
                        irreps_after_tp += [(mul_in1, curr_irreps_out)]
                        instructions += [(i, j, k, mode, trainable)]

        # we will likely have constructed irreps in a non-l-increasing order
        # so we sort them to be in a l-increasing order
        irreps_after_tp, p, _inv = E3Irreps(irreps_after_tp).sort()

        # if we sort the target irreps, we will have to sort the instructions
        # acoordingly, using the permutation indices
        sorted_instructions = []

        for irreps_in1, irreps_in2, irreps_out, mode, trainable in instructions:
            sorted_instructions += [
                (
                    irreps_in1,
                    irreps_in2,
                    p[irreps_out],
                    mode,
                    trainable,
                )
            ]

        tp = FunctionalTensorProduct(
            irreps_in1=edge_features.irreps,
            irreps_in2=edge_sh.irreps,
            irreps_out=irreps_after_tp,
            instructions=sorted_instructions,
        )

        n_tp_weights = 0

        for ins in tp.instructions:
            if ins.has_weight:
                n_tp_weights += ft.reduce(operator.mul, ins.path_shape, 1)

        # Instead of thinking of these RBFs as inputs to an MLP, we can think of them as a function
        # basis we can use to represent our outputs. We enforce that the output has to be a linear
        # function of the RBFs, so the cutoff is applied properly.

        # each of the TP weights is a linear function of the input embedding
        n_mlp_out = n_tp_weights * radial_embedding.shape[-1]

        # there are a couple options for node embedding here:
        # - the original species embeddings
        # - the scalars from the node embeddings
        # - the norms of the node embeddings
        # For now, we just use the scalars.

        # this is a huge batched matrix multiplication, and we only need half precision
        node_scalars = node_feats.filter('0e').array.astype(jnp.bfloat16)
        src_nodes, dst_nodes = jnp.broadcast_arrays(
            node_scalars[..., None, :],  # n_nodes 1 n_scalars
            node_scalars[receivers],  # n_nodes k n_scalars
        )
        mlp_in = jnp.concatenate([src_nodes, dst_nodes], axis=-1)  # n_nodes k n_scalars*2

        fc: LazyInMLP = self.node_feature_mlp.copy(out_dim=n_mlp_out)

        # the TP weights (v dimension) are given by the FC
        mlp_weight = fc(mlp_in, ctx=ctx)  # n_nodes k (n_tp n_basis)
        mlp_weight = mlp_weight.reshape(
            *mlp_weight.shape[:-1], n_tp_weights, radial_embedding.shape[-1]
        )

        weight = jnp.einsum('...ktb,...kb->...kt', mlp_weight, radial_embedding.array)

        # debug_structure(weight=weight, edge_features=edge_features, sh=edge_sh)

        # tp between node features that have been mapped onto edges and edge RSH
        # weighted by FC weight, we vmap over the dimension of the edges
        edge_features = e3nn.vmap(e3nn.vmap(tp.left_right, in_axes=(0, None, 0)))(
            weight, edge_features, edge_sh
        )
        edge_features = jax.tree.map(lambda x: x.astype(node_feats.dtype), edge_features)

        h = node_feats
        # aggregate edges onto nodes after tp using e3nn-jax's index_add
        h_type = h.dtype

        e = edge_features.remove_zero_chunks().simplify()
        h = e3nn.index_add(receivers, e, out_dim=h.shape[0])
        h = h.astype(h_type)

        # normalize by the average (not local) number of neighbors
        h = h / avg_num_neighbors

        return h


class SimpleMixMLPConv(MPConv):
    radial_mix: LazyInMLP
    radius_power: float = 1

    @nn.compact
    def __call__(
        self,
        vectors: E3IrrepsArray,  # [n_nodes, k, 3]
        node_feats: E3IrrepsArray,  # [n_nodes, irreps]
        radial_embedding: E3IrrepsArray,  # [n_nodes, k, radial_embedding_dim]
        receivers: jnp.ndarray,  # [n_nodes, k]
        avg_num_neighbors: jnp.ndarray,  # 1
        ctx: Context,
    ) -> E3IrrepsArray:
        """-> n_nodes irreps"""
        avg_num_neighbors = (
            jnp.array([self.avg_num_neighbors])
            if self.avg_num_neighbors is not None
            else avg_num_neighbors
        )
        messages_broadcast = node_feats[
            jnp.repeat(jnp.arange(node_feats.shape[0])[..., None], 16, axis=-1)
        ]
        # debug_structure(msgs=messages, vecs=vectors)

        inner_irreps = e3nn.Irreps.spherical_harmonics(self.max_ell)

        msg_prefix = messages_broadcast.filter(inner_irreps)
        vec_harms = e3nn.tensor_product(
            messages_broadcast,
            # e3nn.spherical_harmonics(range(1, self.max_ell + 1), vectors, True),
            vectors,
            filter_ir_out=inner_irreps,
        )

        # debug_structure(
        #     msg=messages_broadcast, vecs=vectors, msg_pref=msg_prefix, vec_harm=vec_harms
        # )

        messages = e3nn.concatenate(
            [msg_prefix, vec_harms],
            axis=-1,
        ).regroup()  # [n_nodes, irreps]

        radial = self.radial_mix.copy(out_dim=messages.irreps.num_irreps)(
            radial_embedding, ctx
        )  # [n_nodes, k, num_irreps]

        # debug_structure(messages=messages, mix=radial, rad=radial_embedding.array)
        # debug_stat(messages=messages.array, mix=mix.array, rad=radial_embedding.array)
        # radial = nn.sigmoid(radial.array)
        # radial = nn.tanh(radial.array)
        messages = messages * radial  # [n_nodes, k, irreps]

        # debug_stat(messages=messages, radial=radial)

        zeros = E3IrrepsArray.zeros(messages.irreps, node_feats.shape[:1], messages.dtype)
        # TODO flip this perhaps?
        node_feats = zeros.at[receivers].add(messages)  # [n_nodes, irreps]
        node_feats = node_feats / (avg_num_neighbors**self.radius_power)

        return node_feats


class SimpleInteraction(IrrepsModule):
    conv: MPConv
    linear_intro: bool
    linear_outro: bool

    @nn.compact
    def __call__(
        self,
        vectors: E3IrrepsArray,  # [n_edges, 3]
        node_feats: E3IrrepsArray,  # [n_nodes, irreps]
        radial_embedding: jnp.ndarray,  # [n_edges, radial_embedding_dim]
        receivers: jnp.ndarray,  # [n_edges, ]
        avg_num_neighbors: jnp.ndarray,  # 1
        ctx: Context,
    ) -> E3IrrepsArray:
        """-> n_nodes irreps"""
        if self.linear_intro:
            new_node_feats = Linear(node_feats.irreps, name='linear_intro')(node_feats)
        else:
            new_node_feats = node_feats

        new_node_feats = self.conv.copy(irreps_out=self.ir_out)(
            vectors, new_node_feats, radial_embedding, receivers, avg_num_neighbors, ctx
        )

        if self.linear_outro:
            new_node_feats = Linear(self.ir_out, name='linear_outro', force_irreps_out=True)(
                new_node_feats
            )

        return new_node_feats  # [n_nodes, target_irreps]


class ResidualInteraction(IrrepsModule):
    interaction: IrrepsModule

    @nn.compact
    def __call__(
        self,
        vectors: E3IrrepsArray,  # [n_edges, 3]
        node_feats: E3IrrepsArray,  # [n_nodes, irreps]
        radial_embedding: jnp.ndarray,  # [n_edges, radial_embedding_dim]
        receivers: jnp.ndarray,  # [n_edges, ]
        avg_num_neighbors: jnp.ndarray,  # 1
        ctx: Context,
    ) -> E3IrrepsArray:
        """-> n_nodes irreps"""
        resid = Linear(self.ir_out, name='resid_adapter', force_irreps_out=True)(node_feats)
        new_node_feats = self.interaction.copy(irreps_out=self.ir_out)(
            vectors=vectors,
            node_feats=node_feats,
            radial_embedding=radial_embedding,
            receivers=receivers,
            avg_num_neighbors=avg_num_neighbors,
            ctx=ctx,
        )

        x = new_node_feats + resid
        # x = E3DyTanh(DyTanh())(x, ctx)
        return x
