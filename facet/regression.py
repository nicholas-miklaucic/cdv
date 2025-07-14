"""Loss function for formation energy/force/stress regression."""

from typing import Callable

import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode
from jaxtyping import Array, Float

from facet.data.databatch import CrystalGraphs


class EFSOutput(PyTreeNode):
    energy: Float[Array, ' graphs 1']
    force: Float[Array, ' nodes 3']
    stress: Float[Array, ' graphs 3 3']

    def rotate(self, rots, cg) -> 'EFSOutput':
        return EFSOutput(
            energy=self.energy,  # scalar
            force=jnp.einsum('ij,ijk->ik', self.force, rots[cg.nodes.graph_i]),
            stress=jnp.einsum('ijk,ikl->ijl', self.stress, rots),
        )


class EFSWrapper(PyTreeNode):
    compute_forces: bool
    compute_stress: bool

    def __call__(
        self, apply_fn: Callable, variables, cg: CrystalGraphs, *args, **kwargs
    ) -> EFSOutput:
        """Evaluates the model with the variables on the graph, calculating the forces and stress
        from the gradient."""

        def energy_fn(cart, lat):
            energy = apply_fn(
                variables,
                cg.replace(
                    nodes=cg.nodes.replace(cart=cart), graph_data=cg.graph_data.replace(lat=lat)
                ),
                *args,
                **kwargs,
            )
            return jnp.sum(energy, where=cg.padding_mask), energy

        argnums = []
        if self.compute_forces:
            argnums.append(0)
        if self.compute_stress:
            argnums.append(1)

        if argnums:
            grads, energy = jax.grad(energy_fn, argnums=argnums, has_aux=True)(
                cg.nodes.cart, cg.graph_data.lat
            )
        else:
            _energy_sum, energy = energy_fn(cg.nodes.cart, cg.graph_data.lat)

        if self.compute_forces:
            fgrad = grads[0]
        else:
            fgrad = jnp.zeros_like(cg.nodes.cart)

        if self.compute_stress:
            sgrad = grads[-1]
        else:
            sgrad = jnp.zeros_like(cg.graph_data.lat)

        # https://github.com/MDIL-SNU/SevenNet/blob/afb56e10b6a27190f7c3ce25cbf666cf9175608e/sevenn/nn/force_output.py#L72
        # https://github.com/ACEsuit/mace/blob/575af0171369e2c19e04b115140b9901f83eb00c/mace/modules/utils.py#L60

        force = -fgrad

        volume = jax.vmap(
            jnp.linalg.det,
        )(cg.graph_data.lat.reshape(-1, 3, 3)).reshape(*cg.graph_data.lat.shape[:-2], 1, 1)

        stress = jnp.where(
            volume == 0, -sgrad, -sgrad / jnp.where(volume == 0, jnp.ones_like(volume), volume)
        )
        stress = sgrad * 0.1  # convert to GPa

        return EFSOutput(energy=energy, force=force, stress=stress)


class EFSLoss(PyTreeNode):
    """Calculates loss for the energy/force/stress."""

    loss_fn: Callable
    energy_weight: float
    force_weight: float
    stress_weight: float

    def __call__(self, cg: CrystalGraphs, pred: EFSOutput):
        # https://github.com/MDIL-SNU/SevenNet/blob/afb56e10b6a27190f7c3ce25cbf666cf9175608e/sevenn/nn/force_output.py#L82
        voigt_i = jnp.array([0, 1, 2, 0, 1, 0])
        voigt_j = jnp.array([0, 1, 2, 1, 2, 2])
        loss = {
            'energy': self.loss_fn(pred.energy[..., 0], cg.e_form, cg.padding_mask),
            'force': self.loss_fn(
                pred.force,
                cg.target_data.force,
                cg.padding_mask[cg.nodes.graph_i],
            ),
            'stress': self.loss_fn(
                pred.stress[..., voigt_i, voigt_j],
                cg.target_data.stress[..., voigt_i, voigt_j],
                cg.padding_mask,
            ),
        }

        loss['loss'] = (
            self.energy_weight * loss['energy']
            + self.force_weight * loss['force']
            + self.stress_weight * loss['stress']
        )

        return loss
