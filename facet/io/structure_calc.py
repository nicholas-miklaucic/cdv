"""Convenient inference wrapper for trained models operating on single structures."""

import os
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import pyrallis
from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from vesin import NeighborList

from facet.checkpointing import best_ckpt
from facet.config import MainConfig
from facet.data.databatch import (
    CrystalData,
    CrystalGraphs,
    EdgeData,
    NodeData,
    TargetInfo,
)
from facet.data.metadata import DatasetMetadata
from facet.layers import Context
from facet.mace.mace import MaceModel
from facet.utils import load_pytree


def structure_to_cg(
    struct: Structure, metadata: DatasetMetadata, rmax: float, k: int = 32
) -> CrystalGraphs:
    # positions can be anything compatible with numpy's ndarray
    positions = struct.cart_coords
    box = struct.lattice.matrix

    calculator = NeighborList(cutoff=rmax, full_list=True)
    P, S, d = calculator.compute(
        points=positions, box=box, periodic=True, quantities='PSd', copy=True
    )
    d_sort = np.argsort(d)
    P, S, d = P[d_sort], S[d_sort], d[d_sort]

    n_nodes = struct.num_sites

    z_to_i = {z: i for i, z in enumerate(metadata.atomic_numbers)}

    # pad with edges that are very long so they won't be included
    to_jimage = np.ones((n_nodes, k, 3), dtype=np.int32) * 100
    receiver = np.zeros((n_nodes, k), dtype=np.uint32)
    species = np.zeros((n_nodes,), dtype=np.uint32)

    for i in range(n_nodes):
        mask = np.argwhere(P[:, 0] == i).reshape(-1)
        mask = mask[:k]
        n = len(mask)
        to_jimage[[i], :n, :] = S[mask]
        receiver[i, :n] = P[mask, 1]
        species[i] = z_to_i[struct.species[i].Z]

    cg_s = CrystalGraphs(
        NodeData(jnp.array(species), jnp.array(struct.cart_coords), jnp.zeros_like(species)),
        EdgeData(jnp.array(to_jimage), jnp.array(receiver)),
        n_node=jnp.array([n_nodes]),
        padding_mask=jnp.array([1]),
        graph_data=CrystalData(
            dataset_id=jnp.array([0]),
            abc=jnp.array(struct.lattice.abc)[None, ...],
            angles_rad=jnp.array(np.deg2rad(struct.lattice.angles)[None, ...]),
            lat=jnp.array(struct.lattice.matrix)[None, ...],
        ),
        target_data=TargetInfo(
            jnp.array([0.0]), jnp.zeros_like(struct.cart_coords), jnp.zeros((1, 3, 3))
        ),
    )

    return cg_s


class FacetModel:
    """Trained Facet model, ready for inference."""

    model_dir: Path
    model: MaceModel
    params: dict[str, Any]
    config: MainConfig
    use_ema: bool
    rmax: float
    k: int
    apply_func: Callable[[CrystalGraphs], jax.Array]

    def __init__(
        self,
        config_path: os.PathLike | str,
        params_path: os.PathLike | str,
        data_dir: os.PathLike | str,
        use_ema=True,
    ):
        """Inferences using a directory with config.toml and checkpoints.
        If checkpoint_params is True, uses .ckpt file instead of the checkpoint from folder."""
        self.use_ema = use_ema
        self.config_path = Path(config_path)
        self.params_path = Path(params_path)
        with open(config_path) as f:
            self.config: MainConfig = pyrallis.cfgparsing.load(MainConfig, f)

        self.config.data.data_folder = Path(data_dir)

        self.model = self.config.build_regressor()

        if self.params_path.is_dir():
            ckpt = best_ckpt(self.params_path)
            if self.use_ema:
                self.params = ckpt['state']['opt_state'][-1]['ema']['params']
            else:
                self.params = ckpt['state']['params']
        else:
            self.params = load_pytree(self.params_path)
            if 'params' in self.params:
                self.params = self.params['params']
            self.params = jax.tree.map(jnp.array, self.params)

        default_rmax = self.config.model.edge_embed.r_max
        self.rmax = self.params['edge_embedding'].get('rmax', jnp.array(default_rmax)).item()
        self.k = self.config.data.k
        self.k = self.k or 48
        self.apply_func = jax.jit(
            lambda cg: self.model.apply({'params': self.params}, cg=cg, ctx=Context(training=False))
        )

    @classmethod
    def new_facet(cls):
        return cls(
            '/home/nmiklaucic/cdv/logs/enb-198/config.toml',
            '/home/nmiklaucic/cdv/logs/enb-198/',
            '/home/nmiklaucic/cdv/precomputed',
        )

    @classmethod
    def new_sevennet_streamlined(cls):
        return cls(
            '/home/nmiklaucic/cdv/logs/enb-217/config.toml',
            '/home/nmiklaucic/cdv/logs/enb-217',
            '/home/nmiklaucic/cdv/precomputed',
        )

    @classmethod
    def new_sevennet(cls):
        return cls(
            '/home/nmiklaucic/cdv/configs/sevennet.toml',
            '/home/nmiklaucic/cdv/precomputed/sevennet.ckpt',
            '/home/nmiklaucic/cdv/precomputed',
        )

    def __repr__(self):
        return f'FacetModel(config_path={self.config_path}, params_path={self.params_path}, use_ema={self.use_ema}, rmax={self.rmax:.3f}, k={self.k})'

    def predict_structure(self, struct: Structure) -> tuple[float, Any]:
        cg: CrystalGraphs = structure_to_cg(struct, self.config.data.metadata, self.rmax, self.k)
        coords = cg.nodes.cart

        def force_val(coords):
            return self.apply_func(cg.replace(nodes=cg.nodes.replace(cart=coords))).reshape()

        # print(cg)
        # debug_structure(cg=cg)
        # debug_structure(params=self.params)
        val, grad = jax.value_and_grad(force_val)(coords)
        return val.item(), np.array(grad)

    def predict_atoms(self, atoms: Atoms) -> tuple[float, Any]:
        return self.predict_structure(AseAtomsAdaptor.get_structure(atoms))


if __name__ == '__main__':
    # from mp_api.client import MPRester

    # mp_id = "mp-30580"
    # with MPRester() as mpr:
    #     data = mpr.materials.search(material_ids=[mp_id], fields=["structure"])
    #     thermo_docs = mpr.materials.thermo.search(
    #         material_ids=[mp_id], thermo_types=["GGA_GGA+U", "GGA_GGA+U_R2SCAN", "R2SCAN"]
    #     )

    # struct = data[0].structure
    # print(struct.composition)
    # for doc in thermo_docs:
    #     print(f"{doc.energy_type:>15}", f"{doc.energy_per_atom:>10.4f}")
    # struct.to('data/SrTiO3.cif')

    struct = Structure.from_file('data/SrTiO3.cif')

    for method in (
        FacetModel.new_sevennet,
        FacetModel.new_sevennet_streamlined,
        FacetModel.new_facet,
    ):
        model = method()
        print(model)
        print(model.predict_structure(struct))
