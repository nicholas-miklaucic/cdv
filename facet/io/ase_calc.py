"""ASE Calculator interface."""

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes, all_properties

from facet.io.structure_calc import FacetModel


class FacetCalculator(Calculator):
    """ASE Calculator interface for Facet models."""

    implemented_properties = ['energy', 'forces']

    def __init__(self, structure_calc: FacetModel, **kwargs):
        super().__init__(**kwargs)
        self.structure_calc = structure_calc

    def calculate(
        self,
        atoms: Atoms,
        properties: list | None = None,
        system_changes: list | None = None,
    ) -> None:
        """Calculate formation energy with Facet model."""
        properties = properties or all_properties
        system_changes = system_changes or all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)

        energy, force = self.structure_calc.predict_atoms(atoms)
        tot_energy = energy * len(atoms)
        self.results.update(energy=tot_energy, free_energy=tot_energy, forces=force)
