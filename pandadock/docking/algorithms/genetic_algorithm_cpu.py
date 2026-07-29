"""
Genetic algorithm docking (compatibility shim).

The previous implementation seeded a large fraction of its initial population by
jittering the input conformer around a reference point, and its chromosome encoded
only translation and orientation -- ligand torsions were never genes, so the
population could not evolve a conformation. Combined with a fitness function that
rewarded proximity to the reference, the search converged on the box centre
regardless of the receptor.

`PandaCoreDocker` supersedes it with a Monte Carlo search that does encode
torsions and is driven purely by the interaction energy. If a population-based
search is wanted specifically, the pieces to build one correctly now exist in
`pandadock.docking.search`: `TorsionTree` supplies the full DOF encoding and
`DockingObjective` supplies energies and analytic gradients.
"""

from .pandacore import PandaCoreDocker


class GeneticAlgorithmDocker(PandaCoreDocker):
    """Deprecated alias for `PandaCoreDocker`."""

    def __init__(self, name: str = "genetic_algorithm_cpu"):
        super().__init__(name)
