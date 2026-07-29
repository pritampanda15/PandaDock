"""
Monte Carlo docking (compatibility shim).

The previous implementation under this name was not a Monte Carlo search. It
generated a fixed pool of poses, of which 70% came from jittering the input
conformer within ~1 A and ~10 degrees of a reference point, and the remaining 30%
from a "random" generator that placed 80% of its samples near the box centre.
Poses were then filtered by an energy that added a bonus for proximity to that
same reference point and was clamped to [-15, 10] kcal/mol. There was no
Metropolis criterion, no temperature schedule, and no torsional sampling.

`PandaCoreDocker` implements the intended algorithm properly: independent Monte
Carlo runs with Metropolis acceptance, quasi-Newton local optimization at every
step, and search over translation, orientation and ligand torsions. Set
`temperature` and `n_steps` through its `dock()` kwargs to control the schedule.
"""

from .pandacore import PandaCoreDocker


class MonteCarloDocker(PandaCoreDocker):
    """Deprecated alias for `PandaCoreDocker`, which performs the real search."""

    def __init__(self, name: str = "monte_carlo_cpu"):
        super().__init__(name)
