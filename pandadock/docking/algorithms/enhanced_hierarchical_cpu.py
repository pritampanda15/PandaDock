"""
Enhanced hierarchical docking (compatibility shim).

This module previously contained a three-stage Glide-style pipeline (site-point
filtering, greedy scoring, energy minimization). That pipeline was unreachable:
`dock()` returned from a branch above it, so the roughly 500 lines implementing
it never executed. What did execute was a loop that jittered the input conformer
around the box centre and kept the first poses whose energy fell under a
threshold, with no optimization of any kind.

The parts that did run had further defects that made their output meaningless:

- Ligand flexibility was advertised but absent. `_full_flexibility_minimization`
  and `_minimize_with_rotamers` both fell through to rigid-body minimization.
- `_rigid_minimization` read coordinates from the input conformer rather than
  from the pose it was given, discarding the pose being minimized.
- The scoring function was called without the ligand, so the Vina-style function
  returned 0.0 for every pose.
- Site points were truncated with `valid_site_points[:50]` after being generated
  in lexicographic x, y, z order, which restricted sampling to one face of the box.

Rather than repair code that never ran, this name now delegates to
`PandaCoreDocker`. Import that class directly for new work.
"""

from .pandacore import PandaCoreDocker


class EnhancedHierarchicalDocker(PandaCoreDocker):
    """Deprecated alias for `PandaCoreDocker`."""

    def __init__(self, name: str = "enhanced_hierarchical_cpu"):
        super().__init__(name)
