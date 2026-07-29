"""
Crystal-guided docking (removed).

This module sampled poses within a couple of Angstrom and a few degrees of a
reference coordinate, then added an energy bonus for staying close to it -- its
own documentation described this as "crystal similarity scoring to bias toward
experimental poses".

In a redocking benchmark the reference is derived from the crystal ligand, so the
procedure reports a perturbed copy of the answer rather than a prediction. Any
accuracy figure produced this way measures the bias, not the method. On a novel
target, where no reference exists, the same code falls back to the box centre and
performs no orientational or conformational search at all.

The class is retained only so that existing imports fail loudly rather than
silently producing biased poses.

For unbiased docking use `PandaCoreDocker`. If you genuinely need to bias a search
toward a known reference -- scaffold-constrained docking against a template, for
instance -- use `pandadock.tethered`, where the constraint is explicit, requested
by the user, and recorded in the output rather than hidden inside the scoring
function.
"""

_REMOVAL_MESSAGE = (
    "CrystalGuidedDocker has been removed. It restricted sampling to the "
    "neighbourhood of a reference coordinate and added an energy bonus for "
    "proximity to it, which invalidates any benchmark it appears in. Use "
    "pandadock.docking.algorithms.PandaCoreDocker for unbiased docking, or "
    "pandadock.tethered for explicitly constrained docking."
)


class CrystalGuidedDocker:
    """Removed. Use `PandaCoreDocker`, or `pandadock.tethered` for constrained runs."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_REMOVAL_MESSAGE)
