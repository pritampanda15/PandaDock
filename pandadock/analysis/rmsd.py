"""
Pose RMSD for docking evaluation.

Docking accuracy is conventionally reported as symmetry-corrected heavy-atom
RMSD to the crystal pose, with the standard success threshold at 2.0 A.

Symmetry correction is not optional. A phenyl ring can be labelled six ways that
all describe the same physical structure, so a naive atom-index RMSD reports a
large error for a pose that is in fact perfect. Ignoring this inflates RMSD for
exactly the aromatic and symmetric ligands that are most common in drug
discovery, which makes an engine look far worse than it is.

No superposition is performed: docking RMSD compares poses in the receptor's
frame, so aligning them first would discard the very error being measured.
"""

import logging
from typing import List, Optional, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign

logger = logging.getLogger("pandadock.analysis.rmsd")


def heavy_atom_coords(mol: Chem.Mol, conf_id: int = 0) -> np.ndarray:
    """Heavy-atom coordinates of a conformer."""
    conf = mol.GetConformer(conf_id)
    positions = np.asarray(conf.GetPositions())
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    return positions[heavy]


def simple_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """In-place RMSD between two coordinate sets with matching atom order."""
    if coords_a.shape != coords_b.shape:
        raise ValueError(
            f"Coordinate shape mismatch: {coords_a.shape} vs {coords_b.shape}"
        )
    return float(np.sqrt(np.mean(np.sum((coords_a - coords_b) ** 2, axis=1))))


def symmetry_corrected_rmsd(
    probe: Chem.Mol,
    reference: Chem.Mol,
    probe_conf_id: int = 0,
    ref_conf_id: int = 0,
    max_matches: int = 10000,
) -> float:
    """
    Minimum heavy-atom RMSD over all symmetry-equivalent atom assignments.

    Both molecules must describe the same species. Hydrogens are stripped, since
    their positions are not determined by X-ray crystallography and are excluded
    from the standard metric.

    Falls back to a substructure-match search if RDKit's own routine cannot map
    the two molecules, and raises if they are genuinely different species.
    """
    probe_h = Chem.RemoveHs(Chem.Mol(probe))
    ref_h = Chem.RemoveHs(Chem.Mol(reference))

    if probe_h.GetNumAtoms() != ref_h.GetNumAtoms():
        raise ValueError(
            f"Cannot compare molecules with {probe_h.GetNumAtoms()} and "
            f"{ref_h.GetNumAtoms()} heavy atoms"
        )

    try:
        # GetBestRMS enumerates symmetry-equivalent matches. It aligns by default,
        # so `prbTransform=False`-style behaviour is obtained by comparing in place
        # via CalcRMS, which is the correct choice for docking.
        return float(
            rdMolAlign.CalcRMS(
                probe_h,
                ref_h,
                prbId=probe_conf_id,
                refId=ref_conf_id,
                maxMatches=max_matches,
            )
        )
    except Exception as exc:
        logger.debug("CalcRMS failed (%s); falling back to substructure matching", exc)

    return _fallback_rmsd(probe_h, ref_h, probe_conf_id, ref_conf_id, max_matches)


def _fallback_rmsd(
    probe: Chem.Mol,
    reference: Chem.Mol,
    probe_conf_id: int,
    ref_conf_id: int,
    max_matches: int,
) -> float:
    """Enumerate substructure matches directly when CalcRMS cannot map the pair."""
    matches = reference.GetSubstructMatches(
        probe, uniquify=False, useChirality=False, maxMatches=max_matches
    )
    if not matches:
        # Compare by topology alone, ignoring bond orders, for cases where
        # protonation or kekulisation differs between the two files.
        probe_generic = _to_generic(probe)
        ref_generic = _to_generic(reference)
        matches = ref_generic.GetSubstructMatches(
            probe_generic, uniquify=False, maxMatches=max_matches
        )
    if not matches:
        raise ValueError("Probe and reference molecules are not isomorphic")

    probe_pos = np.asarray(probe.GetConformer(probe_conf_id).GetPositions())
    ref_pos = np.asarray(reference.GetConformer(ref_conf_id).GetPositions())

    best = np.inf
    for match in matches:
        diff = probe_pos - ref_pos[list(match)]
        rmsd = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
        best = min(best, rmsd)
    return best


def _to_generic(mol: Chem.Mol) -> Chem.Mol:
    """Copy with all bonds set to single and aromaticity cleared."""
    generic = Chem.RWMol(mol)
    for bond in generic.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in generic.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
        atom.SetFormalCharge(0)
    return generic.GetMol()


def rmsd_to_reference(
    poses: Sequence[np.ndarray],
    template: Chem.Mol,
    reference: Chem.Mol,
) -> List[float]:
    """
    Symmetry-corrected RMSD of each pose to a reference structure.

    Args:
        poses: Full-molecule coordinate arrays, ordered as `template`'s atoms.
        template: Molecule whose atom ordering the poses follow.
        reference: Crystal (or other reference) pose.
    """
    values: List[float] = []
    for coords in poses:
        mol = Chem.Mol(template)
        mol.RemoveAllConformers()
        conf = Chem.Conformer(template.GetNumAtoms())
        for i in range(template.GetNumAtoms()):
            conf.SetAtomPosition(i, coords[i].tolist())
        mol.AddConformer(conf, assignId=True)
        values.append(symmetry_corrected_rmsd(mol, reference))
    return values


def success_rate(rmsds: Sequence[float], threshold: float = 2.0) -> float:
    """Fraction of values at or below `threshold`. Returns NaN for empty input."""
    finite = [r for r in rmsds if np.isfinite(r)]
    if not finite:
        return float("nan")
    return float(np.mean(np.asarray(finite) <= threshold))


def top_n_success(
    ranked_rmsds: Sequence[float], n: int = 1, threshold: float = 2.0
) -> bool:
    """
    Whether any of the top `n` ranked poses is within `threshold`.

    `n=1` gives the strict top-pose criterion, which is what "% poses <= 2 A"
    means in published redocking tables unless stated otherwise.
    """
    for rmsd in list(ranked_rmsds)[:n]:
        if np.isfinite(rmsd) and rmsd <= threshold:
            return True
    return False
