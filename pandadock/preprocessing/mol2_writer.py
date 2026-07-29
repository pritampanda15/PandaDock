"""
MOL2 output for GNN input preparation.

PandaDock-GNN is trained on MOL2 triples (protein, ligand, site) carrying
hydrogens, explicit bonds and real SYBYL atom types. Feeding it a receptor parsed
from PDB instead loses all three: PDB records carry no bonds, crystal structures
usually omit hydrogens, and atom types have to be inferred.

Measured on one ULVSH complex, dropping bonds shifted the predicted pEC50 by
+0.29 and dropping bonds and hydrogens together by +0.52. Across a benchmark of
differently sized proteins that is enough to destroy the correlation: the model
scores r = 0.82 on its own MOL2 test split and r = 0.17 through a PDB-derived
pipeline.

Binding sites are cut at residue granularity, keeping every residue with any atom
inside the radius. Cutting mid-residue would leave dangling bonds and partial
functional groups, which is worse than a slightly larger site.
"""

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

logger = logging.getLogger("pandadock.preprocessing.mol2")


class OpenBabelUnavailable(RuntimeError):
    """Raised when MOL2 conversion is requested without Open Babel installed."""


def _pybel():
    try:
        from openbabel import pybel
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise OpenBabelUnavailable(
            "Open Babel is required to write MOL2 files. Install it with:\n"
            "    conda install -c conda-forge openbabel"
        ) from exc
    return pybel


def openbabel_available() -> bool:
    try:
        _pybel()
        return True
    except OpenBabelUnavailable:
        return False


def convert_to_mol2(
    input_path: str,
    output_path: str,
    add_hydrogens: bool = True,
    ph: Optional[float] = 7.4,
) -> Path:
    """
    Convert a structure file to MOL2, assigning SYBYL types and hydrogens.

    Args:
        input_path: Source file (PDB, SDF, MOL2, ...).
        output_path: Destination MOL2 path.
        add_hydrogens: Add hydrogens if absent. The model was trained with them.
        ph: Protonate for this pH. None keeps the input's protonation.
    """
    pybel = _pybel()

    suffix = Path(input_path).suffix.lower().lstrip(".")
    molecules = list(pybel.readfile(suffix, str(input_path)))
    if not molecules:
        raise ValueError(f"Open Babel could not read any molecule from {input_path}")

    molecule = molecules[0]
    if add_hydrogens:
        if ph is not None:
            molecule.OBMol.AddHydrogens(False, True, ph)
        else:
            molecule.OBMol.AddHydrogens()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    molecule.write("mol2", str(output_path), overwrite=True)
    return output_path


def write_ligand_mol2(
    mol,
    output_path: str,
    add_hydrogens: bool = True,
) -> Path:
    """
    Write an RDKit molecule to MOL2, preserving its conformer.

    RDKit has no MOL2 writer, so the molecule is routed through an SDF and
    converted, which keeps bond orders intact.
    """
    from rdkit import Chem

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_sdf = output_path.with_suffix(".tmp.sdf")
    writer = Chem.SDWriter(str(tmp_sdf))
    try:
        writer.write(mol)
    finally:
        writer.close()

    try:
        convert_to_mol2(str(tmp_sdf), str(output_path), add_hydrogens=add_hydrogens,
                        ph=None)
    finally:
        tmp_sdf.unlink(missing_ok=True)
    return output_path


def extract_site_pdb(
    receptor_path: str,
    centroid: Sequence[float],
    output_path: str,
    radius: float = 10.0,
) -> Path:
    """
    Write a PDB containing every residue with an atom within `radius`.

    Selection is by residue, not by atom: cutting a residue in half would leave
    dangling bonds and partial functional groups, and Open Babel would then
    invent hydrogens to satisfy the broken valences.
    """
    from Bio.PDB import PDBIO, PDBParser, Select

    centroid = np.asarray(centroid, dtype=float)
    structure = PDBParser(QUIET=True).get_structure("receptor", receptor_path)

    keep = set()
    for residue in structure.get_residues():
        for atom in residue:
            if np.linalg.norm(np.asarray(atom.get_coord()) - centroid) <= radius:
                keep.add(residue.get_full_id())
                break

    if not keep:
        logger.warning(
            "No residue lies within %.1f A of the site centre; writing the whole receptor",
            radius,
        )

    class _SiteSelect(Select):
        def accept_residue(self, residue):
            return 1 if (not keep or residue.get_full_id() in keep) else 0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path), _SiteSelect())
    return output_path


def write_gnn_inputs(
    receptor_path: str,
    ligand_mol,
    output_dir: str,
    site_radius: float = 10.0,
    add_hydrogens: bool = True,
) -> dict:
    """
    Write the protein / ligand / site MOL2 triple the GNN expects.

    Returns the three paths. The site centre is the ligand's heavy-atom centroid.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    heavy = [a.GetIdx() for a in ligand_mol.GetAtoms() if a.GetAtomicNum() > 1]
    positions = np.asarray(ligand_mol.GetConformer().GetPositions())
    centroid = positions[heavy].mean(axis=0) if heavy else positions.mean(axis=0)

    ligand_mol2 = write_ligand_mol2(
        ligand_mol, output_dir / "ligand.mol2", add_hydrogens=add_hydrogens
    )
    protein_mol2 = convert_to_mol2(
        receptor_path, output_dir / "protein.mol2", add_hydrogens=add_hydrogens
    )

    site_pdb = output_dir / "_site.pdb"
    extract_site_pdb(receptor_path, centroid, site_pdb, radius=site_radius)
    site_mol2 = convert_to_mol2(
        str(site_pdb), output_dir / "site.mol2", add_hydrogens=add_hydrogens
    )
    site_pdb.unlink(missing_ok=True)

    return {
        "protein_mol2": str(protein_mol2),
        "ligand_mol2": str(ligand_mol2),
        "site_mol2": str(site_mol2),
    }
