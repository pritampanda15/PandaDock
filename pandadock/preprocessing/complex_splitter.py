"""
Split PDB entries into a receptor and a dockable ligand.

Benchmark sets are usually distributed as whole PDB entries. Turning those into
redocking inputs requires deciding which HETATM group is the ligand, and that
decision is where most of the error in a benchmark comes from. Picking the
largest non-water HETATM -- the obvious heuristic -- is wrong often enough to
distort a published table:

- Modified residues (MSE, SEP, TPO, CME, M3L) are HETATM records but are
  covalently part of the protein chain. "Docking" one means asking the engine to
  reproduce a residue's position, which it cannot do and should not be asked to.
- N-linked glycans (NAG, BMA, MAN) are covalently attached to asparagine.
- Membrane components (CLR, PCW, POV, OLC) sit on the protein surface in GPCR and
  transporter structures, not in a pocket.

This module resolves component identity against the PDB Chemical Component
Dictionary, which records for each code whether it is a polymer-linking residue,
a saccharide, or a genuine non-polymer ligand -- and supplies the SMILES needed to
assign bond orders, which PDB coordinate records do not carry.

Ligands are selected by burial (contacts with the protein) rather than by size, so
a surface-bound lipid loses to a buried inhibitor.
"""

import json
import logging
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger("pandadock.preprocessing.complex_splitter")

CCD_ENDPOINT = "https://data.rcsb.org/rest/v1/core/chemcomp/{code}"

# Solvent, ions, cryoprotectants and buffers. Never the ligand of interest, and
# excluded before any network lookup so the common cases cost nothing.
SOLVENT_AND_IONS: Set[str] = {
    "HOH", "DOD", "WAT", "SO4", "PO4", "GOL", "EDO", "PEG", "PG4", "PGE", "1PE",
    "2PE", "MPD", "ACT", "ACY", "FMT", "DMS", "IMD", "TRS", "MES", "EPE", "BME",
    "DTT", "CIT", "TAR", "MLI", "NO3", "CO3", "SCN", "AZI", "BCT", "EOH", "IPA",
    "MOH", "DIO", "P6G", "XPE", "UNX", "UNL", "UNK", "NH4", "PER", "OXY",
    "NA", "K", "MG", "CA", "ZN", "MN", "FE", "FE2", "CD", "NI", "CU", "CU1",
    "CO", "HG", "CL", "BR", "IOD", "F", "CS", "RB", "SR", "BA", "AU", "PT",
    "AG", "LI", "AL", "GA", "IN", "TL", "PB", "W", "MO", "V",
}

# Lipids, detergents and membrane mimetics. These are large and pass any size
# filter, but they decorate the protein surface rather than occupying a site.
LIPIDS_AND_DETERGENTS: Set[str] = {
    "CLR", "CHL", "CHS", "PCW", "PC1", "POV", "PGV", "PEE", "PEF", "LMT", "LMN",
    "OLC", "OLA", "OLB", "PLM", "MYR", "STE", "PAM", "DAO", "LDA", "C8E", "DDQ",
    "BOG", "HTG", "SDS", "CXS", "NHE", "TRD", "UND", "HEX", "D10", "D12", "DD9",
    "MC3", "PX4", "LHG", "DGA", "SQD", "CDL", "PSC", "Y01", "3PH", "6PL", "9PE",
    "L2P", "L3P", "LPP", "PIO", "PIP", "PSF", "HEZ", "12P", "15P", "7PE",
}

# Ions and metals that belong WITH the receptor: a metalloenzyme's catalytic zinc
# is part of the binding site, and removing it changes the site's chemistry.
RECEPTOR_COFACTOR_CODES: Set[str] = {
    "ZN", "MG", "CA", "MN", "FE", "FE2", "CO", "NI", "CU", "CU1", "NA", "K",
    "HEM", "HEC", "SF4", "FES", "F3S", "MO", "W", "CD",
}


@dataclass
class ComponentInfo:
    """Chemical Component Dictionary record for one three-letter code."""

    code: str
    name: str = ""
    type: str = ""
    formula_weight: float = 0.0
    smiles: str = ""

    @property
    def is_polymer_residue(self) -> bool:
        """
        True for components that link into a polymer chain.

        The CCD type string ends in "linking" for anything that bonds into a
        chain: "L-peptide linking" for modified residues, "DNA linking",
        "D-saccharide, beta linking" for N-glycans. Free ligands are
        "non-polymer", and peptidomimetic inhibitors are "peptide-like", so
        keying on "linking" keeps the inhibitors while dropping chain members.
        """
        return "linking" in self.type.lower()

    @property
    def is_saccharide(self) -> bool:
        return "saccharide" in self.type.lower()


@dataclass
class LigandCandidate:
    """One HETATM residue instance considered as the ligand."""

    code: str
    chain: str
    resseq: str
    atom_lines: List[str]
    coords: np.ndarray
    contacts: int = 0
    info: Optional[ComponentInfo] = None

    @property
    def n_heavy(self) -> int:
        return len(self.atom_lines)

    @property
    def key(self) -> str:
        return f"{self.code}_{self.chain}_{self.resseq}"


# ------------------------------------------------------------------- CCD lookup


class ComponentCache:
    """
    Chemical Component Dictionary lookups, cached on disk.

    Entries are fetched once and reused, so a benchmark run costs a few hundred
    requests total regardless of how many structures reference each component.
    """

    def __init__(self, cache_path: Path):
        self.cache_path = Path(cache_path)
        self._data: Dict[str, dict] = {}
        if self.cache_path.exists():
            try:
                self._data = json.loads(self.cache_path.read_text())
            except json.JSONDecodeError:
                logger.warning("Component cache at %s is corrupt; refetching", cache_path)

    def get(self, code: str) -> Optional[ComponentInfo]:
        raw = self._data.get(code)
        if raw is None:
            return None
        return ComponentInfo(**raw)

    def fetch_missing(self, codes: Iterable[str], max_workers: int = 8) -> None:
        """Fetch any codes not already cached, then persist the cache."""
        wanted = {c for c in codes if c not in self._data}
        if not wanted:
            return

        logger.info("Fetching %d chemical components from RCSB", len(wanted))

        def fetch(code: str) -> Tuple[str, Optional[dict]]:
            url = CCD_ENDPOINT.format(code=code)
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    payload = json.load(response)
            except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
                logger.debug("Lookup failed for %s: %s", code, exc)
                return code, None

            comp = payload.get("chem_comp", {}) or {}
            desc = payload.get("rcsb_chem_comp_descriptor", {}) or {}
            return code, {
                "code": code,
                "name": comp.get("name", "") or "",
                "type": comp.get("type", "") or "",
                "formula_weight": float(comp.get("formula_weight") or 0.0),
                "smiles": desc.get("SMILES_stereo") or desc.get("SMILES") or "",
            }

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for code, record in pool.map(fetch, sorted(wanted)):
                if record is not None:
                    self._data[code] = record

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._data, indent=0, sort_keys=True))
        logger.info("Component cache now holds %d entries", len(self._data))


# ---------------------------------------------------------------- PDB splitting


def _is_first_model_only(lines: Sequence[str]) -> Tuple[List[str], bool]:
    """
    Keep only the first model of a multi-model (typically NMR) entry.

    Returns the trimmed lines and whether the entry had multiple models. NMR
    ensembles have no crystallographic ligand pose, so callers usually reject
    them outright rather than docking against model 1.
    """
    out: List[str] = []
    n_models = 0
    for line in lines:
        if line.startswith("MODEL "):
            n_models += 1
            if n_models > 1:
                break
        if line.startswith("ENDMDL") and n_models >= 1:
            break
        out.append(line)
    return out, n_models > 1


def _altloc_ok(line: str) -> bool:
    return line[16] in (" ", "A")


def parse_candidates(
    lines: Sequence[str], cache: ComponentCache
) -> Tuple[List[LigandCandidate], np.ndarray, List[str]]:
    """Collect ligand candidates, protein coordinates, and receptor atom lines."""
    groups: Dict[Tuple[str, str, str], LigandCandidate] = {}
    protein_coords: List[List[float]] = []
    receptor_lines: List[str] = []

    for line in lines:
        record = line[:6]
        if record not in ("ATOM  ", "HETATM"):
            continue
        if len(line) < 54 or not _altloc_ok(line):
            continue

        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        resname = line[17:20].strip()

        if element == "H" or element == "D":
            continue

        try:
            xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        except ValueError:
            continue

        if record == "ATOM  ":
            protein_coords.append(xyz)
            receptor_lines.append(line)
            continue

        # HETATM from here on.
        if resname in SOLVENT_AND_IONS or resname in LIPIDS_AND_DETERGENTS:
            if resname in RECEPTOR_COFACTOR_CODES:
                receptor_lines.append(line)
                protein_coords.append(xyz)
            continue

        info = cache.get(resname)
        if info is not None and info.is_polymer_residue:
            # Modified residue or glycan: part of the polymer, so it belongs to
            # the receptor rather than being a docking target.
            protein_coords.append(xyz)
            receptor_lines.append(line)
            continue

        key = (resname, line[21], line[22:27].strip())
        candidate = groups.get(key)
        if candidate is None:
            candidate = LigandCandidate(
                code=resname, chain=line[21], resseq=line[22:27].strip(),
                atom_lines=[], coords=np.empty((0, 3)), info=info,
            )
            groups[key] = candidate
        candidate.atom_lines.append(line)

    for candidate in groups.values():
        candidate.coords = np.array(
            [[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in candidate.atom_lines]
        )

    return list(groups.values()), np.array(protein_coords), receptor_lines


def select_ligand(
    candidates: Sequence[LigandCandidate],
    protein_coords: np.ndarray,
    min_heavy_atoms: int = 6,
    max_heavy_atoms: int = 120,
    contact_cutoff: float = 4.5,
    min_contacts: int = 20,
) -> Optional[LigandCandidate]:
    """
    Choose the most buried candidate of acceptable size.

    Burial is the count of protein heavy atoms within `contact_cutoff` of any
    ligand atom. Ranking by burial rather than by size is what separates a bound
    inhibitor from a surface-adsorbed lipid or cryoprotectant of similar mass.
    """
    if len(protein_coords) == 0:
        return None

    viable: List[LigandCandidate] = []
    for candidate in candidates:
        if not (min_heavy_atoms <= candidate.n_heavy <= max_heavy_atoms):
            continue
        distances = np.linalg.norm(
            candidate.coords[:, None, :] - protein_coords[None, :, :], axis=2
        )
        candidate.contacts = int(np.sum(distances < contact_cutoff))
        if candidate.contacts >= min_contacts:
            viable.append(candidate)

    if not viable:
        return None

    # Normalise by ligand size so a large, loosely bound group does not outrank a
    # small one that is fully enclosed.
    return max(viable, key=lambda c: c.contacts / max(c.n_heavy, 1))


def build_ligand_mol(candidate: LigandCandidate, sanitize: bool = True):
    """
    Build an RDKit molecule for the selected ligand, with bond orders.

    PDB coordinate records carry no bond orders, so a molecule read straight from
    them has every bond single and no aromaticity. That corrupts atom typing (an
    aromatic carbon is a distinct Vina type) and defeats symmetry-corrected RMSD,
    which relies on ring perception to enumerate equivalent atom mappings. Bond
    orders are therefore taken from the component's CCD SMILES.

    Crystal ligands are frequently modelled with atoms missing, where density was
    too weak to place them. Such a ligand is a substructure of its CCD template
    rather than isomorphic to it, and template matching fails outright. Dropping
    those entries would bias the benchmark toward well-ordered, tightly bound
    ligands, so two fallbacks follow: perceive bond orders from the geometry, and
    failing that keep the connectivity with single bonds. Which route was used is
    reported back to the caller, since the fallbacks degrade atom typing.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    block = "".join(candidate.atom_lines) + "END\n"
    mol = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=False)
    if mol is None:
        return None, "could not parse ligand coordinates"

    smiles = candidate.info.smiles if candidate.info else ""

    if smiles:
        template = Chem.MolFromSmiles(smiles)
        if template is not None:
            try:
                assigned = AllChem.AssignBondOrdersFromTemplate(template, mol)
                if sanitize:
                    Chem.SanitizeMol(assigned)
                return assigned, ""
            except Exception:
                logger.debug(
                    "%s: CCD template did not match the modelled atoms; "
                    "falling back to perception",
                    candidate.code,
                )

        partial = _assign_from_partial_template(template, mol)
        if partial is not None:
            return partial, "bond orders from partial template match (atoms missing in the model)"

    try:
        fallback = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=False)
        fallback.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            fallback,
            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )
        return fallback, "connectivity only; bond orders unresolved"
    except Exception as exc:
        return None, f"could not build ligand ({type(exc).__name__}: {exc})"


def _strip_to_skeleton(mol):
    """Copy with all bonds single, no aromaticity and no charges."""
    from rdkit import Chem

    skeleton = Chem.RWMol(mol)
    for bond in skeleton.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in skeleton.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    return skeleton.GetMol()


def _assign_from_partial_template(template, mol):
    """
    Copy bond orders from the CCD template when the ligand is modelled with atoms
    missing.

    Disordered crystal ligands are substructures of their template, so exact
    matching fails. Comparing bond-order-agnostic skeletons instead finds where
    the observed fragment sits in the template, and the bond orders and formal
    charges are copied across that mapping. This is a direct lookup rather than a
    perception search, so it is both deterministic and fast -- unlike
    geometry-based perception, which is combinatorial and stalls on large ligands.
    """
    from rdkit import Chem

    try:
        template_skeleton = _strip_to_skeleton(template)
        mol_skeleton = _strip_to_skeleton(mol)
        match = template_skeleton.GetSubstructMatch(mol_skeleton, useChirality=False)
    except Exception:
        return None

    if not match:
        return None

    # match[i] is the template atom corresponding to observed atom i.
    editable = Chem.RWMol(mol)
    for atom in editable.GetAtoms():
        template_atom = template.GetAtomWithIdx(match[atom.GetIdx()])
        atom.SetFormalCharge(template_atom.GetFormalCharge())
        atom.SetIsAromatic(template_atom.GetIsAromatic())
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)

    for bond in editable.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        template_bond = template.GetBondBetweenAtoms(match[begin], match[end])
        if template_bond is None:
            continue
        bond.SetBondType(template_bond.GetBondType())
        bond.SetIsAromatic(template_bond.GetIsAromatic())

    result = editable.GetMol()
    try:
        Chem.SanitizeMol(result)
    except Exception:
        return None
    return result


def split_complex(
    pdb_path: Path,
    out_dir: Path,
    cache: ComponentCache,
    min_heavy_atoms: int = 6,
    max_heavy_atoms: int = 120,
    reject_nmr: bool = True,
) -> Tuple[Optional[Path], Optional[Path], str, dict]:
    """
    Write a receptor PDB and a ligand SDF for one entry.

    Returns (receptor_path, ligand_path, reason_if_skipped, metadata).
    """
    from rdkit import Chem

    lines = pdb_path.read_text(errors="ignore").splitlines(keepends=True)
    lines, is_nmr = _is_first_model_only(lines)
    if is_nmr and reject_nmr:
        return None, None, "multi-model (NMR) entry, no crystallographic pose", {}

    candidates, protein_coords, receptor_lines = parse_candidates(lines, cache)
    if len(protein_coords) == 0:
        return None, None, "no protein atoms", {}
    if not candidates:
        return None, None, "no ligand candidates (apo structure)", {}

    chosen = select_ligand(
        candidates, protein_coords,
        min_heavy_atoms=min_heavy_atoms, max_heavy_atoms=max_heavy_atoms,
    )
    if chosen is None:
        return None, None, "no candidate met size and burial criteria", {}

    mol, note = build_ligand_mol(chosen)
    if mol is None:
        return None, None, note, {"ligand_code": chosen.code}

    # Every candidate that was NOT chosen stays out of the receptor: leaving a
    # second copy of the ligand in the site would let the search score against it.
    out_dir.mkdir(parents=True, exist_ok=True)
    receptor_path = out_dir / f"{pdb_path.stem}_receptor.pdb"
    ligand_path = out_dir / f"{pdb_path.stem}_ligand.sdf"

    receptor_path.write_text("".join(receptor_lines) + "END\n")

    mol.SetProp("_Name", f"{pdb_path.stem}_{chosen.code}")
    writer = Chem.SDWriter(str(ligand_path))
    try:
        writer.write(mol)
    finally:
        writer.close()

    metadata = {
        "ligand_code": chosen.code,
        "ligand_chain": chosen.chain,
        "ligand_resseq": chosen.resseq,
        "n_heavy_atoms": chosen.n_heavy,
        "contacts": chosen.contacts,
        "ligand_name": chosen.info.name if chosen.info else "",
        "n_candidates": len(candidates),
        "note": note,
    }
    return receptor_path, ligand_path, "", metadata


def collect_het_codes(pdb_paths: Iterable[Path]) -> Set[str]:
    """Distinct HETATM codes worth looking up across a set of entries."""
    codes: Set[str] = set()
    for path in pdb_paths:
        for line in path.read_text(errors="ignore").splitlines():
            if line.startswith("HETATM") and len(line) > 20:
                code = line[17:20].strip()
                if code and code not in SOLVENT_AND_IONS:
                    codes.add(code)
    return codes
