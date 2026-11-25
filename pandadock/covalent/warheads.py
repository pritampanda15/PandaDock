"""
Covalent warhead reaction library and SMARTS-based adduct builder.

Each warhead transforms an unreacted small-molecule inhibitor into the
covalently bound adduct carrying a dummy anchor atom (`[*:99]`) at the
nucleophile attachment position (e.g., Cys Sγ or Ser Oγ).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions

logger = logging.getLogger("pandadock.covalent.warheads")


@dataclass(frozen=True)
class WarheadReaction:
    """Definition of a covalent warhead electrophile and its nucleophile reaction."""

    name: str
    description: str
    nucleophiles: Tuple[str, ...]
    detect_smarts: str
    reaction_smarts: str  # reactant -> adduct containing [*:99] as nucleophile anchor
    bond_length_A: float = 1.82  # typical C-S (~1.82 A) or C-O (~1.43 A) bond length


# Ordered from most specific to most general for automatic warhead detection
WARHEAD_LIBRARY: Dict[str, WarheadReaction] = {
    "michael_acceptor": WarheadReaction(
        name="michael_acceptor",
        description="α,β-unsaturated carbonyl / acrylamide / vinyl sulfone (1,4-Michael addition)",
        nucleophiles=("CYS", "LYS", "SER", "HIS"),
        detect_smarts="[CX3H2,CX3H1]=[CX3H1,CX3H0][CX3,SX4](=[OX1])",
        reaction_smarts="[C:1]=[C:2][C,S:3](=[O:4])>>[*:99][C:1][C:2][C,S:3](=[O:4])",
        bond_length_A=1.82,
    ),
    "chloroacetamide": WarheadReaction(
        name="chloroacetamide",
        description="α-haloacetamide / α-haloketone (SN2 displacement of Cl/Br/I)",
        nucleophiles=("CYS", "LYS", "HIS"),
        detect_smarts="[Cl,Br,I][CH2][CX3](=[OX1])",
        reaction_smarts="[Cl,Br,I][CH2:1][C:2](=[O:3])>>[*:99][CH2:1][C:2](=[O:3])",
        bond_length_A=1.82,
    ),
    "sulfonyl_fluoride": WarheadReaction(
        name="sulfonyl_fluoride",
        description="Sulfonyl fluoride / SuFEx warhead (S-F displacement)",
        nucleophiles=("SER", "THR", "TYR", "LYS", "HIS"),
        detect_smarts="[SX4](=[OX1])(=[OX1])[FX1]",
        reaction_smarts="[S:1](=[O:2])(=[O:3])[F]>>[*:99][S:1](=[O:2])(=[O:3])",
        bond_length_A=1.60,
    ),
    "nitrile": WarheadReaction(
        name="nitrile",
        description="Electrophilic nitrile (reversible covalent thioimidate / imidate adduct)",
        nucleophiles=("CYS", "SER"),
        detect_smarts="[CX2]#[NX1]",
        reaction_smarts="[C:1]#[N:2]>>[*:99][C:1]=[NH:2]",
        bond_length_A=1.78,
    ),
    "epoxide": WarheadReaction(
        name="epoxide",
        description="Epoxide / oxirane ring opening",
        nucleophiles=("CYS", "SER", "THR", "HIS", "ASP", "GLU"),
        detect_smarts="[CX4]1[OX2][CX4]1",
        reaction_smarts="[C:1]1[O:2][C:3]1>>[*:99][C:1][C:3][OH:2]",
        bond_length_A=1.82,
    ),
    "boronic_acid": WarheadReaction(
        name="boronic_acid",
        description="Boronic acid / boronate ester tetrahedral adduct",
        nucleophiles=("SER", "THR", "CYS"),
        detect_smarts="[BX3]([OX2H1,OX2])[OX2H1,OX2]",
        reaction_smarts="[B:1]([O:2])[O:3]>>[*:99][B-:1]([O:2])[O:3]",
        bond_length_A=1.48,
    ),
    "aldehyde": WarheadReaction(
        name="aldehyde",
        description="Aldehyde / α-ketoamide (hemithioacetal / hemiacetal / Schiff base)",
        nucleophiles=("CYS", "SER", "THR", "LYS"),
        detect_smarts="[CX3H1](=[OX1])[#6]",
        reaction_smarts="[CH1:1](=[O:2])[#6:3]>>[*:99][CH1:1]([OH:2])[#6:3]",
        bond_length_A=1.82,
    ),
    "disulfide": WarheadReaction(
        name="disulfide",
        description="Disulfide exchange warhead",
        nucleophiles=("CYS",),
        detect_smarts="[SX2][SX2]",
        reaction_smarts="[S:1][S:2]>>[*:99][S:1]",
        bond_length_A=2.05,
    ),
}

NUCLEOPHILE_ATOM_DEFAULTS: Dict[str, Tuple[str, str, float]] = {
    # resname -> (reactive_atom_name, parent_atom_name, typical_bond_length_A)
    "CYS": ("SG", "CB", 1.82),
    "CYX": ("SG", "CB", 1.82),
    "SER": ("OG", "CB", 1.43),
    "THR": ("OG1", "CB", 1.43),
    "TYR": ("OH", "CZ", 1.43),
    "LYS": ("NZ", "CE", 1.47),
    "HIS": ("NE2", "CD2", 1.47),
}


def detect_warhead(mol: Chem.Mol) -> Optional[WarheadReaction]:
    """Automatically detect the first matching covalent warhead in `mol`."""
    for wh in WARHEAD_LIBRARY.values():
        pat = Chem.MolFromSmarts(wh.detect_smarts)
        if pat is not None and mol.HasSubstructMatch(pat):
            return wh
    return None


def react_ligand_with_warhead(
    mol: Chem.Mol,
    warhead: Optional[str] = None,
    nucleophile_element: str = "S",
) -> Tuple[Chem.Mol, int, int, WarheadReaction]:
    """Convert unreacted ligand into a 3D covalent adduct with an anchor atom.

    The dummy `[*:99]` produced by the reaction SMARTS is replaced by a real
    element (`S` for Cys, `O` for Ser/Thr/Tyr, `N` for Lys/His) representing the
    receptor's nucleophilic atom so RDKit ETKDGv3 + MMFF94 generates an accurate
    3D geometry around the newly formed covalent bond.

    Returns:
        `(adduct_mol_with_3D, anchor_atom_idx, attached_ligand_atom_idx, warhead_reaction)`
    """
    if warhead and warhead.lower() != "auto":
        key = warhead.lower()
        if key not in WARHEAD_LIBRARY:
            raise ValueError(
                f"Unknown warhead '{warhead}'. Available: {', '.join(WARHEAD_LIBRARY.keys())}"
            )
        wh = WARHEAD_LIBRARY[key]
    else:
        wh = detect_warhead(mol)
        if wh is None:
            raise ValueError(
                "Could not auto-detect a supported covalent warhead in ligand. "
                f"Specify --warhead explicitly ({', '.join(WARHEAD_LIBRARY.keys())})."
            )

    rxn = rdChemReactions.ReactionFromSmarts(wh.reaction_smarts)
    mol_no_h = Chem.RemoveHs(Chem.Mol(mol))
    products = rxn.RunReactants((mol_no_h,))
    if not products:
        raise ValueError(
            f"Warhead '{wh.name}' ({wh.description}) did not match ligand structure."
        )

    # Pick the largest sanitized product containing the [*:99] anchor atom
    chosen = None
    for prod_tuple in products:
        cand = prod_tuple[0]
        try:
            Chem.SanitizeMol(cand)
        except Exception:
            continue
        if any(a.GetAtomMapNum() == 99 or a.GetAtomicNum() == 0 for a in cand.GetAtoms()):
            chosen = cand
            break

    if chosen is None:
        raise ValueError(f"Failed to sanitize covalent adduct for warhead '{wh.name}'")

    # Replace the dummy anchor with the receptor nucleophile element (capped with methyl for geometry,
    # or simple H-capped heteroatom: e.g. -SH / -OH / -NH2; we mark its index cleanly).
    rw = Chem.RWMol(chosen)
    anchor_idx = None
    for atom in rw.GetAtoms():
        if atom.GetAtomMapNum() == 99 or atom.GetAtomicNum() == 0:
            anchor_idx = atom.GetIdx()
            atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(nucleophile_element))
            atom.SetAtomMapNum(0)
            atom.SetFormalCharge(0)
            break

    if anchor_idx is None:
        raise ValueError("Anchor atom lost during covalent adduct assembly")

    adduct = rw.GetMol()
    Chem.SanitizeMol(adduct)

    # Identify the ligand heavy atom directly bonded to the nucleophile anchor
    anchor_atom = adduct.GetAtomWithIdx(anchor_idx)
    neighbors = [n.GetIdx() for n in anchor_atom.GetNeighbors() if n.GetAtomicNum() > 1]
    if not neighbors:
        raise ValueError("Covalent anchor atom has no bonded ligand heavy atom")
    attached_idx = int(neighbors[0])

    # Add hydrogens and generate a clean 3D conformer
    adduct_3d = Chem.AddHs(adduct)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FA1
    if AllChem.EmbedMolecule(adduct_3d, params) != 0:
        if AllChem.EmbedMolecule(adduct_3d, useRandomCoords=True, randomSeed=0xC0FA1) != 0:
            raise ValueError("Failed to generate 3D conformer for covalent adduct")
    try:
        AllChem.MMFFOptimizeMolecule(adduct_3d, maxIters=500)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(adduct_3d, maxIters=500)
        except Exception:
            pass

    if mol.HasProp("_Name"):
        adduct_3d.SetProp("_Name", f"{mol.GetProp('_Name')}_covalent_{wh.name}")
    else:
        adduct_3d.SetProp("_Name", f"ligand_covalent_{wh.name}")

    return adduct_3d, int(anchor_idx), int(attached_idx), wh
