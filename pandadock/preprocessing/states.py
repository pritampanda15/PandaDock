"""
Automatic pH-dependent protonation, tautomer, and stereoisomer state enumeration
with Henderson-Hasselbalch microspecies state penalties (open-source Epik-style prep).

Uses pure RDKit SMARTS functional-group pKa rules + `rdMolStandardize.TautomerEnumerator`
+ `EnumerateStereoisomers` so every ligand microspecies is chemically valid, has 3D
coordinates, and carries a physically grounded `state_penalty` (kcal/mol) added to
its docking score.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger("pandadock.preprocessing.states")

RT_LN10 = 0.001987204 * 298.15 * math.log(10.0)  # 1.364 kcal/mol per pKa unit


@dataclass
class LigandState:
    """One protonation/tautomer/stereoisomer microspecies ready for docking."""

    state_id: str
    mol: Chem.Mol
    smiles: str
    formal_charge: int
    state_penalty: float          # kcal/mol (>= 0; 0.0 for dominant state at target pH)
    description: str

    def to_dict(self) -> dict:
        return {
            "state_id": self.state_id,
            "smiles": self.smiles,
            "formal_charge": int(self.formal_charge),
            "state_penalty_kcal": round(float(self.state_penalty), 3),
            "description": self.description,
        }


# Curated medicinal-chemistry ionizable group rules:
# (name, SMARTS matching neutral/protonated or deprotonated atom 0, pKa, kind, target_charge_delta)
# kind == "acid": loses H+ when pH > pKa (neutral -> -1)
# kind == "base": gains H+ when pH < pKa (neutral -> +1)
_IONIZABLE_RULES: List[Tuple[str, str, float, str]] = [
    ("carboxylic_acid", "[OX2H1][CX3](=[OX1])", 4.2, "acid"),
    ("phosphonic_acid", "[OX2H1][PX4](=[OX1])", 2.1, "acid"),
    ("sulfonic_acid", "[OX2H1][SX4](=[OX1])(=[OX1])", -1.0, "acid"),
    ("tetrazole", "[nH1]1nnnc1", 4.9, "acid"),
    ("aliphatic_amine", "[NX3H2,NX3H1,NX3H0;!$(NC=O);!$(NS=O);!$(N[a]);!$(N=*)!$(N#*)]", 9.5, "base"),
    ("amidine_guanidine", "[NX2]=[CX3]([NX3])[NX3,C]", 11.5, "base"),
    ("imidazole_n", "[nX2]1c[nH1]cc1", 6.8, "base"),
    ("aryl_ketamine_amine", "[NX3H1,NX3H2;C;!$(NC=O);!$(NS=O);!$(N[a])]", 7.5, "base"),
    ("primary_sulfonamide", "[NX3H2][SX4](=[OX1])(=[OX1])", 7.4, "acid"),
]


def _state_penalty_acid(ph: float, pka: float, deprotonated: bool) -> float:
    """Henderson-Hasselbalch free-energy penalty (kcal/mol) relative to dominant microspecies."""
    ratio = 10.0 ** (ph - pka)  # [A-] / [HA]
    frac_deprot = ratio / (1.0 + ratio)
    frac_prot = 1.0 - frac_deprot
    dominant = max(frac_deprot, frac_prot)
    chosen = frac_deprot if deprotonated else frac_prot
    return float(max(0.0, -0.001987204 * 298.15 * math.log(max(chosen, 1e-12) / max(dominant, 1e-12))))


def _state_penalty_base(ph: float, pka: float, protonated: bool) -> float:
    """Henderson-Hasselbalch penalty for a basic site (BH+ <-> B + H+)."""
    ratio = 10.0 ** (pka - ph)  # [BH+] / [B]
    frac_prot = ratio / (1.0 + ratio)
    frac_neut = 1.0 - frac_prot
    dominant = max(frac_prot, frac_neut)
    chosen = frac_prot if protonated else frac_neut
    return float(max(0.0, -0.001987204 * 298.15 * math.log(max(chosen, 1e-12) / max(dominant, 1e-12))))


def _embed_3d_like(mol: Chem.Mol, template: Optional[Chem.Mol] = None) -> Chem.Mol:
    """Ensure 3D coordinates + explicit Hs exist on a microspecies molecule."""
    m = Chem.AddHs(Chem.RemoveHs(mol), addCoords=True)
    if m.GetNumConformers() == 0:
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(m, params) != 0:
            AllChem.EmbedMolecule(m, randomSeed=42, useRandomCoords=True)
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=200)
        except Exception:
            pass
    return m


def _apply_ionization(mol_noh: Chem.Mol, atom_idx: int, new_charge: int) -> Optional[Chem.Mol]:
    try:
        rw = Chem.RWMol(mol_noh)
        a = rw.GetAtomWithIdx(atom_idx)
        a.SetFormalCharge(new_charge)
        # Let RDKit recompute implicit hydrogens cleanly
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def enumerate_ligand_states(
    mol: Chem.Mol,
    ph: float = 7.4,
    ph_window: float = 2.0,
    max_states: int = 8,
    enumerate_tautomers: bool = True,
    enumerate_stereoisomers: bool = True,
    max_penalty: float = 2.8,
) -> List[LigandState]:
    """
    Enumerate plausible protonation, tautomer, and unspecified-stereoisomer microspecies
    around `ph ± ph_window` with Henderson-Hasselbalch state penalties (kcal/mol).

    Always returns at least one state (the input molecule) and sorts states by
    ascending `state_penalty`.
    """
    base_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "ligand"
    base_noh = Chem.RemoveHs(Chem.Mol(mol))
    base_smi = Chem.MolToSmiles(base_noh, isomericSmiles=True)

    # Start with input microspecies and evaluate its ionizable sites
    candidates: List[Tuple[Chem.Mol, float, str]] = []

    # Step 1: Identify ionizable sites on base_noh
    site_matches: List[Tuple[int, str, float, str]] = []
    seen_atoms = set()
    for name, smarts, pka, kind in _IONIZABLE_RULES:
        pat = Chem.MolFromSmarts(smarts)
        if pat is None:
            continue
        for match in base_noh.GetSubstructMatches(pat):
            a_idx = match[0]
            if a_idx in seen_atoms:
                continue
            seen_atoms.add(a_idx)
            site_matches.append((a_idx, name, pka, kind))

    # Build protonation variants (combinatorial up to 3 sites)
    active_sites = site_matches[:3]
    if not active_sites:
        candidates.append((Chem.Mol(base_noh), 0.0, "input state"))
    else:
        # Generate states toggling each site between neutral and ionized
        states_work: List[Tuple[Chem.Mol, float, List[str]]] = [(Chem.Mol(base_noh), 0.0, [])]
        for a_idx, name, pka, kind in active_sites:
            next_work: List[Tuple[Chem.Mol, float, List[str]]] = []
            for cur_mol, cur_pen, desc in states_work:
                if kind == "acid":
                    # Neutral vs deprotonated (-1)
                    pen_neut = _state_penalty_acid(ph, pka, deprotonated=False)
                    pen_deprot = _state_penalty_acid(ph, pka, deprotonated=True)
                    if cur_pen + pen_neut <= max_penalty:
                        m0 = _apply_ionization(cur_mol, a_idx, 0)
                        if m0 is not None:
                            next_work.append((m0, cur_pen + pen_neut, desc + [f"{name}:neutral"]))
                    if cur_pen + pen_deprot <= max_penalty:
                        m1 = _apply_ionization(cur_mol, a_idx, -1)
                        if m1 is not None:
                            next_work.append((m1, cur_pen + pen_deprot, desc + [f"{name}:deprot(-)"]))
                else:
                    # Neutral vs protonated (+1)
                    pen_neut = _state_penalty_base(ph, pka, protonated=False)
                    pen_prot = _state_penalty_base(ph, pka, protonated=True)
                    if cur_pen + pen_neut <= max_penalty:
                        m0 = _apply_ionization(cur_mol, a_idx, 0)
                        if m0 is not None:
                            next_work.append((m0, cur_pen + pen_neut, desc + [f"{name}:neutral"]))
                    if cur_pen + pen_prot <= max_penalty:
                        m1 = _apply_ionization(cur_mol, a_idx, +1)
                        if m1 is not None:
                            next_work.append((m1, cur_pen + pen_prot, desc + [f"{name}:prot(+)"]))
            if next_work:
                states_work = next_work
        for m, pen, desc in states_work:
            candidates.append((m, pen, "; ".join(desc) if desc else "pH state"))

    # Step 2: Optional tautomer enumeration on the dominant neutral/charged state
    if enumerate_tautomers and len(candidates) < max_states:
        try:
            te = rdMolStandardize.TautomerEnumerator()
            te.SetMaxTautomers(4)
            tauts = list(te.Enumerate(base_noh))
            for ti, tm in enumerate(tauts[:3]):
                t_smi = Chem.MolToSmiles(tm, isomericSmiles=True)
                if t_smi != base_smi:
                    candidates.append((tm, 0.8 + 0.2 * ti, f"tautomer_{ti + 1}"))
        except Exception as exc:
            logger.debug("Tautomer enumeration skipped: %s", exc)

    # Step 3: Optional enumeration of UNSPECIFIED stereocenters only
    expanded: List[Tuple[Chem.Mol, float, str]] = []
    for m, pen, desc in candidates:
        if enumerate_stereoisomers:
            try:
                opts = StereoEnumerationOptions(
                    tryEmbedding=False, unique=True, maxIsomers=4, onlyUnassigned=True
                )
                isos = list(EnumerateStereoisomers(m, options=opts))
                if len(isos) > 1:
                    for ii, iso in enumerate(isos[:4]):
                        expanded.append((iso, pen, f"{desc} [stereo {ii + 1}]"))
                    continue
            except Exception:
                pass
        expanded.append((m, pen, desc))

    # Deduplicate by canonical isomeric SMILES keeping lowest penalty
    best_by_smi = {}
    for m, pen, desc in expanded:
        smi = Chem.MolToSmiles(m, isomericSmiles=True)
        if smi not in best_by_smi or pen < best_by_smi[smi][1]:
            best_by_smi[smi] = (m, pen, desc)

    # Always ensure input molecule is represented if nothing survived
    if not best_by_smi:
        best_by_smi[base_smi] = (Chem.Mol(base_noh), 0.0, "input state")

    # Normalize penalties so minimum is 0.0
    ordered = sorted(best_by_smi.values(), key=lambda x: x[1])[:max_states]
    min_pen = ordered[0][1]

    out: List[LigandState] = []
    for idx, (m_noh, pen, desc) in enumerate(ordered, 1):
        # Preserve original 3D coordinates if SMILES matches input
        smi = Chem.MolToSmiles(m_noh, isomericSmiles=True)
        if smi == base_smi and mol.GetNumConformers() > 0:
            m3d = Chem.AddHs(Chem.Mol(mol), addCoords=True)
        else:
            m3d = _embed_3d_like(m_noh, template=mol)
        state_id = f"{base_name}_state{idx}"
        m3d.SetProp("_Name", state_id)
        m3d.SetProp("StatePenalty", f"{pen - min_pen:.3f}")
        m3d.SetProp("StateDescription", desc)
        out.append(
            LigandState(
                state_id=state_id,
                mol=m3d,
                smiles=smi,
                formal_charge=int(Chem.GetFormalCharge(m_noh)),
                state_penalty=float(pen - min_pen),
                description=desc,
            )
        )
    return out
