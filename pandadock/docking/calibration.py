"""
Per-target few-shot scoring weight calibration (`pandadock calibrate`).

Given a receptor, a small training set of docked or crystallographic ligands
(as few as 4-25 compounds) and measured labels (experimental pIC50 / ΔG or
reference scores), extracts the exact 5 unweighted Vina interaction terms
(`gauss1`, `gauss2`, `repulsion`, `hydrophobic`, `hydrogen`) plus rotatable
bond count `n_rot`, and fits Ridge-regularized target-specific weights shrunk
toward the standard Vina prior via leave-one-out cross-validation (LOOCV).

The resulting `calibrated_weights.json` can be passed directly to:
    pandadock dock --weights-file calibrated_weights.json ...
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import Lipinski
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr

from .scoring.vina_scoring import VinaScoring

logger = logging.getLogger("pandadock.docking.calibration")

TERM_NAMES = ("gauss1", "gauss2", "repulsion", "hydrophobic", "hydrogen")


def extract_vina_terms(
    ligand_mol: Chem.Mol,
    receptor_structure,
    scoring: Optional[VinaScoring] = None,
) -> Dict[str, float]:
    """Extract the 5 raw unweighted Vina interaction terms and `n_rot` for a 3D ligand pose."""
    sc = scoring or VinaScoring(preset="vina")
    conf = ligand_mol.GetConformer(0)
    lig_coords_full = np.asarray(conf.GetPositions(), dtype=np.float64)

    heavy_idx = [a.GetIdx() for a in ligand_mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy_idx:
        return {k: 0.0 for k in TERM_NAMES} | {"n_rot": 0.0, "default_vina": 0.0}

    # Full ligand arrays so we reuse VinaScoring's exact hydrophobic & H-bond routines
    lig_types = sc._get_ligand_atom_types(ligand_mol)
    lig_radii = np.array([sc._get_vdw_radius(t) for t in lig_types], dtype=np.float64)

    # Receptor atoms within cutoff + 3 A bounding box
    all_rec_atoms = list(receptor_structure.get_atoms())
    all_rec_coords = np.asarray([a.get_coord() for a in all_rec_atoms], dtype=np.float64)
    lo = lig_coords_full.min(axis=0) - (sc.cutoff + 3.0)
    hi = lig_coords_full.max(axis=0) + (sc.cutoff + 3.0)
    mask = np.all((all_rec_coords >= lo) & (all_rec_coords <= hi), axis=1)
    rec_atoms = [all_rec_atoms[i] for i in np.where(mask)[0]]
    rec_coords = all_rec_coords[mask]

    if len(rec_coords) == 0:
        return {k: 0.0 for k in TERM_NAMES} | {"n_rot": float(Lipinski.NumRotatableBonds(ligand_mol)), "default_vina": 0.0}

    rec_types = sc._get_receptor_atom_types(rec_atoms)
    rec_radii = np.array([sc._get_vdw_radius(t) for t in rec_types], dtype=np.float64)

    dists = cdist(lig_coords_full, rec_coords)
    surf = dists - (lig_radii[:, None] + rec_radii[None, :])

    g1 = float(sc._calculate_gauss(surf, sc.gauss1_offset, sc.gauss1_width))
    g2 = float(sc._calculate_gauss(surf, sc.gauss2_offset, sc.gauss2_width))
    rep = float(sc._calculate_repulsion(surf))
    hydro = float(sc._calculate_hydrophobic(surf, lig_types, rec_types, ligand_mol, rec_atoms))
    hbond = float(sc._calculate_hydrogen_bonds(surf, lig_types, rec_types, ligand_mol, rec_atoms))

    n_rot = float(sc._count_rotatable_bonds(ligand_mol))
    denom = 1.0 + sc.n_rot_weight * n_rot
    raw_terms = {
        "gauss1": g1 / denom,
        "gauss2": g2 / denom,
        "repulsion": rep / denom,
        "hydrophobic": hydro / denom,
        "hydrogen": hbond / denom,
    }
    default_vina = sum(sc.weights[k] * raw_terms[k] for k in TERM_NAMES)
    raw_terms["n_rot"] = n_rot
    raw_terms["default_vina"] = float(default_vina)
    return raw_terms


def _load_labels_csv(labels_csv: str) -> Dict[str, float]:
    """Load `name,target` (or `ligand,affinity` / `id,dg`) CSV labels."""
    labels: Dict[str, float] = {}
    with open(labels_csv, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"Empty labels CSV: {labels_csv}")
        name_col = next(
            (c for c in reader.fieldnames if c.strip().lower() in ("name", "ligand", "id", "compound", "title")),
            reader.fieldnames[0],
        )
        val_col = next(
            (c for c in reader.fieldnames if c.strip().lower() in ("dg", "delta_g", "affinity", "score", "pic50", "target", "y", "label")),
            reader.fieldnames[1],
        )
        for row in reader:
            k = (row.get(name_col) or "").strip()
            v_str = (row.get(val_col) or "").strip()
            if k and v_str:
                labels[k] = float(v_str)
    return labels


def _safe_corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan")
    r, _ = pearsonr(x, y)
    rho, _ = spearmanr(x, y)
    return float(r), float(rho)


def calibrate_scoring_weights(
    receptor_file: str,
    poses_sdf: str,
    labels_csv: str,
    output_json: str,
    preset: str = "vina",
    higher_is_better: bool = False,
    alphas: Sequence[float] = (0.05, 0.25, 1.0, 5.0, 20.0, 100.0),
) -> Dict[str, Any]:
    """Fit Ridge-regularized target-specific Vina weights with LOOCV validation.

    The Ridge penalty shrinks `(w - w_prior)` toward standard Vina/Vinardo
    weights `w_prior`, so even on small congeneric sets (N=4-20) the calibrated
    weights remain physically well-behaved and cannot explode.
    """
    labels = _load_labels_csv(labels_csv)
    parser = PDBParser(QUIET=True)
    receptor = parser.get_structure("rec", receptor_file)
    base_scoring = VinaScoring(preset=preset)
    w_prior = np.array([base_scoring.weights[k] for k in TERM_NAMES], dtype=np.float64)

    supplier = Chem.SDMolSupplier(str(poses_sdf), removeHs=False)
    mols = [m for m in supplier if m is not None]
    if not mols:
        raise ValueError(f"No valid molecules in {poses_sdf}")

    rows: List[Dict[str, Any]] = []
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    default_preds: List[float] = []

    for idx, mol in enumerate(mols):
        name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else f"ligand_{idx+1}"
        # Match exact name or case-insensitive prefix
        label_val = labels.get(name)
        if label_val is None:
            low_map = {k.lower(): v for k, v in labels.items()}
            label_val = low_map.get(name.lower())
        if label_val is None:
            continue
        target_y = -float(label_val) if higher_is_better else float(label_val)
        terms = extract_vina_terms(mol, receptor, base_scoring)
        x_vec = np.array([terms[k] for k in TERM_NAMES], dtype=np.float64)
        X_list.append(x_vec)
        y_list.append(target_y)
        default_preds.append(terms["default_vina"])
        rows.append({"name": name, "target_energy": target_y, "default_vina": terms["default_vina"], **terms})

    n = len(X_list)
    if n < 3:
        raise ValueError(
            f"Need at least 3 matched ligands between {poses_sdf} and {labels_csv}; matched {n}"
        )

    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=np.float64)
    y_default = np.asarray(default_preds, dtype=np.float64)

    # Standardize target scale into Vina energy range if labels are e.g. pIC50 or Glide scores
    # Fit: y ≈ X @ w + intercept, with Ridge prior shrinkage toward w_prior:
    #   min || (X w + b) - y ||^2 + alpha * || w - w_prior ||^2
    # Center X and y for intercept b, solve (Xc^T Xc + alpha I) (w - w_prior) = Xc^T (yc - Xc w_prior)
    def fit_ridge_prior(X_tr: np.ndarray, y_tr: np.ndarray, alpha: float) -> Tuple[np.ndarray, float]:
        xm = X_tr.mean(axis=0)
        ym = y_tr.mean()
        Xc = X_tr - xm
        yc = y_tr - ym
        A = Xc.T @ Xc + alpha * np.eye(X_tr.shape[1])
        rhs = Xc.T @ (yc - Xc @ w_prior)
        w = w_prior + np.linalg.solve(A, rhs)
        # Keep repulsion non-negative for physical clash safety
        w[2] = max(0.25, float(w[2]))
        b = float(ym - xm @ w)
        return w, b

    best_alpha = float(alphas[0])
    best_loocv_rmse = float("inf")
    best_loocv_preds = np.zeros(n, dtype=np.float64)

    for alpha in alphas:
        preds = np.zeros(n, dtype=np.float64)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            w_i, b_i = fit_ridge_prior(X[mask], y[mask], float(alpha))
            preds[i] = float(X[i] @ w_i + b_i)
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        if rmse < best_loocv_rmse:
            best_loocv_rmse = rmse
            best_alpha = float(alpha)
            best_loocv_preds = preds

    w_final, b_final = fit_ridge_prior(X, y, best_alpha)
    train_preds = X @ w_final + b_final

    def_r, def_rho = _safe_corr(y_default, y)
    loo_r, loo_rho = _safe_corr(best_loocv_preds, y)
    tr_r, tr_rho = _safe_corr(train_preds, y)

    payload: Dict[str, Any] = {
        "preset": preset,
        "n_ligands": n,
        "regularization_alpha": best_alpha,
        "weights": {k: round(float(w_final[i]), 6) for i, k in enumerate(TERM_NAMES)},
        "prior_weights": {k: round(float(w_prior[i]), 6) for i, k in enumerate(TERM_NAMES)},
        "n_rot_weight": base_scoring.n_rot_weight,
        "intercept": round(float(b_final), 4),
        "metrics": {
            "default_pearson_r": None if np.isnan(def_r) else round(def_r, 4),
            "default_spearman_rho": None if np.isnan(def_rho) else round(def_rho, 4),
            "loocv_pearson_r": None if np.isnan(loo_r) else round(loo_r, 4),
            "loocv_spearman_rho": None if np.isnan(loo_rho) else round(loo_rho, 4),
            "loocv_rmse": round(best_loocv_rmse, 4),
            "train_pearson_r": None if np.isnan(tr_r) else round(tr_r, 4),
            "train_spearman_rho": None if np.isnan(tr_rho) else round(tr_rho, 4),
        },
        "ligands": [
            {
                "name": rows[i]["name"],
                "target": round(float(y[i]), 4),
                "default_vina": round(float(y_default[i]), 4),
                "loocv_predicted": round(float(best_loocv_preds[i]), 4),
                "calibrated_predicted": round(float(train_preds[i]), 4),
            }
            for i in range(n)
        ],
    }

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_calibrated_weights(weights_file: str) -> Dict[str, float]:
    """Load calibrated weights dict from a `calibrated_weights.json` file."""
    data = json.loads(Path(weights_file).read_text(encoding="utf-8"))
    weights = dict(data.get("weights", data))
    if "n_rot_weight" in data:
        weights["n_rot_weight"] = float(data["n_rot_weight"])
    return weights
