"""
Docking run report: plots and an HTML summary.

Produces the diagnostics that inform a decision about a docking run, rather than
restating the score in several forms:

- Score by rank, with the clustering gap visible
- Energy terms with an explicit favourable/unfavourable split
- Pose diversity, as a heavy-atom RMSD matrix between returned modes
- Interaction fingerprint: which residues each pose touches, and how

Deliberately omitted: plots of Kd or IC50 against the docking score. Those are a
fixed algebraic transform of one another (Kd = exp(dG/RT)), so the plot is a
straight line by construction and carries no information beyond the conversion.
Presenting it next to experimental axis labels implies a validation that has not
been performed. A docking score is also not a binding free energy: PandaDock's
own cross-family benchmark puts the empirical score's correlation with measured
affinity near zero, so converting it to a picomolar IC50 and plotting the result
overstates what the number supports.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger("pandadock.visualization.report")

# Terms that stabilise the complex versus those that penalise it. Keeping the
# split explicit prevents the sign error that makes a steric clash look
# favourable when every bar is drawn below the axis.
FAVOURABLE_TERMS = {"intermolecular", "gauss1", "gauss2", "hydrophobic", "hydrogen"}
UNFAVOURABLE_TERMS = {"intramolecular", "repulsion", "clash", "strain", "entropy"}


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_pose_scores(poses: Sequence, output_file: Path, title: str = "") -> Optional[Path]:
    """Score by rank, annotated with the gap between best and runner-up."""
    if not poses:
        return None
    plt = _matplotlib()

    scores = [p.energy for p in poses]
    ranks = np.arange(1, len(scores) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ranks, scores, "o--", color="#4C72B0", markersize=9, linewidth=1.5)
    ax.plot(ranks[0], scores[0], "*", color="#DD8452", markersize=20, label="Best pose")

    if len(scores) > 1:
        gap = scores[1] - scores[0]
        ax.annotate(
            f"gap to rank 2: {gap:.2f} kcal/mol",
            xy=(1, scores[0]), xytext=(1.5, scores[0] - 0.05 * (max(scores) - min(scores) + 1)),
            fontsize=9, color="#555555",
        )

    ax.set_xlabel("Pose rank")
    ax.set_ylabel("Score (kcal/mol)")
    ax.set_title(title or "Pose scores")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def plot_energy_components(poses: Sequence, output_file: Path) -> Optional[Path]:
    """
    Per-pose energy terms, split by whether they stabilise or destabilise.

    Favourable terms are drawn below the axis and penalties above it, so a clash
    or strain term reads as a cost. Drawing every term downward -- as the earlier
    plots did -- makes a steric clash look like a binding contribution.
    """
    components = [p.energy_components for p in poses if p.energy_components]
    if not components:
        return None
    plt = _matplotlib()

    names = sorted({k for c in components for k in c if k != "total"})
    if not names:
        return None

    ranks = np.arange(1, len(components) + 1)
    width = 0.8 / max(len(names), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")

    for i, name in enumerate(names):
        values = [c.get(name, 0.0) for c in components]
        offset = (i - (len(names) - 1) / 2) * width
        favourable = name in FAVOURABLE_TERMS
        label = f"{name} ({'favourable' if favourable else 'penalty'})"
        ax.bar(ranks + offset, values, width=width, label=label,
               color=cmap(i % 10), edgecolor="black", linewidth=0.4)

    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xlabel("Pose rank")
    ax.set_ylabel("Energy contribution (kcal/mol)")
    ax.set_title("Energy terms per pose (negative stabilises the complex)")
    ax.set_xticks(ranks)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def pose_rmsd_matrix(poses: Sequence, heavy_atoms: Optional[np.ndarray] = None) -> np.ndarray:
    """Pairwise in-place heavy-atom RMSD between returned poses."""
    coords = [p.coordinates for p in poses]
    if heavy_atoms is not None:
        coords = [c[heavy_atoms] for c in coords]

    n = len(coords)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rmsd = float(np.sqrt(np.mean(np.sum((coords[i] - coords[j]) ** 2, axis=1))))
            matrix[i, j] = matrix[j, i] = rmsd
    return matrix


def plot_pose_diversity(poses: Sequence, output_file: Path,
                        heavy_atoms: Optional[np.ndarray] = None) -> Optional[Path]:
    """
    RMSD between returned poses.

    Shows whether the returned set represents genuinely different binding modes
    or minor variations on one. A run where every pair sits just above the
    clustering cutoff has found one mode, not several, however many poses it
    reports.
    """
    if len(poses) < 2:
        return None
    plt = _matplotlib()

    matrix = pose_rmsd_matrix(poses, heavy_atoms)
    n = len(poses)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, cmap="viridis", origin="upper")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(1, n + 1))
    ax.set_yticklabels(range(1, n + 1))
    ax.set_xlabel("Pose rank")
    ax.set_ylabel("Pose rank")
    ax.set_title("Pairwise heavy-atom RMSD between poses (A)")

    if n <= 12:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if matrix[i, j] < matrix.max() * 0.6 else "black")

    fig.colorbar(im, ax=ax, label="RMSD (A)")
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def plot_interaction_fingerprint(per_pose_interactions: List[Dict], output_file: Path,
                                 max_residues: int = 25) -> Optional[Path]:
    """
    Residue-by-pose contact map.

    Reveals whether the top poses engage the same residues. Poses that score
    similarly but contact different residues represent a real ambiguity that a
    single score cannot express.
    """
    if not per_pose_interactions:
        return None

    residue_counts: Dict[str, int] = {}
    for pose_data in per_pose_interactions:
        for interactions in pose_data.get("interaction_details", {}).values():
            for item in interactions:
                residue = item.get("residue")
                if residue:
                    residue_counts[residue] = residue_counts.get(residue, 0) + 1

    if not residue_counts:
        return None

    residues = [r for r, _ in sorted(residue_counts.items(),
                                     key=lambda kv: -kv[1])[:max_residues]]
    residues.sort()

    matrix = np.zeros((len(residues), len(per_pose_interactions)))
    for j, pose_data in enumerate(per_pose_interactions):
        for interactions in pose_data.get("interaction_details", {}).values():
            for item in interactions:
                residue = item.get("residue")
                if residue in residues:
                    matrix[residues.index(residue), j] += 1

    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(max(6, len(per_pose_interactions) * 0.8),
                                    max(4, len(residues) * 0.28)))
    im = ax.imshow(matrix, cmap="magma", aspect="auto", origin="upper")
    ax.set_xticks(range(len(per_pose_interactions)))
    ax.set_xticklabels(range(1, len(per_pose_interactions) + 1))
    ax.set_yticks(range(len(residues)))
    ax.set_yticklabels(residues, fontsize=8)
    ax.set_xlabel("Pose rank")
    ax.set_title("Interactions per residue")
    fig.colorbar(im, ax=ax, label="Interaction count")
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


HTML_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>PandaDock report - {name}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem auto;max-width:1000px;
      line-height:1.5;color:#222;padding:0 1rem}}
 h1{{margin-bottom:.2rem}} .sub{{color:#666;margin-top:0}}
 table{{border-collapse:collapse;margin:1rem 0;width:100%}}
 th,td{{border:1px solid #ddd;padding:.4rem .6rem;text-align:right;font-variant-numeric:tabular-nums}}
 th{{background:#f5f5f5;text-align:left}} td:first-child,th:first-child{{text-align:left}}
 img{{max-width:100%;border:1px solid #eee;border-radius:4px;margin:.5rem 0}}
 .note{{background:#fff8e6;border-left:4px solid #e0a800;padding:.6rem 1rem;margin:1rem 0}}
 code{{background:#f4f4f4;padding:.1rem .3rem;border-radius:3px}}
</style>
<h1>{name}</h1>
<p class="sub">{receptor}</p>

<h2>Run</h2>
<table>
<tr><th>Best score</th><td>{best:.2f} kcal/mol</td></tr>
<tr><th>Ensemble &Delta;G</th><td>{ensemble:.2f} kcal/mol</td></tr>
<tr><th>Binding modes returned</th><td>{n_poses}</td></tr>
<tr><th>Rotatable bonds searched</th><td>{n_torsions}</td></tr>
<tr><th>Exhaustiveness</th><td>{exhaustiveness}</td></tr>
<tr><th>Energy evaluations</th><td>{evaluations:,}</td></tr>
<tr><th>Runtime</th><td>{runtime:.1f} s</td></tr>
<tr><th>Seed</th><td>{seed}</td></tr>
</table>

<div class="note">
The score is an empirical docking score in kcal/mol. It ranks poses; it is not a
measured binding free energy and should not be converted to a Kd or IC50 and
reported as a potency prediction.
</div>

{sections}

<h2>Files</h2>
<ul>
<li><code>poses.sdf</code> &mdash; all poses with bond orders, scores and ranks</li>
<li><code>pose{{N}}.pdb</code> &mdash; individual ligand poses</li>
<li><code>complex{{N}}.pdb</code> &mdash; receptor plus ligand</li>
<li><code>interaction_analysis.json</code> &mdash; detected interactions</li>
</ul>
"""


def build_html_report(output_dir: Path, result, images: Dict[str, Path],
                      ligand_name: str = "ligand") -> Optional[Path]:
    """Assemble the generated plots into a single self-describing HTML page."""
    try:
        params = result.parameters or {}
        sections = []
        captions = {
            "scores": ("Pose scores",
                       "Ranked scores. A large gap to rank 2 indicates a "
                       "well-separated best pose."),
            "components": ("Energy terms",
                           "Contribution of each term. Negative values stabilise "
                           "the complex; positive values are penalties."),
            "diversity": ("Pose diversity",
                          "Pairwise RMSD between returned poses. Low values "
                          "throughout mean one binding mode, not several."),
            "fingerprint": ("Interaction fingerprint",
                            "Residues contacted by each pose."),
            "pandamap": ("Protein-ligand interaction map",
                         "2D interaction diagram for the top pose (PandaMap)."),
        }
        for key, path in images.items():
            if path is None:
                continue
            title, caption = captions.get(key, (key, ""))
            sections.append(
                f"<h2>{title}</h2>\n<p>{caption}</p>\n"
                f'<img src="{Path(path).name}" alt="{title}">'
            )

        html = HTML_TEMPLATE.format(
            name=ligand_name,
            receptor=result.receptor_file,
            best=result.poses[0].energy if result.poses else float("nan"),
            ensemble=result.ensemble_binding_energy,
            n_poses=len(result.poses),
            n_torsions=params.get("n_torsions", "?"),
            exhaustiveness=params.get("exhaustiveness", "?"),
            evaluations=int(params.get("energy_evaluations", 0)),
            runtime=result.runtime_seconds,
            seed=params.get("seed") if params.get("seed") is not None else "random",
            sections="\n".join(sections),
        )
        path = output_dir / "report.html"
        path.write_text(html)
        return path
    except Exception as exc:
        logger.error("Could not build HTML report: %s", exc)
        return None


def generate_report(result, output_dir: Path, ligand_mol=None,
                    receptor_file: Optional[str] = None,
                    interaction_analyses: Optional[List[Dict]] = None,
                    run_pandamap: bool = True) -> Dict[str, Path]:
    """
    Generate all plots and the HTML report for a docking result.

    Returns the mapping of generated artefacts. Failures are logged and skipped
    individually so one unavailable plotting dependency cannot lose the rest.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images: Dict[str, Path] = {}

    heavy = None
    if ligand_mol is not None:
        heavy = np.array(
            [a.GetIdx() for a in ligand_mol.GetAtoms() if a.GetAtomicNum() > 1],
            dtype=np.int64,
        )

    ligand_name = getattr(result, "ligand_name", "ligand")

    tasks = [
        ("scores", lambda: plot_pose_scores(result.poses, output_dir / "pose_scores.png",
                                            f"Pose scores - {ligand_name}")),
        ("components", lambda: plot_energy_components(
            result.poses, output_dir / "energy_components.png")),
        ("diversity", lambda: plot_pose_diversity(
            result.poses, output_dir / "pose_diversity.png", heavy)),
    ]
    if interaction_analyses:
        tasks.append(("fingerprint", lambda: plot_interaction_fingerprint(
            interaction_analyses, output_dir / "interaction_fingerprint.png")))

    for key, task in tasks:
        try:
            path = task()
            if path is not None:
                images[key] = path
        except Exception as exc:
            logger.warning("Could not generate %s plot: %s", key, exc)

    if run_pandamap:
        try:
            path = _run_pandamap(output_dir)
            if path is not None:
                images["pandamap"] = path
        except Exception as exc:
            logger.warning("PandaMap visualisation unavailable: %s", exc)

    build_html_report(output_dir, result, images, ligand_name)
    return images


def _run_pandamap(output_dir: Path) -> Optional[Path]:
    """Generate a 2D interaction diagram for the top complex, if PandaMap is present."""
    # `dock` writes complex1.pdb while `hybrid` writes complex_1.pdb; accept both
    # rather than silently skipping the diagram for one of them.
    candidates = sorted(output_dir.glob("complex1.pdb")) or sorted(
        output_dir.glob("complex_1.pdb")
    )
    if not candidates:
        return None

    from ..visualization.pandamap.pandamap_integration import PandaMapIntegration

    integration = PandaMapIntegration(str(output_dir))
    if not integration._check_pandamap_availability():
        logger.info("PandaMap is not installed; skipping the interaction diagram. "
                    "Install it with: pip install pandamap")
        return None

    integration.analyze_complex_pdbs(str(output_dir), top_n=1, generate_3d=False)
    for candidate in sorted(output_dir.glob("pandamap_2d_*.png")):
        return candidate
    return None


def load_result_from_dir(results_dir: Path):
    """
    Reconstruct a docking result from the JSON a run writes.

    Lets the reporting commands operate on an ordinary `pandadock dock` output
    directory. `pandadock-report plots` previously accepted only an
    algorithm-comparison layout and reported "No valid results found" for a
    normal run -- then printed a success banner and wrote nothing.

    Returns None when the directory holds no docking output, so callers can fall
    back to another loader rather than treating absence as failure.
    """
    from ..docking.core import DockingResult, Pose

    results_dir = Path(results_dir)
    summaries = sorted(results_dir.glob("*_summary.json"))
    poses_files = sorted(results_dir.glob("*_poses.json"))
    if not summaries or not poses_files:
        return None

    try:
        summary = json.loads(summaries[0].read_text())
        poses_blob = json.loads(poses_files[0].read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read docking output in %s: %s", results_dir, exc)
        return None

    raw_poses = poses_blob.get("poses", poses_blob) if isinstance(poses_blob, dict) else poses_blob
    if not raw_poses:
        return None

    poses = []
    for entry in raw_poses:
        try:
            poses.append(
                Pose(
                    coordinates=np.asarray(entry["coordinates"], dtype=float),
                    center=np.asarray(entry.get("center", [0, 0, 0]), dtype=float),
                    rotation=np.asarray(entry.get("rotation", [0, 0, 0, 1]), dtype=float),
                    conformer_id=int(entry.get("conformer_id", 0)),
                    energy=float(entry.get("energy", 0.0)),
                    energy_components=entry.get("energy_components") or {},
                    internal_strain=float(entry.get("internal_strain", 0.0)),
                    confidence=float(entry.get("confidence", 0.0)),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("Skipping malformed pose entry: %s", exc)

    if not poses:
        return None

    result = DockingResult(
        ligand_name=summary.get("ligand_name", "ligand"),
        receptor_file=summary.get("receptor_file", ""),
        grid_center=np.asarray(summary.get("grid_center", [0, 0, 0]), dtype=float),
        grid_dimensions=np.asarray(summary.get("grid_dimensions", [0, 0, 0]), dtype=float),
        algorithm_used=summary.get("algorithm_used", "pandadock"),
        scoring_function=summary.get("scoring_function", "vina"),
        poses=poses,
        runtime_seconds=float(summary.get("runtime_seconds", 0.0)),
        parameters=summary.get("parameters", {}),
    )
    result.ensemble_binding_energy = float(summary.get("ensemble_binding_energy", 0.0))
    result.ensemble_confidence = float(summary.get("ensemble_confidence", 0.0))
    return result


def load_interaction_analyses(results_dir: Path) -> List[Dict]:
    """Read per-pose interaction analyses if a run recorded them."""
    path = Path(results_dir) / "interaction_analysis.json"
    if not path.exists():
        return []
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(blob, dict) and "all_poses" in blob:
        return blob["all_poses"]
    return [blob] if isinstance(blob, dict) else []
