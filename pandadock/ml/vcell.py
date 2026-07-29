"""
Inspection for the SAIR-trained cell-context model (`model_vcell_complex.pt`).

This checkpoint is architecturally unrelated to `PandaDockGNN` and cannot be
loaded by it. This module reports what the checkpoint actually contains, derived
from tensor shapes rather than from its stored config -- which does not match the
weights.

What the model is
-----------------
A cell-line-conditioned potency regressor. Three parts:

1. `complex_encoder` -- 5-layer GATv2 over a protein-ligand complex graph
   (52-dim node features, 4-dim edge features, hidden 256), with separate protein,
   ligand and interface readouts fused to a 256-d complex embedding.
2. `cell_encoder` -- multi-omic encoder over a cell line: expression (1000),
   mutation hotspots (554), damaging mutations (1509), copy number (500) and
   CRISPR essentiality (1000), plus a 3-layer GATv2 over a gene graph
   (65-dim nodes, 1-dim edges). Produces a 64-d cell embedding.
3. `bilinear` (256 x 256 x 64) followed by `fusion_mlp` -> a single scalar.

What it can and cannot do for docking
-------------------------------------
It cannot generate or rank poses. The output is one scalar per
(complex, cell line) pair: there is no per-atom output, no coordinate update, and
no pose-quality or confidence head. Pose accuracy is determined by the search, not
by any affinity model.

It also cannot be evaluated from a receptor and a ligand alone. Every prediction
requires ~4500 cell-line omics features, so "which cell line?" must be answered
before the model produces a number at all.

Most importantly, its target is a different physical quantity from a docking
score. A cellular IC50 folds in permeability, efflux, protein binding and
off-target effects on top of target engagement, so it is not interchangeable with
a binding free energy. PandaDock has already seen what happens when incompatible
affinity scales are pooled: mixing pKd and pEC50 training data dropped test
Pearson R from 0.81 to 0.49. Reporting IC50 predictions alongside docking scores
is fine; averaging or co-training them without accounting for the scale
difference is not.

Reproducibility gap
-------------------
The stored config disagrees with the weights, so the model cannot be rebuilt from
the checkpoint alone. `check_config_consistency` reports the mismatches. The
graph featurization (which 52 node features, which 4 edge features, atom ordering,
edge construction cutoff) is not recoverable from the weights at any level of
effort, and neither is the omics feature ordering -- the gene lists behind the
1000/554/1509/500/1000 vectors must match the training data exactly or the
predictions are meaningless without being obviously wrong. Both must come from the
original SAIR training code, referenced in the checkpoint as
``/home/ubuntu/SAIR/train_sair_gnn/``.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("pandadock.ml.vcell")


# Input widths recovered from the weight shapes.
OMICS_INPUT_DIMS = {
    "expression": 1000,
    "mutation_hotspot": 554,
    "mutation_damaging": 1509,
    "copy_number": 500,
    "crispr": 1000,
}

COMPLEX_GRAPH_SPEC = {
    "node_features": 52,
    "edge_features": 4,
    "hidden_dim": 256,
    "num_layers": 5,
    "attention_heads": 4,
    "readouts": ("protein", "ligand", "interface"),
    "embedding_dim": 256,
}

CELL_GRAPH_SPEC = {
    "node_features": 65,
    "edge_features": 1,
    "num_layers": 3,
    "attention_heads": 2,
    "embedding_dim": 64,
}


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load the checkpoint without instantiating a model."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        raise ValueError(
            f"{path} does not look like a vcell checkpoint "
            f"(keys: {sorted(checkpoint)[:8]})"
        )
    return checkpoint


def describe(path: str) -> Dict[str, Any]:
    """Summarise a vcell checkpoint's architecture and inputs."""
    checkpoint = load_checkpoint(path)
    state = checkpoint["state_dict"]

    n_params = sum(t.numel() for t in state.values() if hasattr(t, "numel"))
    output_dim = None
    for key, tensor in state.items():
        if key.startswith("fusion_mlp") and key.endswith(".weight") and tensor.dim() == 2:
            output_dim = tensor.shape[0]

    return {
        "path": str(path),
        "stored_config": checkpoint.get("config", {}),
        "n_parameters": n_params,
        "n_tensors": len(state),
        "complex_graph": COMPLEX_GRAPH_SPEC,
        "cell_graph": CELL_GRAPH_SPEC,
        "omics_inputs": OMICS_INPUT_DIMS,
        "total_omics_features": sum(OMICS_INPUT_DIMS.values()),
        "output_dim": output_dim,
        "config_mismatches": check_config_consistency(path, checkpoint),
        "provides_pose_scoring": False,
        "requires_cell_line_context": True,
    }


def check_config_consistency(
    path: str, checkpoint: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Report where the stored config disagrees with the actual weights.

    The config is not a reliable description of this checkpoint; anything rebuilt
    from it will fail to load or, worse, load into the wrong shape silently.
    """
    checkpoint = checkpoint or load_checkpoint(path)
    state = checkpoint["state_dict"]
    config = checkpoint.get("config", {}) or {}

    mismatches: List[str] = []

    def actual(key: str, dim: int) -> Optional[int]:
        tensor = state.get(key)
        return int(tensor.shape[dim]) if tensor is not None else None

    checks: List[Tuple[str, Optional[int], str]] = [
        ("hidden_dim", actual("complex_encoder.node_embed.0.weight", 0),
         "complex encoder hidden width"),
        ("node_dim", actual("complex_encoder.node_embed.0.weight", 1),
         "complex graph node features (config value describes the CELL graph)"),
        ("edge_dim", actual("complex_encoder.layers.0.edge_proj.0.weight", 1),
         "complex graph edge features (config value describes the CELL graph)"),
    ]

    for key, found, note in checks:
        stated = config.get(key)
        if stated is not None and found is not None and stated != found:
            mismatches.append(
                f"config['{key}']={stated} but weights require {found} -- {note}"
            )

    return mismatches


def format_report(path: str) -> str:
    """Human-readable summary, including what is needed to actually run the model."""
    info = describe(path)
    lines: List[str] = []

    lines.append(f"vcell checkpoint: {Path(info['path']).name}")
    lines.append(f"  parameters:        {info['n_parameters']:,} in {info['n_tensors']} tensors")
    lines.append(f"  output:            {info['output_dim']} scalar per (complex, cell line)")
    lines.append("")
    lines.append("  complex encoder:   GATv2, "
                 f"{info['complex_graph']['num_layers']} layers, "
                 f"hidden {info['complex_graph']['hidden_dim']}, "
                 f"{info['complex_graph']['node_features']}-d nodes, "
                 f"{info['complex_graph']['edge_features']}-d edges")
    lines.append(f"                     readouts: {', '.join(info['complex_graph']['readouts'])}")
    lines.append("  cell encoder:      multi-omic + gene-graph GATv2 -> "
                 f"{info['cell_graph']['embedding_dim']}-d")
    for name, dim in info["omics_inputs"].items():
        lines.append(f"                       {name:<20} {dim:>5} features")
    lines.append(f"                     total omics input: {info['total_omics_features']} features")
    lines.append("")
    lines.append("  pose scoring:      NO -- single scalar output, no per-atom or "
                 "confidence head")
    lines.append("  usable from a receptor + ligand alone: NO -- requires cell-line omics")

    if info["config_mismatches"]:
        lines.append("")
        lines.append("  stored config does NOT match the weights:")
        for mismatch in info["config_mismatches"]:
            lines.append(f"    - {mismatch}")
        lines.append("    The model cannot be rebuilt from this checkpoint alone.")

    lines.append("")
    lines.append("  To use this model you still need, from the SAIR training code:")
    lines.append("    1. the complex graph featurizer (which 52 node / 4 edge features,")
    lines.append("       atom ordering, and the edge construction cutoff)")
    lines.append("    2. the gene list and ordering behind each omics vector")
    lines.append("    3. the gene graph used by the cell encoder")
    lines.append("    4. the forward wiring of the cell encoder, which the tensor")
    lines.append("       shapes alone do not determine unambiguously")
    lines.append("    5. the target transform (log units, and IC50 vs pIC50 sign)")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m pandadock.ml.vcell <model_vcell_complex.pt>")
        raise SystemExit(2)
    print(format_report(sys.argv[1]))
