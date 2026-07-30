"""
SAIR dataset for PandaDock-GNN training.

SAIR pairs co-folded protein-ligand structures (ModelCIF) with measured pIC50.
It is read directly from CIF: the graph builder consumes `ParsedMolecule`
objects, not MOL2 files, so converting ~900k complexes to 2.7M MOL2 files on
disk would cost days of Open Babel subprocess time and buy nothing. Protein atom
types come from the residue-aware SYBYL table, and ligand bond orders from the
SMILES in the parquet, which is more reliable than perceiving them from a
predicted geometry.

Two properties of the data drive the design:

- SAIR CIFs carry **no hydrogens**. Models trained here are heavy-atom models and
  are not interchangeable with the ULVSH/PDBbind models, which were trained with
  hydrogens present.
- There are ~8.4 measurements per structure. Rows are aggregated to one label per
  `entry_id`, because feeding the same coordinates with several different labels
  teaches the model to predict their average while penalising it for doing so.

Splitting is by protein sequence. With ~1M complexes over far fewer distinct
targets, a random split puts the same protein in train and test, and the reported
correlation then largely measures memorised targets.
"""

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("pandadock.gnn.data.sair")

# _atom_site column order in SAIR ModelCIF files.
ATOM_SITE_FIELDS = [
    "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
    "label_comp_id", "label_seq_id", "auth_seq_id", "pdbx_PDB_ins_code",
    "label_asym_id", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy",
    "label_entity_id", "auth_asym_id", "auth_comp_id", "B_iso_or_equiv",
    "pdbx_PDB_model_num",
]

LIGAND_COMP_ID = "LIG"


@dataclass
class SAIREntry:
    """One SAIR complex: a structure path plus its measured affinity."""

    entry_id: int
    cif_path: str
    pic50: float
    smiles: str
    sequence: str
    n_measurements: int = 1



def _open_cif(path: str):
    """
    Open a CIF from local disk or S3.

    S3 paths are streamed rather than synced. Each CIF is read exactly once, to
    build its graph; training thereafter reads only the cache. Syncing the full
    set would cost roughly 432 GB of local disk for data touched once.
    """
    import io

    if not str(path).startswith("s3://"):
        return open(path, "r", errors="ignore")

    import boto3

    bucket, key = str(path)[5:].split("/", 1)
    client = _s3_client()
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return io.StringIO(body.decode("utf-8", errors="ignore"))


_S3_CLIENT = None


def _s3_client():
    """One boto3 client per process; creating one per file dominates runtime."""
    global _S3_CLIENT
    if _S3_CLIENT is None:
        import boto3
        from botocore.config import Config

        _S3_CLIENT = boto3.client(
            "s3",
            config=Config(max_pool_connections=64, retries={"max_attempts": 5}),
        )
    return _S3_CLIENT


# --------------------------------------------------------------------- CIF read


def parse_sair_cif(cif_path: str) -> Tuple[List[dict], List[dict]]:
    """
    Read a SAIR ModelCIF into protein and ligand atom records.

    Returns (protein_atoms, ligand_atoms), each a list of dicts with element,
    name, residue name, residue number and coordinates. Only the first model is
    read; SAIR ships one model per file.
    """
    protein: List[dict] = []
    ligand: List[dict] = []

    index = {name: i for i, name in enumerate(ATOM_SITE_FIELDS)}
    in_atom_site = False
    header: List[str] = []

    with _open_cif(cif_path) as handle:
        for line in handle:
            stripped = line.strip()

            if stripped.startswith("_atom_site."):
                header.append(stripped.split(".", 1)[1])
                in_atom_site = True
                continue

            if not stripped.startswith(("ATOM ", "HETATM")):
                if in_atom_site and stripped in ("#", "loop_"):
                    in_atom_site = False
                continue

            fields = stripped.split()
            # Prefer the file's own header if it deviates from the expected order.
            layout = {name: i for i, name in enumerate(header)} if header else index
            try:
                record = {
                    "element": fields[layout["type_symbol"]].upper(),
                    "name": fields[layout["label_atom_id"]],
                    "resname": fields[layout["label_comp_id"]],
                    "resnum": fields[layout["auth_seq_id"]],
                    "x": float(fields[layout["Cartn_x"]]),
                    "y": float(fields[layout["Cartn_y"]]),
                    "z": float(fields[layout["Cartn_z"]]),
                }
            except (KeyError, IndexError, ValueError):
                continue

            if record["element"] in ("H", "D"):
                continue

            if record["resname"] == LIGAND_COMP_ID:
                ligand.append(record)
            else:
                protein.append(record)

    return protein, ligand


def protein_to_parsed(atoms: Sequence[dict], name: str = "protein"):
    """Build a ParsedMolecule for the protein, typed by residue and atom name."""
    from .graph_builder import protein_sybyl_type
    from .mol2_parser import Atom, ParsedMolecule

    parsed_atoms = []
    for i, record in enumerate(atoms):
        parsed_atoms.append(
            Atom(
                id=i + 1,
                name=record["name"],
                x=record["x"], y=record["y"], z=record["z"],
                atom_type=protein_sybyl_type(
                    record["resname"], record["name"], record["element"]
                ),
                charge=0.0,
                residue_name=record["resname"],
                residue_id=int(record["resnum"]) if record["resnum"].lstrip("-").isdigit() else 1,
            )
        )

    molecule = ParsedMolecule(name=name, atoms=parsed_atoms, bonds=[])
    molecule.num_atoms = len(parsed_atoms)
    return molecule


def ligand_to_parsed(atoms: Sequence[dict], smiles: str, name: str = "ligand"):
    """
    Build a ParsedMolecule for the ligand, with bond orders from SMILES.

    The CIF gives elements and coordinates but no bonds. Matching the SMILES
    template recovers aromaticity and hybridisation, which the SYBYL typing then
    depends on: without it every carbon types as C.3, and the protein half of the
    graph is what the model reads most strongly.

    Falls back to element-only typing when the template cannot be matched, and
    reports which route was taken so callers can filter.
    """
    from rdkit import Chem

    from .mol2_parser import Atom, ParsedMolecule

    coords = np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=float)
    elements = [a["element"].capitalize() for a in atoms]

    sybyl: List[str] = []
    matched = False

    template = Chem.MolFromSmiles(smiles) if smiles else None
    if template is not None:
        template = Chem.RemoveHs(template)
        if template.GetNumAtoms() == len(atoms):
            hyb_map = {
                Chem.HybridizationType.SP3: "3",
                Chem.HybridizationType.SP2: "2",
                Chem.HybridizationType.SP: "1",
            }
            # Atom order in the CIF need not match the SMILES, but SAIR writes
            # the ligand in template order; verify by element before trusting it.
            template_elements = [a.GetSymbol() for a in template.GetAtoms()]
            if template_elements == elements:
                for atom in template.GetAtoms():
                    symbol = atom.GetSymbol()
                    if atom.GetIsAromatic():
                        sybyl.append(f"{symbol}.ar")
                    else:
                        sybyl.append(f"{symbol}.{hyb_map.get(atom.GetHybridization(), '3')}")
                matched = True

    if not matched:
        sybyl = [f"{e}.3" if e not in ("F", "Cl", "Br", "I") else e for e in elements]

    parsed_atoms = [
        Atom(
            id=i + 1,
            name=atoms[i]["name"],
            x=float(coords[i][0]), y=float(coords[i][1]), z=float(coords[i][2]),
            atom_type=sybyl[i],
            charge=0.0,
            residue_name=LIGAND_COMP_ID,
            residue_id=1,
        )
        for i in range(len(atoms))
    ]

    molecule = ParsedMolecule(name=name, atoms=parsed_atoms, bonds=[])
    molecule.num_atoms = len(parsed_atoms)
    return molecule, matched


def build_complex_graph(cif_path: str, smiles: str, graph_builder,
                        site_radius: float = 10.0):
    """Build the heterogeneous graph for one SAIR complex, or None on failure."""
    from .graph_builder import extract_binding_site

    protein_atoms, ligand_atoms = parse_sair_cif(cif_path)
    if not protein_atoms or not ligand_atoms:
        return None

    ligand, _ = ligand_to_parsed(ligand_atoms, smiles)
    protein = protein_to_parsed(protein_atoms)

    centroid = np.array([[a.x, a.y, a.z] for a in ligand.atoms]).mean(axis=0)
    site = extract_binding_site(protein, centroid, radius=site_radius)

    return graph_builder.build_graph(site, ligand)


# ------------------------------------------------------------------- index/label


def load_entries(
    parquet_path: str,
    cif_index_path: str,
    cif_root: str,
    min_pic50: Optional[float] = None,
    drop_floor: bool = True,
) -> List[SAIREntry]:
    """
    Join the affinity table to the downloaded structures.

    Args:
        drop_floor: Discard measurements sitting exactly on the assay floor
            (pIC50 == 4.0). These are censored -- the true value is "weaker than
            4", not "equal to 4" -- and training on them as point measurements
            teaches the model a value that was never observed.
    """
    import pyarrow.parquet as pq

    with open(cif_index_path) as handle:
        index = json.load(handle)
    available = {int(k): v for k, v in index.items()}

    table = pq.read_table(
        parquet_path, columns=["entry_id", "pIC50", "SMILES", "sequence"]
    )
    entry_ids = table["entry_id"].to_numpy()
    pic50 = table["pIC50"].to_numpy(zero_copy_only=False).astype(float)
    smiles = table["SMILES"].to_pylist()
    sequences = table["sequence"].to_pylist()

    grouped: Dict[int, dict] = {}
    for i in range(len(entry_ids)):
        eid = int(entry_ids[i])
        if eid not in available:
            continue
        value = pic50[i]
        if not math.isfinite(value):
            continue
        if drop_floor and value <= 4.0:
            continue
        if min_pic50 is not None and value < min_pic50:
            continue

        record = grouped.setdefault(
            eid, {"values": [], "smiles": smiles[i], "sequence": sequences[i]}
        )
        record["values"].append(value)

    entries = []
    for eid, record in grouped.items():
        entries.append(
            SAIREntry(
                entry_id=eid,
                cif_path=(
                    f"{cif_root.rstrip('/')}/{os.path.basename(available[eid])}"
                    if str(cif_root).startswith("s3://")
                    else os.path.join(cif_root, os.path.basename(available[eid]))
                ),
                # Median across replicate measurements: a structure has one
                # geometry, so it must carry one label.
                pic50=float(np.median(record["values"])),
                smiles=record["smiles"],
                sequence=record["sequence"],
                n_measurements=len(record["values"]),
            )
        )

    logger.info(
        "SAIR: %d complexes with structures and labels (from %d CIFs)",
        len(entries), len(available),
    )
    return entries


def split_by_sequence(
    entries: Sequence[SAIREntry],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Dict[str, List[SAIREntry]]:
    """
    Split so that no protein sequence appears in more than one split.

    A random split over complexes would place the same target in train and test.
    With roughly a million complexes over far fewer distinct targets, the
    resulting test correlation largely measures memorised targets rather than
    generalisation to new ones.
    """
    rng = np.random.default_rng(seed)

    by_sequence: Dict[str, List[SAIREntry]] = {}
    for entry in entries:
        by_sequence.setdefault(entry.sequence, []).append(entry)

    sequences = sorted(by_sequence)
    rng.shuffle(sequences)

    n_train = int(len(sequences) * ratios[0])
    n_val = int(len(sequences) * ratios[1])

    assignment = {
        "train": sequences[:n_train],
        "val": sequences[n_train:n_train + n_val],
        "test": sequences[n_train + n_val:],
    }

    splits = {
        name: [e for s in seqs for e in by_sequence[s]]
        for name, seqs in assignment.items()
    }
    for name, members in splits.items():
        logger.info(
            "%s: %d complexes over %d targets", name, len(members), len(assignment[name])
        )
    return splits


# ----------------------------------------------------------------------- dataset


class SAIRDataset:
    """
    PyTorch Geometric dataset over SAIR complexes.

    Graphs are built on demand from CIF and cached to disk, because building
    ~900k of them takes far longer than an epoch and the result is deterministic.
    """

    def __init__(
        self,
        parquet_path: str,
        cif_index_path: str,
        cif_root: str,
        split: str = "train",
        cache_dir: Optional[str] = None,
        site_radius: float = 10.0,
        graph_config=None,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        limit: Optional[int] = None,
        drop_floor: bool = True,
    ):
        from .graph_builder import HeterogeneousGraphBuilder

        entries = load_entries(parquet_path, cif_index_path, cif_root,
                               drop_floor=drop_floor)
        splits = split_by_sequence(entries, split_ratios, seed)
        if split not in splits:
            raise ValueError(f"Unknown split '{split}'; expected train, val or test")

        self.entries = splits[split]
        if limit:
            self.entries = self.entries[:limit]

        self.split = split
        self.site_radius = site_radius
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.graph_builder = HeterogeneousGraphBuilder(graph_config)

    def __len__(self) -> int:
        return len(self.entries)

    def _cache_path(self, entry: SAIREntry) -> Optional[Path]:
        if not self.cache_dir:
            return None
        # Shard so no directory holds a million files.
        shard = f"{entry.entry_id % 1000:03d}"
        directory = self.cache_dir / shard
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{entry.entry_id}.pt"

    def __getitem__(self, index: int):
        import torch

        entry = self.entries[index]
        cache_path = self._cache_path(entry)

        if cache_path is not None and cache_path.exists():
            graph = torch.load(cache_path, weights_only=False)
        else:
            graph = build_complex_graph(
                entry.cif_path, entry.smiles, self.graph_builder, self.site_radius
            )
            if graph is None:
                raise RuntimeError(f"Could not build a graph for entry {entry.entry_id}")
            if cache_path is not None:
                torch.save(graph, cache_path)

        graph.y = torch.tensor([entry.pic50], dtype=torch.float)
        graph.entry_id = entry.entry_id
        return graph

    def get_entry(self, index: int) -> SAIREntry:
        return self.entries[index]
