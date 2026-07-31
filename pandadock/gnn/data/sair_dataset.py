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

        # y_affinity is the name the trainer reads; `y` alone yields no loss
        # terms at all, which surfaces as a total loss of 0.0 rather than an error.
        graph.y_affinity = torch.tensor([entry.pic50], dtype=torch.float32)
        graph.y = graph.y_affinity
        graph.entry_id = entry.entry_id
        return graph

    def get_entry(self, index: int) -> SAIREntry:
        return self.entries[index]


# ------------------------------------------------------------- compact caching

# Caching built graphs costs ~195 KB per complex, because a materialised
# HeteroData carries ~600 site atoms of 56-dimensional float features. Caching
# the parsed atoms instead and featurising at load time costs roughly 5 KB, which
# is the difference between 172 GB and 5 GB over the full set. Featurisation is
# cheap once the CIF has been parsed, so nothing is lost by deferring it.

SHARD_SIZE = 1000


def complex_to_record(
    entry_id: int,
    cif_path: str,
    smiles: str,
    pic50: float,
    site_radius: float = 10.0,
) -> Optional[dict]:
    """
    Parse one complex into the compact form that gets cached.

    Coordinates are float32 and atom types are interned strings; the graph is
    rebuilt from this at load time.
    """
    protein_atoms, ligand_atoms = parse_sair_cif(cif_path)
    if not protein_atoms or not ligand_atoms:
        return None

    ligand, matched = ligand_to_parsed(ligand_atoms, smiles)
    protein = protein_to_parsed(protein_atoms)

    ligand_xyz = np.array([[a.x, a.y, a.z] for a in ligand.atoms], dtype=np.float32)
    centroid = ligand_xyz.mean(axis=0)

    # Cut the site here rather than at load time: it is the whole reason the
    # record is small, and the radius is not something training should vary.
    keep = [
        a for a in protein.atoms
        if (a.x - centroid[0]) ** 2 + (a.y - centroid[1]) ** 2
        + (a.z - centroid[2]) ** 2 <= site_radius ** 2
    ]
    if not keep:
        return None

    return {
        "entry_id": entry_id,
        "pic50": float(pic50),
        "smiles_matched": bool(matched),
        "site_xyz": np.array([[a.x, a.y, a.z] for a in keep], dtype=np.float32),
        "site_types": [a.atom_type for a in keep],
        "site_names": [a.name for a in keep],
        "site_resnames": [a.residue_name for a in keep],
        "site_resids": np.array([a.residue_id for a in keep], dtype=np.int32),
        "lig_xyz": ligand_xyz,
        "lig_types": [a.atom_type for a in ligand.atoms],
        "lig_names": [a.name for a in ligand.atoms],
    }


def record_to_molecules(record: dict):
    """Rebuild the site and ligand ParsedMolecules from a cached record."""
    from .mol2_parser import Atom, ParsedMolecule

    def build(prefix: str, resnames, resids, name: str):
        xyz = record[f"{prefix}_xyz"]
        types = record[f"{prefix}_types"]
        names = record[f"{prefix}_names"]
        atoms = [
            Atom(
                id=i + 1,
                name=names[i],
                x=float(xyz[i][0]), y=float(xyz[i][1]), z=float(xyz[i][2]),
                atom_type=types[i],
                charge=0.0,
                residue_name=resnames[i] if resnames is not None else LIGAND_COMP_ID,
                residue_id=int(resids[i]) if resids is not None else 1,
            )
            for i in range(len(types))
        ]
        molecule = ParsedMolecule(name=name, atoms=atoms, bonds=[])
        molecule.num_atoms = len(atoms)
        return molecule

    site = build("site", record["site_resnames"], record["site_resids"], "site")
    ligand = build("lig", None, None, "ligand")
    return site, ligand


def shard_path(cache_dir, shard_id: int) -> Path:
    return Path(cache_dir) / f"shard_{shard_id:04d}.pkl.gz"


def save_shard(cache_dir, shard_id: int, records: dict) -> Path:
    """Write one shard of records. Gzipped: these are mostly repeated strings."""
    import gzip
    import pickle

    path = shard_path(cache_dir, shard_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wb", compresslevel=4) as handle:
        pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return path


def load_shard(cache_dir, shard_id: int) -> dict:
    import gzip
    import pickle

    path = shard_path(cache_dir, shard_id)
    if not path.exists():
        return {}
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


INDEX_NAME = "index.pkl.gz"


def build_index(cache_dir, rebuild: bool = False) -> dict:
    """
    Index the shard cache: which shard each entry lives in, and its target
    sequence so splits can be made target-disjoint.

    Persisted, because building it means decompressing every shard -- several
    minutes over the full 920k set, and it would otherwise be paid three times
    per run, once for each of train/val/test.

    Sequences are stored once in a table and referenced by position; the same
    target recurs across thousands of complexes, so storing the string per entry
    would make the index larger than it needs to be by two orders of magnitude.
    """
    import glob
    import gzip
    import pickle

    cache_dir = Path(cache_dir)
    path = cache_dir / INDEX_NAME

    if path.exists() and not rebuild:
        with gzip.open(path, "rb") as handle:
            return pickle.load(handle)

    shards = sorted(glob.glob(str(cache_dir / "shard_*.pkl.gz")))
    if not shards:
        raise FileNotFoundError(f"No shards under {cache_dir}")

    entries: List[Tuple[int, int, int]] = []
    sequence_ids: Dict[str, int] = {}
    logger.info("Indexing %d shards (one-off, result is cached)", len(shards))

    for shard_file in shards:
        shard_id = int(Path(shard_file).name.split("_")[1].split(".")[0])
        for entry_id, record in load_shard(cache_dir, shard_id).items():
            sequence = record.get("sequence") or str(entry_id)
            seq_id = sequence_ids.setdefault(sequence, len(sequence_ids))
            entries.append((shard_id, entry_id, seq_id))

    index = {
        "entries": entries,
        "n_sequences": len(sequence_ids),
        "n_shards": len(shards),
    }

    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wb", compresslevel=4) as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)

    logger.info("Indexed %d complexes over %d targets",
                len(entries), len(sequence_ids))
    return index


class SAIRCachedDataset:
    """
    Dataset over the sharded record cache.

    Shards are loaded lazily and the most recently used ones kept, so access in
    shard order touches each file once. Random access across 920 shards would
    decompress a shard per sample, so shuffle with ShardBlockSampler rather than
    with DataLoader(shuffle=True).
    """

    def __init__(
        self,
        cache_dir: str,
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        graph_config=None,
        max_open_shards: int = 4,
        index: Optional[dict] = None,
    ):
        from .graph_builder import HeterogeneousGraphBuilder

        self.cache_dir = Path(cache_dir)
        self.graph_builder = HeterogeneousGraphBuilder(graph_config)
        self.max_open_shards = max_open_shards
        self._open: Dict[int, dict] = {}

        if index is None:
            index = build_index(self.cache_dir)

        # Split on targets, not complexes: SAIR carries many ligands against the
        # same protein, so a random split would put near-identical complexes on
        # both sides and report a score the model has not earned.
        rng = np.random.default_rng(seed)
        order = rng.permutation(index["n_sequences"])
        n_train = int(index["n_sequences"] * split_ratios[0])
        n_val = int(index["n_sequences"] * split_ratios[1])
        assigned = {
            "train": set(order[:n_train].tolist()),
            "val": set(order[n_train:n_train + n_val].tolist()),
            "test": set(order[n_train + n_val:].tolist()),
        }[split]

        # Sorted by shard so the default sequential order is also the cheapest.
        self.index: List[Tuple[int, int]] = [
            (shard_id, entry_id)
            for shard_id, entry_id, seq_id in index["entries"]
            if seq_id in assigned
        ]
        logger.info("%s: %d complexes from %d shards",
                    split, len(self.index), index["n_shards"])

    def __len__(self) -> int:
        return len(self.index)

    def _records(self, shard_id: int) -> dict:
        if shard_id not in self._open:
            if len(self._open) >= self.max_open_shards:
                self._open.pop(next(iter(self._open)))
            self._open[shard_id] = load_shard(self.cache_dir, shard_id)
        return self._open[shard_id]

    def __getitem__(self, index: int):
        import torch

        shard_id, entry_id = self.index[index]
        record = self._records(shard_id)[entry_id]

        site, ligand = record_to_molecules(record)
        graph = self.graph_builder.build_graph(site, ligand)
        # See the note in SAIRDataset.__getitem__: the trainer reads y_affinity.
        graph.y_affinity = torch.tensor([record["pic50"]], dtype=torch.float32)
        graph.y = graph.y_affinity
        graph.entry_id = entry_id
        return graph

    @property
    def shard_ids(self) -> List[int]:
        return [shard_id for shard_id, _ in self.index]


class ShardBlockSampler:
    """
    Shuffle within a sliding block of shards instead of across the whole set.

    DataLoader(shuffle=True) draws uniformly over 920k samples, so consecutive
    samples land in different shards and each one costs a gzip decompression --
    the cache holds four shards and thrashes immediately. Shuffling shard order,
    then shuffling within a block of `block_shards` at a time, keeps every
    decompressed shard in use for its whole block.

    With the default block of 16 shards the shuffle pool is 16k complexes, which
    is well past the point where batch composition correlates with shard order.
    Reshuffled every epoch.
    """

    def __init__(self, dataset: "SAIRCachedDataset", block_shards: int = 16, seed: int = 42):
        self.block_shards = block_shards
        self.seed = seed
        self.epoch = 0

        self.by_shard: Dict[int, List[int]] = {}
        for position, (shard_id, _) in enumerate(dataset.index):
            self.by_shard.setdefault(shard_id, []).append(position)
        self.n = len(dataset.index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        shards = list(self.by_shard)
        rng.shuffle(shards)

        for start in range(0, len(shards), self.block_shards):
            block: List[int] = []
            for shard_id in shards[start:start + self.block_shards]:
                block.extend(self.by_shard[shard_id])
            rng.shuffle(block)
            yield from block


def create_sair_dataloaders(
    cache_dir: str,
    batch_size: int = 32,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    num_workers: int = 0,
    block_shards: int = 16,
    graph_config=None,
):
    """
    Build train/val/test loaders over the shard cache.

    The index is built once and shared, rather than rebuilt per split.
    """
    from torch_geometric.loader import DataLoader as PyGDataLoader

    index = build_index(cache_dir)

    # Enough open shards to cover a block, plus room for the loader to read
    # ahead. Each shard is ~3 MB decompressed, so a block of 16 costs ~50 MB per
    # worker -- negligible against a 23 GB card, and it removes the thrashing.
    max_open = block_shards + 2

    datasets = {
        split: SAIRCachedDataset(
            cache_dir, split=split, split_ratios=split_ratios, seed=seed,
            graph_config=graph_config, max_open_shards=max_open, index=index,
        )
        for split in ("train", "val", "test")
    }

    train_sampler = ShardBlockSampler(datasets["train"], block_shards, seed)
    loaders = {
        "train": PyGDataLoader(
            datasets["train"], batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, persistent_workers=num_workers > 0,
        )
    }
    # Validation and test run in shard order: no shuffle needed, and sequential
    # access is the cheapest way through the cache.
    for split in ("val", "test"):
        loaders[split] = PyGDataLoader(
            datasets[split], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, persistent_workers=num_workers > 0,
        )

    return loaders["train"], loaders["val"], loaders["test"]
