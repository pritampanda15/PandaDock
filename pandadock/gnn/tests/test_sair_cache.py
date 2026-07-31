"""
Tests for the SAIR shard cache: indexing, splitting, shuffling, and the
graph/target contract the trainer relies on.

These use a synthetic cache rather than real CIFs so they run without S3 or the
920k-complex dataset.
"""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from pandadock.gnn.data.sair_dataset import (  # noqa: E402
    SAIRCachedDataset,
    ShardBlockSampler,
    build_index,
    save_shard,
)

N_SHARDS = 6
PER_SHARD = 50
N_TARGETS = 20


@pytest.fixture
def cache(tmp_path):
    """A synthetic shard cache: 300 complexes over 20 targets in 6 shards."""
    rng = np.random.default_rng(0)
    entry_id = 0
    for shard in range(N_SHARDS):
        records = {}
        for _ in range(PER_SHARD):
            n_site, n_lig = 40, 12
            records[entry_id] = {
                "entry_id": entry_id,
                "pic50": float(rng.uniform(4, 10)),
                "smiles_matched": True,
                "site_xyz": (rng.normal(size=(n_site, 3)) * 3).astype(np.float32),
                "site_types": ["C.3"] * n_site,
                "site_names": ["CA"] * n_site,
                "site_resnames": ["ALA"] * n_site,
                "site_resids": np.arange(n_site, dtype=np.int32),
                "lig_xyz": rng.normal(size=(n_lig, 3)).astype(np.float32),
                "lig_types": ["C.3"] * n_lig,
                "lig_names": [f"C{i}" for i in range(n_lig)],
                "sequence": f"SEQ{entry_id % N_TARGETS}",
            }
            entry_id += 1
        save_shard(tmp_path, shard, records)
    return tmp_path


def test_index_is_persisted_and_stable(cache):
    """
    The index must be written to disk and reused.

    Building it decompresses every shard, which over the full 920k set takes
    minutes; recomputing it per split would pay that three times per run.
    """
    first = build_index(cache)
    assert (cache / "index.pkl.gz").exists()
    assert len(first["entries"]) == N_SHARDS * PER_SHARD
    assert first["n_sequences"] == N_TARGETS
    assert first["n_shards"] == N_SHARDS

    assert build_index(cache) == first


def test_splits_are_target_disjoint(cache):
    """
    No target may appear in more than one split.

    SAIR holds many ligands against the same protein. A split over complexes
    rather than targets puts near-identical entries on both sides, and the
    resulting validation score measures memorisation instead of generalisation.
    """
    splits = {
        name: SAIRCachedDataset(str(cache), split=name)
        for name in ("train", "val", "test")
    }

    def targets(dataset):
        return {f"SEQ{entry_id % N_TARGETS}" for _, entry_id in dataset.index}

    train, val, test = (targets(splits[n]) for n in ("train", "val", "test"))
    assert not train & val
    assert not train & test
    assert not val & test

    assert sum(len(d) for d in splits.values()) == N_SHARDS * PER_SHARD


def test_splits_are_deterministic_for_a_seed(cache):
    a = SAIRCachedDataset(str(cache), split="train", seed=7)
    b = SAIRCachedDataset(str(cache), split="train", seed=7)
    c = SAIRCachedDataset(str(cache), split="train", seed=8)
    assert a.index == b.index
    assert a.index != c.index


def test_sampler_covers_every_sample_and_reshuffles(cache):
    dataset = SAIRCachedDataset(str(cache), split="train")
    sampler = ShardBlockSampler(dataset, block_shards=2, seed=1)

    first = list(sampler)
    assert sorted(first) == list(range(len(dataset)))
    assert len(first) == len(sampler)

    sampler.set_epoch(1)
    assert list(sampler) != first


def test_sampler_keeps_samples_grouped_by_shard(cache):
    """
    Shuffling must preserve shard locality.

    Uniform shuffling over the full set puts consecutive samples in different
    shards, so each one costs a gzip decompression and the four-shard cache
    thrashes. Within a block, consecutive samples should mostly share a shard.
    """
    dataset = SAIRCachedDataset(str(cache), split="train")
    sampler = ShardBlockSampler(dataset, block_shards=1, seed=1)

    shards = [dataset.index[i][0] for i in sampler]
    transitions = sum(1 for a, b in zip(shards, shards[1:]) if a != b)

    # With one shard per block, the only transitions are between blocks.
    assert transitions < len(set(shards)) + 1


def test_graph_carries_the_target_name_the_trainer_reads(cache):
    """
    Graphs must set y_affinity, not just y.

    GNNTrainer._get_targets looks for y_affinity. A dataset setting only `y`
    produces no loss terms, so the total stays the float it was initialised to
    and training either crashes on .backward() or silently learns nothing.
    """
    dataset = SAIRCachedDataset(str(cache), split="train")
    graph = dataset[0]

    assert hasattr(graph, "y_affinity")
    assert graph.y_affinity.dtype.is_floating_point
    assert 4.0 <= float(graph.y_affinity) <= 10.0


def test_missing_target_raises_instead_of_training_on_nothing():
    """A batch with no recognised target must fail loudly, not silently."""
    import torch
    from torch_geometric.data import HeteroData

    from pandadock.gnn.training.trainer import GNNTrainer

    batch = HeteroData()
    batch["ligand"].x = torch.zeros(3, 4)
    batch.y = torch.tensor([5.0])  # `y` alone: the mistake this guards against

    with pytest.raises(ValueError, match="no training target"):
        GNNTrainer._get_targets(object.__new__(GNNTrainer), batch)
