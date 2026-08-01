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

    # Compared field by field: the index holds a numpy array, so dict equality
    # raises rather than returning False.
    second = build_index(cache)
    assert second["entries"] == first["entries"]
    assert second["n_sequences"] == first["n_sequences"]
    assert second["n_shards"] == first["n_shards"]
    assert np.array_equal(second["labels"], first["labels"])


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


def test_target_centering_removes_between_target_variance(cache):
    """
    Centred labels must have zero mean within every target.

    That is the entire point: what remains is the ligand-to-ligand variation,
    which is what the model should be spending its capacity on.
    """
    from collections import defaultdict

    plain = SAIRCachedDataset(str(cache), split="train")
    centered = SAIRCachedDataset(str(cache), split="train", center_targets=True)

    assert plain.index == centered.index

    groups = defaultdict(list)
    for position in range(len(centered)):
        _, entry_id = centered.index[position]
        groups[centered.entry_target[entry_id]].append(
            float(centered[position].y_affinity)
        )

    for target, values in groups.items():
        assert abs(float(np.mean(values))) < 1e-4, (
            f"target {target} has non-zero mean after centring"
        )

    # And the ordering within a target is untouched by the shift.
    first = centered.index[0][1]
    offset = centered.target_mean[centered.entry_target[first]]
    assert float(plain[0].y_affinity) - offset == pytest.approx(
        float(centered[0].y_affinity), abs=1e-5
    )


def test_target_means_use_only_the_given_entries(cache):
    """
    A split's means must come from that split alone.

    Splits are target-disjoint, so this cannot leak in practice -- but the
    function must not quietly average over the whole cache, because that would
    make the guarantee accidental rather than enforced.
    """
    from pandadock.gnn.data.sair_dataset import build_index, target_means

    index = build_index(str(cache))
    train = SAIRCachedDataset(str(cache), split="train")
    test = SAIRCachedDataset(str(cache), split="test")

    train_means = target_means(index, train.index)
    test_means = target_means(index, test.index)

    assert set(train_means) & set(test_means) == set()
    assert len(train_means) + len(test_means) < index["n_sequences"] + 1


def test_index_rebuilds_when_the_version_changes(cache, monkeypatch):
    """A cache indexed by an older version must rebuild, not fail on a key."""
    import gzip
    import pickle

    from pandadock.gnn.data import sair_dataset

    index = sair_dataset.build_index(str(cache))
    assert "labels" in index

    stale = {k: v for k, v in index.items() if k != "labels"}
    stale["version"] = 1
    with gzip.open(cache / sair_dataset.INDEX_NAME, "wb") as handle:
        pickle.dump(stale, handle)

    rebuilt = sair_dataset.build_index(str(cache))
    assert rebuilt["version"] == sair_dataset.INDEX_VERSION
    assert "labels" in rebuilt


def test_within_target_loss_ignores_absolute_error():
    """
    The ranking term must respond to ordering, not to absolute offset.

    A prediction that is wrong by a constant per target but perfectly ordered
    within it scores zero; a prediction with the right mean but reversed order
    does not.
    """
    import torch

    from pandadock.gnn.training.losses import within_target_loss

    truth = torch.tensor([2.0, 4.0, 6.0])
    group = torch.tensor([1, 1, 1])

    shifted = torch.tensor([102.0, 104.0, 106.0])
    assert float(within_target_loss(shifted, truth, group)) == pytest.approx(0.0, abs=1e-6)

    reversed_order = torch.tensor([6.0, 4.0, 2.0])
    assert float(within_target_loss(reversed_order, truth, group)) > 1.0

    # Singletons carry no ordering information and must not contribute.
    assert within_target_loss(truth, truth, torch.tensor([1, 2, 3])) is None
