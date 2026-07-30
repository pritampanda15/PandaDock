#!/usr/bin/env python
"""
Build the SAIR graph cache in parallel.

Measured at 279 ms per CIF, a single-threaded pass over ~900k structures takes
about 72 hours. The cost is almost entirely S3 round-trip latency rather than
compute, so the work is I/O bound and scales close to linearly with workers:
64 workers brings it to roughly an hour.

The job is resumable. Each complex is written to its own cache file and existing
files are skipped, so an interrupted run continues where it stopped rather than
starting over -- which matters when the full pass is measured in hours.

Usage:
    python benchmarking/sair_preprocess.py \
        --parquet sair.parquet \
        --cif-index cif_index.json \
        --cif-root s3://revilico-virtual-cell/SAIR/cif/ \
        --cache graph_cache/ --workers 64

    # validate the pipeline on a subset first
    python benchmarking/sair_preprocess.py ... --limit 50000
"""

import argparse
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger("pandadock.sair.preprocess")

_LOCAL = threading.local()


def _builder():
    """
    One graph builder per worker thread.

    Threads rather than processes: the job waits ~279 ms on S3 per structure, so
    it is I/O bound and threads reach the same throughput. Processes each load
    their own copy of torch and RDKit, which at 64 workers exhausted memory and
    the run was OOM-killed partway through.
    """
    if not hasattr(_LOCAL, "builder"):
        from pandadock.gnn.data.graph_builder import HeterogeneousGraphBuilder

        _LOCAL.builder = HeterogeneousGraphBuilder()
    return _LOCAL.builder


def process_one(task) -> tuple:
    """Parse one complex into a cache record. Returns (entry_id, status, record)."""
    entry_id, cif_path, smiles, pic50, sequence = task

    from pandadock.gnn.data.sair_dataset import complex_to_record

    try:
        record = complex_to_record(entry_id, cif_path, smiles, pic50)
        if record is None:
            return entry_id, "empty", None
        # Kept so the dataset can split by target without re-reading the parquet.
        record["sequence"] = sequence
        return entry_id, "built", record
    except Exception as exc:
        return entry_id, f"error:{type(exc).__name__}: {exc}", None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--cif-index", required=True)
    parser.add_argument("--cif-root", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default=None,
                        choices=["train", "val", "test"],
                        help="Only build one split; default builds everything")
    parser.add_argument("--keep-floor", action="store_true",
                        help="Keep pIC50 == 4.0 censored measurements")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild shards that already exist")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s: %(message)s")

    from pandadock.gnn.data.sair_dataset import (
        SHARD_SIZE, load_entries, load_shard, save_shard, shard_path, split_by_sequence,
    )

    print("Indexing...", flush=True)
    entries = load_entries(args.parquet, args.cif_index, args.cif_root,
                           drop_floor=not args.keep_floor)

    if args.split:
        entries = split_by_sequence(entries)[args.split]
    if args.limit:
        entries = entries[: args.limit]

    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (e.entry_id, e.cif_path, e.smiles, e.pic50, e.sequence) for e in entries
    ]
    del entries

    print(f"{len(tasks):,} complexes, {args.workers} worker threads", flush=True)

    counts = {"built": 0, "cached": 0, "empty": 0, "error": 0}
    error_samples: dict = {}
    start = time.time()
    done = 0

    # One shard per SHARD_SIZE complexes. Writing a file per complex produced
    # ~195 KB each -- a materialised graph carries ~600 site atoms of 56-dim
    # float features -- which is 172 GB over the full set. Caching parsed atoms
    # in gzipped shards is roughly 5 KB each, and featurising at load time costs
    # little once the CIF has been parsed.
    for shard_id in range(0, (len(tasks) + SHARD_SIZE - 1) // SHARD_SIZE):
        if shard_path(cache_dir, shard_id).exists() and not args.rebuild:
            existing = load_shard(cache_dir, shard_id)
            counts["cached"] += len(existing)
            done += len(existing)
            continue

        wave = tasks[shard_id * SHARD_SIZE:(shard_id + 1) * SHARD_SIZE]
        records = {}

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(process_one, t) for t in wave]):
                entry_id, status, record = future.result()
                done += 1
                if status == "built":
                    counts["built"] += 1
                    records[entry_id] = record
                elif status == "empty":
                    counts["empty"] += 1
                else:
                    counts["error"] += 1
                    key = status.split(":", 1)[-1].strip()[:90]
                    error_samples[key] = error_samples.get(key, 0) + 1

        if records:
            save_shard(cache_dir, shard_id, records)

        elapsed = time.time() - start
        rate = done / elapsed if elapsed else 0
        remaining = (len(tasks) - done) / rate if rate else 0
        print(
            f"  shard {shard_id:04d}  {done:,}/{len(tasks):,}  {rate:.0f}/s  "
            f"eta {remaining/3600:.1f} h  built {counts['built']:,} "
            f"cached {counts['cached']:,} empty {counts['empty']:,} "
            f"error {counts['error']:,}",
            flush=True,
        )

        # A run where everything fails should stop in seconds, not hours. An
        # expired AWS token previously produced 22,422 identical failures over
        # eleven minutes before anyone saw the cause.
        if counts["error"] >= 200 and counts["built"] == 0:
            print("\nEvery attempt has failed. Most common:", flush=True)
            for message, count in sorted(error_samples.items(), key=lambda kv: -kv[1])[:3]:
                print(f"  {count:>6,}  {message}", flush=True)
            return 1

    if error_samples:
        print("\nMost common failures:")
        for message, count in sorted(error_samples.items(), key=lambda kv: -kv[1])[:5]:
            print(f"  {count:>7,}  {message}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/3600:.2f} h")
    for name, value in counts.items():
        print(f"  {name:8} {value:,}")

    usable = counts["built"] + counts["cached"]
    if usable < len(tasks) * 0.9:
        print(
            f"\nOnly {100*usable/max(len(tasks),1):.0f}% of complexes produced a graph. "
            "Check the failures before training: a systematic parsing problem "
            "will look like a smaller dataset rather than an error."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
