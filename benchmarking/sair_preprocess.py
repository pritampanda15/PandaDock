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
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger("pandadock.sair.preprocess")

_BUILDER = None


def _builder():
    """One graph builder per worker process."""
    global _BUILDER
    if _BUILDER is None:
        from pandadock.gnn.data.graph_builder import HeterogeneousGraphBuilder

        _BUILDER = HeterogeneousGraphBuilder()
    return _BUILDER


def cache_path(cache_dir: Path, entry_id: int) -> Path:
    """Shard by id so no directory holds a million files."""
    return cache_dir / f"{entry_id % 1000:03d}" / f"{entry_id}.pt"


def process_one(task) -> tuple:
    """Build and cache one complex. Returns (entry_id, status)."""
    entry_id, cif_path, smiles, pic50, cache_dir = task
    import torch

    from pandadock.gnn.data.sair_dataset import build_complex_graph

    target = cache_path(Path(cache_dir), entry_id)
    if target.exists():
        return entry_id, "cached"

    try:
        graph = build_complex_graph(cif_path, smiles, _builder())
        if graph is None:
            return entry_id, "empty"

        graph.y = torch.tensor([pic50], dtype=torch.float)
        graph.entry_id = entry_id

        target.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary name first: a worker killed mid-write would
        # otherwise leave a truncated file that the resume logic counts as done.
        tmp = target.with_suffix(".tmp")
        torch.save(graph, tmp)
        tmp.replace(target)
        return entry_id, "built"
    except Exception as exc:
        logger.debug("entry %s failed: %s", entry_id, exc)
        return entry_id, f"error:{type(exc).__name__}"


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
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s: %(message)s")

    from pandadock.gnn.data.sair_dataset import load_entries, split_by_sequence

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
        (e.entry_id, e.cif_path, e.smiles, e.pic50, str(cache_dir)) for e in entries
    ]
    print(f"{len(tasks):,} complexes, {args.workers} workers", flush=True)

    counts = {"built": 0, "cached": 0, "empty": 0, "error": 0}
    start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_one, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            _, status = future.result()
            counts[status if status in counts else "error"] += 1

            if done % 1000 == 0 or done == len(tasks):
                elapsed = time.time() - start
                rate = done / elapsed
                remaining = (len(tasks) - done) / rate if rate else 0
                print(
                    f"  {done:,}/{len(tasks):,}  {rate:.0f}/s  "
                    f"eta {remaining/3600:.1f} h  "
                    f"built {counts['built']:,} cached {counts['cached']:,} "
                    f"empty {counts['empty']:,} error {counts['error']:,}",
                    flush=True,
                )

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
