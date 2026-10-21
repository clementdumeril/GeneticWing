"""Merge dataset shards written by ``phase3_dataset.py --first ... --count ...``.

Run: ``python scenarios/phase3_merge.py SHARD_DIR [MORE_DIRS_OR_FILES ...] --out dataset.npz``

Every shard must come from the same tuned law (same ``theta``); instance ids
must not repeat across shards.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="+", type=Path, help="shard files or directories of shards")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    files = []
    for src in args.sources:
        files += sorted(src.glob("*.npz")) if src.is_dir() else [src]
    if not files:
        raise SystemExit("no shard found")
    shards = [np.load(f) for f in files]
    theta = shards[0]["theta"]
    for f, d in zip(files, shards):
        if not np.allclose(d["theta"], theta):
            raise SystemExit(f"{f} was generated with a different law")
    keys = [k for k in shards[0].files if k.startswith(("rec_", "teacher_"))]
    merged = {}
    for k in keys:
        parts = [d[k] for d in shards if d[k].size]
        merged[k] = np.concatenate(parts) if parts else shards[0][k]
    ids = merged["teacher_instance"]
    if len(np.unique(ids)) != len(ids):
        raise SystemExit("an instance appears in two shards")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, theta=theta, **merged)
    split = merged["rec_split"]
    print(
        f"merged {len(files)} shards: {len(ids)} instances, {len(split)} solutions "
        f"({(split == 'train').sum()} train, {(split == 'test').sum()} test) -> {args.out}"
    )


if __name__ == "__main__":
    main()
