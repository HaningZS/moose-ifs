#!/usr/bin/env python3
"""Merge old 220-case MooseBench results with rerun records for modified cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METHOD_ORDER = {
    "A": 0,
    "AE": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "AReg": 5,
    "ExecRepairReg": 6,
    "DReg": 7,
}


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_ids(path: Path) -> set[str]:
    return set(json.loads(path.read_text(encoding="utf-8")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Original full-result JSONL.")
    parser.add_argument(
        "--modified",
        type=Path,
        action="append",
        required=True,
        help="Modified-case rerun JSONL. May be passed multiple times; later files override earlier duplicates.",
    )
    parser.add_argument(
        "--modified-ids",
        type=Path,
        default=Path("experiments/moosebench_clean/modified_source_ids.json"),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Optional method filter applied to both base and modified records.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modified_ids = _load_ids(args.modified_ids)
    methods = set(args.methods) if args.methods else None
    merged: dict[tuple[str, str], dict] = {}

    for row in _load_jsonl(args.base):
        if methods and row["method"] not in methods:
            continue
        if row["id"] not in modified_ids:
            merged[(row["id"], row["method"])] = row

    for path in args.modified:
        for row in _load_jsonl(path):
            if methods and row["method"] not in methods:
                continue
            if row["id"] in modified_ids:
                merged[(row["id"], row["method"])] = row

    rows = sorted(
        merged.values(),
        key=lambda row: (row["id"], METHOD_ORDER.get(row["method"], 99), row["method"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    by_method: dict[str, int] = {}
    by_method_ids: dict[str, set[str]] = {}
    for row in rows:
        by_method[row["method"]] = by_method.get(row["method"], 0) + 1
        by_method_ids.setdefault(row["method"], set()).add(row["id"])

    print(f"Wrote {args.output}")
    print(f"Records: {len(rows)}")
    for method in sorted(by_method, key=lambda m: METHOD_ORDER.get(m, 99)):
        print(f"  {method}: {by_method[method]} records, {len(by_method_ids[method])} ids")


if __name__ == "__main__":
    main()
