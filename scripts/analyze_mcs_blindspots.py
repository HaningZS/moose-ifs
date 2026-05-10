#!/usr/bin/env python3
"""Read-only MCS blind-spot analyzer for MooseBench JSONL outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from codmos.multiagent.pde.ifs_engine import compute_material_consistency


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL records, skipping malformed lines."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _score(record: dict[str, Any]) -> float | None:
    value = record.get("ifs")
    return float(value) if value is not None else None


def _resolve_code_path(record: dict[str, Any], repo_root: Path) -> Path | None:
    raw = record.get("code_path")
    if not raw:
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = repo_root / path
    return path if path.exists() else None


def _source_path(record: dict[str, Any], source_dir: Path) -> Path:
    return source_dir / f"{record['id']}.i"


def _mismatch_props(mismatched: list[dict[str, Any]]) -> str:
    return ";".join(str(item.get("property", "")) for item in mismatched)


def _mismatch_reasons(mismatched: list[dict[str, Any]]) -> str:
    return ";".join(str(item.get("reason", "")) for item in mismatched)


def _mismatch_bucket(prop: str) -> str:
    return prop.split(":", 1)[0] if ":" in prop else prop or "unknown"


def build_rows(
    records: list[dict[str, Any]],
    *,
    source_dir: Path,
    repo_root: Path,
    high_ifs: float,
    low_mcs: float,
) -> list[dict[str, Any]]:
    """Re-score records with current MCS and return analyzer rows."""
    rows: list[dict[str, Any]] = []
    for record in records:
        if not record.get("parse"):
            continue

        source_path = _source_path(record, source_dir)
        code_path = _resolve_code_path(record, repo_root)
        if not source_path.exists() or code_path is None:
            continue

        try:
            mcs = compute_material_consistency(
                source_path.read_text(encoding="utf-8"),
                code_path.read_text(encoding="utf-8"),
            )
            error = None
        except Exception as exc:  # pragma: no cover - defensive batch path
            mcs = None
            error = str(exc)

        if mcs is None:
            row = {
                "id": record.get("id"),
                "method": record.get("method"),
                "llm": record.get("llm"),
                "family": record.get("family"),
                "complexity": record.get("complexity"),
                "ifs": _score(record),
                "mcs": None,
                "mcs_total": None,
                "mcs_matched": None,
                "mcs_mismatched": None,
                "high_ifs_low_mcs": False,
                "mismatch_properties": "",
                "mismatch_reasons": error or "",
                "first_mismatch": "",
                "code_path": str(code_path),
            }
            rows.append(row)
            continue

        applicable = mcs.total_properties > 0
        score = mcs.score if applicable else None
        ifs = _score(record)
        high_ifs_low_mcs = (
            score is not None
            and ifs is not None
            and ifs >= high_ifs
            and score < low_mcs
        )
        rows.append({
            "id": record.get("id"),
            "method": record.get("method"),
            "llm": record.get("llm"),
            "family": record.get("family"),
            "complexity": record.get("complexity"),
            "ifs": ifs,
            "mcs": score,
            "mcs_total": mcs.total_properties,
            "mcs_matched": mcs.matched_properties,
            "mcs_mismatched": len(mcs.mismatched),
            "high_ifs_low_mcs": high_ifs_low_mcs,
            "mismatch_properties": _mismatch_props(mcs.mismatched),
            "mismatch_reasons": _mismatch_reasons(mcs.mismatched),
            "first_mismatch": json.dumps(
                mcs.mismatched[0] if mcs.mismatched else {},
                ensure_ascii=False,
                sort_keys=True,
            ),
            "code_path": str(code_path),
        })
    return rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize MCS rows by applicability, family, and mismatch bucket."""
    comparable = [row for row in rows if row.get("mcs") is not None]
    low_rows = [
        row for row in comparable
        if row.get("mcs") is not None and float(row["mcs"]) < 0.5
    ]
    high_ifs_low = [row for row in comparable if row.get("high_ifs_low_mcs")]

    bucket_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for row in comparable:
        props = [prop for prop in str(row.get("mismatch_properties", "")).split(";") if prop]
        reasons = [reason for reason in str(row.get("mismatch_reasons", "")).split(";") if reason]
        bucket_counts.update(_mismatch_bucket(prop) for prop in props)
        reason_counts.update(reasons)

    family_scores: dict[str, list[float]] = defaultdict(list)
    for row in comparable:
        family_scores[str(row.get("family") or "unknown")].append(float(row["mcs"]))

    return {
        "scanned": len(rows),
        "comparable": len(comparable),
        "mean_mcs": _mean([float(row["mcs"]) for row in comparable]),
        "low_mcs_count": len(low_rows),
        "high_ifs_low_mcs_count": len(high_ifs_low),
        "mismatch_buckets": dict(bucket_counts),
        "mismatch_reasons": dict(reason_counts),
        "family_mean_mcs": {
            family: _mean(values)
            for family, values in sorted(family_scores.items())
        },
    }


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write analyzer rows as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "method",
        "llm",
        "family",
        "complexity",
        "ifs",
        "mcs",
        "mcs_total",
        "mcs_matched",
        "mcs_mismatched",
        "high_ifs_low_mcs",
        "mismatch_properties",
        "mismatch_reasons",
        "first_mismatch",
        "code_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: dict[str, Any], rows: list[dict[str, Any]], *, top: int) -> None:
    """Print a compact human-readable summary."""
    print("MCS Blind-Spot Summary")
    print(f"scanned rows: {summary['scanned']}")
    print(f"comparable rows: {summary['comparable']}")
    mean_mcs = summary["mean_mcs"]
    print(f"mean MCS: {mean_mcs:.4f}" if mean_mcs is not None else "mean MCS: n/a")
    print(f"low MCS (<0.5): {summary['low_mcs_count']}")
    print(f"high IFS / low MCS: {summary['high_ifs_low_mcs_count']}")
    print(f"mismatch buckets: {summary['mismatch_buckets']}")
    print(f"mismatch reasons: {summary['mismatch_reasons']}")
    print(f"family mean MCS: {summary['family_mean_mcs']}")

    ranked = sorted(
        [row for row in rows if row.get("high_ifs_low_mcs")],
        key=lambda row: (-(float(row["ifs"]) if row.get("ifs") is not None else 0.0), float(row["mcs"])),
    )
    if ranked:
        print("\nTop high-IFS / low-MCS cases:")
        for row in ranked[:top]:
            print(
                f"- {row['id']} method={row['method']} ifs={float(row['ifs']):.4f} "
                f"mcs={float(row['mcs']):.4f} props={row['mismatch_properties']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="MooseBench result JSONL")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("experiments/moosebench_clean/source_files"),
        help="Directory containing reference .i files named by case id",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repo root used to resolve relative code_path fields",
    )
    parser.add_argument("--output-csv", type=Path, help="Optional CSV output path")
    parser.add_argument("--high-ifs", type=float, default=0.9)
    parser.add_argument("--low-mcs", type=float, default=0.5)
    parser.add_argument("--top", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.jsonl)
    rows = build_rows(
        records,
        source_dir=args.source_dir,
        repo_root=args.repo_root,
        high_ifs=args.high_ifs,
        low_mcs=args.low_mcs,
    )
    summary = build_summary(rows)
    if args.output_csv:
        write_csv(rows, args.output_csv)
    print_summary(summary, rows, top=args.top)


if __name__ == "__main__":
    main()
