#!/usr/bin/env python3
"""Read-only MooseBench JSONL analyzer.

This script summarizes method means, retry cohorts, extraction-quality bins,
and D-C decomposition without changing runner outputs or score definitions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

METHOD_ORDER = ("A", "AE", "B", "C", "D")
GAP_PAIRS = (("B", "A"), ("D", "B"), ("C", "D"), ("D", "A"), ("C", "A"))


def load_records(path: Path) -> list[dict]:
    """Load non-empty JSONL records, skipping malformed lines."""
    records: list[dict] = []
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


def _score(record: dict | None) -> float | None:
    if record is None:
        return None
    value = record.get("ifs")
    return float(value) if value is not None else 0.0


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _by_case(records: list[dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in records:
        grouped[record["id"]][record["method"]] = record
    return dict(grouped)


def _bin_extraction(value: float) -> str:
    if value < 0.5:
        return "<0.5"
    if value < 0.7:
        return "0.5-0.7"
    if value < 0.9:
        return "0.7-0.9"
    return ">=0.9"


def build_summary(records: list[dict]) -> dict:
    """Build aggregate summaries for a MooseBench JSONL record list."""
    summary: dict = {
        "n_records": len(records),
        "n_cases": len({record["id"] for record in records}),
        "method_means": {},
        "gap_means": {},
        "dc_decomposition": {},
        "retry": {},
        "extraction_bins": {},
    }

    for method in METHOD_ORDER:
        recs = [record for record in records if record.get("method") == method]
        if not recs:
            continue
        scores = [_score(record) for record in recs]
        score_vals = [value for value in scores if value is not None]
        parsed = sum(1 for record in recs if record.get("parse"))
        summary["method_means"][method] = {
            "n": len(recs),
            "mean_ifs": _mean(score_vals),
            "parse_rate": parsed / len(recs) if recs else None,
        }

    grouped = _by_case(records)
    for left, right in GAP_PAIRS:
        deltas: list[float] = []
        for methods in grouped.values():
            left_score = _score(methods.get(left))
            right_score = _score(methods.get(right))
            if left_score is not None and right_score is not None:
                deltas.append(left_score - right_score)
        if deltas:
            summary["gap_means"][f"{left}-{right}"] = _mean(deltas)

    dc_rows: list[dict] = []
    for case_id, methods in grouped.items():
        c_score = _score(methods.get("C"))
        d = methods.get("D")
        d_score = _score(d)
        if c_score is None or d_score is None or d is None:
            continue
        ext = d.get("extraction_ifs")
        internal = d.get("internal_ifs")
        row = {
            "id": case_id,
            "c_minus_d": c_score - d_score,
            "extraction_loss": 1.0 - ext if ext is not None else None,
            "generation_loss": 1.0 - internal if internal is not None else None,
        }
        dc_rows.append(row)

    if dc_rows:
        summary["dc_decomposition"] = {
            "n": len(dc_rows),
            "mean_c_minus_d": _mean([row["c_minus_d"] for row in dc_rows]),
            "mean_extraction_loss": _mean([
                row["extraction_loss"] for row in dc_rows
                if row["extraction_loss"] is not None
            ]),
            "mean_generation_loss": _mean([
                row["generation_loss"] for row in dc_rows
                if row["generation_loss"] is not None
            ]),
        }
    else:
        summary["dc_decomposition"] = {
            "n": 0,
            "mean_c_minus_d": None,
            "mean_extraction_loss": None,
            "mean_generation_loss": None,
        }

    for label, predicate in (
        ("retried", lambda r: bool(r.get("extraction_retried"))),
        ("not_retried", lambda r: not bool(r.get("extraction_retried"))),
    ):
        recs = [
            record for record in records
            if record.get("method") in {"B", "D"} and predicate(record)
        ]
        scores = [_score(record) for record in recs]
        ext_vals = [
            float(record["extraction_ifs"]) for record in recs
            if record.get("extraction_ifs") is not None
        ]
        summary["retry"][label] = {
            "n": len(recs),
            "mean_ifs": _mean([score for score in scores if score is not None]),
            "mean_extraction_ifs": _mean(ext_vals),
        }

    extraction_bins = {
        "<0.5": [],
        "0.5-0.7": [],
        "0.7-0.9": [],
        ">=0.9": [],
    }
    for record in records:
        if record.get("method") not in {"B", "D"}:
            continue
        ext = record.get("extraction_ifs")
        if ext is None:
            continue
        extraction_bins[_bin_extraction(float(ext))].append(record)
    for label, recs in extraction_bins.items():
        scores = [_score(record) for record in recs]
        summary["extraction_bins"][label] = {
            "n": len(recs),
            "mean_ifs": _mean([score for score in scores if score is not None]),
            "mean_extraction_ifs": _mean([
                float(record["extraction_ifs"]) for record in recs
            ]),
        }

    return summary


def build_tidy_rows(records: list[dict]) -> list[dict]:
    """Build tidy rows for method scores and per-case gap metrics."""
    rows: list[dict] = []
    for record in records:
        rows.append({
            "id": record["id"],
            "family": record.get("family"),
            "complexity": record.get("complexity"),
            "llm": record.get("llm"),
            "method": record.get("method"),
            "metric": "ifs",
            "value": _score(record),
        })

    grouped = _by_case(records)
    for case_id, methods in grouped.items():
        base = next(iter(methods.values()))
        for left, right in GAP_PAIRS:
            left_score = _score(methods.get(left))
            right_score = _score(methods.get(right))
            if left_score is None or right_score is None:
                continue
            rows.append({
                "id": case_id,
                "family": base.get("family"),
                "complexity": base.get("complexity"),
                "llm": base.get("llm"),
                "method": "",
                "metric": f"{left}-{right}",
                "value": left_score - right_score,
            })
    return rows


def write_tidy_csv(rows: list[dict], output_path: Path) -> None:
    """Write tidy analyzer rows as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "family", "complexity", "llm", "method", "metric", "value"]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: dict) -> None:
    """Print a compact human-readable summary."""
    print(f"Records: {summary['n_records']}  Cases: {summary['n_cases']}")

    print("\nMethod means:")
    for method in METHOD_ORDER:
        data = summary["method_means"].get(method)
        if not data:
            continue
        mean = data["mean_ifs"]
        parse = data["parse_rate"]
        mean_s = f"{mean:.4f}" if mean is not None else "N/A"
        parse_s = f"{100 * parse:.1f}%" if parse is not None else "N/A"
        print(f"  {method:<2} n={data['n']:<4} mean={mean_s:<8} parse={parse_s}")

    print("\nMean gaps:")
    for label, value in summary["gap_means"].items():
        print(f"  {label:<4} {value:+.4f}")

    dc = summary["dc_decomposition"]
    print("\nD-C decomposition:")
    print(f"  n={dc['n']}")
    for key in ("mean_c_minus_d", "mean_extraction_loss", "mean_generation_loss"):
        value = dc.get(key)
        value_s = f"{value:.4f}" if value is not None else "N/A"
        print(f"  {key}: {value_s}")

    print("\nRetry groups (B/D):")
    for label, data in summary["retry"].items():
        mean = data["mean_ifs"]
        ext = data["mean_extraction_ifs"]
        mean_s = f"{mean:.4f}" if mean is not None else "N/A"
        ext_s = f"{ext:.4f}" if ext is not None else "N/A"
        print(f"  {label:<11} n={data['n']:<4} mean={mean_s:<8} ext={ext_s}")

    print("\nExtraction bins (B/D):")
    for label, data in summary["extraction_bins"].items():
        mean = data["mean_ifs"]
        ext = data["mean_extraction_ifs"]
        mean_s = f"{mean:.4f}" if mean is not None else "N/A"
        ext_s = f"{ext:.4f}" if ext is not None else "N/A"
        print(f"  {label:<7} n={data['n']:<4} mean={mean_s:<8} ext={ext_s}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MooseBench JSONL results.")
    parser.add_argument("input", type=Path, help="MooseBench JSONL file to analyze.")
    parser.add_argument(
        "--tidy-output",
        type=Path,
        default=None,
        help="Optional CSV path for tidy method/gap rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    summary = build_summary(records)
    print_summary(summary)
    if args.tidy_output is not None:
        write_tidy_csv(build_tidy_rows(records), args.tidy_output)
        print(f"\nTidy CSV: {args.tidy_output}")


if __name__ == "__main__":
    main()
