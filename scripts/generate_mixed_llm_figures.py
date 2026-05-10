#!/usr/bin/env python3
"""Generate mixed-model 2x2 diagnostics from completed D-only JSONL runs.

The release artifact includes this executable analysis script but does not
bundle cached mixed2x2 result files. Place the completed JSONL files in
``experiments/results/current`` or pass ``--results-dir`` before
running.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "experiments" / "results" / "current"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "figures"


@dataclass(frozen=True)
class RunSpec:
    extractor: str
    generator: str
    filename: str
    note: str = ""


RUN_SPECS: dict[str, tuple[RunSpec, ...]] = {
    "gpt-deepseek": (
        RunSpec("GPT-5.4", "GPT-5.4", "mixed2x2_20260506_gpt54_self_d.jsonl"),
        RunSpec("GPT-5.4", "DeepSeek V4 Flash", "mixed2x2_20260506_gpt54_ext_deepseek_gen_d_clean_20260507.jsonl"),
        RunSpec("DeepSeek V4 Flash", "GPT-5.4", "mixed2x2_20260506_deepseek_ext_gpt54_gen_d_clean_20260507.jsonl"),
        RunSpec("DeepSeek V4 Flash", "DeepSeek V4 Flash", "mixed2x2_20260506_deepseek_self_d_clean_20260507.jsonl"),
    ),
    "gemini-deepseek": (
        RunSpec("Gemini 3.1 Flash Lite", "Gemini 3.1 Flash Lite", "mixed2x2_20260506_gemini31_lite_self_d.jsonl"),
        RunSpec(
            "Gemini 3.1 Flash Lite",
            "DeepSeek V4 Flash",
            "mixed2x2_20260506_gemini31_lite_ext_deepseek_gen_d_retrymerged_20260507.jsonl",
            "retry-merged cross cell",
        ),
        RunSpec(
            "DeepSeek V4 Flash",
            "Gemini 3.1 Flash Lite",
            "mixed2x2_20260506_deepseek_ext_gemini31_lite_gen_d_retrymerged_20260507.jsonl",
            "retry-merged cross cell",
        ),
        RunSpec("DeepSeek V4 Flash", "DeepSeek V4 Flash", "mixed2x2_20260506_deepseek_self_d_clean_20260507.jsonl"),
    ),
}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def score(row: dict) -> float:
    value = row.get("ifs")
    return float(value) if value is not None else 0.0


def summarize(spec: RunSpec, rows: list[dict]) -> dict[str, object]:
    scores = [score(row) for row in rows]
    parse_ok = sum(bool(row.get("parse")) for row in rows)
    errors = sum(1 for row in rows if row.get("error"))
    zeros = sum(1 for value in scores if value == 0.0)
    extraction_vals = [float(row["extraction_ifs"]) for row in rows if row.get("extraction_ifs") is not None]
    return {
        "extractor": spec.extractor,
        "generator": spec.generator,
        "filename": spec.filename,
        "rows": len(rows),
        "mean_ifs": float(np.mean(scores)) if scores else float("nan"),
        "parse_rate": parse_ok / len(rows) if rows else float("nan"),
        "error_rows": errors,
        "zero_ifs_rows": zeros,
        "mean_extraction_ifs": float(np.mean(extraction_vals)) if extraction_vals else float("nan"),
        "note": spec.note,
    }


def load_pair(pair: str, results_dir: Path) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    missing: list[Path] = []
    for spec in RUN_SPECS[pair]:
        path = results_dir / spec.filename
        if not path.exists():
            missing.append(path)
            continue
        summaries.append(summarize(spec, load_jsonl(path)))
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing mixed2x2 result files. The artifact ships code/scripts only; "
            "place completed JSONL files first or pass --results-dir.\n" + formatted
        )
    return summaries


def write_summary_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "extractor",
        "generator",
        "filename",
        "rows",
        "mean_ifs",
        "parse_rate",
        "error_rows",
        "zero_ifs_rows",
        "mean_extraction_ifs",
        "note",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmap(pair: str, rows: list[dict[str, object]], output_dir: Path) -> None:
    extractors = sorted({str(row["extractor"]) for row in rows})
    generators = sorted({str(row["generator"]) for row in rows})
    matrix = np.full((len(extractors), len(generators)), np.nan)
    parse = np.full_like(matrix, np.nan, dtype=float)
    for row in rows:
        i = extractors.index(str(row["extractor"]))
        j = generators.index(str(row["generator"]))
        matrix[i, j] = float(row["mean_ifs"])
        parse[i, j] = float(row["parse_rate"])

    fig, ax = plt.subplots(figsize=(5.3, 3.7))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_xticks(np.arange(len(generators)))
    ax.set_xticklabels(generators, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(extractors)))
    ax.set_yticklabels(extractors)
    ax.set_xlabel("Generator")
    ax.set_ylabel("Extractor")
    ax.set_title(f"Mixed-model 2x2 D-only IFS ({pair})", loc="left", fontweight="bold")
    for i in range(len(extractors)):
        for j in range(len(generators)):
            if np.isfinite(matrix[i, j]):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.3f}\nparse {parse[i, j] * 100:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean final IFS")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"mixed2x2_{pair}_diagnostics.{ext}", bbox_inches="tight", dpi=450)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=sorted(RUN_SPECS), default="gpt-deepseek")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = load_pair(args.pair, args.results_dir)
    summary_csv = args.summary_csv or args.output_dir / f"mixed2x2_{args.pair}_summary.csv"
    write_summary_csv(summaries, summary_csv)
    plot_heatmap(args.pair, summaries, args.output_dir)
    print(f"Wrote {summary_csv}")
    print(f"Wrote {args.output_dir / f'mixed2x2_{args.pair}_diagnostics.pdf'}")
    print(f"Wrote {args.output_dir / f'mixed2x2_{args.pair}_diagnostics.png'}")


if __name__ == "__main__":
    main()
