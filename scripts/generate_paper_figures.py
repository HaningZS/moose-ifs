#!/usr/bin/env python3
"""Generate paper figures from MooseBench experiment results.

Produces publication-quality figures (DPI>400) with seaborn styling.
Output: results/figures/

Usage:
    uv run python scripts/generate_paper_figures.py --results-dir experiments/results/current
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RESULTS_DIR = _ROOT / "experiments" / "results" / "current"
_IMGS_DIR = _ROOT / "results" / "figures"

DPI = 450
sns.set_theme(style="whitegrid", font_scale=1.1, rc={
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.edgecolor": ".3",
    "axes.linewidth": 0.8,
    "grid.alpha": 0.3,
})

def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _current_representative_runs(results_dir: Path) -> dict[str, list[dict]]:
    return {
        "Sonnet": _load_jsonl(results_dir / "merged220_claude_abd.jsonl"),
        "DeepSeek V4 Flash": (
            _load_jsonl(results_dir / "merged220_deepseek-flash_a.jsonl")
            + _load_jsonl(results_dir / "merged220_deepseek-flash_d.jsonl")
        ),
    }


def _rolling_mean(xs: list[float], ys: list[float]) -> tuple[np.ndarray, np.ndarray]:
    pairs = sorted(zip(xs, ys, strict=False))
    x_arr = np.array([p[0] for p in pairs], dtype=float)
    y_arr = np.array([p[1] for p in pairs], dtype=float)
    window = max(len(x_arr) // 8, 5)
    kernel = np.ones(window) / window
    return (
        np.convolve(x_arr, kernel, mode="valid"),
        np.convolve(y_arr, kernel, mode="valid"),
    )


def fig_pipeline_diagnostics_compact(results_dir: Path) -> None:
    """Compact two-panel replacement for the old standalone Fig. 5 and Fig. 6."""
    runs = _current_representative_runs(results_dir)
    colors = {"Sonnet": "#2F5F9E", "DeepSeek V4 Flash": "#D6A44A"}
    plt.rcParams.update({
        "font.size": 7.4,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.4,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 5.7,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.12))
    ax_ext, ax_gain = axes

    for label, records in runs.items():
        d_recs = [
            r for r in records
            if r.get("method") == "D"
            and r.get("extraction_ifs") is not None
            and r.get("ifs") is not None
            and float(r["extraction_ifs"]) > 0
        ]
        ax_ext.scatter(
            [float(r["extraction_ifs"]) for r in d_recs],
            [float(r.get("ifs") or 0.0) for r in d_recs],
            s=14,
            alpha=0.43,
            color=colors[label],
            edgecolors="none",
            label=label,
        )

        direct = {
            r["id"]: float(r.get("ifs") or 0.0)
            for r in records
            if r.get("method") == "A"
        }
        refine = {
            r["id"]: float(r.get("ifs") or 0.0)
            for r in records
            if r.get("method") == "D"
        }
        common = sorted(set(direct) & set(refine))
        x = [direct[i] for i in common]
        y = [refine[i] - direct[i] for i in common]
        ax_gain.scatter(
            x,
            y,
            s=14,
            alpha=0.38,
            color=colors[label],
            edgecolors="none",
        )
        sx, sy = _rolling_mean(x, y)
        ax_gain.plot(sx, sy, lw=1.7, color=colors[label], alpha=0.88, label=label)

    ax_ext.plot([0, 1], [0, 1], color="0.42", lw=0.8, ls=":", alpha=0.72)
    ax_ext.set_title("(a) Extraction quality", loc="left", fontweight="bold", pad=4)
    ax_ext.set_xlabel("Extraction IFS")
    ax_ext.set_ylabel("Final code IFS")
    ax_ext.set_xlim(-0.02, 1.03)
    ax_ext.set_ylim(-0.02, 1.03)
    ax_ext.legend(loc="upper left", frameon=True, framealpha=0.92,
                  borderpad=0.25, handletextpad=0.3)
    ax_ext.grid(alpha=0.22)

    ax_gain.axhline(0, color="0.25", lw=0.8)
    ax_gain.axvline(0.7, color="0.45", lw=0.8, ls="--")
    ceiling_x = np.linspace(0.0, 1.0, 100)
    ax_gain.plot(ceiling_x, 1.0 - ceiling_x, color="0.52", lw=0.75, ls=":", alpha=0.55)
    ax_gain.text(0.08, 0.86, "max possible gain", color="0.42", fontsize=5.8)
    ax_gain.set_title("(b) Conditional benefit", loc="left", fontweight="bold", pad=4)
    ax_gain.set_xlabel("Direct IFS")
    ax_gain.set_ylabel("PDE-Refine gain")
    ax_gain.set_xlim(-0.02, 1.03)
    ax_gain.set_ylim(-0.55, 1.03)
    ax_gain.legend(loc="upper right", frameon=True, framealpha=0.92,
                   borderpad=0.25, handletextpad=0.3)
    ax_gain.grid(alpha=0.22)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        ax.tick_params(direction="out", top=False, right=False, length=3.0, width=0.9)

    fig.tight_layout(w_pad=1.0)
    fig.savefig(_IMGS_DIR / "pipeline_diagnostics_compact.pdf", bbox_inches="tight")
    fig.savefig(_IMGS_DIR / "pipeline_diagnostics_compact.png", bbox_inches="tight", dpi=450)
    plt.close(fig)
    print("  pipeline_diagnostics_compact.pdf/png")


def fig_execution_audit_quadrants() -> None:
    rows = [
        ("Sonnet\nExec-Repair+Reg", 43.2, 40.0, 16.8),
        ("Sonnet\nPDE-Reg", 60.0, 31.4, 8.6),
        ("GPT-5.4\nExec-Repair+Reg", 42.3, 40.0, 17.7),
        ("GPT-5.4\nPDE-Reg", 56.8, 30.0, 13.2),
        ("DeepSeek V4 Flash\nExec-Repair+Reg", 35.0, 39.1, 25.9),
        ("DeepSeek V4 Flash\nPDE-Reg", 60.0, 27.7, 12.3),
    ]
    labels = [r[0] for r in rows]
    good = np.array([r[1] for r in rows])
    false = np.array([r[2] for r in rows])
    no_exec = np.array([r[3] for r in rows])
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7.1, 2.9))
    width = 0.72
    colors = {
        "GoodExec": "#4E88C7",
        "FalseExec": "#D39C3F",
        "Non-executing": "#D7D7D7",
    }
    ax.bar(x, good, width, color=colors["GoodExec"], label="GoodExec")
    ax.bar(x, false, width, bottom=good, color=colors["FalseExec"], label="FalseExec")
    ax.bar(x, no_exec, width, bottom=good + false, color=colors["Non-executing"], label="Non-executing")

    for i, (g, f, n) in enumerate(zip(good, false, no_exec, strict=True)):
        ax.text(i, g / 2, f"{g:.1f}", ha="center", va="center", fontsize=7, color="white", fontweight="bold")
        ax.text(i, g + f / 2, f"{f:.1f}", ha="center", va="center", fontsize=7, color="white", fontweight="bold")
        ax.text(i, g + f + n / 2, f"{n:.1f}", ha="center", va="center", fontsize=7, color="0.25")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Cases (%)", fontsize=8)
    ax.set_ylim(0, 100)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(_IMGS_DIR / "execution_audit_quadrants.pdf", bbox_inches="tight")
    fig.savefig(_IMGS_DIR / "execution_audit_quadrants.png", bbox_inches="tight", dpi=450)
    plt.close(fig)
    print("  execution_audit_quadrants.pdf/png")


# ---------------------------------------------------------------------------
# Figure 3: Error Decomposition Stacked Bar
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    _IMGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Output: {_IMGS_DIR}")
    print(f"Input results: {args.results_dir}")
    print()

    print("Generating current paper figures:")
    fig_pipeline_diagnostics_compact(args.results_dir)
    fig_execution_audit_quadrants()

    print(f"\nDone. {len(list(_IMGS_DIR.glob('*.pdf')))} PDFs + {len(list(_IMGS_DIR.glob('*.png')))} PNGs in {_IMGS_DIR}")


if __name__ == "__main__":
    main()
