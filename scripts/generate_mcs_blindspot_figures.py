#!/usr/bin/env python3
"""Generate MCS blind-spot and repair-smoke figures for the paper."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER_IMGS = ROOT / "results" / "figures"
L2_DATA = ROOT / "results" / "derived" / "ifs_vs_l2_data.json"


def _to_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def load_blindspot_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_repair_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def primary_category(row: dict[str, Any]) -> str:
    props = str(row.get("mismatch_properties") or "").split(";")
    buckets = {prop.split(":", 1)[0] for prop in props if prop}
    if "material_model" in buckets:
        return "Constitutive model"
    if "bc" in buckets:
        return "BC coefficient"
    if "kernel" in buckets:
        return "Kernel coefficient"
    if "material" in buckets:
        return "Material parameter"
    return "No MCS mismatch"


def mismatch_bucket_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for prop in str(row.get("mismatch_properties") or "").split(";"):
            if not prop:
                continue
            counts[prop.split(":", 1)[0]] += 1
    return counts


def figure(
    blind_rows: list[dict[str, Any]],
    repair_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8.1,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    })
    comparable = [
        row for row in blind_rows
        if _to_float(row.get("ifs")) is not None and _to_float(row.get("mcs")) is not None
    ]
    high_blind = [
        row for row in comparable
        if float(row["ifs"]) >= 0.9 and float(row["mcs"]) < 0.5
    ]
    category_order = [
        "No MCS mismatch",
        "Kernel coefficient",
        "BC coefficient",
        "Material parameter",
        "Constitutive model",
    ]
    colors = {
        "No MCS mismatch": "#BDBDBD",
        "Kernel coefficient": "#4E88C7",
        "BC coefficient": "#D39C3F",
        "Material parameter": "#8E6BBE",
        "Constitutive model": "#CC79A7",
    }

    fig = plt.figure(figsize=(7.1, 4.35))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.42, 1.0], hspace=0.56, wspace=0.34)
    ax_scatter = fig.add_subplot(gs[:, 0])
    ax_mcs = fig.add_subplot(gs[0, 1])
    ax_ifs = fig.add_subplot(gs[1, 1])

    ax_scatter.axvspan(0.9, 1.01, ymin=0.0, ymax=0.5, color="#EEE7F6", alpha=0.62)
    ax_scatter.axvline(0.9, color="0.35", lw=0.8, ls="--")
    ax_scatter.axhline(0.5, color="0.35", lw=0.8, ls="--")
    for category in category_order:
        xs = [
            float(row["ifs"]) for row in comparable
            if primary_category(row) == category
        ]
        ys = [
            float(row["mcs"]) for row in comparable
            if primary_category(row) == category
        ]
        if not xs:
            continue
        ax_scatter.scatter(
            xs,
            ys,
            s=15,
            alpha=0.72,
            color=colors[category],
            edgecolors="none",
            label=category,
        )
    repaired_ids = {row["id"] for row in repair_rows}
    repaired_points = [row for row in comparable if row["id"] in repaired_ids]
    ax_scatter.scatter(
        [float(row["ifs"]) for row in repaired_points],
        [float(row["mcs"]) for row in repaired_points],
        s=35,
        facecolors="none",
        edgecolors="black",
        linewidths=0.8,
        label="repair smoke",
    )
    ax_scatter.set_xlim(0.45, 1.02)
    ax_scatter.set_ylim(-0.03, 1.03)
    ax_scatter.set_xlabel("IFS")
    ax_scatter.set_ylabel("MCS")
    ax_scatter.set_title(
        f"(a) IFS-blind coefficient failures (n={len(high_blind)})",
        loc="left",
        fontsize=8.4,
        fontweight="bold",
        pad=5,
    )
    ax_scatter.grid(alpha=0.22)
    ax_scatter.legend(loc="lower left", frameon=True, ncol=1)

    ids = [row["id"] for row in repair_rows]
    order = np.argsort([row["before"]["mcs"] for row in repair_rows])
    xs = np.arange(len(ids))
    mcs_before = np.array([repair_rows[i]["before"]["mcs"] for i in order], dtype=float)
    mcs_after = np.array([repair_rows[i]["best"]["mcs"] for i in order], dtype=float)
    ifs_before = np.array([repair_rows[i]["before"]["ifs"] for i in order], dtype=float)
    ifs_after = np.array([repair_rows[i]["best"]["ifs"] for i in order], dtype=float)

    for i, (before, after) in enumerate(zip(mcs_before, mcs_after, strict=True)):
        ax_mcs.plot([i, i], [before, after], color="#3182BD", alpha=0.55, lw=1.0)
    ax_mcs.scatter(xs, mcs_before, s=16, color="#9ECAE1", label="before")
    ax_mcs.scatter(xs, mcs_after, s=18, color="#08519C", label="after")
    ax_mcs.set_ylim(-0.03, 1.05)
    ax_mcs.set_xticks([])
    ax_mcs.set_ylabel("MCS")
    ax_mcs.set_title(
        f"(b) MCS repair\n"
        f"mean {mcs_before.mean():.3f}->{mcs_after.mean():.3f}; "
        f"{sum(abs(mcs_after - 1.0) < 1e-9)}/{len(repair_rows)} at 1.0",
        loc="left",
        fontsize=8.1,
        fontweight="bold",
        pad=5,
    )
    ax_mcs.grid(axis="y", alpha=0.22)
    ax_mcs.legend(fontsize=6.5, loc="lower right", frameon=True)

    for i, (before, after) in enumerate(zip(ifs_before, ifs_after, strict=True)):
        ax_ifs.plot([i, i], [before, after], color="#8E6BBE", alpha=0.50, lw=1.0)
    ax_ifs.scatter(xs, ifs_before, s=16, color="#C8B7E4", label="before")
    ax_ifs.scatter(xs, ifs_after, s=18, color="#6D4FA3", label="after")
    ax_ifs.set_ylim(0.78, 1.02)
    ax_ifs.set_xticks([])
    ax_ifs.set_ylabel("IFS")
    ax_ifs.set_title(
        f"(c) IFS preserved\n"
        f"mean {ifs_before.mean():.3f}->{ifs_after.mean():.3f}; "
        f"{sum(ifs_after >= ifs_before - 1e-12)}/{len(repair_rows)} non-decrease",
        loc="left",
        fontsize=8.1,
        fontweight="bold",
        pad=5,
    )
    ax_ifs.grid(axis="y", alpha=0.22)
    ax_ifs.legend(fontsize=6.5, loc="lower right", frameon=True)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.92)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = output_dir / "mcs_blindspot_repair.pdf"
    out_png = output_dir / "mcs_blindspot_repair.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=450)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")

    buckets = mismatch_bucket_counts(comparable)
    print("Blindspot survey:")
    print(f"- comparable={len(comparable)}")
    print(f"- high_ifs_low_mcs={len(high_blind)}")
    print(f"- mean_mcs={np.mean([float(row['mcs']) for row in comparable]):.4f}")
    print(f"- buckets={dict(buckets)}")
    print("Repair smoke:")
    print(f"- n={len(repair_rows)}")
    print(f"- accepted={sum(bool(row['accepted']) for row in repair_rows)}")
    print(f"- mean_mcs={mcs_before.mean():.4f}->{mcs_after.mean():.4f}")
    print(f"- mean_ifs={ifs_before.mean():.4f}->{ifs_after.mean():.4f}")


def compact_figure(
    blind_rows: list[dict[str, Any]],
    repair_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate a half-width version for the main paper."""
    plt.rcParams.update({
        "font.size": 7.4,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.4,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 5.7,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    })
    comparable = [
        row for row in blind_rows
        if _to_float(row.get("ifs")) is not None and _to_float(row.get("mcs")) is not None
    ]

    category_order = [
        "No MCS mismatch",
        "Kernel coefficient",
        "BC coefficient",
        "Material parameter",
        "Constitutive model",
    ]
    colors = {
        "No MCS mismatch": "#BDBDBD",
        "Kernel coefficient": "#4E88C7",
        "BC coefficient": "#D39C3F",
        "Material parameter": "#8E6BBE",
        "Constitutive model": "#CC79A7",
    }
    label_map = {
        "No MCS mismatch": "No mismatch",
        "Kernel coefficient": "Kernel coeff.",
        "BC coefficient": "BC coeff.",
        "Material parameter": "Material param.",
        "Constitutive model": "Constitutive",
    }

    fig, ax = plt.subplots(figsize=(3.45, 2.70))
    ax.axvspan(0.9, 1.01, ymin=0.0, ymax=0.5, color="#EEE7F6", alpha=0.62)
    ax.axvline(0.9, color="0.35", lw=0.75, ls="--")
    ax.axhline(0.5, color="0.35", lw=0.75, ls="--")
    for category in category_order:
        xs = [
            float(row["ifs"]) for row in comparable
            if primary_category(row) == category
        ]
        ys = [
            float(row["mcs"]) for row in comparable
            if primary_category(row) == category
        ]
        if not xs:
            continue
        ax.scatter(
            xs,
            ys,
            s=13,
            alpha=0.72,
            color=colors[category],
            edgecolors="none",
            label=label_map[category],
        )

    repaired_ids = {row["id"] for row in repair_rows}
    repaired_points = [row for row in comparable if row["id"] in repaired_ids]
    ax.scatter(
        [float(row["ifs"]) for row in repaired_points],
        [float(row["mcs"]) for row in repaired_points],
        s=31,
        facecolors="none",
        edgecolors="black",
        linewidths=0.75,
        label="Repair subset",
    )

    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("IFS")
    ax.set_ylabel("MCS")
    ax.set_title("MCS coefficient blind-spot repair", loc="left", fontweight="bold", pad=4)
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.92,
              borderpad=0.25, handletextpad=0.25, columnspacing=0.55)
    ax.grid(alpha=0.22)

    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.15, top=0.90)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = output_dir / "mcs_blindspot_repair.pdf"
    out_png = output_dir / "mcs_blindspot_repair.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=450)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


def _plot_ifs_l2_panel(ax: plt.Axes, l2_rows: list[dict[str, Any]]) -> None:
    cat_style = {
        "structural": {
            "color": "#8E6BBE",
            "marker": "D",
            "label": "Structural",
            "zorder": 6,
        },
        "bc_value": {
            "color": "#D39C3F",
            "marker": "o",
            "label": "BC value",
            "zorder": 5,
        },
        "coefficient": {
            "color": "#4E88C7",
            "marker": "s",
            "label": "Coefficient",
            "zorder": 7,
        },
    }
    ifs_blind_threshold = 0.90
    l2_blind_threshold = 0.50
    ymin, ymax = -0.08, 2.0

    ax.axvspan(0.40, ifs_blind_threshold, alpha=0.045, color="#EEE7F6", zorder=0)
    ax.fill_between(
        [ifs_blind_threshold, 1.06],
        l2_blind_threshold,
        ymax,
        color="#E8F1FA",
        alpha=0.12,
        zorder=0,
    )
    ax.axvline(ifs_blind_threshold, color="0.35", linestyle="--", linewidth=1.0, alpha=0.9)
    ax.axhline(l2_blind_threshold, color="0.35", linestyle="--", linewidth=1.0, alpha=0.9)

    for category in ["structural", "bc_value", "coefficient"]:
        points = [row for row in l2_rows if row.get("category", "structural") == category]
        if not points:
            continue
        style = cat_style[category]
        ax.scatter(
            [float(row["ifs"]) for row in points],
            [float(row["l2_error"]) for row in points],
            c=style["color"],
            marker=style["marker"],
            s=24,
            edgecolors="white",
            linewidths=0.25,
            alpha=0.85,
            label=style["label"],
            zorder=style["zorder"],
        )

    structural = [row for row in l2_rows if row.get("category", "structural") == "structural"]
    if len(structural) > 4:
        s_ifs = np.array([float(row["ifs"]) for row in structural])
        s_l2 = np.array([float(row["l2_error"]) for row in structural])
        coeffs = np.polyfit(s_ifs, s_l2, 1)
        x_fit = np.linspace(0.45, 0.92, 50)
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(
            x_fit,
            np.clip(y_fit, 0, 3),
            "--",
            color="#8E6BBE",
            linewidth=0.9,
            alpha=0.35,
            label="Structural trend",
        )

    ax.text(
        0.87,
        0.90,
        "blind quadrant\nhigh IFS / high $E_{L^2}$",
        transform=ax.transAxes,
        fontsize=5.8,
        fontstyle="italic",
        color="#4C72B0",
        ha="center",
        va="top",
        alpha=0.8,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#4C72B0", alpha=0.08),
    )
    ax.set_xlabel("IFS")
    ax.set_ylabel("$E_{L^2}$")
    ax.set_xlim(0.42, 1.06)
    ax.set_ylim(ymin, ymax)
    ax.legend(loc="upper left", ncol=1, framealpha=0.92, borderpad=0.25, handletextpad=0.3)
    ax.set_title("(a) IFS structural-error detection", loc="left", fontweight="bold", pad=4)


def _plot_mcs_panel(
    ax: plt.Axes,
    blind_rows: list[dict[str, Any]],
    repair_rows: list[dict[str, Any]],
) -> None:
    comparable = [
        row for row in blind_rows
        if _to_float(row.get("ifs")) is not None and _to_float(row.get("mcs")) is not None
    ]
    category_order = [
        "No MCS mismatch",
        "Kernel coefficient",
        "BC coefficient",
        "Material parameter",
        "Constitutive model",
    ]
    colors = {
        "No MCS mismatch": "#BDBDBD",
        "Kernel coefficient": "#4E88C7",
        "BC coefficient": "#D39C3F",
        "Material parameter": "#8E6BBE",
        "Constitutive model": "#CC79A7",
    }
    label_map = {
        "No MCS mismatch": "No mismatch",
        "Kernel coefficient": "Kernel coeff.",
        "BC coefficient": "BC coeff.",
        "Material parameter": "Material param.",
        "Constitutive model": "Constitutive",
    }

    ax.axvspan(0.9, 1.01, ymin=0.0, ymax=0.5, color="#EEE7F6", alpha=0.62)
    ax.axvline(0.9, color="0.35", lw=0.75, ls="--")
    ax.axhline(0.5, color="0.35", lw=0.75, ls="--")
    for category in category_order:
        points = [row for row in comparable if primary_category(row) == category]
        if not points:
            continue
        ax.scatter(
            [float(row["ifs"]) for row in points],
            [float(row["mcs"]) for row in points],
            s=13,
            alpha=0.72,
            color=colors[category],
            edgecolors="none",
            label=label_map[category],
        )

    repaired_ids = {row["id"] for row in repair_rows}
    repaired_points = [row for row in comparable if row["id"] in repaired_ids]
    ax.scatter(
        [float(row["ifs"]) for row in repaired_points],
        [float(row["mcs"]) for row in repaired_points],
        s=31,
        facecolors="none",
        edgecolors="black",
        linewidths=0.75,
        label="Repair subset",
    )

    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("IFS")
    ax.set_ylabel("MCS")
    ax.set_title("(b) MCS coefficient blind-spot repair", loc="left", fontweight="bold", pad=4)
    ax.legend(
        loc="upper left",
        ncol=2,
        frameon=True,
        framealpha=0.92,
        borderpad=0.25,
        handletextpad=0.25,
        columnspacing=0.55,
    )


def combined_main_figure(
    blind_rows: list[dict[str, Any]],
    repair_rows: list[dict[str, Any]],
    output_dir: Path,
    l2_data_path: Path = L2_DATA,
) -> None:
    """Generate the main-paper two-panel IFS/MCS validation figure."""
    l2_rows = json.loads(l2_data_path.read_text())
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
    _plot_ifs_l2_panel(axes[0], l2_rows)
    _plot_mcs_panel(axes[1], blind_rows, repair_rows)
    for ax in axes:
        ax.grid(alpha=0.22)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        ax.tick_params(width=0.9)

    fig.tight_layout(w_pad=1.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = output_dir / "ifs_mcs_validation_compact.pdf"
    out_png = output_dir / "ifs_mcs_validation_compact.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=450)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blindspot-csv",
        type=Path,
        default=ROOT / "results" / "derived" / "mcs_blindspots_gpt54_ds_v4.csv",
    )
    parser.add_argument(
        "--repair-summary",
        type=Path,
        default=ROOT / "results" / "derived" / "mcs_repair_smoke_summary_v3.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=PAPER_IMGS)
    parser.add_argument("--l2-data", type=Path, default=L2_DATA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blind_rows = load_blindspot_rows(args.blindspot_csv)
    repair_rows = load_repair_rows(args.repair_summary)
    compact_figure(blind_rows, repair_rows, args.output_dir)
    combined_main_figure(blind_rows, repair_rows, args.output_dir, args.l2_data)


if __name__ == "__main__":
    main()
