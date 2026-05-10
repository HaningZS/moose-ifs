#!/usr/bin/env python3
"""Small DeepSeek-driven repair smoke for MCS coefficient blind spots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from codmos.multiagent.pde.ifs_engine import (  # noqa: E402
    compute_material_consistency,
    extract_coefficient_contract,
)
from experiments.llm import extract_code  # noqa: E402
from scripts.analyze_mcs_blindspots import build_rows, load_records  # noqa: E402
from scripts.run_moosebench import evaluate_code, make_llm  # noqa: E402
from scripts.validate_moosebench import load_gt  # noqa: E402

SYSTEM_PROMPT = """You repair MOOSE input files by fixing coefficient and material-fidelity facts.

Return ONLY a complete MOOSE HIT input file. Do not include prose or markdown.
Preserve the candidate's mesh, variables, kernels, BC topology, ICs, executioner,
and outputs unless a listed coefficient/material fact requires a local edit.
Do not rename variables unless required by the current code.
"""


def _mcs_score(mcs: Any) -> float | None:
    return mcs.score if mcs.total_properties > 0 else None


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in rows}


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    cases: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    by_id = _index_rows(rows)
    if cases:
        return [by_id[case_id] for case_id in cases if case_id in by_id]

    ranked = sorted(
        [row for row in rows if row.get("mcs") is not None],
        key=lambda row: (
            not bool(row.get("high_ifs_low_mcs")),
            -(float(row["ifs"]) if row.get("ifs") is not None else 0.0),
            float(row["mcs"]),
        ),
    )
    return ranked[:limit]


def _repair_prompt(
    *,
    case_id: str,
    source_contract: list[dict[str, Any]],
    mismatches: list[dict[str, Any]],
    current_code: str,
    previous_note: str = "",
) -> str:
    return f"""Case: {case_id}

Reference coefficient/material contract:
{json.dumps(source_contract, ensure_ascii=False, indent=2, sort_keys=True)}

Current mismatches to fix:
{json.dumps(mismatches, ensure_ascii=False, indent=2, sort_keys=True)}

Repair hints:
- For bc:*:t_infinity and bc:*:heat_transfer_coefficient, edit or add the corresponding ConvectiveHeatFluxBC parameter on the listed boundary.
- For kernel:* facts, set the effective material-backed coefficient value. Prefer adding/updating [Materials] with MOOSE-standard property names over changing PDE topology.
- For material_model:elasticity_tensor=isotropic, use ComputeIsotropicElasticityTensor.
- For material_model:strain_model=small_strain, use ComputeSmallStrain.
- For material_model:strain_model=plane_small_strain, use ComputePlaneSmallStrain.
- For material_model:stress_model=linear_elastic, use ComputeLinearElasticStress.
- For material_model:stress_model=multiple_inelastic, use ComputeMultipleInelasticStress with the needed inelastic model name.
- For material_model:inelastic_model=isotropic_plasticity, use IsotropicPlasticityStressUpdate.
- For material:* facts, set the exact numeric value from the reference contract.
{previous_note}

Candidate MOOSE input to repair:
{current_code}
"""


def _evaluate_attempt(
    *,
    code: str,
    source_code: str,
    gt_path: Path,
) -> dict[str, Any]:
    gt = load_gt(gt_path)
    ifs_result = evaluate_code(code, gt)
    try:
        mcs = compute_material_consistency(source_code, code)
    except Exception as exc:
        return {
            "parse": False,
            "ifs": ifs_result.get("ifs"),
            "mcs": None,
            "mcs_total": None,
            "mcs_matched": None,
            "mcs_mismatched": None,
            "mismatches": [],
            "error": f"{ifs_result.get('error') or ''}; mcs_error={exc}",
        }
    return {
        "parse": ifs_result.get("parse"),
        "ifs": ifs_result.get("ifs"),
        "mcs": _mcs_score(mcs),
        "mcs_total": mcs.total_properties,
        "mcs_matched": mcs.matched_properties,
        "mcs_mismatched": len(mcs.mismatched),
        "mismatches": mcs.mismatched,
        "error": ifs_result.get("error"),
    }


def _is_better(
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    min_ifs_drop: float,
) -> bool:
    if not after.get("parse"):
        return False

    before_mcs = before.get("mcs")
    after_mcs = after.get("mcs")
    if before_mcs is None or after_mcs is None or float(after_mcs) <= float(before_mcs):
        return False

    before_ifs = before.get("ifs")
    after_ifs = after.get("ifs")
    if before_ifs is None or after_ifs is None:
        return True
    return float(after_ifs) >= float(before_ifs) - min_ifs_drop


def run_case(
    *,
    row: dict[str, Any],
    llm: Any,
    source_dir: Path,
    gt_dir: Path,
    output_dir: Path,
    attempts: int,
    min_ifs_drop: float,
) -> dict[str, Any]:
    case_id = str(row["id"])
    source_path = source_dir / f"{case_id}.i"
    gt_path = gt_dir / f"{case_id}.json"
    candidate_path = Path(str(row["code_path"]))
    source_code = _read(source_path)
    current_code = _read(candidate_path)

    before = _evaluate_attempt(code=current_code, source_code=source_code, gt_path=gt_path)
    source_contract = extract_coefficient_contract(source_code)

    best_code = current_code
    best_metrics = before
    attempt_rows: list[dict[str, Any]] = []
    previous_note = ""

    for attempt_idx in range(1, attempts + 1):
        if best_metrics["mcs_mismatched"] == 0:
            break
        prompt = _repair_prompt(
            case_id=case_id,
            source_contract=source_contract,
            mismatches=best_metrics["mismatches"],
            current_code=best_code,
            previous_note=previous_note,
        )
        raw_response = llm.generate(SYSTEM_PROMPT, prompt, temperature=0)
        repaired_code = extract_code(raw_response)
        metrics = _evaluate_attempt(
            code=repaired_code,
            source_code=source_code,
            gt_path=gt_path,
        )
        attempt_path = output_dir / case_id / f"attempt_{attempt_idx}.i"
        _write(attempt_path, repaired_code)
        attempt_rows.append({
            "attempt": attempt_idx,
            "path": str(attempt_path),
            **{key: value for key, value in metrics.items() if key != "mismatches"},
        })

        if _is_better(before=best_metrics, after=metrics, min_ifs_drop=min_ifs_drop):
            best_code = repaired_code
            best_metrics = metrics
            previous_note = (
                "\nPrevious repair improved MCS. Continue only if remaining mismatches "
                "can be fixed with local coefficient/material edits."
            )
        else:
            previous_note = (
                "\nPrevious repair was not acceptable. Keep the original structure and "
                "make smaller edits that only address the listed mismatches."
            )

    accepted = best_code != current_code
    if accepted:
        _write(output_dir / case_id / "accepted.i", best_code)

    return {
        "id": case_id,
        "family": row.get("family"),
        "complexity": row.get("complexity"),
        "accepted": accepted,
        "before": {key: value for key, value in before.items() if key != "mismatches"},
        "best": {key: value for key, value in best_metrics.items() if key != "mismatches"},
        "attempts": attempt_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="MooseBench JSONL with code_path fields")
    parser.add_argument("--llm", default="deepseek-flash")
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--min-ifs-drop", type=float, default=0.05)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("experiments/moosebench_clean/source_files"),
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("experiments/moosebench_clean/ground_truth"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/mcs_repair_smoke"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(
        load_records(args.jsonl),
        source_dir=args.source_dir,
        repo_root=Path.cwd(),
        high_ifs=0.9,
        low_mcs=0.5,
    )
    selected = _select_rows(rows, cases=args.cases, limit=args.limit)
    if not selected:
        raise SystemExit("No candidate rows selected")

    llm = make_llm(args.llm)
    summaries = [
        run_case(
            row=row,
            llm=llm,
            source_dir=args.source_dir,
            gt_dir=args.gt_dir,
            output_dir=args.output_dir,
            attempts=args.attempts,
            min_ifs_drop=args.min_ifs_drop,
        )
        for row in selected
    ]

    summary_path = args.output_dir / "summary.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        for summary in summaries:
            fh.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")

    for summary in summaries:
        before = summary["before"]
        best = summary["best"]
        print(
            f"{summary['id']}: accepted={summary['accepted']} "
            f"IFS {before['ifs']} -> {best['ifs']} "
            f"MCS {before['mcs']} -> {best['mcs']} "
            f"mismatches {before['mcs_mismatched']} -> {best['mcs_mismatched']}"
        )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
