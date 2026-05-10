#!/usr/bin/env python3
"""Validate the released clean MooseBench artifact.

The release benchmark lives in ``experiments/moosebench_clean`` and
contains the active 220-case index, ground-truth physics contracts, and source
MOOSE input files. Older prompt/ground-truth benchmark snapshots are not part of
the artifact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codmos.multiagent.pde.ifs_engine import compute_ifs
from codmos.multiagent.pde.representation import (
    BoundaryCondition,
    InitialCondition,
    PDERepresentation,
    PDETerm,
)

ROOT = Path(__file__).resolve().parents[1]
MOOSEBENCH_CLEAN = ROOT / "experiments" / "moosebench_clean"
INDEX_PATH = MOOSEBENCH_CLEAN / "index.json"
GT_DIR = MOOSEBENCH_CLEAN / "ground_truth"
SOURCE_DIR = MOOSEBENCH_CLEAN / "source_files"


def load_gt(gt_path: Path) -> PDERepresentation:
    """Build a PDERepresentation from a ground-truth JSON file."""
    data = json.loads(gt_path.read_text(encoding="utf-8"))

    terms = [
        PDETerm(
            variable=t["variable"],
            operator=t["operator"],
            coefficient=t.get("coefficient"),
            coupled_variable=t.get("coupled_variable"),
            kernel_type=t.get("kernel_type"),
            severity=t.get("severity", "medium"),
        )
        for t in data.get("terms", [])
    ]

    boundary_conditions = [
        BoundaryCondition(
            variable=bc["variable"],
            boundary=bc["boundary"],
            bc_type=bc["bc_type"],
            value=bc.get("value"),
            moose_bc_class=bc.get("moose_type"),
            severity=bc.get("severity", "medium"),
        )
        for bc in data.get("boundary_conditions", [])
    ]

    initial_conditions = [
        InitialCondition(
            variable=ic["variable"],
            ic_type=ic.get("ic_type", "constant"),
            value=ic.get("value"),
            severity=ic.get("severity", "medium"),
        )
        for ic in data.get("initial_conditions", [])
    ]

    return PDERepresentation(
        terms=terms,
        boundary_conditions=boundary_conditions,
        initial_conditions=initial_conditions,
        time_scheme=data.get("time_scheme", "steady"),
        variables=data.get("variables", []),
        dimensions=data.get("domain_dim", 2),
    )


def _active_ids() -> list[str]:
    data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    ids = data.get("ids")
    if not isinstance(ids, list):
        raise ValueError(f"{INDEX_PATH} must contain an 'ids' list")
    expected_total = data.get("total")
    if expected_total != len(ids):
        raise ValueError(f"{INDEX_PATH} total={expected_total!r} but ids has {len(ids)} entries")
    return [str(case_id) for case_id in ids]


def validate_case(case_id: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    gt_path = GT_DIR / f"{case_id}.json"
    source_path = SOURCE_DIR / f"{case_id}.i"

    if not gt_path.exists():
        errors.append(f"missing ground-truth contract: {gt_path}")
    if not source_path.exists():
        errors.append(f"missing source input: {source_path}")
    if errors:
        return False, errors

    try:
        gt = load_gt(gt_path)
    except Exception as exc:
        return False, [f"cannot load GT: {exc}"]

    if not gt.terms:
        errors.append("GT has no PDE terms")
    if not gt.boundary_conditions:
        errors.append("GT has no boundary conditions")

    try:
        result = compute_ifs(gt, gt)
    except Exception as exc:
        errors.append(f"compute_ifs(gt, gt) raised: {exc}")
        return False, errors

    if abs(result.ifs_score - 1.0) > 1e-9:
        errors.append(f"self-IFS != 1.0 (got {result.ifs_score:.6f})")

    return len(errors) == 0, errors


def main() -> None:
    ids = _active_ids()
    n_ok = 0
    n_fail = 0

    print(f"Validating {len(ids)} clean MooseBench case(s) in {MOOSEBENCH_CLEAN}\n")
    print(f"{'Case':<20}  {'Status':<6}  Notes")
    print("-" * 72)

    for case_id in ids:
        passed, errors = validate_case(case_id)
        if passed:
            n_ok += 1
            print(f"{case_id:<20}  {'OK':<6}")
        else:
            n_fail += 1
            print(f"{case_id:<20}  {'FAIL':<6}  {'; '.join(errors)}")

    print("-" * 72)
    print(f"\nSummary: {n_ok} OK, {n_fail} FAIL")
    if n_fail:
        sys.exit(1)
    print("All clean MooseBench cases passed.")


if __name__ == "__main__":
    main()
