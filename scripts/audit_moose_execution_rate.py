#!/usr/bin/env python3
"""Audit MooseBench execution rate with a short no-error window."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ERROR_TOKENS = (
    "error",
    "not a registered object",
    "invalid parameter",
    "is not a valid parameter",
    "required parameter",
    "missing required",
    "unable to find",
    "unknown variable",
    "unknown boundary",
    "nonexistent boundary",
    "material property",
    "mpi_abort",
    "abort(",
    "exception",
    "segmentation fault",
    "solve failed",
)

BENIGN_ERROR_TOKENS = (
    "0 errors",
    "no errors",
    "error estimates",
    "error norms",
)

SOLVE_START_TOKENS = (
    "nonlinear |r|",
    "linear |r|",
    "time step",
    "time step:",
    "solve::",
    "solve started",
)


def _first_error_match(text: str) -> tuple[int, str] | None:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        lowered = line.lower()
        if any(token in lowered for token in BENIGN_ERROR_TOKENS):
            continue
        if any(token in lowered for token in ERROR_TOKENS):
            start = max(0, index - 2)
            end = min(len(lines), index + 5)
            return index, "\n".join(lines[start:end])[:1200]
    return None


def _first_error_excerpt(text: str) -> str | None:
    match = _first_error_match(text)
    return match[1] if match is not None else None


def _first_solve_start_match(text: str) -> tuple[int, str] | None:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        lowered = line.lower()
        if any(token in lowered for token in SOLVE_START_TOKENS):
            start = max(0, index - 2)
            end = min(len(lines), index + 5)
            return index, "\n".join(lines[start:end])[:1200]
    return None


def _decode_timeout_stream(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return value.decode(errors="ignore")


def _copy_runtime_inputs(runtime_input_dir: Path, work_dir: Path) -> None:
    """Copy mesh/table sidecars that generated inputs may reference by basename."""
    if not runtime_input_dir.exists():
        return
    for path in runtime_input_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".e", ".csv"}:
            shutil.copy2(path, work_dir / path.name)


def audit_record(
    row: dict,
    *,
    moose_app: Path,
    timeout: int,
    runtime_input_dir: Path,
    pass_on_solve_start: bool,
) -> dict:
    code_path = row.get("code_path")
    record = {
        "id": row["id"],
        "method": row["method"],
        "ifs": row.get("ifs"),
        "parse": row.get("parse"),
        "registry_l2_pass_after": row.get("registry_l2_pass_after"),
        "code_path": code_path,
        "metric": "init_exec" if pass_on_solve_start else "first_window_no_error",
        "timeout_s": timeout,
        "first_window_pass": False,
        "status": None,
        "returncode": None,
        "error_excerpt": None,
        "solve_start_excerpt": None,
    }
    if not code_path or not Path(code_path).exists():
        record["status"] = "missing_code_path"
        record["error_excerpt"] = "missing code_path"
        return record

    with tempfile.TemporaryDirectory(prefix=f"cm_fw_{row['id']}_{row['method']}_", dir="/tmp") as tmp:
        _copy_runtime_inputs(runtime_input_dir, Path(tmp))
        try:
            completed = subprocess.run(
                [str(moose_app), "-i", str(Path(code_path).resolve()), "--allow-unused"],
                cwd=tmp,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            text = _decode_timeout_stream(exc.stdout) + "\n" + _decode_timeout_stream(exc.stderr)
            error_match = _first_error_match(text)
            solve_match = _first_solve_start_match(text)
            record["error_excerpt"] = error_match[1] if error_match is not None else None
            record["solve_start_excerpt"] = solve_match[1] if solve_match is not None else None
            solve_before_error = (
                pass_on_solve_start
                and solve_match is not None
                and (error_match is None or solve_match[0] < error_match[0])
            )
            if solve_before_error:
                record["status"] = "timeout_solve_started_before_error"
                record["first_window_pass"] = True
            else:
                record["status"] = (
                    "timeout_no_error" if record["error_excerpt"] is None else "timeout_with_error_text"
                )
                record["first_window_pass"] = record["error_excerpt"] is None
            record["returncode"] = "timeout"
            return record

    text = (completed.stdout or "") + "\n" + (completed.stderr or "")
    record["returncode"] = completed.returncode
    error_match = _first_error_match(text)
    solve_match = _first_solve_start_match(text)
    record["error_excerpt"] = error_match[1] if error_match is not None else None
    record["solve_start_excerpt"] = solve_match[1] if solve_match is not None else None
    solve_before_error = (
        pass_on_solve_start
        and solve_match is not None
        and (error_match is None or solve_match[0] < error_match[0])
    )
    if pass_on_solve_start and completed.returncode == 0:
        record["status"] = (
            "completed_exit0"
            if record["error_excerpt"] is None
            else "completed_exit0_with_error_text"
        )
        record["first_window_pass"] = True
    elif completed.returncode == 0 and record["error_excerpt"] is None:
        record["status"] = "completed_no_error"
        record["first_window_pass"] = True
    elif completed.returncode == 0:
        record["status"] = "completed_with_error_text"
    elif solve_before_error:
        record["status"] = "failed_after_solve_start"
        record["first_window_pass"] = True
    else:
        record["status"] = "failed_fast"
    return record


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def print_summary(records: list[dict]) -> None:
    by_method: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_method[record["method"]].append(record)

    for method in sorted(by_method):
        rows = by_method[method]
        pass_rate = sum(row["first_window_pass"] for row in rows) / len(rows)
        mean_ifs = sum((row.get("ifs") or 0.0) for row in rows) / len(rows)
        l2_rate = sum(row.get("registry_l2_pass_after") is True for row in rows) / len(rows)
        statuses: dict[str, int] = defaultdict(int)
        for row in rows:
            statuses[row["status"]] += 1
        print(
            f"{method:<6} n={len(rows):>4} "
            f"exec={pass_rate:>6.1%} mean_ifs={mean_ifs:.4f} "
            f"l2={l2_rate:>6.1%} statuses={dict(statuses)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="MooseBench JSONL with code_path fields.")
    parser.add_argument("--moose-app", type=Path, required=True, help="Path to app opt binary.")
    parser.add_argument("--output", type=Path, required=True, help="Audit JSON output path.")
    parser.add_argument("--timeout", type=int, default=2, help="No-error window in seconds. Default: 2.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel MOOSE processes. Default: 4.")
    parser.add_argument(
        "--pass-on-solve-start",
        action="store_true",
        default=True,
        help=(
            "Use InitExec semantics: pass on exit_code=0, or if nonlinear/linear "
            "solve starts before the first error, or if the timeout window emits no error. "
            "This is the default."
        ),
    )
    parser.add_argument(
        "--first-window-only",
        dest="pass_on_solve_start",
        action="store_false",
        help=(
            "Disable solve-start acceptance and use the stricter first-window no-error "
            "metric."
        ),
    )
    parser.add_argument(
        "--runtime-input-dir",
        type=Path,
        default=Path("experiments/moosebench_clean/source_files"),
        help="Directory containing mesh/table sidecars (.e, .csv) to copy beside each run.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Optional method filter. Records from other methods are ignored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_jsonl(args.results)
    if args.methods:
        methods = set(args.methods)
        rows = [row for row in rows if row.get("method") in methods]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                audit_record,
                row,
                moose_app=args.moose_app,
                timeout=args.timeout,
                runtime_input_dir=args.runtime_input_dir,
                pass_on_solve_start=args.pass_on_solve_start,
            )
            for row in rows
        ]
        audits = [future.result() for future in as_completed(futures)]

    method_order = {"A": 0, "D": 1, "AReg": 2, "ExecRepairReg": 3, "DReg": 4}
    audits.sort(key=lambda row: (row["id"], method_order.get(row["method"], 99)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audits, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print_summary(audits)


if __name__ == "__main__":
    main()
