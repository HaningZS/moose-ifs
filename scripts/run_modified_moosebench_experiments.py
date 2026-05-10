#!/usr/bin/env python3
"""Run the current MooseBench update slice across model/method configurations.

The case slice defaults to moosebench_clean/modified_source_ids.json, produced
by scripts/sync_moosebench_final_sources.py. Each run excludes cases outside
that slice so the generated JSONL files contain only the selected active subset.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "experiments" / "moosebench"
CLEAN = ROOT / "experiments" / "moosebench_clean"
RESULTS = ROOT / "experiments" / "results"


@dataclass(frozen=True)
class RunConfig:
    name: str
    llm: str
    methods: tuple[str, ...]
    workers: int
    registry: bool = False


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _prompt_ids() -> list[str]:
    return sorted(path.stem for path in (BENCH / "prompts").glob("*.json"))


def _exclude_file(case_ids: set[str], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded = [case_id for case_id in _prompt_ids() if case_id not in case_ids]
    path = output_dir / "exclude_outside_current_slice.json"
    path.write_text(json.dumps(excluded, indent=2) + "\n", encoding="utf-8")
    return path


def _configs(args: argparse.Namespace) -> list[RunConfig]:
    ordinary_models = [m.strip() for m in args.ordinary_models.split(",") if m.strip()]
    registry_models = [m.strip() for m in args.registry_models.split(",") if m.strip()]
    configs: list[RunConfig] = []
    for model in ordinary_models:
        workers = args.rate_limited_workers if model.startswith("claude") else args.workers
        configs.append(
            RunConfig(
                name=f"{model}_abd",
                llm=model,
                methods=("A", "B", "D"),
                workers=workers,
            )
        )
    registry_methods = ("AReg", "ExecRepairReg", "DReg") if args.include_areg else ("ExecRepairReg", "DReg")
    for model in registry_models:
        workers = args.rate_limited_workers if model.startswith("claude") else args.workers
        configs.append(
            RunConfig(
                name=f"{model}_regexec",
                llm=model,
                methods=registry_methods,
                workers=workers,
                registry=True,
            )
        )
    return configs


def _run_command(cmd: list[str], *, env: dict[str, str]) -> tuple[int, str]:
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.returncode, completed.stdout


def _run_generation(config: RunConfig, args: argparse.Namespace, exclude: Path, env: dict[str, str]) -> Path:
    output = args.output_dir / f"moosebench_v4_current_{config.name}.jsonl"
    cmd = [
        sys.executable,
        "scripts/run_moosebench.py",
        "--llm",
        config.llm,
        "--methods",
        *config.methods,
        "--exclude",
        str(exclude),
        "--fallback-threshold",
        str(args.fallback_threshold),
        "--workers",
        str(config.workers),
        "--smoke-timeout",
        str(args.smoke_timeout),
        "--output",
        str(output),
    ]
    if config.registry:
        cmd.extend(["--registry-json", str(args.registry_json), "--moose-app", str(args.moose_app)])
    code, log = _run_command(cmd, env=env)
    log_path = output.with_suffix(output.suffix + ".log")
    log_path.write_text(log, encoding="utf-8", errors="replace")
    if code != 0:
        raise RuntimeError(f"{config.name} failed with exit code {code}; see {log_path}")
    return output


def _run_audit(result_path: Path, args: argparse.Namespace, env: dict[str, str]) -> Path:
    output = result_path.with_name(result_path.stem + f"_exec{args.smoke_timeout}.json")
    cmd = [
        sys.executable,
        "scripts/audit_moose_execution_rate.py",
        "--results",
        str(result_path),
        "--moose-app",
        str(args.moose_app),
        "--output",
        str(output),
        "--timeout",
        str(args.smoke_timeout),
        "--workers",
        str(args.audit_workers),
        "--runtime-input-dir",
        str(args.runtime_input_dir),
    ]
    code, log = _run_command(cmd, env=env)
    log_path = output.with_suffix(output.suffix + ".log")
    log_path.write_text(log, encoding="utf-8", errors="replace")
    if code != 0:
        raise RuntimeError(f"audit failed for {result_path.name}; see {log_path}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-ids", type=Path, default=CLEAN / "modified_source_ids.json")
    parser.add_argument(
        "--limit-cases",
        type=int,
        default=0,
        help="Use only the first N modified case IDs for smoke testing. Default: all.",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "current")
    parser.add_argument(
        "--ordinary-models",
        default="claude,gpt-5.4,deepseek-flash,gpt,claude-haiku",
        help="Comma-separated LLM names for A/B/D reruns.",
    )
    parser.add_argument(
        "--registry-models",
        default="claude,gpt-5.4,deepseek-flash,gpt",
        help="Comma-separated LLM names for registry execution reruns.",
    )
    parser.add_argument("--include-areg", action="store_true", help="Also run AReg in registry reruns.")
    parser.add_argument("--workers", type=int, default=3, help="Default per-run workers.")
    parser.add_argument(
        "--rate-limited-workers",
        type=int,
        default=1,
        help="Per-run workers for rate-limited model providers.",
    )
    parser.add_argument("--max-jobs", type=int, default=3, help="Parallel model/run subprocesses.")
    parser.add_argument("--audit-workers", type=int, default=4)
    parser.add_argument("--fallback-threshold", type=float, default=0.0)
    parser.add_argument("--smoke-timeout", type=int, default=1)
    parser.add_argument("--registry-json", type=Path, default=Path(os.environ.get("MOOSE_REGISTRY_JSON", "combined_syntax_full.txt")))
    parser.add_argument(
        "--moose-app",
        type=Path,
        default=Path(os.environ.get("MOOSE_APP", "combined-opt")),
    )
    parser.add_argument("--runtime-input-dir", type=Path, default=BENCH / "source_files")
    parser.add_argument("--skip-audit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_ids = set(_load_json(args.case_ids))
    if not case_ids:
        raise ValueError(f"No case IDs found in {args.case_ids}")
    if args.limit_cases > 0:
        case_ids = set(sorted(case_ids)[: args.limit_cases])
    exclude = _exclude_file(case_ids, args.output_dir)
    configs = _configs(args)
    env = os.environ.copy()

    print(f"Modified cases: {len(case_ids)}")
    print(f"Exclude file: {exclude}")
    print(f"Run configs: {[config.name for config in configs]}")

    generated: list[Path] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.max_jobs) as pool:
        futures = {
            pool.submit(_run_generation, config, args, exclude, env): config
            for config in configs
        }
        for future in as_completed(futures):
            config = futures[future]
            try:
                path = future.result()
            except Exception as exc:  # noqa: BLE001 - keep independent model runs alive.
                failures.append(f"{config.name}: {exc}")
                print(f"[failed] {config.name}: {exc}")
                continue
            generated.append(path)
            print(f"[done] {config.name}: {path}")

    if args.skip_audit or not generated:
        if failures:
            print("Failures:")
            for failure in failures:
                print(f"  - {failure}")
        return

    with ThreadPoolExecutor(max_workers=min(args.max_jobs, len(generated))) as pool:
        futures = {pool.submit(_run_audit, path, args, env): path for path in generated}
        for future in as_completed(futures):
            source = futures[future]
            try:
                audit = future.result()
            except Exception as exc:  # noqa: BLE001 - report all audit failures.
                failures.append(f"audit {source.name}: {exc}")
                print(f"[failed] audit {source.name}: {exc}")
                continue
            print(f"[audit] {source.name}: {audit}")

    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")


if __name__ == "__main__":
    main()
