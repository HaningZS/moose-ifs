#!/usr/bin/env python3
"""Conservatively refresh prompts for expert-modified MooseBench source files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "experiments" / "moosebench" / "prompts"
SOURCES = ROOT / "experiments" / "moosebench" / "source_files"
CLEAN = ROOT / "experiments" / "moosebench_clean"


SYSTEM = """You update natural-language benchmark prompts for MOOSE simulations.

Rules:
- Preserve the same benchmark case ID, physics_family, complexity, and JSON shape.
- Update nl_description only when the new source file changes the physical problem.
- Do not mention MOOSE object names, block names, kernels, input syntax, files, meshes, or solver settings.
- Describe the intended physics at a domain level: variables, domain, governing effects, coefficients if explicit, boundary/initial conditions, and steady/transient choice.
- Keep the prompt concise enough for a benchmark item.
- Return ONLY JSON with keys: nl_description, expected_operators, notes.
"""


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("LLM response did not contain a JSON object")
    return json.loads(match.group(0))


def _make_llm(model: str):
    if model == "claude":
        from experiments.llm import AnthropicLLM

        return AnthropicLLM(model="claude-sonnet-4-6")
    if model == "claude-haiku":
        from experiments.llm import AnthropicLLM

        return AnthropicLLM(model="claude-haiku-4-5-20251001")
    if model == "deepseek-flash":
        from scripts.run_moosebench import make_llm

        return make_llm("deepseek-flash")
    raise ValueError("model must be one of: claude, claude-haiku, deepseek-flash")


def _prompt_for_case(case_id: str, prompt: dict, source: str) -> str:
    return f"""Case ID: {case_id}

Current prompt JSON:
{json.dumps(prompt, indent=2, ensure_ascii=False)}

Updated expert-runnable source file:
```text
{source}
```

Return the updated JSON object now."""


def update_case(case_id: str, *, llm, dry_run: bool) -> dict:
    prompt_path = PROMPTS / f"{case_id}.json"
    source_path = SOURCES / f"{case_id}.i"
    prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
    source = source_path.read_text(encoding="utf-8", errors="replace")
    response = llm.generate(SYSTEM, _prompt_for_case(case_id, prompt, source), temperature=0)
    patch = _extract_json(response)

    updated = dict(prompt)
    updated["nl_description"] = str(patch["nl_description"]).strip()
    if isinstance(patch.get("expected_operators"), list):
        updated["expected_operators"] = patch["expected_operators"]
    if patch.get("notes"):
        updated["notes"] = str(patch["notes"]).strip()
    updated["source_revision"] = "source_files_final"

    if not dry_run:
        prompt_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "id": case_id,
        "changed_nl": updated.get("nl_description") != prompt.get("nl_description"),
        "old_nl": prompt.get("nl_description"),
        "new_nl": updated.get("nl_description"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-ids", type=Path, default=CLEAN / "modified_source_ids.json")
    parser.add_argument("--model", choices=["claude", "claude-haiku", "deepseek-flash"], default="claude")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=CLEAN / "prompt_update_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(ROOT))
    case_ids = json.loads(args.case_ids.read_text(encoding="utf-8"))
    if args.limit > 0:
        case_ids = case_ids[: args.limit]
    llm = _make_llm(args.model)
    report = []
    for index, case_id in enumerate(case_ids, start=1):
        print(f"[{index}/{len(case_ids)}] updating {case_id}", flush=True)
        report.append(update_case(case_id, llm=llm, dry_run=args.dry_run))
    if not args.dry_run:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"updated": len(report), "changed_nl": sum(r["changed_nl"] for r in report)}, indent=2))


if __name__ == "__main__":
    main()
