#!/usr/bin/env python3
"""MooseBench evaluation runner.

Runs MooseBench evaluation: methods A/B/D x LLMs -> JSONL results.

Usage:
    uv run python scripts/run_moosebench.py --llm claude --methods A B D
    uv run python scripts/run_moosebench.py --llm gpt --methods A --limit 5
    uv run python scripts/run_moosebench.py --llm deepseek-flash --methods A AE B C D --workers 4
    uv run python scripts/run_moosebench.py --summary experiments/results/moosebench_results.jsonl
"""

from __future__ import annotations

import argparse
import ast
import copy
import functools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

# Make project and src packages importable when running directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codmos.multiagent.moose_registry import MooseRegistry
from codmos.multiagent.object_realization import (
    ObjectRealizationPlan,
    build_object_plan,
    format_registry_issues,
    validate_and_repair_code,
)
from codmos.multiagent.pde.ifs_engine import (
    compute_ifs,
    compute_material_consistency,
    format_violations_for_code,
)
from codmos.multiagent.pde.reconstruction import reconstruct_pde
from codmos.multiagent.pde.representation import (
    BoundaryCondition,
    InitialCondition,
    PDERepresentation,
    PDETerm,
)
from codmos.multiagent.validators.hit_parser import HITParseError
from codmos.multiagent.validators.hit_syntax import HITSyntaxValidator
from codmos.multiagent.validators.moose_type import MOOSETypeValidator
from experiments.llm import extract_code
from scripts.audit_moose_execution_rate import _decode_timeout_stream, _first_error_excerpt
from scripts.validate_moosebench import load_gt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[1]
_MOOSEBENCH_DIR = _ROOT / "experiments" / "moosebench"
_PROMPTS_DIR = _MOOSEBENCH_DIR / "prompts"
_GT_DIR = _MOOSEBENCH_DIR / "ground_truth"
_SOURCE_FILES_DIR = _MOOSEBENCH_DIR / "source_files"
_RESULTS_DIR = _ROOT / "experiments" / "results"
_METHOD_NAMES = ("A", "AE", "B", "C", "D", "AReg", "ExecRepairReg", "DReg")
_REGISTRY_METHODS = {"AReg", "ExecRepairReg", "DReg"}
_SMOKE_EXEC_METHODS = {"ExecRepairReg"}
_PROMPT_VERSION = "moosebench-2026-05-02-registry-object-plan-v3-exec-repair"

# ---------------------------------------------------------------------------
# Prompts (copied from experiments/smoke_test_methods_v2.py)
# ---------------------------------------------------------------------------

EXTRACT_SPEC_SYSTEM = """You are a MOOSE simulation physics expert. Given a natural language description of a simulation problem, extract the PDE specification as a JSON object.

Return ONLY valid JSON with this structure:
{
  "variables": ["var1", "var2"],
  "terms": [
    {"variable": "var1", "operator": "diffusion", "coefficient": 45.0},
    {"variable": "var1", "operator": "time_derivative", "coefficient": null}
  ],
  "boundary_conditions": [
    {"variable": "var1", "boundary": "left", "bc_type": "Dirichlet", "value": 300.0}
  ],
  "initial_conditions": [
    {"variable": "var1", "value": 300.0, "ic_type": "constant"}
  ],
  "time_scheme": "transient",
  "dimensions": 2
}

Valid operators (choose the most specific one that matches the physics):
  General:     diffusion, time_derivative, source, reaction, advection, stress_divergence
  Coupling:    coupled_force (use when one variable drives a forcing term in another variable's equation)
  Porous flow: pf_darcy_flux, pf_effective_stress
  Phase field: allen_cahn, cahn_hilliard
  Fluid (NS):  navier_stokes_mass, navier_stokes_momentum

Valid bc_type: Dirichlet, Neumann, Robin

Variable naming — preserve MOOSE conventions exactly as described:
  Pore pressure          → pp   (NOT p)
  Displacements          → disp_x, disp_y, disp_z   (NOT u, v, w)
  Phase-field order par. → eta
  Temperature            → T
  Concentration          → c
  Pressure (NS)          → p"""

EXTRACT_SELF_CHECK_SYSTEM = """You are checking a first-pass PDE extraction for a MOOSE simulation benchmark.

Return ONLY valid JSON with this structure:
{
  "confidence": 0.0,
  "needs_retry": true,
  "flags": ["possible missing coupled term"],
  "checklist": ["recheck variables"]
}

Your job is to judge whether the extraction may have omitted explicitly described physics.
Do not repair the extraction.
Do not invent physics that is not in the natural language description.
If uncertain about an omission, set needs_retry=true so extraction can reread the problem.
If the extraction is adequate, set needs_retry=false and confidence close to 1.0."""

SPEC_GUIDED_SYSTEM = """You are a MOOSE simulation expert. Given a structured PDE specification, generate a complete MOOSE input file (.i format) that implements exactly this physics.

The specification tells you:
- What variables to create
- What kernels (PDE terms) to include
- What boundary conditions to apply
- Whether the problem is transient or steady-state

CRITICAL — completeness: implement every variable, PDE term, boundary condition, initial condition, and time scheme listed in the specification. Do NOT omit terms, variables, BCs, ICs, dimensions, or couplings to make the file shorter or easier to run. If a registry object plan is provided, realize every listed PDE item with a valid MOOSE object unless one object explicitly implements the same PDE item.

CRITICAL — coefficients: the coefficient values listed in the specification are EXACT. You MUST use them verbatim in [Materials] (e.g. as 'diffusivity', 'thermal_conductivity', 'youngs_modulus', etc.). Do NOT substitute 1.0 or any default value for a coefficient that is explicitly given.

Generate a complete, syntactically valid MOOSE .i file."""

REGISTRY_MECHANICAL_REPAIR_TEMPLATE = """The following MOOSE input file failed mechanical MOOSE registry validation.

Current code:
```
{code}
```

Mechanical registry violations:
{violations}

Registry-grounded object plan:
{object_plan}

STRICT REPAIR RULES:
1. You may fix only object names, block placement/nesting, required object parameters, variable/boundary references, and non-physics boilerplate.
2. Do NOT add or remove governing PDE terms, BC types, ICs, source terms, coefficient values, material models, or the time scheme unless the registry violation is purely mechanical.
3. Do NOT replace the PDE specification with a different physical model.
4. Preserve every numerical value already present in the code unless the registry violation says the parameter name is mechanical wrong.
5. Do NOT use unregistered object names. For coupled_force, use CoupledForce with its coef parameter rather than inventing CoefCoupledForce.
6. For PETSc options with values, prefer paired petsc_options_iname / petsc_options_value lists; do not put '-option value' pairs into petsc_options.
7. Avoid inline comments after parameter values; return clean HIT input only.
8. Return ONLY the corrected MOOSE input file.
"""

REFINE_TEMPLATE = """The MOOSE input file you generated has specific physics-level errors detected by automated PDE verification.

Current code:
```
{code}
```

Violations detected:
{violations}

CRITICAL INSTRUCTIONS:
1. Make ONLY the minimal changes needed to fix the listed violations above.
2. Do NOT modify, remove, or rewrite any kernel, BC, IC, material, or executioner setting that is NOT mentioned in the violations.
3. Do NOT change variable names, mesh parameters, or numerical values unless a violation specifically requires it.
4. If a violation says "missing term", ADD the missing kernel without removing existing ones.
5. If a violation says "extra term", REMOVE only that specific kernel.
6. Preserve the overall structure of the input file.
7. For coefficient violations: set the EXACT numerical value from the problem description into the corresponding Material parameter — do NOT use 1.0 or a default value.

Return the corrected MOOSE input file."""

EXEC_REFINE_TEMPLATE = """The MOOSE input file you generated may have errors.

Current code:
```
{code}
```

The code may have issues with missing physics terms, incorrect boundary conditions, or wrong parameters. Please review and fix any problems. Return the corrected MOOSE input file."""

EXEC_REPAIR_REG_TEMPLATE = """The following MOOSE input file failed a short MOOSE runtime smoke check.

Current code:
```
{code}
```

MOOSE smoke-check status:
{status}

Runtime error excerpt:
```
{runtime_log}
```

Repair objective:
Make the input file pass the same short MOOSE no-error smoke check.

RULES:
1. Use only the current code and the MOOSE runtime error. You are not given PDE/IFS feedback.
2. Prefer minimal runtime/schema fixes: object names, required parameters, variable/material/boundary references, block structure, and solver/output boilerplate.
3. Preserve the intended problem where possible, but prioritize MOOSE acceptance over completeness if the runtime error cannot be repaired from local context.
4. Do not invent unregistered MOOSE object names.
5. Avoid inline comments after parameter values; return clean HIT input only.
6. Return ONLY the corrected MOOSE input file.
"""

# 方案三: fallback to Method A when extraction IFS is below this threshold
_EXTRACTION_IFS_FALLBACK_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# LLM construction
# ---------------------------------------------------------------------------


def make_llm(llm_name: str):
    """Construct an LLM backend by name."""
    if llm_name == "claude":
        from experiments.llm import AnthropicLLM
        return AnthropicLLM(model="claude-sonnet-4-6")
    elif llm_name == "claude-haiku":
        from experiments.llm import AnthropicLLM
        return AnthropicLLM(model="claude-haiku-4-5-20251001")
    elif llm_name == "gpt":
        from experiments.llm import OpenAICompatibleLLM
        return OpenAICompatibleLLM(model="gpt-4.1-mini")
    elif llm_name == "gpt-5.4":
        from experiments.llm import OpenAICompatibleLLM
        return OpenAICompatibleLLM(model="gpt-5.4", use_completion_tokens=True)
    elif llm_name == "gemini":
        from experiments.llm import GeminiLLM
        return GeminiLLM(model="gemini-2.5-flash")
    elif llm_name == "gemini3":
        from experiments.llm import GeminiLLM
        return GeminiLLM(model="gemini-3-flash-preview")
    elif llm_name in {"gemini-3-pro", "gemini3-pro"}:
        from experiments.llm import GeminiLLM
        return GeminiLLM(model="gemini-3-pro-preview")
    elif llm_name in {"gemini-3.1-pro-preview", "gemini31-pro"}:
        from experiments.llm import GeminiLLM
        return GeminiLLM(model="gemini-3.1-pro-preview")
    elif llm_name in {"gemini-3.1-flash-lite-preview", "gemini31-flash-lite"}:
        from experiments.llm import GeminiLLM
        return GeminiLLM(model="gemini-3.1-flash-lite-preview")
    elif llm_name == "deepseek-pro":
        from experiments.llm import OpenAICompatibleLLM
        return OpenAICompatibleLLM(
            model="deepseek-v4-pro",
            base_url="https://api.deepseek.com",
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
        )
    elif llm_name == "deepseek-flash":
        from experiments.llm import OpenAICompatibleLLM
        return OpenAICompatibleLLM(
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
        )
    else:
        raise ValueError(f"Unknown LLM: {llm_name!r}. Choose claude, claude-haiku, gpt, gemini, gemini3, gemini-3-pro, gemini-3.1-pro-preview, gemini-3.1-flash-lite-preview, deepseek-pro, or deepseek-flash.")


# ---------------------------------------------------------------------------
# PDE extraction helper
# ---------------------------------------------------------------------------


@dataclass
class ExtractionAttempt:
    """Runtime-visible metadata for a PDE extraction attempt."""

    pde: PDERepresentation | None
    raw_json: str
    self_check_confidence: float | None = None
    self_check_flags: list[str] = field(default_factory=list)
    retried: bool = False
    retry_reason: str | None = None
    retry_accepted: bool | None = None


@dataclass
class ExtractionSelfCheck:
    """Parsed extraction self-check response."""

    confidence: float | None
    needs_retry: bool
    flags: list[str] = field(default_factory=list)
    checklist: list[str] = field(default_factory=list)


def parse_pde_llm(json_str: str) -> PDERepresentation | None:
    """Parse LLM-extracted JSON into PDERepresentation."""
    try:
        text = _extract_json_payload(json_str)
        spec = _loads_pde_json(text)

        terms = [
            PDETerm(
                variable=t["variable"],
                operator=t["operator"],
                coefficient=_json_scalar_or_none(t.get("coefficient")),
                coupled_variable=t.get("coupled_variable"),
                kernel_type=None,
                severity="medium",
            )
            for t in spec.get("terms", [])
        ]
        bcs = [
            BoundaryCondition(
                variable=bc["variable"],
                boundary=bc["boundary"],
                bc_type=bc["bc_type"],
                value=_json_scalar_or_none(bc.get("value")),
                moose_bc_class=None,
                severity="high",
            )
            for bc in spec.get("boundary_conditions", [])
        ]
        ics = [
            InitialCondition(
                variable=ic["variable"],
                ic_type=ic.get("ic_type", "constant"),
                value=_json_scalar_or_none(ic.get("value")),
                severity="medium",
            )
            for ic in spec.get("initial_conditions", [])
        ]
        return PDERepresentation(
            terms=terms,
            boundary_conditions=bcs,
            initial_conditions=ics,
            time_scheme=spec.get("time_scheme", "steady"),
            variables=spec.get("variables", []),
            dimensions=spec.get("dimensions", 2),
        )
    except Exception as exc:
        print(f"    [WARN] PDE_llm parse failed: {exc}", flush=True)
        return None


def _loads_pde_json(text: str) -> dict:
    """Load extraction JSON with a narrow repair for numeric expressions."""
    comment_repaired = _replace_json_block_comments_with_null(text)
    candidates = [
        text,
        _repair_json_numeric_expressions(text),
        comment_repaired,
        _repair_json_numeric_expressions(comment_repaired),
    ]

    last_error: json.JSONDecodeError | None = None
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("no JSON candidates", text, 0)


def _json_scalar_or_none(value):
    """Keep only scalar values supported by PDERepresentation fields."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value
    return None


def _repair_json_numeric_expressions(text: str) -> str:
    """Evaluate simple unquoted numeric expressions in PDE numeric fields.

    Some smaller models emit JSON-like fields such as
    ``"coefficient": 7200.0 * 500.0``. This repair is limited to scalar
    numeric fields and accepts only Python AST arithmetic over numeric literals,
    so it cannot rewrite operators, variables, BC types, or other semantic
    labels.
    """

    numeric_field = re.compile(
        r'(?P<prefix>"(?:coefficient|value)"\s*:\s*)'
        r'(?P<expr>(?!["{\[])[^,\n\r}\]]+)'
    )

    def replace(match: re.Match[str]) -> str:
        expr = match.group("expr").strip()
        if not _looks_like_numeric_expression(expr):
            return match.group(0)
        value = _safe_eval_numeric_expression(expr)
        if value is None:
            return match.group(0)
        return f"{match.group('prefix')}{format(value, '.17g')}"

    return numeric_field.sub(replace, text)


def _replace_json_block_comments_with_null(text: str) -> str:
    """Replace C-style block comments outside JSON strings with ``null``."""
    result: list[str] = []
    index = 0
    in_string = False
    escape = False
    while index < len(text):
        char = text[index]
        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue

        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            if end < 0:
                return text
            result.append("null")
            index = end + 2
            continue

        result.append(char)
        index += 1
    return "".join(result)


def _looks_like_numeric_expression(expr: str) -> bool:
    """Return True when a JSON value looks like arithmetic over literals."""
    if not re.fullmatch(r"[0-9eE+\-*/().\s]+", expr):
        return False
    if not re.search(r"\d", expr):
        return False
    return any(operator in expr for operator in ("*", "/", "(", ")")) or bool(
        re.search(r"(?<![eE])\s[+\-]\s", expr)
    )


def _safe_eval_numeric_expression(expr: str) -> float | None:
    """Safely evaluate a simple arithmetic expression over numeric literals."""
    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
            value = eval_node(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Sub | ast.Mult | ast.Div | ast.Pow):
            left = eval_node(node.left)
            right = eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            return left ** right
        raise ValueError(f"unsupported numeric expression node: {type(node).__name__}")

    try:
        value = eval_node(parsed)
    except (ArithmeticError, OverflowError, ValueError):
        return None
    if not math.isfinite(value) or abs(value) > 1.0e100:
        return None
    return value


def _extract_json_payload(text: str) -> str:
    """Return raw JSON from a plain response, fenced block, or prose wrapper."""
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    payload = match.group(1) if match else text
    payload = payload.strip()
    if payload.startswith("{") and payload.endswith("}"):
        return payload
    balanced = _extract_first_json_object(payload)
    return balanced if balanced is not None else payload


def _extract_first_json_object(text: str) -> str | None:
    """Extract the first balanced JSON object, respecting quoted strings."""
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def _parse_self_check(response: str) -> ExtractionSelfCheck | None:
    """Parse the optional extraction self-check. Invalid JSON disables retry."""
    try:
        data = json.loads(_extract_json_payload(response))
    except Exception:
        return None

    raw_confidence = data.get("confidence")
    confidence = None
    if isinstance(raw_confidence, int | float):
        confidence = float(raw_confidence)

    flags = data.get("flags", [])
    if not isinstance(flags, list):
        flags = []
    checklist = data.get("checklist", [])
    if not isinstance(checklist, list):
        checklist = []

    return ExtractionSelfCheck(
        confidence=confidence,
        needs_retry=bool(data.get("needs_retry", False)),
        flags=[str(flag) for flag in flags],
        checklist=[str(item) for item in checklist],
    )


def _build_self_check_prompt(nl: str, raw_json: str) -> str:
    """Build the self-check prompt from runtime-visible information only."""
    return (
        "Natural language description:\n"
        f"{nl}\n\n"
        "First-pass extraction JSON:\n"
        f"```json\n{raw_json}\n```\n\n"
        "Check whether this extraction may have omitted variables, coupling terms, "
        "time terms, boundary/initial conditions, or explicitly stated material "
        "coefficients. Return only the self-check JSON."
    )


def _detect_risk_families(nl: str) -> set[str]:
    """Detect conservative PF/THM risk words from the natural language prompt."""
    lowered = nl.lower()
    families: set[str] = set()
    phase_field_terms = (
        "phase-field",
        "phase field",
        "order parameter",
        "fracture",
        "allen-cahn",
        "cahn-hilliard",
    )
    thm_terms = (
        "thermo-hydro-mechanical",
        "thm",
        "porous",
        "pore pressure",
        "darcy",
        "effective stress",
    )
    if any(term in lowered for term in phase_field_terms):
        families.add("phase_field")
    if any(term in lowered for term in thm_terms):
        families.add("thm")
    return families


def _is_overly_generic_for_risk(pde: PDERepresentation, families: set[str]) -> bool:
    """Return True when a risky prompt was reduced to only generic operators."""
    if not families or not pde.terms:
        return False

    operators = {term.operator for term in pde.terms}
    domain_specific = {
        "phase_field": {"allen_cahn", "cahn_hilliard", "coupled_force"},
        "thm": {"pf_darcy_flux", "pf_effective_stress", "coupled_force"},
    }
    return any(operators.isdisjoint(domain_specific[family]) for family in families)


def _extraction_quality_score(
    pde: PDERepresentation | None,
    families: set[str],
) -> float:
    """Score extraction completeness using runtime-visible structure only."""
    if pde is None:
        return -1.0

    score = 0.0
    score += 2.0 if pde.variables else 0.0
    score += 3.0 if pde.terms else 0.0
    score += min(len(pde.variables), 6) * 0.25
    score += min(len(pde.terms), 8) * 0.5
    score += min(len(pde.boundary_conditions), 8) * 0.15
    score += min(len(pde.initial_conditions), 4) * 0.1

    if pde.time_scheme and pde.time_scheme != "steady":
        score += 0.25

    operators = {term.operator for term in pde.terms}
    domain_specific = {
        "phase_field": {"allen_cahn", "cahn_hilliard", "coupled_force"},
        "thm": {"pf_darcy_flux", "pf_effective_stress", "coupled_force"},
    }
    for family in families:
        family_ops = domain_specific.get(family, set())
        score += len(operators & family_ops) * 2.0
        if family_ops and operators and operators.isdisjoint(family_ops):
            score -= 2.0

    return score


def _should_accept_retry(
    first_pde: PDERepresentation | None,
    retry_pde: PDERepresentation | None,
    families: set[str],
) -> bool:
    """Accept retry only when it is structurally better than the first attempt."""
    if retry_pde is None:
        return False
    if first_pde is None:
        return True
    return (
        _extraction_quality_score(retry_pde, families)
        > _extraction_quality_score(first_pde, families)
    )


def _retry_reason(
    nl: str,
    pde: PDERepresentation | None,
    self_check: ExtractionSelfCheck | None,
) -> tuple[str | None, set[str]]:
    """Decide whether to retry without using GT or extraction_ifs."""
    families = _detect_risk_families(nl)

    if pde is None:
        return "parse failed", families

    if self_check is not None:
        if self_check.needs_retry:
            flags = ", ".join(self_check.flags[:3])
            suffix = f" ({flags})" if flags else ""
            return f"self-check requested retry{suffix}", families
        if self_check.confidence is not None and self_check.confidence < 0.70:
            return f"self-check low confidence {self_check.confidence:.2f}", families

    if not pde.variables:
        return "local sanity check: no variables extracted", families
    if not pde.terms:
        return "local sanity check: no PDE terms extracted", families

    if _is_overly_generic_for_risk(pde, families):
        family_label = "/".join(sorted(families))
        return f"local sanity check: {family_label} prompt reduced to generic operators", families

    return None, families


def _build_extraction_retry_prompt(
    nl: str,
    first_json: str,
    self_check: ExtractionSelfCheck | None,
    triggered_families: set[str],
) -> str:
    """Build a conservative retry prompt with targeted context only when triggered."""
    prompt_parts = [
        "Re-read the problem and redo the PDE extraction from scratch.",
        "",
        "Natural language description:",
        nl,
        "",
        "First-pass extraction JSON:",
        "```json",
        first_json,
        "```",
        "",
        "Retry checklist:",
        "- Re-read the problem and check whether any variable is missing.",
        "- Check whether every explicitly described coupling has a term.",
        "- Check whether transient wording implies a time derivative.",
        "- Check whether BC/IC and coefficients were preserved.",
        "- If unsure, prefer a faithful explicit term over silently omitting it.",
    ]

    if self_check is not None and (self_check.flags or self_check.checklist):
        prompt_parts.extend([
            "",
            "Self-check notes:",
            json.dumps({
                "confidence": self_check.confidence,
                "flags": self_check.flags,
                "checklist": self_check.checklist,
            }, indent=2),
        ])

    if "phase_field" in triggered_families:
        prompt_parts.extend([
            "",
            "Phase-field retry context:",
            "For order-parameter or fracture descriptions, keep the order "
            "parameter variable and any explicitly described evolution or "
            "coupling structure. Do not collapse phase evolution into generic "
            "diffusion/reaction if the prompt describes a more specific process.",
        ])

    if "thm" in triggered_families:
        prompt_parts.extend([
            "",
            "Porous-flow / THM retry context:",
            "For thermo-hydro-mechanical descriptions, preserve temperature, "
            "pore pressure, and displacement variables such as T, pp, disp_x, "
            "and disp_y when they are described. Recheck flow/mechanics coupling, "
            "Darcy flux, and effective stress rather than silently omitting them.",
        ])

    prompt_parts.extend([
        "",
        "Return ONLY the corrected PDE extraction JSON using the required schema.",
    ])
    return "\n".join(prompt_parts)


def _extract_pde_llm(ext_llm, nl: str) -> ExtractionAttempt:
    """Extract PDE_llm, optionally retrying once using runtime-visible checks."""
    first_json = ext_llm.generate(EXTRACT_SPEC_SYSTEM, nl, temperature=0)
    pde = parse_pde_llm(first_json)
    attempt = ExtractionAttempt(pde=pde, raw_json=first_json)

    self_check = None
    if pde is not None:
        self_check_response = ext_llm.generate(
            EXTRACT_SELF_CHECK_SYSTEM,
            _build_self_check_prompt(nl, first_json),
            temperature=0,
        )
        self_check = _parse_self_check(self_check_response)
        if self_check is not None:
            attempt.self_check_confidence = self_check.confidence
            attempt.self_check_flags = self_check.flags

    reason, families = _retry_reason(nl, pde, self_check)
    if reason is None:
        return attempt

    retry_json = ext_llm.generate(
        EXTRACT_SPEC_SYSTEM,
        _build_extraction_retry_prompt(nl, first_json, self_check, families),
        temperature=0,
    )
    retry_pde = parse_pde_llm(retry_json)

    attempt.retried = True
    attempt.retry_reason = reason
    attempt.retry_accepted = _should_accept_retry(pde, retry_pde, families)
    if attempt.retry_accepted:
        attempt.pde = retry_pde
        attempt.raw_json = retry_json
    return attempt


def _spec_summary(pde_llm: PDERepresentation) -> str:
    """Serialize a compact PDERepresentation summary for prompts."""
    return json.dumps(
        {
            "variables": pde_llm.variables,
            "terms": [
                {"var": t.variable, "op": t.operator, "coeff": t.coefficient}
                for t in pde_llm.terms
            ],
            "bcs": [
                {
                    "var": bc.variable,
                    "boundary": bc.boundary,
                    "type": bc.bc_type,
                    "value": bc.value,
                }
                for bc in pde_llm.boundary_conditions
            ],
            "time_scheme": pde_llm.time_scheme,
        },
        indent=2,
    )


def _build_spec_guided_user_prompt(
    pde_llm: PDERepresentation,
    *,
    object_plan: ObjectRealizationPlan | None = None,
) -> str:
    """Build the user prompt for spec-guided code generation."""
    spec_summary = _spec_summary(pde_llm)
    prompt = f"PDE Specification:\n```json\n{spec_summary}\n```\n"
    if object_plan is not None:
        prompt += f"\n{object_plan.to_prompt_section()}\n"
        prompt += (
            "\nCompleteness requirement:\n"
            f"- Define all variables: {', '.join(pde_llm.variables)}.\n"
            f"- Realize all {len(pde_llm.terms)} PDE terms, all "
            f"{len(pde_llm.boundary_conditions)} boundary conditions, and all "
            f"{len(pde_llm.initial_conditions)} initial conditions listed above.\n"
            "- Do not return a runnable subset or simplified surrogate of the PDE.\n"
        )
    prompt += "\nGenerate the MOOSE input file."
    return prompt


def _generate_from_pde(llm, pde_llm: PDERepresentation) -> str:
    """Generate MOOSE code from an extracted PDERepresentation."""
    return llm.generate(
        SPEC_GUIDED_SYSTEM,
        _build_spec_guided_user_prompt(pde_llm),
        temperature=0,
    )


def _generate_from_pde_with_registry(
    llm,
    pde_llm: PDERepresentation,
    registry: MooseRegistry,
) -> tuple[str, ObjectRealizationPlan]:
    """Generate MOOSE code with targeted registry object schemas."""
    object_plan = build_object_plan(pde_llm, registry)
    return (
        llm.generate(
            SPEC_GUIDED_SYSTEM,
            _build_spec_guided_user_prompt(pde_llm, object_plan=object_plan),
            temperature=0,
        ),
        object_plan,
    )


def _extract_and_generate(llm, nl: str, *, extractor_llm=None) -> tuple[ExtractionAttempt, str | None]:
    """Stage 1+2 shared by methods B and D: extract PDE_llm, generate spec-guided code.

    If *extractor_llm* is provided, Stage 1 (PDE extraction) uses that model
    while Stage 2 (code generation) uses *llm*.  This enables mixed-LLM ablation.
    """
    _ext = extractor_llm if extractor_llm is not None else llm
    attempt = _extract_pde_llm(_ext, nl)
    pde_llm = attempt.pde
    if pde_llm is None:
        return attempt, None

    return attempt, _generate_from_pde(llm, pde_llm)


def _extract_and_generate_reg(
    llm,
    nl: str,
    registry: MooseRegistry,
    *,
    extractor_llm=None,
) -> tuple[ExtractionAttempt, str | None, ObjectRealizationPlan | None]:
    """Stage 1+2 for registry-grounded DReg."""
    _ext = extractor_llm if extractor_llm is not None else llm
    attempt = _extract_pde_llm(_ext, nl)
    pde_llm = attempt.pde
    if pde_llm is None:
        return attempt, None, None

    code, object_plan = _generate_from_pde_with_registry(llm, pde_llm, registry)
    return attempt, code, object_plan


def _add_extraction_metadata(result: dict, attempt: ExtractionAttempt) -> dict:
    """Attach retry metadata to a result dict."""
    result["extraction_retried"] = attempt.retried
    result["extraction_retry_reason"] = attempt.retry_reason
    result["extraction_retry_accepted"] = attempt.retry_accepted
    result["_pde_llm_json"] = attempt.raw_json if attempt.pde is not None else None
    return result


def _get_runner_commit() -> str:
    """Return the current git commit for result provenance."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def _artifact_path(
    artifact_dir: Path,
    *,
    case_id: str,
    method: str,
    llm_name: str,
    suffix: str,
) -> Path:
    safe_llm = re.sub(r"[^A-Za-z0-9_.-]+", "_", llm_name)
    case_dir = artifact_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir / f"{method}_{safe_llm}{suffix}"


def _write_artifacts(
    task: dict,
    llm_name: str,
    result: dict,
    artifact_dir: Path,
) -> dict[str, str | None]:
    """Persist generated code and extracted PDE JSON sidecars."""
    paths: dict[str, str | None] = {
        "code_path": None,
        "pde_llm_path": None,
        "object_plan_path": None,
    }

    code = result.get("_code")
    if isinstance(code, str):
        code_path = _artifact_path(
            artifact_dir,
            case_id=task["case_id"],
            method=task["method"],
            llm_name=llm_name,
            suffix=".i",
        )
        code_path.write_text(code, encoding="utf-8")
        paths["code_path"] = str(code_path)

    pde_json = result.get("_pde_llm_json")
    if isinstance(pde_json, str):
        pde_path = _artifact_path(
            artifact_dir,
            case_id=task["case_id"],
            method=task["method"],
            llm_name=llm_name,
            suffix=".pde_llm.json",
        )
        pde_path.write_text(_extract_json_payload(pde_json), encoding="utf-8")
        paths["pde_llm_path"] = str(pde_path)

    object_plan = result.get("_object_plan")
    if isinstance(object_plan, str):
        object_plan_path = _artifact_path(
            artifact_dir,
            case_id=task["case_id"],
            method=task["method"],
            llm_name=llm_name,
            suffix=".object_plan.txt",
        )
        object_plan_path.write_text(object_plan, encoding="utf-8")
        paths["object_plan_path"] = str(object_plan_path)

    return paths


def _material_consistency_fields(task: dict, result: dict) -> dict:
    """Compute supplementary MCS fields when reference and candidate code exist."""
    source_path = task.get("source_path")
    code = result.get("_code")
    fields = {
        "mcs": None,
        "mcs_applicable": None,
        "mcs_total": None,
        "mcs_matched": None,
        "mcs_mismatched": None,
        "mcs_error": None,
    }
    if source_path is None or not Path(source_path).exists() or not isinstance(code, str):
        return fields

    try:
        ref_code = Path(source_path).read_text(encoding="utf-8")
        mcs = compute_material_consistency(ref_code, code)
    except Exception as exc:
        fields["mcs_error"] = str(exc)
        return fields

    mcs_applicable = mcs.total_properties > 0
    fields.update({
        "mcs": mcs.score if mcs_applicable else None,
        "mcs_applicable": mcs_applicable,
        "mcs_total": mcs.total_properties,
        "mcs_matched": mcs.matched_properties,
        "mcs_mismatched": len(mcs.mismatched),
    })
    return fields


# ---------------------------------------------------------------------------
# IFS evaluation
# ---------------------------------------------------------------------------


def _evaluate_code_once(code: str, ref: PDERepresentation) -> dict:
    """Run one IFS evaluation attempt without code repair."""
    try:
        pde_code = reconstruct_pde(code)
        ifs = compute_ifs(ref, pde_code)
        return {
            "parse": True,
            "ifs": ifs.ifs_score,
            "term": ifs.ifs_term,
            "coeff": ifs.ifs_coeff,
            "bc": ifs.ifs_bc,
            "ic": ifs.ifs_ic,
            "time_dim": ifs.ifs_time,
            "passed": ifs.num_passed,
            "total": ifs.num_checkpoints,
            "error": None,
            "_ifs_result": ifs,
        }
    except HITParseError as exc:
        return _parse_failure_result(str(exc))
    except Exception as exc:
        return _parse_failure_result(str(exc))


def _parse_failure_result(error: str) -> dict:
    return {
        "parse": False,
        "ifs": None,
        "term": None,
        "coeff": None,
        "bc": None,
        "ic": None,
        "time_dim": None,
        "passed": 0,
        "total": 0,
        "error": error,
    }


def _iter_code_repair_candidates(code: str):
    """Yield fallback-only code candidates after an initial parse failure."""
    seen: set[str] = set()

    def emit(candidate: str):
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            return None
        seen.add(candidate)
        return candidate

    extracted = extract_code(code)
    candidate = emit(extracted)
    if candidate is not None:
        yield candidate, "strip_markdown_fence"

    for base, reason in ((code, "trim_to_hit_region"), (extracted, "trim_extracted_to_hit_region")):
        trimmed = _trim_to_parseable_hit_region(base)
        candidate = emit(trimmed) if trimmed is not None else None
        if candidate is not None:
            yield candidate, reason

    for base, reason in ((extracted, "hit_syntax_repair"), (code, "hit_syntax_repair_raw")):
        syntax = HITSyntaxValidator().validate(base)
        if syntax.passed:
            candidate = emit(syntax.final_code)
            if candidate is not None:
                yield candidate, reason


def _trim_to_parseable_hit_region(code: str) -> str | None:
    """Trim prose before/after a HIT block and return a syntax-valid region."""
    lines = code.splitlines()
    start = next((i for i, line in enumerate(lines) if line.lstrip().startswith("[")), None)
    if start is None:
        return None

    validator = HITSyntaxValidator()
    for end in range(len(lines), start, -1):
        candidate = "\n".join(lines[start:end]).strip()
        if not candidate:
            continue
        syntax = validator.validate(candidate)
        if syntax.passed:
            return syntax.final_code
    return None


def evaluate_code(code: str, ref: PDERepresentation) -> dict:
    """Run IFS evaluation of generated code against a reference PDERepresentation.

    Returns a dict with parse, ifs, sub-scores, counts, and optionally error.
    Preserves _ifs_result for violation extraction.
    Applies code repair only after the original code fails to parse.
    """
    result = _evaluate_code_once(code, ref)
    if result["parse"]:
        result["code_repair_applied"] = False
        result["code_repair_reason"] = None
        result["_evaluated_code"] = code
        return result

    original_error = result.get("error")
    for candidate, reason in _iter_code_repair_candidates(code):
        repaired = _evaluate_code_once(candidate, ref)
        if repaired["parse"]:
            repaired["code_repair_applied"] = True
            repaired["code_repair_reason"] = reason
            repaired["code_repair_original_error"] = original_error
            repaired["_evaluated_code"] = candidate
            return repaired

    result["code_repair_applied"] = False
    result["code_repair_reason"] = None
    result["_evaluated_code"] = code
    return result


def _evaluated_code(result: dict, fallback: str) -> str:
    """Return the code that actually parsed/evaluated, if repair was applied."""
    code = result.get("_evaluated_code")
    return code if isinstance(code, str) else fallback


def _score_or_zero(result: dict) -> float:
    """Return an IFS score with parse/failure counted as zero."""
    value = result.get("ifs")
    return float(value) if value is not None else 0.0


def _convergence_snapshot(
    *,
    stage: str,
    iteration: int,
    code: str,
    internal_result: dict,
    gt: PDERepresentation,
    accepted: bool | None = None,
    candidate_internal_ifs: float | None = None,
    candidate_gt_ifs: float | None = None,
    stop_reason: str | None = None,
) -> dict:
    """Record convergence diagnostics without affecting refinement decisions."""
    gt_result = evaluate_code(code, gt)
    return {
        "stage": stage,
        "iteration": iteration,
        "internal_ifs": _score_or_zero(internal_result),
        "gt_ifs": _score_or_zero(gt_result),
        "internal_parse": bool(internal_result.get("parse", False)),
        "gt_parse": bool(gt_result.get("parse", False)),
        "accepted": accepted,
        "candidate_internal_ifs": candidate_internal_ifs,
        "candidate_gt_ifs": candidate_gt_ifs,
        "stop_reason": stop_reason,
    }


def _syntax_prepare_for_registry(code: str) -> tuple[str, str | None]:
    """Apply fallback-only HIT syntax prep before L2 registry validation."""
    syntax = HITSyntaxValidator().validate(code)
    if syntax.passed:
        return syntax.final_code, None

    for candidate, reason in _iter_code_repair_candidates(code):
        if HITSyntaxValidator().validate(candidate).passed:
            return candidate, reason
    return code, None


def _empty_registry_meta(*, enabled: bool = False) -> dict:
    return {
        "registry_enabled": enabled,
        "registry_preplan_applied": False,
        "registry_l2_pass_before": None,
        "registry_l2_pass_after": None,
        "registry_issue_count": None,
        "registry_issue_kinds": None,
        "registry_repair_applied": False,
        "registry_repair_changes": [],
        "registry_repair_stages": [],
        "registry_llm_repair_attempted": False,
        "registry_llm_repair_accepted": False,
    }


def _merge_registry_summary(meta: dict, stage: str, summary) -> None:
    meta["registry_enabled"] = True
    meta["registry_l2_pass_before"] = summary.before_passed
    meta["registry_l2_pass_after"] = summary.after_passed
    meta["registry_issue_count"] = summary.issue_count
    meta["registry_issue_kinds"] = summary.issue_kinds
    meta["registry_repair_applied"] = bool(meta.get("registry_repair_applied")) or summary.changed
    changes = list(meta.get("registry_repair_changes") or [])
    changes.extend(summary.changes)
    meta["registry_repair_changes"] = changes[:24]
    stages = list(meta.get("registry_repair_stages") or [])
    stages.append({
        "stage": stage,
        "before_passed": summary.before_passed,
        "after_passed": summary.after_passed,
        "changed": summary.changed,
        "issue_count": summary.issue_count,
        "issue_kinds": summary.issue_kinds,
        "syntax_repair_reason": summary.syntax_repair_reason,
        "changes": list(summary.changes[:8]),
    })
    meta["registry_repair_stages"] = stages


def _apply_registry_guard(
    code: str,
    registry: MooseRegistry,
    meta: dict,
    *,
    stage: str,
) -> str:
    """Run syntax prep + deterministic L2 repair and merge metadata."""
    prepared, syntax_reason = _syntax_prepare_for_registry(code)
    summary = validate_and_repair_code(
        prepared,
        registry,
        syntax_repair_reason=syntax_reason,
    )
    _merge_registry_summary(meta, stage, summary)
    return summary.repaired_code if summary.changed or syntax_reason else prepared


def _registry_issues_for_prompt(code: str, registry: MooseRegistry) -> str:
    prepared, _ = _syntax_prepare_for_registry(code)
    result = validate_and_repair_code(prepared, registry)
    validator = MOOSETypeValidator(registry)
    issues = validator.validate(result.repaired_code).issues
    return format_registry_issues(issues)


def _try_registry_llm_repair(
    llm,
    code: str,
    pde_ref: PDERepresentation,
    registry: MooseRegistry,
    meta: dict,
    *,
    object_plan_text: str | None,
    stage: str,
) -> tuple[str, dict]:
    """Try one constrained mechanical LLM repair with IFS regression guard."""
    if meta.get("registry_l2_pass_after") is True:
        return code, evaluate_code(code, pde_ref)

    meta_before = copy.deepcopy(meta)
    before_issue_count = meta.get("registry_issue_count")
    if not isinstance(before_issue_count, int):
        before_issue_count = 10**9
    before_eval = evaluate_code(code, pde_ref)
    before_ifs = before_eval.get("ifs") or 0.0
    object_plan = object_plan_text or "No object plan is available for this stage."
    prompt = REGISTRY_MECHANICAL_REPAIR_TEMPLATE.format(
        code=code,
        violations=_registry_issues_for_prompt(code, registry),
        object_plan=object_plan,
    )
    meta["registry_llm_repair_attempted"] = True
    repaired = llm.generate(SPEC_GUIDED_SYSTEM, prompt, temperature=0)
    repaired = _apply_registry_guard(repaired, registry, meta, stage=f"{stage}:llm_repair")
    repaired_eval = evaluate_code(repaired, pde_ref)
    repaired_ifs = repaired_eval.get("ifs") or 0.0
    after_issue_count = meta.get("registry_issue_count")
    issue_count_decreased = (
        isinstance(after_issue_count, int)
        and after_issue_count < before_issue_count
    )

    if repaired_ifs >= before_ifs and (
        meta.get("registry_l2_pass_after") is True
        or issue_count_decreased
    ):
        meta["registry_llm_repair_accepted"] = True
        return repaired, repaired_eval

    meta["registry_llm_repair_accepted"] = False
    attempted = True
    meta.clear()
    meta.update(meta_before)
    meta["registry_llm_repair_attempted"] = attempted
    meta["registry_llm_repair_accepted"] = False
    return code, before_eval


@dataclass(frozen=True)
class SmokeExecResult:
    """Short first-window MOOSE runtime check result."""

    passed: bool
    status: str
    returncode: int | str | None
    error_excerpt: str | None
    timeout_s: int


def _tail_excerpt(text: str, *, limit: int = 1200) -> str | None:
    text = text.strip()
    if not text:
        return None
    return text[-limit:]


def _copy_benchmark_runtime_inputs(work_dir: Path) -> None:
    """Copy benchmark mesh/table sidecars for generated inputs using basename refs."""
    if not _SOURCE_FILES_DIR.exists():
        return
    for path in _SOURCE_FILES_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in {".e", ".csv"}:
            shutil.copy2(path, work_dir / path.name)


def _smoke_exec_code(
    code: str,
    *,
    moose_app: Path,
    timeout: int = 1,
) -> SmokeExecResult:
    """Run a short MOOSE no-error window for code not yet written as an artifact.

    A timeout with no detected error is a pass.  A fast nonzero return code or
    explicit error text within the window is a failure and can trigger
    execution-only repair.
    """
    with tempfile.TemporaryDirectory(prefix="cm_exec_repair_", dir="/tmp") as tmp:
        _copy_benchmark_runtime_inputs(Path(tmp))
        input_path = Path(tmp) / "candidate.i"
        input_path.write_text(code, encoding="utf-8")
        try:
            completed = subprocess.run(
                [str(moose_app), "-i", str(input_path), "--allow-unused"],
                cwd=tmp,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            text = _decode_timeout_stream(exc.stdout) + "\n" + _decode_timeout_stream(exc.stderr)
            error_excerpt = _first_error_excerpt(text)
            return SmokeExecResult(
                passed=error_excerpt is None,
                status="timeout_no_error" if error_excerpt is None else "timeout_with_error_text",
                returncode="timeout",
                error_excerpt=error_excerpt,
                timeout_s=timeout,
            )

    text = (completed.stdout or "") + "\n" + (completed.stderr or "")
    error_excerpt = _first_error_excerpt(text)
    if completed.returncode == 0 and error_excerpt is None:
        return SmokeExecResult(
            passed=True,
            status="completed_no_error",
            returncode=completed.returncode,
            error_excerpt=None,
            timeout_s=timeout,
        )
    if completed.returncode == 0:
        return SmokeExecResult(
            passed=False,
            status="completed_with_error_text",
            returncode=completed.returncode,
            error_excerpt=error_excerpt,
            timeout_s=timeout,
        )
    return SmokeExecResult(
        passed=False,
        status="failed_fast",
        returncode=completed.returncode,
        error_excerpt=error_excerpt
        or _tail_excerpt(text)
        or f"Process exited with return code {completed.returncode} within {timeout}s.",
        timeout_s=timeout,
    )


def _empty_exec_repair_meta(*, enabled: bool = False, timeout_s: int | None = None) -> dict:
    return {
        "exec_repair_enabled": enabled,
        "exec_repair_attempted": False,
        "exec_repair_accepted": False,
        "exec_repair_rounds": 0,
        "exec_repair_timeout_s": timeout_s,
        "exec_repair_smoke_before_pass": None,
        "exec_repair_smoke_before_status": None,
        "exec_repair_smoke_after_pass": None,
        "exec_repair_smoke_after_status": None,
        "exec_repair_error_excerpt": None,
    }


def _record_smoke_meta(meta: dict, smoke: SmokeExecResult, *, prefix: str) -> None:
    meta[f"exec_repair_smoke_{prefix}_pass"] = smoke.passed
    meta[f"exec_repair_smoke_{prefix}_status"] = smoke.status
    meta["exec_repair_timeout_s"] = smoke.timeout_s
    if prefix == "before":
        meta["exec_repair_error_excerpt"] = smoke.error_excerpt


# ---------------------------------------------------------------------------
# Method implementations
# ---------------------------------------------------------------------------


def run_method_a(llm, nl: str, gt: PDERepresentation) -> dict:
    """Method A (Direct): single-shot code generation from NL."""
    from experiments.prompts import _CODEGEN_SYSTEM, format_codegen_prompt

    code = llm.generate(_CODEGEN_SYSTEM, format_codegen_prompt(nl), temperature=0)
    result = evaluate_code(code, gt)
    result.pop("_ifs_result", None)
    result["internal_ifs"] = None
    result["extraction_ifs"] = None
    result["extraction_retried"] = False
    result["extraction_retry_reason"] = None
    result["extraction_retry_accepted"] = None
    result["_code"] = _evaluated_code(result, code)
    return result


def run_method_b(llm, nl: str, gt: PDERepresentation,
                 fallback_threshold: float = _EXTRACTION_IFS_FALLBACK_THRESHOLD,
                 extractor_llm=None) -> dict:
    """Method B (Spec-Guided): extract PDE spec then generate code, no refinement."""
    attempt, code = _extract_and_generate(llm, nl, extractor_llm=extractor_llm)
    pde_llm = attempt.pde
    if pde_llm is None or code is None:
        return _add_extraction_metadata({
            "parse": False,
            "ifs": None,
            "term": None,
            "coeff": None,
            "bc": None,
            "ic": None,
            "time_dim": None,
            "passed": 0,
            "total": 0,
            "internal_ifs": None,
            "extraction_ifs": None,
            "error": "PDE_llm extraction failed",
        }, attempt)
    # Measure extraction accuracy: IFS(PDE_gt, PDE_llm)
    try:
        ext_result = compute_ifs(gt, pde_llm)
        extraction_ifs = ext_result.ifs_score
    except Exception:
        extraction_ifs = None

    # fallback to Method A if extraction quality is too low (disabled when threshold=0)
    if fallback_threshold > 0 and extraction_ifs is not None and extraction_ifs < fallback_threshold:
        result = run_method_a(llm, nl, gt)
        result["extraction_ifs"] = extraction_ifs
        return _add_extraction_metadata(result, attempt)

    result = evaluate_code(code, gt)
    result.pop("_ifs_result", None)
    result["internal_ifs"] = None
    result["extraction_ifs"] = extraction_ifs
    result["_code"] = _evaluated_code(result, code)
    return _add_extraction_metadata(result, attempt)


def run_method_d(llm, nl: str, gt: PDERepresentation, max_iter: int = 2,
                 fallback_threshold: float = _EXTRACTION_IFS_FALLBACK_THRESHOLD,
                 extractor_llm=None) -> dict:
    """Method D (Full Pipeline): spec-guided + PDE_llm refinement for up to max_iter iters.

    Improvements over naive refinement:
    1. Conservative prompt — only fix listed violations, preserve everything else
    2. Regression guard — reject refinement step if IFS drops vs pre-refine code
    3. Lower skip threshold (0.85) — avoid refining already-good code
    """
    attempt, code = _extract_and_generate(llm, nl, extractor_llm=extractor_llm)
    pde_llm = attempt.pde
    if pde_llm is None or code is None:
        return _add_extraction_metadata({
            "parse": False,
            "ifs": None,
            "term": None,
            "coeff": None,
            "bc": None,
            "ic": None,
            "time_dim": None,
            "passed": 0,
            "total": 0,
            "internal_ifs": None,
            "extraction_ifs": None,
            "error": "PDE_llm extraction failed",
        }, attempt)

    # Measure extraction accuracy: IFS(PDE_gt, PDE_llm)
    try:
        ext_result = compute_ifs(gt, pde_llm)
        extraction_ifs = ext_result.ifs_score
    except Exception:
        extraction_ifs = None

    # fallback to Method A if extraction quality is too low (disabled when threshold=0)
    if fallback_threshold > 0 and extraction_ifs is not None and extraction_ifs < fallback_threshold:
        result = run_method_a(llm, nl, gt)
        result["extraction_ifs"] = extraction_ifs
        result["internal_ifs"] = None
        return _add_extraction_metadata(result, attempt)

    # Refinement loop using IFS(PDE_llm, PDE_code) with regression guard
    result_vs_llm = evaluate_code(code, pde_llm)
    best_code = _evaluated_code(result_vs_llm, code)
    best_ifs = result_vs_llm.get("ifs") or 0.0
    convergence_history = [
        _convergence_snapshot(
            stage="initial",
            iteration=0,
            code=best_code,
            internal_result=result_vs_llm,
            gt=gt,
        )
    ]
    refinement_rounds_attempted = 0
    refinement_rounds_accepted = 0
    refinement_stop_reason = "max_iter"

    for index in range(max_iter):
        # Skip if already good enough (lowered from 0.95 to 0.85)
        if best_ifs >= 0.85:
            refinement_stop_reason = "threshold_reached"
            break
        ifs_obj = result_vs_llm.get("_ifs_result")
        if ifs_obj is None:
            refinement_stop_reason = "missing_internal_ifs"
            break
        violations = format_violations_for_code(ifs_obj)
        if violations == "No violations.":
            refinement_stop_reason = "no_violations"
            break

        refinement_rounds_attempted += 1
        refined_code = llm.generate(
            SPEC_GUIDED_SYSTEM,
            REFINE_TEMPLATE.format(code=best_code, violations=violations),
            temperature=0,
        )
        result_vs_llm = evaluate_code(refined_code, pde_llm)
        refined_code = _evaluated_code(result_vs_llm, refined_code)
        refined_ifs = result_vs_llm.get("ifs") or 0.0
        candidate_gt = evaluate_code(refined_code, gt)
        candidate_gt_ifs = _score_or_zero(candidate_gt)

        # Regression guard: only accept refinement if IFS improves
        if refined_ifs >= best_ifs:
            best_code = refined_code
            best_ifs = refined_ifs
            refinement_rounds_accepted += 1
            convergence_history.append(
                _convergence_snapshot(
                    stage=f"refine_{index + 1}",
                    iteration=index + 1,
                    code=best_code,
                    internal_result=result_vs_llm,
                    gt=gt,
                    accepted=True,
                    candidate_internal_ifs=refined_ifs,
                    candidate_gt_ifs=candidate_gt_ifs,
                )
            )
        else:
            # Reject this refinement step, keep previous best
            previous_result_vs_llm = evaluate_code(best_code, pde_llm)
            convergence_history.append(
                _convergence_snapshot(
                    stage=f"refine_{index + 1}",
                    iteration=index + 1,
                    code=best_code,
                    internal_result=previous_result_vs_llm,
                    gt=gt,
                    accepted=False,
                    candidate_internal_ifs=refined_ifs,
                    candidate_gt_ifs=candidate_gt_ifs,
                    stop_reason="regression_guard",
                )
            )
            result_vs_llm = evaluate_code(best_code, pde_llm)
            refinement_stop_reason = "regression_guard"
            break  # don't try again — refinement is diverging

    internal_ifs = best_ifs

    # Final evaluation against GT
    result_vs_gt = evaluate_code(best_code, gt)
    result_vs_gt.pop("_ifs_result", None)
    result_vs_gt["internal_ifs"] = internal_ifs
    result_vs_gt["extraction_ifs"] = extraction_ifs
    result_vs_gt["convergence_history"] = convergence_history
    result_vs_gt["refinement_rounds_attempted"] = refinement_rounds_attempted
    result_vs_gt["refinement_rounds_accepted"] = refinement_rounds_accepted
    result_vs_gt["refinement_stop_reason"] = refinement_stop_reason
    result_vs_gt["_code"] = _evaluated_code(result_vs_gt, best_code)
    return _add_extraction_metadata(result_vs_gt, attempt)


def run_method_areg(
    llm,
    nl: str,
    gt: PDERepresentation,
    *,
    registry: MooseRegistry,
) -> dict:
    """Method AReg: direct NL generation plus registry mechanical validation.

    This is the registry-only control. It does not use GT or an extracted PDE
    as a repair oracle; the frozen registry guard applies only deterministic
    syntax/L2 normalization.
    """
    from experiments.prompts import _CODEGEN_SYSTEM, format_codegen_prompt

    code = llm.generate(_CODEGEN_SYSTEM, format_codegen_prompt(nl), temperature=0)
    registry_meta = _empty_registry_meta(enabled=True)
    code = _apply_registry_guard(code, registry, registry_meta, stage="post_generate")

    result = evaluate_code(code, gt)
    result.pop("_ifs_result", None)
    result["internal_ifs"] = None
    result["extraction_ifs"] = None
    result["extraction_retried"] = False
    result["extraction_retry_reason"] = None
    result["extraction_retry_accepted"] = None
    result["_code"] = _evaluated_code(result, code)
    result.update(registry_meta)
    return result


def run_method_execrepairreg(
    llm,
    nl: str,
    gt: PDERepresentation,
    *,
    registry: MooseRegistry,
    moose_app: Path,
    smoke_timeout: int = 2,
) -> dict:
    """Method ExecRepairReg: AReg plus one runtime-log repair if the short smoke check errors.

    This is the execution-only engineering baseline. It uses the same frozen
    registry layer as AReg, then runs a short MOOSE no-error window. Only a
    detected error within that window triggers exactly one LLM repair round; no
    PDE/IFS violation feedback is provided to the repair prompt.
    """
    from experiments.prompts import _CODEGEN_SYSTEM, format_codegen_prompt

    registry_meta = _empty_registry_meta(enabled=True)
    exec_meta = _empty_exec_repair_meta(enabled=True, timeout_s=smoke_timeout)

    code = llm.generate(_CODEGEN_SYSTEM, format_codegen_prompt(nl), temperature=0)
    code = _apply_registry_guard(code, registry, registry_meta, stage="post_generate")

    smoke_before = _smoke_exec_code(
        code,
        moose_app=moose_app,
        timeout=smoke_timeout,
    )
    _record_smoke_meta(exec_meta, smoke_before, prefix="before")

    final_code = code
    if not smoke_before.passed and smoke_before.error_excerpt:
        exec_meta["exec_repair_attempted"] = True
        exec_meta["exec_repair_rounds"] = 1
        prompt = EXEC_REPAIR_REG_TEMPLATE.format(
            code=code,
            status=f"{smoke_before.status}, returncode={smoke_before.returncode}",
            runtime_log=smoke_before.error_excerpt,
        )
        repaired = llm.generate(_CODEGEN_SYSTEM, prompt, temperature=0)
        repaired = _apply_registry_guard(
            repaired,
            registry,
            registry_meta,
            stage="exec_repair_1",
        )
        smoke_after = _smoke_exec_code(
            repaired,
            moose_app=moose_app,
            timeout=smoke_timeout,
        )
        _record_smoke_meta(exec_meta, smoke_after, prefix="after")
        if smoke_after.passed:
            exec_meta["exec_repair_accepted"] = True
            final_code = repaired

    result = evaluate_code(final_code, gt)
    result.pop("_ifs_result", None)
    result["internal_ifs"] = None
    result["extraction_ifs"] = None
    result["extraction_retried"] = False
    result["extraction_retry_reason"] = None
    result["extraction_retry_accepted"] = None
    result["_code"] = _evaluated_code(result, final_code)
    result.update(registry_meta)
    result.update(exec_meta)
    return result


def run_method_dreg(
    llm,
    nl: str,
    gt: PDERepresentation,
    max_iter: int = 2,
    fallback_threshold: float = _EXTRACTION_IFS_FALLBACK_THRESHOLD,
    extractor_llm=None,
    *,
    registry: MooseRegistry,
) -> dict:
    """Method DReg: PDE pipeline plus frozen registry object-realization layer."""
    registry_meta = _empty_registry_meta(enabled=True)
    attempt, code, object_plan = _extract_and_generate_reg(
        llm,
        nl,
        registry,
        extractor_llm=extractor_llm,
    )
    pde_llm = attempt.pde
    if pde_llm is None or code is None or object_plan is None:
        result = _add_extraction_metadata({
            "parse": False,
            "ifs": None,
            "term": None,
            "coeff": None,
            "bc": None,
            "ic": None,
            "time_dim": None,
            "passed": 0,
            "total": 0,
            "internal_ifs": None,
            "extraction_ifs": None,
            "error": "PDE_llm extraction failed",
        }, attempt)
        result.update(registry_meta)
        return result

    object_plan_text = object_plan.to_prompt_section()
    registry_meta["registry_preplan_applied"] = True

    try:
        ext_result = compute_ifs(gt, pde_llm)
        extraction_ifs = ext_result.ifs_score
    except Exception:
        extraction_ifs = None

    if fallback_threshold > 0 and extraction_ifs is not None and extraction_ifs < fallback_threshold:
        result = run_method_areg(llm, nl, gt, registry=registry)
        result["extraction_ifs"] = extraction_ifs
        result["internal_ifs"] = None
        result["_object_plan"] = object_plan_text
        return _add_extraction_metadata(result, attempt)

    code = _apply_registry_guard(code, registry, registry_meta, stage="post_generate")
    code, result_vs_llm = _try_registry_llm_repair(
        llm,
        code,
        pde_llm,
        registry,
        registry_meta,
        object_plan_text=object_plan_text,
        stage="post_generate",
    )
    best_code = _evaluated_code(result_vs_llm, code)
    best_ifs = result_vs_llm.get("ifs") or 0.0

    for index in range(max_iter):
        if best_ifs >= 0.85:
            break
        ifs_obj = result_vs_llm.get("_ifs_result")
        if ifs_obj is None:
            break
        violations = format_violations_for_code(ifs_obj)
        if violations == "No violations.":
            break

        refined_code = llm.generate(
            SPEC_GUIDED_SYSTEM,
            REFINE_TEMPLATE.format(code=best_code, violations=violations),
            temperature=0,
        )
        refined_code = _apply_registry_guard(
            refined_code,
            registry,
            registry_meta,
            stage=f"semantic_refine_{index + 1}",
        )
        refined_code, refined_vs_llm = _try_registry_llm_repair(
            llm,
            refined_code,
            pde_llm,
            registry,
            registry_meta,
            object_plan_text=object_plan_text,
            stage=f"semantic_refine_{index + 1}",
        )
        refined_code = _evaluated_code(refined_vs_llm, refined_code)
        refined_ifs = refined_vs_llm.get("ifs") or 0.0

        if refined_ifs >= best_ifs:
            best_code = refined_code
            best_ifs = refined_ifs
            result_vs_llm = refined_vs_llm
        else:
            result_vs_llm = evaluate_code(best_code, pde_llm)
            break

    best_code = _apply_registry_guard(
        best_code,
        registry,
        registry_meta,
        stage="final_guard",
    )
    best_code, final_vs_llm = _try_registry_llm_repair(
        llm,
        best_code,
        pde_llm,
        registry,
        registry_meta,
        object_plan_text=object_plan_text,
        stage="final_guard",
    )
    final_ifs = final_vs_llm.get("ifs") or 0.0
    if final_ifs >= best_ifs:
        best_code = _evaluated_code(final_vs_llm, best_code)
        best_ifs = final_ifs

    result_vs_gt = evaluate_code(best_code, gt)
    result_vs_gt.pop("_ifs_result", None)
    result_vs_gt["internal_ifs"] = best_ifs
    result_vs_gt["extraction_ifs"] = extraction_ifs
    result_vs_gt["_code"] = _evaluated_code(result_vs_gt, best_code)
    result_vs_gt["_object_plan"] = object_plan_text
    result_vs_gt.update(registry_meta)
    return _add_extraction_metadata(result_vs_gt, attempt)


def run_method_ae(llm, nl: str, gt: PDERepresentation, max_iter: int = 2) -> dict:
    """Method A+E (Exec-Refine): direct gen + generic self-refinement (no PDE feedback)."""
    from experiments.prompts import _CODEGEN_SYSTEM, format_codegen_prompt

    code = llm.generate(_CODEGEN_SYSTEM, format_codegen_prompt(nl), temperature=0)
    result = evaluate_code(code, gt)
    code = _evaluated_code(result, code)

    for _ in range(max_iter):
        if result.get("ifs") is not None and result["ifs"] >= 0.95:
            break
        prompt = EXEC_REFINE_TEMPLATE.format(code=code)
        code = llm.generate(_CODEGEN_SYSTEM, prompt, temperature=0)
        result = evaluate_code(code, gt)
        code = _evaluated_code(result, code)

    result.pop("_ifs_result", None)
    result["internal_ifs"] = None
    result["extraction_ifs"] = None
    result["extraction_retried"] = False
    result["extraction_retry_reason"] = None
    result["extraction_retry_accepted"] = None
    result["_code"] = code
    return result


def run_method_c(llm, nl: str, gt: PDERepresentation, max_iter: int = 2) -> dict:
    """Method C (Verif-Only): direct gen + IFS refinement with GT as reference.

    Uses regression guard: reject refinement if IFS drops.
    """
    from experiments.prompts import _CODEGEN_SYSTEM, format_codegen_prompt

    code = llm.generate(_CODEGEN_SYSTEM, format_codegen_prompt(nl), temperature=0)
    result = evaluate_code(code, gt)
    best_code = _evaluated_code(result, code)
    best_ifs = result.get("ifs") or 0.0

    for _ in range(max_iter):
        if best_ifs >= 0.95:
            break
        ifs_obj = result.get("_ifs_result")
        if ifs_obj is None:
            break
        violations = format_violations_for_code(ifs_obj)
        if violations == "No violations.":
            break
        refined_code = llm.generate(
            _CODEGEN_SYSTEM,
            REFINE_TEMPLATE.format(code=best_code, violations=violations),
            temperature=0,
        )
        result = evaluate_code(refined_code, gt)
        refined_code = _evaluated_code(result, refined_code)
        refined_ifs = result.get("ifs") or 0.0

        if refined_ifs >= best_ifs:
            best_code = refined_code
            best_ifs = refined_ifs
        else:
            result = evaluate_code(best_code, gt)
            break

    result.pop("_ifs_result", None)
    result["internal_ifs"] = None
    result["extraction_ifs"] = None
    result["extraction_retried"] = False
    result["extraction_retry_reason"] = None
    result["extraction_retry_accepted"] = None
    result["_code"] = best_code
    return result


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


def load_done_set(output_path: Path) -> set[tuple[str, str, str]]:
    """Return the set of (id, method, llm) triples already present in the JSONL."""
    done: set[tuple[str, str, str]] = set()
    if not output_path.exists():
        return done
    with output_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["id"], rec["method"], rec["llm"]))
            except Exception:
                pass
    return done


def _is_provider_run_abort(exc: Exception) -> bool:
    """Identify provider/account errors that should stop the whole run."""
    text = str(exc).lower()
    fatal_markers = (
        "credit balance is too low",
        "billing",
        "rate_limit_error",
        "too many requests",
        "error code: 429",
    )
    return any(marker in text for marker in fatal_markers)


def _build_method_fns(
    llm,
    *,
    fallback_threshold: float,
    extractor_llm=None,
    registry: MooseRegistry | None = None,
    moose_app: Path | None = None,
    smoke_timeout: int = 2,
) -> dict:
    """Build method dispatch table for a specific LLM client instance."""
    methods = {
        "A": functools.partial(run_method_a, llm),
        "AE": functools.partial(run_method_ae, llm),
        "B": functools.partial(
            run_method_b,
            llm,
            fallback_threshold=fallback_threshold,
            extractor_llm=extractor_llm,
        ),
        "C": functools.partial(run_method_c, llm),
        "D": functools.partial(
            run_method_d,
            llm,
            fallback_threshold=fallback_threshold,
            extractor_llm=extractor_llm,
        ),
    }
    if registry is not None:
        registry_methods = {
            "AReg": functools.partial(run_method_areg, llm, registry=registry),
            "DReg": functools.partial(
                run_method_dreg,
                llm,
                fallback_threshold=fallback_threshold,
                extractor_llm=extractor_llm,
                registry=registry,
            ),
        }
        if moose_app is not None:
            registry_methods["ExecRepairReg"] = functools.partial(
                run_method_execrepairreg,
                llm,
                registry=registry,
                moose_app=moose_app,
                smoke_timeout=smoke_timeout,
            )
        methods.update(registry_methods)
    return methods


def _error_result(exc: Exception) -> dict:
    """Return a result-shaped dict for failed evaluation tasks."""
    result = {
        "parse": False,
        "ifs": None,
        "term": None,
        "coeff": None,
        "bc": None,
        "ic": None,
        "time_dim": None,
        "passed": 0,
        "total": 0,
        "internal_ifs": None,
        "extraction_ifs": None,
        "convergence_history": None,
        "refinement_rounds_attempted": None,
        "refinement_rounds_accepted": None,
        "refinement_stop_reason": None,
        "extraction_retried": False,
        "extraction_retry_reason": None,
        "extraction_retry_accepted": None,
        "code_repair_applied": False,
        "code_repair_reason": None,
        "error": str(exc),
    }
    result.update(_empty_registry_meta())
    result.update(_empty_exec_repair_meta())
    return result


def _make_record(
    task: dict,
    llm_name: str,
    result: dict,
    elapsed: float,
    *,
    artifact_dir: Path,
    runner_commit: str,
    fallback_threshold: float,
    extractor_llm_name: str | None,
) -> dict:
    """Convert a method result into the stable JSONL record schema."""
    artifact_paths = _write_artifacts(task, llm_name, result, artifact_dir)
    mcs_fields = _material_consistency_fields(task, result)
    registry_defaults = _empty_registry_meta()
    exec_repair_defaults = _empty_exec_repair_meta()
    record = {
        "id": task["case_id"],
        "method": task["method"],
        "llm": llm_name,
        "family": task["family"],
        "complexity": task["complexity"],
        "parse": result.get("parse", False),
        "ifs": result.get("ifs"),
        "term": result.get("term"),
        "coeff": result.get("coeff"),
        "bc": result.get("bc"),
        "ic": result.get("ic"),
        "time_dim": result.get("time_dim"),
        "passed": result.get("passed", 0),
        "total": result.get("total", 0),
        "elapsed": round(elapsed, 2),
        "internal_ifs": result.get("internal_ifs"),
        "extraction_ifs": result.get("extraction_ifs"),
        "convergence_history": result.get("convergence_history"),
        "refinement_rounds_attempted": result.get("refinement_rounds_attempted"),
        "refinement_rounds_accepted": result.get("refinement_rounds_accepted"),
        "refinement_stop_reason": result.get("refinement_stop_reason"),
        "extraction_retried": result.get("extraction_retried", False),
        "extraction_retry_reason": result.get("extraction_retry_reason"),
        "extraction_retry_accepted": result.get("extraction_retry_accepted"),
        "code_repair_applied": result.get("code_repair_applied", False),
        "code_repair_reason": result.get("code_repair_reason"),
        "code_path": artifact_paths["code_path"],
        "pde_llm_path": artifact_paths["pde_llm_path"],
        "object_plan_path": artifact_paths["object_plan_path"],
        "prompt_version": _PROMPT_VERSION,
        "runner_commit": runner_commit,
        "generator_llm": llm_name,
        "extractor_llm": extractor_llm_name or llm_name,
        "retry_enabled": task["method"] in {"B", "D", "DReg"},
        "fallback_threshold": fallback_threshold,
        "registry_enabled": result.get(
            "registry_enabled", registry_defaults["registry_enabled"]
        ),
        "registry_preplan_applied": result.get(
            "registry_preplan_applied",
            registry_defaults["registry_preplan_applied"],
        ),
        "registry_l2_pass_before": result.get(
            "registry_l2_pass_before",
            registry_defaults["registry_l2_pass_before"],
        ),
        "registry_l2_pass_after": result.get(
            "registry_l2_pass_after",
            registry_defaults["registry_l2_pass_after"],
        ),
        "registry_issue_count": result.get(
            "registry_issue_count",
            registry_defaults["registry_issue_count"],
        ),
        "registry_issue_kinds": result.get(
            "registry_issue_kinds",
            registry_defaults["registry_issue_kinds"],
        ),
        "registry_repair_applied": result.get(
            "registry_repair_applied",
            registry_defaults["registry_repair_applied"],
        ),
        "registry_repair_changes": result.get(
            "registry_repair_changes",
            registry_defaults["registry_repair_changes"],
        ),
        "registry_repair_stages": result.get(
            "registry_repair_stages",
            registry_defaults["registry_repair_stages"],
        ),
        "registry_llm_repair_attempted": result.get(
            "registry_llm_repair_attempted",
            registry_defaults["registry_llm_repair_attempted"],
        ),
        "registry_llm_repair_accepted": result.get(
            "registry_llm_repair_accepted",
            registry_defaults["registry_llm_repair_accepted"],
        ),
        "exec_repair_enabled": result.get(
            "exec_repair_enabled",
            exec_repair_defaults["exec_repair_enabled"],
        ),
        "exec_repair_attempted": result.get(
            "exec_repair_attempted",
            exec_repair_defaults["exec_repair_attempted"],
        ),
        "exec_repair_accepted": result.get(
            "exec_repair_accepted",
            exec_repair_defaults["exec_repair_accepted"],
        ),
        "exec_repair_rounds": result.get(
            "exec_repair_rounds",
            exec_repair_defaults["exec_repair_rounds"],
        ),
        "exec_repair_timeout_s": result.get(
            "exec_repair_timeout_s",
            exec_repair_defaults["exec_repair_timeout_s"],
        ),
        "exec_repair_smoke_before_pass": result.get(
            "exec_repair_smoke_before_pass",
            exec_repair_defaults["exec_repair_smoke_before_pass"],
        ),
        "exec_repair_smoke_before_status": result.get(
            "exec_repair_smoke_before_status",
            exec_repair_defaults["exec_repair_smoke_before_status"],
        ),
        "exec_repair_smoke_after_pass": result.get(
            "exec_repair_smoke_after_pass",
            exec_repair_defaults["exec_repair_smoke_after_pass"],
        ),
        "exec_repair_smoke_after_status": result.get(
            "exec_repair_smoke_after_status",
            exec_repair_defaults["exec_repair_smoke_after_status"],
        ),
        "exec_repair_error_excerpt": result.get(
            "exec_repair_error_excerpt",
            exec_repair_defaults["exec_repair_error_excerpt"],
        ),
        "error": result.get("error"),
    }
    record.update(mcs_fields)
    return record


def _run_eval_task(
    task: dict,
    get_method_fns,
    llm_name: str,
    *,
    artifact_dir: Path,
    runner_commit: str,
    fallback_threshold: float,
    extractor_llm_name: str | None,
) -> tuple[dict, float]:
    """Run one case/method task and return a JSONL record plus elapsed seconds."""
    t0 = time.time()
    try:
        method_fns = get_method_fns()
        result = method_fns[task["method"]](task["nl"], task["gt"])
    except Exception as exc:
        if _is_provider_run_abort(exc):
            raise
        result = _error_result(exc)
    elapsed = time.time() - t0
    return _make_record(
        task,
        llm_name,
        result,
        elapsed,
        artifact_dir=artifact_dir,
        runner_commit=runner_commit,
        fallback_threshold=fallback_threshold,
        extractor_llm_name=extractor_llm_name,
    ), elapsed


# ---------------------------------------------------------------------------
# Summary mode
# ---------------------------------------------------------------------------


def print_summary(jsonl_path: Path) -> None:
    """Read a JSONL results file and print formatted summary tables."""
    records: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            with suppress(Exception):
                records.append(json.loads(line))

    if not records:
        print(f"No records found in {jsonl_path}")
        return

    print(f"\nSummary of {len(records)} records from {jsonl_path}\n")

    # --- Per-method mean IFS (FAIL = 0) ---
    methods = sorted({r["method"] for r in records})
    print("Per-method mean IFS (FAIL counts as 0):")
    print(f"  {'Method':<6}  {'N':>4}  {'Mean IFS':>10}  {'Parse%':>8}")
    print("  " + "-" * 36)
    for method in methods:
        recs = [r for r in records if r["method"] == method]
        scores = [r["ifs"] if r.get("ifs") is not None else 0.0 for r in recs]
        n_parsed = sum(1 for r in recs if r.get("parse"))
        mean_ifs = sum(scores) / len(scores) if scores else 0.0
        parse_pct = 100 * n_parsed / len(recs) if recs else 0.0
        print(f"  {method:<6}  {len(recs):>4}  {mean_ifs:>10.4f}  {parse_pct:>7.1f}%")

    # --- Per-family breakdown ---
    families = sorted({r.get("family", "unknown") for r in records})
    print("\nPer-family breakdown (mean IFS, FAIL=0):")
    header = f"  {'Family':<14}" + "".join(f"  {m:>8}" for m in methods)
    print(header)
    print("  " + "-" * (14 + 10 * len(methods)))
    for family in families:
        row = f"  {family:<14}"
        for method in methods:
            recs = [r for r in records if r.get("family") == family and r["method"] == method]
            if not recs:
                row += f"  {'N/A':>8}"
            else:
                scores = [r["ifs"] if r.get("ifs") is not None else 0.0 for r in recs]
                row += f"  {sum(scores)/len(scores):>8.4f}"
        print(row)

    # --- Per-complexity breakdown ---
    complexities = sorted({r.get("complexity", "unknown") for r in records})
    print("\nPer-complexity breakdown (mean IFS, FAIL=0):")
    header = f"  {'Complexity':<12}" + "".join(f"  {m:>8}" for m in methods)
    print(header)
    print("  " + "-" * (12 + 10 * len(methods)))
    for complexity in complexities:
        row = f"  {complexity:<12}"
        for method in methods:
            recs = [
                r
                for r in records
                if r.get("complexity") == complexity and r["method"] == method
            ]
            if not recs:
                row += f"  {'N/A':>8}"
            else:
                scores = [r["ifs"] if r.get("ifs") is not None else 0.0 for r in recs]
                row += f"  {sum(scores)/len(scores):>8.4f}"
        print(row)

    print()


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_evaluation(
    llm_name: str,
    methods: list[str],
    limit: int,
    output_path: Path,
    exclude_ids: set[str] | None = None,
    fallback_threshold: float = _EXTRACTION_IFS_FALLBACK_THRESHOLD,
    extractor_llm_name: str | None = None,
    workers: int = 1,
    artifact_dir: Path | None = None,
    registry_json: Path | None = None,
    moose_app: Path | None = None,
    smoke_timeout: int = 2,
) -> None:
    """Run the MooseBench evaluation for the given LLM and methods."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if any(method in _REGISTRY_METHODS for method in methods) and registry_json is None:
        raise ValueError("--registry-json is required for registry-enabled methods")
    if any(method in _SMOKE_EXEC_METHODS for method in methods) and moose_app is None:
        raise ValueError("--moose-app is required for ExecRepairReg")
    if smoke_timeout < 1:
        raise ValueError("smoke_timeout must be >= 1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_dir is None:
        artifact_dir = output_path.parent / f"{output_path.stem}_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    runner_commit = _get_runner_commit()
    registry = None
    if registry_json is not None:
        print(f"Loading MOOSE registry: {registry_json}", flush=True)
        registry = MooseRegistry.from_moose_json_file(registry_json)
        print(f"Registry object types: {len(registry.all_type_names())}", flush=True)

    # Load all prompt files
    prompt_files = sorted(_PROMPTS_DIR.glob("*.json"))
    if not prompt_files:
        print(f"No prompt JSON files found in {_PROMPTS_DIR}")
        sys.exit(1)

    # Filter out excluded cases
    if exclude_ids:
        before = len(prompt_files)
        prompt_files = [
            p for p in prompt_files if p.stem not in exclude_ids
        ]
        print(f"Excluded {before - len(prompt_files)} cases ({len(prompt_files)} remaining)")

    if limit > 0:
        prompt_files = prompt_files[:limit]

    # Check which (id, method, llm) are already done
    done_set = load_done_set(output_path)
    if done_set:
        print(f"Resuming: {len(done_set)} record(s) already in {output_path}")

    thread_local = threading.local()
    shared_method_fns = None

    if workers == 1:
        print(f"Building LLM backend: {llm_name}", flush=True)
        llm = make_llm(llm_name)
        extractor_llm = None
        if extractor_llm_name:
            print(f"Building extractor LLM: {extractor_llm_name}", flush=True)
            extractor_llm = make_llm(extractor_llm_name)
        shared_method_fns = _build_method_fns(
            llm,
            fallback_threshold=fallback_threshold,
            extractor_llm=extractor_llm,
            registry=registry,
            moose_app=moose_app,
            smoke_timeout=smoke_timeout,
        )

        def get_method_fns():
            return shared_method_fns

    else:
        print(
            f"Validating LLM backend: {llm_name} "
            f"(each worker will create its own client)",
            flush=True,
        )
        make_llm(llm_name)
        if extractor_llm_name:
            print(
                f"Validating extractor LLM: {extractor_llm_name} "
                f"(each worker will create its own client)",
                flush=True,
            )
            make_llm(extractor_llm_name)

        def get_method_fns():
            if not hasattr(thread_local, "method_fns"):
                llm = make_llm(llm_name)
                extractor_llm = (
                    make_llm(extractor_llm_name) if extractor_llm_name else None
                )
                thread_local.method_fns = _build_method_fns(
                    llm,
                    fallback_threshold=fallback_threshold,
                    extractor_llm=extractor_llm,
                    registry=registry,
                    moose_app=moose_app,
                    smoke_timeout=smoke_timeout,
                )
            return thread_local.method_fns

    total_cases = len(prompt_files) * len(methods)
    skip_count = 0
    tasks: list[dict] = []

    for prompt_path in prompt_files:
        prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
        case_id = prompt.get("id", prompt_path.stem)
        nl = prompt.get("nl_description", "")
        family = prompt.get("physics_family", "unknown")
        complexity = prompt.get("complexity", "unknown")

        gt_path = _GT_DIR / f"{case_id}.json"
        if not gt_path.exists():
            print(f"  [SKIP] {case_id}: ground truth not found", flush=True)
            continue

        try:
            gt = load_gt(gt_path)
        except Exception as exc:
            print(f"  [SKIP] {case_id}: GT load error: {exc}", flush=True)
            continue

        for method in methods:
            triple = (case_id, method, llm_name)
            if triple in done_set:
                skip_count += 1
                continue

            if method not in _METHOD_NAMES:
                print(f"  [SKIP] Unknown method {method!r}", flush=True)
                continue

            tasks.append({
                "display_index": len(tasks) + skip_count + 1,
                "case_id": case_id,
                "method": method,
                "nl": nl,
                "family": family,
                "complexity": complexity,
                "gt": gt,
                "source_path": _SOURCE_FILES_DIR / f"{case_id}.i",
            })

    if workers > 1:
        print(f"Running {len(tasks)} task(s) with {workers} worker threads", flush=True)

    done_count = 0

    def write_record(out_fh, record: dict, display_index: int, elapsed: float) -> None:
        ifs_str = f"IFS={record['ifs']:.4f}" if record["ifs"] is not None else "FAIL"
        print(
            f"  [{display_index}/{total_cases}] "
            f"{record['id']} method={record['method']} llm={llm_name}"
            f"  -> {ifs_str}  ({elapsed:.1f}s)",
            flush=True,
        )
        out_fh.write(json.dumps(record) + "\n")
        out_fh.flush()

    with output_path.open("a", encoding="utf-8") as out_fh:
        if workers == 1:
            for task in tasks:
                record, elapsed = _run_eval_task(
                    task,
                    get_method_fns,
                    llm_name,
                    artifact_dir=artifact_dir,
                    runner_commit=runner_commit,
                    fallback_threshold=fallback_threshold,
                    extractor_llm_name=extractor_llm_name,
                )
                write_record(out_fh, record, task["display_index"], elapsed)
                done_count += 1
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_task = {
                    executor.submit(
                        _run_eval_task,
                        task,
                        get_method_fns,
                        llm_name,
                        artifact_dir=artifact_dir,
                        runner_commit=runner_commit,
                        fallback_threshold=fallback_threshold,
                        extractor_llm_name=extractor_llm_name,
                    ): task
                    for task in tasks
                }
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    record, elapsed = future.result()
                    write_record(out_fh, record, task["display_index"], elapsed)
                    done_count += 1

    print(
        f"\nDone. {done_count} new record(s) written, {skip_count} skipped (already done)."
    )
    print(f"Output: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MooseBench evaluation runner: methods A/B/D x LLMs -> JSONL results."
    )
    parser.add_argument(
        "--llm",
        choices=[
            "claude",
            "claude-haiku",
            "gpt",
            "gpt-5.4",
            "gemini",
            "gemini3",
            "gemini-3-pro",
            "gemini3-pro",
            "gemini-3.1-pro-preview",
            "gemini31-pro",
            "gemini-3.1-flash-lite-preview",
            "gemini31-flash-lite",
            "deepseek-flash",
        ],
        help="LLM backend to use (required unless --summary).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(_METHOD_NAMES),
        default=["A"],
        help="Evaluation methods to run (default: A).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of cases to run (0 = all, default: 0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL output path (default: experiments/results/moosebench_{llm}.jsonl).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Print summary of existing JSONL file and exit.",
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        default=None,
        help="JSON file listing case IDs to exclude (array of strings).",
    )
    parser.add_argument(
        "--extractor-llm",
        choices=[
            "claude",
            "claude-haiku",
            "gpt",
            "gpt-5.4",
            "gemini",
            "gemini3",
            "gemini-3-pro",
            "gemini3-pro",
            "gemini-3.1-pro-preview",
            "gemini31-pro",
            "gemini-3.1-flash-lite-preview",
            "gemini31-flash-lite",
            "deepseek-flash",
        ],
        default=None,
        help="Use a different LLM for PDE extraction (Stage 1). If omitted, --llm is used for all stages.",
    )
    parser.add_argument(
        "--fallback-threshold",
        type=float,
        default=_EXTRACTION_IFS_FALLBACK_THRESHOLD,
        help=(
            "B/D fallback threshold: if extraction_ifs < this, fall back to Method A. "
            "Set to 0.0 to disable fallback entirely (recommended for weak models). "
            f"Default: {_EXTRACTION_IFS_FALLBACK_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel worker threads for case/method tasks. "
            "Default: 1 (serial)."
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help=(
            "Directory for generated code/PDE sidecar artifacts. "
            "Default: <output_stem>_artifacts beside the JSONL output."
        ),
    )
    parser.add_argument(
        "--registry-json",
        type=Path,
        default=None,
        help=(
            "MOOSE app syntax dump from app-opt --json. Required for registry-enabled methods."
        ),
    )
    parser.add_argument(
        "--moose-app",
        type=Path,
        default=None,
        help="Path to the MOOSE app binary. Required for ExecRepairReg.",
    )
    parser.add_argument(
        "--smoke-timeout",
        type=int,
        default=2,
        help="Short no-error MOOSE window in seconds for ExecRepairReg. Default: 2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Summary mode
    if args.summary is not None:
        print_summary(args.summary)
        return

    # Evaluation mode requires --llm
    if args.llm is None:
        print("ERROR: --llm is required unless --summary is provided.")
        sys.exit(1)
    if args.workers < 1:
        print("ERROR: --workers must be >= 1.")
        sys.exit(1)
    if any(method in _REGISTRY_METHODS for method in args.methods) and args.registry_json is None:
        print("ERROR: --registry-json is required when running registry-enabled methods.")
        sys.exit(1)
    if any(method in _SMOKE_EXEC_METHODS for method in args.methods) and args.moose_app is None:
        print("ERROR: --moose-app is required when running ExecRepairReg.")
        sys.exit(1)
    if args.smoke_timeout < 1:
        print("ERROR: --smoke-timeout must be >= 1.")
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        output_path = _RESULTS_DIR / f"moosebench_{args.llm}.jsonl"

    # Load exclude list
    exclude_ids: set[str] | None = None
    if args.exclude is not None:
        exclude_ids = set(json.loads(args.exclude.read_text(encoding="utf-8")))
        print(f"  Exclude: {len(exclude_ids)} cases from {args.exclude}")

    fallback_threshold = args.fallback_threshold

    print("MooseBench Evaluation Runner")
    print(f"  LLM:               {args.llm}")
    if args.extractor_llm:
        print(f"  Extractor LLM:     {args.extractor_llm}")
    print(f"  Methods:           {args.methods}")
    print(f"  Limit:             {args.limit if args.limit > 0 else 'all'}")
    print(f"  Workers:           {args.workers}")
    print(f"  Output:            {output_path}")
    print(
        "  Artifacts:         "
        f"{args.artifact_dir or output_path.parent / f'{output_path.stem}_artifacts'}"
    )
    if args.registry_json is not None:
        print(f"  Registry JSON:     {args.registry_json}")
    if args.moose_app is not None:
        print(f"  MOOSE app:         {args.moose_app}")
    if any(method in _SMOKE_EXEC_METHODS for method in args.methods):
        print(f"  Smoke timeout:     {args.smoke_timeout}s")
    print(f"  Fallback threshold:{fallback_threshold} ({'disabled' if fallback_threshold == 0 else 'enabled'})")
    print()

    run_evaluation(
        llm_name=args.llm,
        methods=args.methods,
        limit=args.limit,
        output_path=output_path,
        exclude_ids=exclude_ids,
        fallback_threshold=fallback_threshold,
        extractor_llm_name=args.extractor_llm,
        workers=args.workers,
        artifact_dir=args.artifact_dir,
        registry_json=args.registry_json,
        moose_app=args.moose_app,
        smoke_timeout=args.smoke_timeout,
    )


if __name__ == "__main__":
    main()
