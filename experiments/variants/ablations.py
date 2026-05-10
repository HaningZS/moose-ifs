"""Ablation variants for the Full pipeline (Variant D).

D-no-refine:  Full generation, no refinement loop
D-generic:    Refinement with generic "fix errors" (no PDE violations)
D-exec-only:  Refinement based on parse success, not IFS
D-cross:      Different LLMs for extraction vs generation
"""

from __future__ import annotations

import logging

from codmos.multiagent.pde.ifs_engine import IFSResult, compute_ifs
from codmos.multiagent.pde.reconstruction import reconstruct_pde
from codmos.multiagent.pde.representation import PDERepresentation
from codmos.multiagent.validators.hit_parser import HITParseError
from experiments.llm import LLMBackend
from experiments.prompts import (
    _CODEGEN_SYSTEM,
    format_generic_refinement_prompt,
    format_refinement_prompt,
    format_spec_guided_prompt,
    format_violations,
)
from experiments.variants.base import VariantResult

logger = logging.getLogger(__name__)


def run_full_no_refine(
    nl_description: str,
    pde_llm: PDERepresentation,
    llm: LLMBackend,
    temperature: float = 0,
) -> VariantResult:
    """D-no-refine: Spec-guided generation, single attempt, no refinement."""
    user_prompt = format_spec_guided_prompt(pde_llm)
    code = llm.generate(_CODEGEN_SYSTEM, user_prompt, temperature=temperature)

    internal_ifs: IFSResult | None = None
    parse_success = False
    try:
        pde_code = reconstruct_pde(code)
        internal_ifs = compute_ifs(pde_llm, pde_code)
        parse_success = True
    except (HITParseError, Exception):
        pass

    return VariantResult(
        variant="D-no-refine",
        code=code,
        iterations=1,
        internal_ifs=internal_ifs,
        pde_llm=pde_llm,
        parse_success=parse_success,
        moose_result=None,
    )


def run_full_generic_feedback(
    nl_description: str,
    pde_llm: PDERepresentation,
    llm: LLMBackend,
    ifs_threshold: float = 0.85,
    max_iterations: int = 3,
    temperature: float = 0,
) -> VariantResult:
    """D-generic: Spec-guided + refinement with generic feedback only."""
    intermediate_codes: list[str] = []
    code = ""
    internal_ifs: IFSResult | None = None

    for iteration in range(max_iterations):
        if iteration == 0:
            user_prompt = format_spec_guided_prompt(pde_llm)
        else:
            user_prompt = format_generic_refinement_prompt(nl_description, code)

        code = llm.generate(_CODEGEN_SYSTEM, user_prompt, temperature=temperature)
        intermediate_codes.append(code)

        try:
            pde_code = reconstruct_pde(code)
        except (HITParseError, Exception):
            continue

        internal_ifs = compute_ifs(pde_llm, pde_code)
        if internal_ifs.ifs_score >= ifs_threshold:
            break

    parse_success = _try_parse(code)

    return VariantResult(
        variant="D-generic",
        code=code,
        iterations=iteration + 1,
        internal_ifs=internal_ifs,
        pde_llm=pde_llm,
        parse_success=parse_success,
        moose_result=None,
        intermediate_codes=intermediate_codes,
    )


def run_full_exec_only(
    nl_description: str,
    pde_llm: PDERepresentation,
    llm: LLMBackend,
    max_iterations: int = 3,
    temperature: float = 0,
) -> VariantResult:
    """D-exec-only: Spec-guided + stops when code parses (not IFS)."""
    intermediate_codes: list[str] = []
    code = ""
    internal_ifs: IFSResult | None = None

    for iteration in range(max_iterations):
        if iteration == 0:
            user_prompt = format_spec_guided_prompt(pde_llm)
        else:
            user_prompt = format_refinement_prompt(
                nl_description, code, "Code failed to parse. Fix syntax errors."
            )

        code = llm.generate(_CODEGEN_SYSTEM, user_prompt, temperature=temperature)
        intermediate_codes.append(code)

        try:
            pde_code = reconstruct_pde(code)
            internal_ifs = compute_ifs(pde_llm, pde_code)
            break  # Parse success = stop
        except (HITParseError, Exception):
            continue

    parse_success = _try_parse(code)

    return VariantResult(
        variant="D-exec-only",
        code=code,
        iterations=iteration + 1,
        internal_ifs=internal_ifs,
        pde_llm=pde_llm,
        parse_success=parse_success,
        moose_result=None,
        intermediate_codes=intermediate_codes,
    )


def run_full_cross_llm(
    nl_description: str,
    pde_llm: PDERepresentation,
    llm_generate: LLMBackend,
    ifs_threshold: float = 0.85,
    max_iterations: int = 3,
    temperature: float = 0,
) -> VariantResult:
    """D-cross: PDE_llm from one LLM, generation from another."""
    intermediate_codes: list[str] = []
    code = ""
    internal_ifs: IFSResult | None = None
    violation_report = ""

    for iteration in range(max_iterations):
        if iteration == 0:
            user_prompt = format_spec_guided_prompt(pde_llm)
        else:
            user_prompt = format_refinement_prompt(nl_description, code, violation_report)

        code = llm_generate.generate(_CODEGEN_SYSTEM, user_prompt, temperature=temperature)
        intermediate_codes.append(code)

        try:
            pde_code = reconstruct_pde(code)
        except (HITParseError, Exception) as e:
            violation_report = f"Parse error: {e}"
            continue

        internal_ifs = compute_ifs(pde_llm, pde_code)
        if internal_ifs.ifs_score >= ifs_threshold:
            break

        violation_report = format_violations(internal_ifs)

    parse_success = _try_parse(code)

    return VariantResult(
        variant="D-cross",
        code=code,
        iterations=iteration + 1,
        internal_ifs=internal_ifs,
        pde_llm=pde_llm,
        parse_success=parse_success,
        moose_result=None,
        intermediate_codes=intermediate_codes,
    )


def _try_parse(code: str) -> bool:
    try:
        reconstruct_pde(code)
        return True
    except (HITParseError, Exception):
        return False
