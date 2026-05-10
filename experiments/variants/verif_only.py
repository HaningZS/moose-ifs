"""Variant C: Verification-Only.

Flow: NL → LLM → Code → IFS(PDE_llm, PDE_code) → refine loop → eval.
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
    format_codegen_prompt,
    format_refinement_prompt,
    format_violations,
)
from experiments.variants.base import VariantResult

logger = logging.getLogger(__name__)


def run_verif_only(
    nl_description: str,
    pde_llm: PDERepresentation,
    llm: LLMBackend,
    ifs_threshold: float = 0.85,
    max_iterations: int = 3,
    temperature: float = 0,
) -> VariantResult:
    """Run Variant C (Verif-Only): NL → LLM → Code → IFS refine."""
    intermediate_codes: list[str] = []
    code = ""
    internal_ifs: IFSResult | None = None
    violation_report = ""

    for iteration in range(max_iterations):
        if iteration == 0:
            user_prompt = format_codegen_prompt(nl_description)
        else:
            user_prompt = format_refinement_prompt(nl_description, code, violation_report)

        code = llm.generate(_CODEGEN_SYSTEM, user_prompt, temperature=temperature)
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
        variant="verif_only",
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
