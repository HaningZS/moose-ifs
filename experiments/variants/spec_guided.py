"""Variant B: Spec-Guided code generation.

Flow: PDE_llm → LLM(PDE spec) → Code (no verification loop).
"""

from __future__ import annotations

import logging

from codmos.multiagent.pde.reconstruction import reconstruct_pde
from codmos.multiagent.pde.representation import PDERepresentation
from codmos.multiagent.validators.hit_parser import HITParseError
from experiments.llm import LLMBackend
from experiments.prompts import _CODEGEN_SYSTEM, format_spec_guided_prompt
from experiments.variants.base import VariantResult

logger = logging.getLogger(__name__)


def run_spec_guided(
    pde_llm: PDERepresentation,
    llm: LLMBackend,
    temperature: float = 0,
) -> VariantResult:
    """Run Variant B (Spec-Guided): PDE_llm → LLM → Code."""
    user_prompt = format_spec_guided_prompt(pde_llm)
    code = llm.generate(_CODEGEN_SYSTEM, user_prompt, temperature=temperature)
    parse_success = _try_parse(code)

    return VariantResult(
        variant="spec_guided",
        code=code,
        iterations=1,
        internal_ifs=None,
        pde_llm=pde_llm,
        parse_success=parse_success,
        moose_result=None,
    )


def _try_parse(code: str) -> bool:
    try:
        reconstruct_pde(code)
        return True
    except (HITParseError, Exception):
        return False
