"""Variant A: Direct code generation.

Flow: NL → LLM → Code (no PDE extraction, no verification).
Simplest baseline.
"""

from __future__ import annotations

import logging

from codmos.multiagent.pde.reconstruction import reconstruct_pde
from codmos.multiagent.validators.hit_parser import HITParseError
from experiments.llm import LLMBackend
from experiments.prompts import _CODEGEN_SYSTEM, format_codegen_prompt
from experiments.variants.base import VariantResult

logger = logging.getLogger(__name__)


def run_direct(
    nl_description: str,
    llm: LLMBackend,
    temperature: float = 0,
) -> VariantResult:
    """Run Variant A (Direct): NL → LLM → Code."""
    user_prompt = format_codegen_prompt(nl_description)
    code = llm.generate(_CODEGEN_SYSTEM, user_prompt, temperature=temperature)
    parse_success = _try_parse(code)

    return VariantResult(
        variant="direct",
        code=code,
        iterations=1,
        internal_ifs=None,
        pde_llm=None,
        parse_success=parse_success,
        moose_result=None,
    )


def _try_parse(code: str) -> bool:
    try:
        reconstruct_pde(code)
        return True
    except (HITParseError, Exception):
        return False
