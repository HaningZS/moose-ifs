"""Experiment runner: orchestrates variant execution + evaluation.

Runs a single (prompt, variant, llm) combination and evaluates
the result against PDE_gt using compute_ifs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from codmos.multiagent.pde.ifs_engine import IFSResult, compute_ifs
from codmos.multiagent.pde.reconstruction import reconstruct_pde
from codmos.multiagent.pde.representation import PDERepresentation
from codmos.multiagent.validators.hit_parser import HITParseError
from experiments.llm import LLMBackend
from experiments.moosebench_loader import BenchmarkPrompt
from experiments.variants.ablations import (
    run_full_cross_llm,
    run_full_exec_only,
    run_full_generic_feedback,
    run_full_no_refine,
)
from experiments.variants.base import VariantResult
from experiments.variants.direct import run_direct
from experiments.variants.full import run_full
from experiments.variants.spec_guided import run_spec_guided
from experiments.variants.verif_only import run_verif_only

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Complete result for one (prompt, variant, llm) trial."""

    prompt_id: str
    variant: str
    complexity: str
    variant_result: VariantResult
    eval_ifs: IFSResult | None  # IFS(PDE_gt, PDE_code) — None if no GT or parse fails

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "variant": self.variant,
            "complexity": self.complexity,
            "eval_ifs": self.eval_ifs.ifs_score if self.eval_ifs else None,
            **self.variant_result.to_dict(),
        }


def run_single_prompt(
    prompt: BenchmarkPrompt,
    variant: str,
    llm: LLMBackend,
    pde_llm: PDERepresentation | None = None,
    ifs_threshold: float = 0.85,
    max_iterations: int = 3,
    temperature: float = 0,
    llm_generate: LLMBackend | None = None,
) -> EvalResult:
    """Run a single (prompt, variant) trial and evaluate against PDE_gt."""
    variant_result = _dispatch_variant(
        variant=variant,
        nl_description=prompt.nl_description,
        pde_llm=pde_llm,
        llm=llm,
        ifs_threshold=ifs_threshold,
        max_iterations=max_iterations,
        temperature=temperature,
        llm_generate=llm_generate,
    )

    eval_ifs = _evaluate(variant_result.code, prompt.pde_gt)

    return EvalResult(
        prompt_id=prompt.id,
        variant=variant,
        complexity=prompt.complexity,
        variant_result=variant_result,
        eval_ifs=eval_ifs,
    )


def _dispatch_variant(
    variant: str,
    nl_description: str,
    pde_llm: PDERepresentation | None,
    llm: LLMBackend,
    ifs_threshold: float,
    max_iterations: int,
    temperature: float,
    llm_generate: LLMBackend | None,
) -> VariantResult:
    """Dispatch to the appropriate variant function."""
    if variant == "direct":
        return run_direct(nl_description, llm, temperature)
    elif variant == "spec_guided":
        assert pde_llm is not None, "spec_guided requires pde_llm"
        return run_spec_guided(pde_llm, llm, temperature)
    elif variant == "verif_only":
        assert pde_llm is not None, "verif_only requires pde_llm"
        return run_verif_only(nl_description, pde_llm, llm, ifs_threshold, max_iterations, temperature)
    elif variant == "full":
        assert pde_llm is not None, "full requires pde_llm"
        return run_full(nl_description, pde_llm, llm, ifs_threshold, max_iterations, temperature)
    elif variant == "D-no-refine":
        assert pde_llm is not None
        return run_full_no_refine(nl_description, pde_llm, llm, temperature)
    elif variant == "D-generic":
        assert pde_llm is not None
        return run_full_generic_feedback(nl_description, pde_llm, llm, ifs_threshold, max_iterations, temperature)
    elif variant == "D-exec-only":
        assert pde_llm is not None
        return run_full_exec_only(nl_description, pde_llm, llm, max_iterations, temperature)
    elif variant == "D-cross":
        assert pde_llm is not None
        assert llm_generate is not None, "D-cross requires llm_generate"
        return run_full_cross_llm(nl_description, pde_llm, llm_generate, ifs_threshold, max_iterations, temperature)
    else:
        raise ValueError(f"Unknown variant: {variant!r}")


def _evaluate(code: str, pde_gt: PDERepresentation | None) -> IFSResult | None:
    """Evaluate generated code against ground truth PDE."""
    if pde_gt is None:
        return None
    try:
        pde_code = reconstruct_pde(code)
        return compute_ifs(pde_gt, pde_code)
    except (HITParseError, Exception):
        return None
