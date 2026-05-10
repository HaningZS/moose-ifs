"""Shared types for experiment variants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from codmos.multiagent.pde.ifs_engine import IFSResult
from codmos.multiagent.pde.representation import PDERepresentation
from experiments.moose_env import MOOSEResult


@dataclass
class VariantResult:
    """Output from running any variant on a single prompt."""

    variant: str  # "direct" | "spec_guided" | "verif_only" | "full" | ablation name
    code: str  # Final generated MOOSE .i code
    iterations: int  # Number of generation attempts (1 for non-refining)
    internal_ifs: IFSResult | None  # IFS(pde_llm, pde_code) — None for A/B
    pde_llm: PDERepresentation | None  # Extracted PDE — None for A
    parse_success: bool  # Whether final code parsed successfully
    moose_result: MOOSEResult | None  # MOOSE execution result (if available)
    intermediate_codes: list[str] = field(default_factory=list)
    """Codes from each iteration (for analysis)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "code": self.code,
            "iterations": self.iterations,
            "internal_ifs": self.internal_ifs.ifs_score if self.internal_ifs else None,
            "parse_success": self.parse_success,
            "moose_skipped": self.moose_result.skipped if self.moose_result else True,
            "moose_success": self.moose_result.success if self.moose_result else None,
        }
