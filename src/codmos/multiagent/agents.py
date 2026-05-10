"""Minimal paper-artifact agent data models.

The full development repository contains interactive agent orchestration. This
artifact keeps only the structured data containers required by the PDE
conversion/extraction tests and paper reproduction scripts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PhysicsSpecification:
    """Structured physics specification from the modeling stage."""

    problem_description: str
    spatial_dimensionality: str
    modeling_assumptions: list[str]
    physics_modules: list[str]
    primary_variables: list[dict[str, str]]
    governing_equations: str
    boundary_conditions: list[dict[str, str]]
    initial_conditions: list[dict[str, str]]
    material_properties: list[dict[str, Any]]
    time_integration: dict[str, Any]
    numerical_considerations: str
    outputs_and_diagnostics: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class ValidationResult:
    """Lightweight validation result container retained for compatibility."""

    is_valid: bool
    syntax_check: bool
    physics_check: bool
    execution_success: bool
    errors: list[str]
    warnings: list[str]
    output_files: list[str]
    execution_log: str
    feedbacks: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
