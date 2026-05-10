"""Core PDE data structures.

Layer 1 — pure data, no external dependencies.

These types represent PDE content only (what the PDE is).
Evaluation-specific concerns (ranges, ground truth modes) live in ifs_engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PDETerm:
    """A single PDE term (maps to one MOOSE kernel).

    ``operator`` stores the PDE-level normalized name (e.g. "diffusion"),
    while ``kernel_type`` preserves the MOOSE class (e.g. "HeatConduction")
    for traceability and coefficient resolution.
    """

    variable: str
    operator: str
    coefficient: float | str | None
    coupled_variable: str | None
    kernel_type: str | None
    severity: str


@dataclass
class BoundaryCondition:
    """A boundary condition specification."""

    variable: str
    boundary: str
    bc_type: str
    value: float | str | None
    moose_bc_class: str | None
    severity: str


@dataclass
class InitialCondition:
    """An initial condition (transient problems only)."""

    variable: str
    ic_type: str
    value: float | str | None
    severity: str


@dataclass
class PDERepresentation:
    """Complete PDE formal representation.

    Core types are kept clean — no evaluation-specific fields.
    Range handling and ground truth wrappers live in ``ifs_engine``.
    """

    terms: list[PDETerm]
    boundary_conditions: list[BoundaryCondition]
    initial_conditions: list[InitialCondition]
    time_scheme: str
    variables: list[str]
    dimensions: int

    unresolved_kernels: list[str] = field(default_factory=list)
    unresolved_coefficients: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def is_partial(self) -> bool:
        """True if reconstruction encountered unresolvable elements."""
        return bool(self.unresolved_kernels or self.unresolved_coefficients)
