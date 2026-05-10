# src/codmos/multiagent/pde/conversion.py
"""PhysicsSpecification <-> PDERepresentation conversion bridge.

Layer 3 — imports from Layer 1 (representation) and existing agents.py.

Primary conversion uses STRUCTURED fields from PhysicsSpecification only.
The free-text ``governing_equations`` field is NOT parsed — structured
fields (physics_modules, boundary_conditions dicts, time_integration,
material_properties) provide all necessary information.
"""

from __future__ import annotations

import logging
from typing import Any

from codmos.multiagent.agents import PhysicsSpecification
from codmos.multiagent.pde.representation import (
    BoundaryCondition,
    InitialCondition,
    PDERepresentation,
    PDETerm,
)

logger = logging.getLogger(__name__)

# Map physics module names to expected PDE terms.
# Each entry: (operator, default_coefficient_param, severity)
_PHYSICS_MODULE_TERMS: dict[str, list[tuple[str, str | None, str]]] = {
    "heat_conduction": [
        ("diffusion", "thermal_conductivity", "high"),
    ],
    "thermal": [
        ("diffusion", "thermal_conductivity", "high"),
    ],
    "diffusion": [
        ("diffusion", "diffusivity", "high"),
    ],
    "solid_mechanics": [
        ("stress_divergence", None, "high"),
    ],
    "mechanics": [
        ("stress_divergence", None, "high"),
    ],
}

_DIM_MAP: dict[str, int] = {"1D": 1, "2D": 2, "3D": 3, "Axisymmetric": 2}
_DIM_REVERSE: dict[int, str] = {1: "1D", 2: "2D", 3: "3D"}


def physicsspec_to_pde(spec: PhysicsSpecification) -> PDERepresentation:
    """Convert a PhysicsSpecification to PDERepresentation.

    Uses structured fields only. ``governing_equations`` is not parsed.
    """
    variables = [v["name"] for v in spec.primary_variables]
    dimensions = _DIM_MAP.get(spec.spatial_dimensionality, 1)
    time_scheme = "transient" if spec.time_integration.get("type") == "Transient" else "steady"

    mat_lookup = _build_material_lookup(spec.material_properties)
    terms = _infer_terms(spec.physics_modules, variables, mat_lookup, time_scheme)
    bcs = _convert_bcs(spec.boundary_conditions)
    ics = _convert_ics(spec.initial_conditions)

    warnings: list[str] = []

    return PDERepresentation(
        terms=terms,
        boundary_conditions=bcs,
        initial_conditions=ics,
        time_scheme=time_scheme,
        variables=variables,
        dimensions=dimensions,
        warnings=warnings,
    )


def pde_to_physicsspec(pde: PDERepresentation) -> PhysicsSpecification:
    """Convert a PDERepresentation back to PhysicsSpecification."""
    primary_variables = [{"name": v, "type": "scalar"} for v in pde.variables]

    physics_modules = _infer_physics_modules(pde.terms)

    boundary_conditions = [
        {
            "name": bc.boundary,
            "type": bc.bc_type,
            "variable": bc.variable,
            "value": str(bc.value) if bc.value is not None else "",
        }
        for bc in pde.boundary_conditions
    ]

    initial_conditions = [
        {
            "variable": ic.variable,
            "type": ic.ic_type,
            "value": str(ic.value) if ic.value is not None else "",
        }
        for ic in pde.initial_conditions
    ]

    material_properties = _extract_materials(pde.terms)

    governing_eqs = ", ".join(
        f"{t.operator}({t.variable})" for t in pde.terms
    )

    return PhysicsSpecification(
        problem_description="Converted from PDERepresentation",
        spatial_dimensionality=_DIM_REVERSE.get(pde.dimensions, "1D"),
        modeling_assumptions=[],
        physics_modules=physics_modules,
        primary_variables=primary_variables,
        governing_equations=governing_eqs,
        boundary_conditions=boundary_conditions,
        initial_conditions=initial_conditions,
        material_properties=material_properties,
        time_integration={"type": "Transient" if pde.time_scheme == "transient" else "Steady"},
        numerical_considerations="",
        outputs_and_diagnostics=[],
    )


def _build_material_lookup(materials: list[dict[str, Any]]) -> dict[str, float | str]:
    """Build name -> value mapping from material_properties list."""
    lookup: dict[str, float | str] = {}
    for mat in materials:
        name = mat.get("name", "")
        value = mat.get("value")
        if name and value is not None:
            try:
                lookup[name] = float(value)
            except (ValueError, TypeError):
                lookup[name] = str(value)
    return lookup


def _infer_terms(
    physics_modules: list[str],
    variables: list[str],
    mat_lookup: dict[str, float | str],
    time_scheme: str,
) -> list[PDETerm]:
    """Infer PDE terms from physics_modules and time_scheme."""
    terms: list[PDETerm] = []

    for module in physics_modules:
        module_lower = module.lower().replace("-", "_").replace(" ", "_")
        term_defs = _PHYSICS_MODULE_TERMS.get(module_lower, [])

        for operator, coef_param, severity in term_defs:
            coefficient: float | str | None = None
            if coef_param and coef_param in mat_lookup:
                coefficient = mat_lookup[coef_param]

            for var in variables:
                terms.append(PDETerm(
                    variable=var,
                    operator=operator,
                    coefficient=coefficient,
                    coupled_variable=None,
                    kernel_type=None,
                    severity=severity,
                ))

    if time_scheme == "transient":
        for var in variables:
            terms.append(PDETerm(
                variable=var,
                operator="time_derivative",
                coefficient=None,
                coupled_variable=None,
                kernel_type=None,
                severity="critical",
            ))

    return terms


def _convert_bcs(bc_dicts: list[dict[str, str]]) -> list[BoundaryCondition]:
    """Convert BC dicts from PhysicsSpecification to BoundaryCondition objects."""
    bcs: list[BoundaryCondition] = []
    for bc in bc_dicts:
        raw_value = bc.get("value")
        value: float | str | None = None
        if raw_value:
            try:
                value = float(raw_value)
            except ValueError:
                value = raw_value

        bc_type = bc.get("type", "Dirichlet")
        severity = "high" if bc_type == "Dirichlet" else "medium"

        bcs.append(BoundaryCondition(
            variable=bc.get("variable", ""),
            boundary=bc.get("name", bc.get("boundary", "")),
            bc_type=bc_type,
            value=value,
            moose_bc_class=None,
            severity=severity,
        ))
    return bcs


def _convert_ics(ic_dicts: list[dict[str, str]]) -> list[InitialCondition]:
    """Convert IC dicts from PhysicsSpecification to InitialCondition objects."""
    ics: list[InitialCondition] = []
    for ic in ic_dicts:
        raw_value = ic.get("value")
        value: float | str | None = None
        if raw_value:
            try:
                value = float(raw_value)
            except ValueError:
                value = raw_value

        ics.append(InitialCondition(
            variable=ic.get("variable", ""),
            ic_type=ic.get("type", "constant"),
            value=value,
            severity="medium",
        ))
    return ics


def _infer_physics_modules(terms: list[PDETerm]) -> list[str]:
    """Infer physics module names from PDE terms."""
    modules: set[str] = set()
    for term in terms:
        if term.operator == "diffusion" and term.kernel_type in ("HeatConduction", "ADHeatConduction"):
            modules.add("heat_conduction")
        elif term.operator == "diffusion":
            modules.add("diffusion")
        elif term.operator == "stress_divergence":
            modules.add("solid_mechanics")
    return sorted(modules)


def _extract_materials(terms: list[PDETerm]) -> list[dict[str, Any]]:
    """Extract material properties from PDE terms."""
    materials: list[dict[str, Any]] = []
    seen: set[str] = set()
    for term in terms:
        if isinstance(term.coefficient, (int, float)) and term.kernel_type:
            # Use a descriptive name based on operator
            name = f"{term.operator}_coefficient"
            if name not in seen:
                materials.append({"name": name, "value": term.coefficient})
                seen.add(name)
    return materials
