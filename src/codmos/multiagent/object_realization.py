"""Registry-grounded object realization support.

This module is an execution-control layer, not a semantic verifier. It uses a
frozen MOOSE app registry to expose compact object schemas before generation
and to run mechanical validation/repair as part of the same registry-controlled
workflow.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from codmos.multiagent.moose_registry import MooseObjectSpec, MooseRegistry
from codmos.multiagent.pde.kernel_map import KernelMap
from codmos.multiagent.pde.representation import (
    BoundaryCondition,
    InitialCondition,
    PDERepresentation,
    PDETerm,
)
from codmos.multiagent.validators import MOOSETypeIssue, MOOSETypeValidator

_KERNEL_CONTEXTS = ("Kernels/*", "FVKernels/*", "DGKernels/*", "InterfaceKernels/*")
_BC_CONTEXTS = ("BCs/*", "FVBCs/*")
_IC_CONTEXTS = ("ICs/*",)

_COMMON_NONSEMANTIC_PARAMS = {
    "active",
    "inactive",
    "control_tags",
    "enable",
    "execute_on",
    "matrix_only",
    "use_displaced_mesh",
    "vector_tags",
    "extra_vector_tags",
    "absolute_value_vector_tags",
    "save_in",
    "diag_save_in",
}

_OPERATOR_PREFERENCES: dict[str, tuple[str, ...]] = {
    "diffusion": (
        "Diffusion",
        "MatDiffusion",
        "CoefDiffusion",
        "ADDiffusion",
        "HeatConduction",
        "ADHeatConduction",
    ),
    "time_derivative": (
        "TimeDerivative",
        "CoefTimeDerivative",
        "ADTimeDerivative",
        "HeatConductionTimeDerivative",
        "ADHeatConductionTimeDerivative",
    ),
    "source": ("BodyForce", "MatBodyForce", "ADBodyForce"),
    "reaction": ("CoefReaction", "Reaction", "MatReaction", "ADReaction"),
    "advection": ("ConservativeAdvection", "ADConservativeAdvection"),
    "stress_divergence": ("StressDivergenceTensors", "ADStressDivergenceTensors"),
    "pf_darcy_flux": ("PorousFlowAdvectiveFlux", "PorousFlowFullySaturatedDarcyFlow"),
    "pf_effective_stress": ("PorousFlowEffectiveStressCoupling",),
    "allen_cahn": ("AllenCahn", "ADAllenCahn"),
    "cahn_hilliard": ("CahnHilliard", "ADCahnHilliard"),
    "navier_stokes_mass": ("PINSFVMassAdvection", "INSFVMassAdvection"),
    "navier_stokes_momentum": ("PINSFVMomentumAdvection", "INSFVMomentumAdvection"),
    "coupled_force": ("CoupledForce", "ADCoupledForce"),
}

_HEAT_OPERATOR_PREFERENCES: dict[str, tuple[str, ...]] = {
    "diffusion": (
        "HeatConduction",
        "Diffusion",
        "MatDiffusion",
        "CoefDiffusion",
        "ADHeatConduction",
        "ADDiffusion",
    ),
    "time_derivative": (
        "HeatConductionTimeDerivative",
        "TimeDerivative",
        "ADHeatConductionTimeDerivative",
        "ADTimeDerivative",
    ),
}

_BC_PREFERENCES: dict[str, tuple[str, ...]] = {
    "Dirichlet": ("DirichletBC", "ADDirichletBC", "FunctionDirichletBC"),
    "Neumann": ("NeumannBC", "ADNeumannBC", "FunctionNeumannBC"),
    "Robin": ("ADRobinBC", "ConvectiveFluxBC"),
}

_IC_PREFERENCES: dict[str, tuple[str, ...]] = {
    "constant": ("ConstantIC", "ADConstantIC"),
    "function": ("FunctionIC", "ADFunctionIC"),
}


@dataclass(frozen=True)
class ObjectCandidate:
    """Compact schema for one admissible MOOSE object candidate."""

    object_type: str
    context: str
    required_params: tuple[str, ...] = ()
    semantic_params: tuple[str, ...] = ()
    optional_params: tuple[str, ...] = ()
    coefficient_param: str | None = None
    note: str | None = None

    def format_line(self) -> str:
        pieces = [f"{self.object_type} [{self.context}]"]
        if self.required_params:
            pieces.append(f"required={list(self.required_params)}")
        if self.semantic_params:
            pieces.append(f"semantic={list(self.semantic_params)}")
        if self.coefficient_param:
            pieces.append(f"coefficient_param={self.coefficient_param}")
        if self.optional_params:
            pieces.append(f"optional={list(self.optional_params)}")
        if self.note:
            pieces.append(f"note={self.note}")
        return "; ".join(pieces)


@dataclass(frozen=True)
class ObjectPlanItem:
    """Object-realization candidates for one PDE item."""

    role: str
    description: str
    candidates: tuple[ObjectCandidate, ...]


@dataclass
class ObjectRealizationPlan:
    """Targeted registry plan for a PDERepresentation."""

    items: list[ObjectPlanItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_prompt_section(self) -> str:
        lines = [
            "Frozen MOOSE object-realization plan:",
            "- Use only listed object names for the corresponding PDE item when possible.",
            "- Treat the first listed candidate as the canonical/default realization for that PDE item.",
            "- Prefer non-AD and generic objects unless the PDE item, variable, or existing code explicitly requires AD or heat-transfer-specific objects.",
            "- Do not invent MOOSE object names or parameter names.",
            "- For HeatConductionTimeDerivative, represent rho*cp with density_name and specific_heat material properties; do not write density_specific_heat as a MOOSE parameter.",
            "- For temperature Robin/convective boundaries, prefer ConvectiveHeatFluxBC with T_infinity and heat_transfer_coefficient material properties.",
            "- For ADRobinBC, use the coefficient parameter; do not write alpha, beta, or gamma.",
            "- The registry is PDE-neutral: it does not choose new terms, BC types, coefficients, materials, or time scheme.",
            "- If a semantic value is unknown, keep the PDE specification value or leave an explicit placeholder rather than substituting a default.",
            "- You may add non-physics boilerplate such as [Mesh], [Variables], [Executioner], [Outputs].",
        ]
        if self.warnings:
            lines.append("")
            lines.append("Registry warnings:")
            lines.extend(f"- {warning}" for warning in self.warnings[:8])
        lines.append("")
        lines.append("Required boilerplate:")
        lines.append("- [Mesh]: use type=GeneratedMesh directly, or [Mesh/*] type=GeneratedMeshGenerator.")
        lines.append("- [Variables]: define every PDE variable.")
        lines.append("- [Executioner]: use Steady for steady problems, Transient for transient problems.")
        lines.append("- [Executioner]: nl_abs_tol=1e-8 is acceptable solver boilerplate.")
        lines.append("- [Outputs]: exodus=true is acceptable boilerplate.")
        lines.append("")
        lines.append("PDE-item object schemas:")
        for index, item in enumerate(self.items, start=1):
            lines.append(f"{index}. {item.role}: {item.description}")
            if not item.candidates:
                lines.append("   - no registry-backed candidates found")
                continue
            for candidate in item.candidates:
                lines.append(f"   - {candidate.format_line()}")
        return "\n".join(lines)


@dataclass(frozen=True)
class RegistryRepairSummary:
    """Mechanical registry validation/repair summary."""

    before_passed: bool
    after_passed: bool
    changed: bool
    repaired_code: str
    changes: tuple[str, ...]
    issue_kinds: dict[str, int]
    issue_count: int
    syntax_repair_reason: str | None = None


def build_object_plan(
    pde: PDERepresentation,
    registry: MooseRegistry,
    *,
    kernel_map: KernelMap | None = None,
    max_candidates: int = 4,
) -> ObjectRealizationPlan:
    """Build a compact registry-grounded plan for each PDE item."""
    kernel_map = kernel_map or KernelMap()
    plan = ObjectRealizationPlan()

    for term in pde.terms:
        candidates = _term_candidates(term, registry, kernel_map, max_candidates)
        if not candidates:
            plan.warnings.append(f"No registered object candidates for term {term.operator!r}.")
        plan.items.append(ObjectPlanItem(
            role=f"term:{term.operator}:{term.variable}",
            description=_describe_term(term),
            candidates=tuple(candidates),
        ))

    for bc in pde.boundary_conditions:
        candidates = _bc_candidates(bc, registry, kernel_map, max_candidates)
        if not candidates:
            plan.warnings.append(f"No registered object candidates for BC {bc.bc_type!r}.")
        plan.items.append(ObjectPlanItem(
            role=f"bc:{bc.bc_type}:{bc.variable}:{bc.boundary}",
            description=_describe_bc(bc),
            candidates=tuple(candidates),
        ))

    for ic in pde.initial_conditions:
        candidates = _ic_candidates(ic, registry, kernel_map, max_candidates)
        if not candidates:
            plan.warnings.append(f"No registered object candidates for IC {ic.ic_type!r}.")
        plan.items.append(ObjectPlanItem(
            role=f"ic:{ic.ic_type}:{ic.variable}",
            description=_describe_ic(ic),
            candidates=tuple(candidates),
        ))

    return plan


def validate_and_repair_code(
    code: str,
    registry: MooseRegistry,
    *,
    syntax_repair_reason: str | None = None,
) -> RegistryRepairSummary:
    """Run L2 validation and conservative deterministic repair."""
    validator = MOOSETypeValidator(registry)
    repair = validator.repair(code)
    issues = repair.before.issues
    issue_kinds = Counter(issue.kind for issue in issues)
    return RegistryRepairSummary(
        before_passed=repair.before.passed,
        after_passed=repair.after.passed,
        changed=repair.changed,
        repaired_code=repair.repaired_code,
        changes=tuple(repair.changes),
        issue_kinds=dict(issue_kinds),
        issue_count=len(issues),
        syntax_repair_reason=syntax_repair_reason,
    )


def format_registry_issues(issues: list[MOOSETypeIssue], *, limit: int = 12) -> str:
    """Format L2 issues for constrained mechanical repair prompts."""
    if not issues:
        return "No mechanical registry violations."

    lines: list[str] = []
    for issue in issues[:limit]:
        suggestion = f" suggestions={issue.suggestions[:4]}" if issue.suggestions else ""
        parameter = f" parameter={issue.parameter}" if issue.parameter else ""
        object_type = f" object={issue.object_type}" if issue.object_type else ""
        lines.append(
            f"- {issue.kind} at {issue.path}{object_type}{parameter}: "
            f"{issue.message}{suggestion}"
        )
    if len(issues) > limit:
        lines.append(f"- ... {len(issues) - limit} more issue(s) omitted")
    return "\n".join(lines)


def _term_candidates(
    term: PDETerm,
    registry: MooseRegistry,
    kernel_map: KernelMap,
    max_candidates: int,
) -> list[ObjectCandidate]:
    mappings = list(getattr(kernel_map, "_kernels", {}).values())
    matched = [mapping for mapping in mappings if mapping.operator == term.operator]
    by_name = {mapping.kernel_class: mapping for mapping in matched}

    ordered_names: list[str] = []
    if term.kernel_type:
        ordered_names.append(term.kernel_type)
    ordered_names.extend(_operator_preferences_for_term(term))
    ordered_names.extend(mapping.kernel_class for mapping in matched)
    ordered_names = _dedupe(ordered_names)

    candidates: list[ObjectCandidate] = []
    for name in ordered_names:
        spec = registry.get(name)
        if spec is None:
            continue
        context = _first_context(spec, _KERNEL_CONTEXTS)
        if context is None:
            continue
        mapping = by_name.get(name)
        coefficient_param = mapping.coefficient_param if mapping else None
        display_coefficient_param = coefficient_param
        semantic_params = _semantic_params(
            spec,
            (
                "variable",
                "v",
                "component",
                "displacements",
                coefficient_param,
                "density_name",
                "specific_heat",
                "thermal_conductivity",
                "coef",
                "Coefficient",
                "coefficient",
                "rate",
                "value",
            ),
        )
        note = None
        if term.coefficient not in (None, 1.0, "1.0") and coefficient_param:
            note = "preserve exact coefficient via this parameter or matching material property"
        if coefficient_param == "density_specific_heat":
            display_coefficient_param = None
            note = (
                "preserve rho*cp by defining density_name and specific_heat "
                "material properties; do not use density_specific_heat as a parameter"
            )
        candidates.append(_candidate_from_spec(
            spec,
            context,
            semantic_params=semantic_params,
            coefficient_param=display_coefficient_param,
            note=note,
        ))
        if len(candidates) >= max_candidates:
            break
    return candidates


def _operator_preferences_for_term(term: PDETerm) -> tuple[str, ...]:
    if _is_heat_term(term):
        return _HEAT_OPERATOR_PREFERENCES.get(
            term.operator,
            _OPERATOR_PREFERENCES.get(term.operator, ()),
        )
    return _OPERATOR_PREFERENCES.get(term.operator, ())


def _is_heat_term(term: PDETerm) -> bool:
    if term.kernel_type and "heat" in term.kernel_type.lower():
        return True
    return _is_heat_variable(term.variable)


def _is_heat_variable(variable: str) -> bool:
    return variable.lower() in {"t", "temp", "temperature"}


def _bc_candidates(
    bc: BoundaryCondition,
    registry: MooseRegistry,
    kernel_map: KernelMap,
    max_candidates: int,
) -> list[ObjectCandidate]:
    normalized = _normalize_bc_type(bc.bc_type)
    mappings = list(getattr(kernel_map, "_bcs", {}).values())
    matched = [mapping for mapping in mappings if mapping.bc_type == normalized]
    by_name = {mapping.bc_class: mapping for mapping in matched}

    ordered_names: list[str] = []
    if bc.moose_bc_class:
        ordered_names.append(bc.moose_bc_class)
    ordered_names.extend(_bc_preferences_for_bc(bc, normalized))
    ordered_names.extend(mapping.bc_class for mapping in matched)
    ordered_names = _dedupe(ordered_names)

    candidates: list[ObjectCandidate] = []
    for name in ordered_names:
        spec = registry.get(name)
        if spec is None:
            continue
        context = _first_context(spec, _BC_CONTEXTS)
        if context is None:
            continue
        mapping = by_name.get(name)
        value_param = mapping.value_param if mapping else None
        semantic_params = _semantic_params(
            spec,
            ("variable", "boundary", value_param, "value", "function", "T_infinity", "heat_transfer_coefficient"),
        )
        candidates.append(_candidate_from_spec(
            spec,
            context,
            semantic_params=semantic_params,
            coefficient_param=value_param,
        ))
        if len(candidates) >= max_candidates:
            break
    return candidates


def _bc_preferences_for_bc(
    bc: BoundaryCondition,
    normalized: str,
) -> tuple[str, ...]:
    if normalized == "Robin" and _is_heat_variable(bc.variable):
        return ("ConvectiveHeatFluxBC", "ADRobinBC", "ConvectiveFluxBC")
    return _BC_PREFERENCES.get(normalized, ())


def _ic_candidates(
    ic: InitialCondition,
    registry: MooseRegistry,
    kernel_map: KernelMap,
    max_candidates: int,
) -> list[ObjectCandidate]:
    normalized = (ic.ic_type or "constant").lower()
    mappings = list(getattr(kernel_map, "_ics", {}).values())
    matched = [mapping for mapping in mappings if mapping.ic_type.lower() == normalized]
    by_name = {mapping.ic_class: mapping for mapping in matched}

    ordered_names: list[str] = []
    ordered_names.extend(_IC_PREFERENCES.get(normalized, ()))
    ordered_names.extend(mapping.ic_class for mapping in matched)
    ordered_names = _dedupe(ordered_names)

    candidates: list[ObjectCandidate] = []
    for name in ordered_names:
        spec = registry.get(name)
        if spec is None:
            continue
        context = _first_context(spec, _IC_CONTEXTS)
        if context is None:
            continue
        mapping = by_name.get(name)
        value_param = mapping.value_param if mapping else None
        semantic_params = _semantic_params(spec, ("variable", value_param, "value", "function"))
        candidates.append(_candidate_from_spec(
            spec,
            context,
            semantic_params=semantic_params,
            coefficient_param=value_param,
        ))
        if len(candidates) >= max_candidates:
            break
    return candidates


def _candidate_from_spec(
    spec: MooseObjectSpec,
    context: str,
    *,
    semantic_params: tuple[str, ...],
    coefficient_param: str | None = None,
    note: str | None = None,
) -> ObjectCandidate:
    required = tuple(
        name
        for name, param in spec.params.items()
        if param.required and name != "type"
    )
    optional = tuple(
        name
        for name in spec.params
        if name not in required
        and name != "type"
        and name not in semantic_params
        and name in _COMMON_NONSEMANTIC_PARAMS
    )[:5]
    return ObjectCandidate(
        object_type=spec.type_name,
        context=context,
        required_params=required,
        semantic_params=semantic_params,
        optional_params=optional,
        coefficient_param=coefficient_param,
        note=note,
    )


def _first_context(spec: MooseObjectSpec, preferred: tuple[str, ...]) -> str | None:
    for context in preferred:
        if context in spec.contexts:
            return context
    return sorted(spec.contexts)[0] if spec.contexts else None


def _semantic_params(spec: MooseObjectSpec, names: tuple[str | None, ...]) -> tuple[str, ...]:
    return tuple(name for name in _dedupe([n for n in names if n]) if name in spec.params)


def _dedupe(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value is None or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _normalize_bc_type(value: str) -> str:
    lowered = value.lower()
    if lowered.startswith("dirichlet"):
        return "Dirichlet"
    if lowered.startswith("neumann"):
        return "Neumann"
    if lowered.startswith("robin"):
        return "Robin"
    return value.capitalize()


def _describe_term(term: PDETerm) -> str:
    coeff = f", coefficient={term.coefficient}" if term.coefficient is not None else ""
    coupled = f", coupled_variable={term.coupled_variable}" if term.coupled_variable else ""
    return f"{term.operator} on variable {term.variable}{coeff}{coupled}"


def _describe_bc(bc: BoundaryCondition) -> str:
    value = f", value={bc.value}" if bc.value is not None else ""
    return f"{bc.bc_type} on boundary {bc.boundary} for variable {bc.variable}{value}"


def _describe_ic(ic: InitialCondition) -> str:
    value = f", value={ic.value}" if ic.value is not None else ""
    return f"{ic.ic_type} initial condition for variable {ic.variable}{value}"
