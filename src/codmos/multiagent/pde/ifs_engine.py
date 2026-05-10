"""IFS (Intent Fidelity Score) computation engine.

Layer 2 — imports Layer 1 (representation, boundary_matcher).

Compares two PDERepresentations across five dimensions (terms,
coefficients, BCs, ICs, time scheme) and produces a severity-weighted
fidelity score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from codmos.multiagent.pde.boundary_matcher import match_boundaries
from codmos.multiagent.pde.representation import (
    BoundaryCondition,
    InitialCondition,
    PDERepresentation,
    PDETerm,
)

SEVERITY_WEIGHTS: dict[str, float] = {
    # Schema v3.0 (6-level, expert-defined from collaborator's kernel_map)
    "highest": 4.0,
    "very_high": 3.0,
    "high": 2.0,
    "medium": 0.7,
    "medium_low": 1.0,
    "low": 0.5,
    # Legacy aliases (backward compatibility with old GT/test data)
    "critical": 4.0,
}

_EPSILON = 1e-12
_MCS_KERNEL_MAP_CACHE: Any | None = None

_BC_COEFFICIENT_PARAMS: frozenset[str] = frozenset({
    "alpha",
    "coef",
    "coefficient",
    "flux",
    "heat_transfer_coefficient",
    "heat_transfer_coefficient_dT",
    "heat_transfer_coefficient_functor",
    "penalty",
    "T_infinity",
    "T_infinity_functor",
    "transfer_coefficient",
})
_BC_COEFFICIENT_ALIASES: dict[str, str] = {
    "heat_transfer_coefficient_functor": "heat_transfer_coefficient",
    "T_infinity_functor": "T_infinity",
}
_GENERIC_BC_VALUE_PARAMS: frozenset[str] = frozenset({
    "value",
    "values",
    "invalue",
    "inside",
    "outside",
})
_MATERIAL_MODEL_SIGNATURES: dict[str, tuple[str, str]] = {
    "ADComputeFiniteStrainElasticStress": ("stress_model", "finite_strain_elastic"),
    "ADComputeIsotropicElasticityTensor": ("elasticity_tensor", "isotropic"),
    "ADComputeLinearElasticStress": ("stress_model", "linear_elastic"),
    "ADComputeMultipleInelasticStress": ("stress_model", "multiple_inelastic"),
    "ADComputePlaneSmallStrain": ("strain_model", "plane_small_strain"),
    "ADComputeSmallStrain": ("strain_model", "small_strain"),
    "ADIsotropicPlasticityStressUpdate": ("inelastic_model", "isotropic_plasticity"),
    "ComputeElasticityTensor": ("elasticity_tensor", "generic"),
    "ComputeEigenstrain": ("eigenstrain_model", "generic"),
    "ComputeFiniteStrain": ("strain_model", "finite_strain"),
    "ComputeFiniteStrainElasticStress": ("stress_model", "finite_strain_elastic"),
    "ComputeIsotropicElasticityTensor": ("elasticity_tensor", "isotropic"),
    "ComputeLinearElasticPFFractureStress": ("stress_model", "linear_elastic_pf_fracture"),
    "ComputeLinearElasticStress": ("stress_model", "linear_elastic"),
    "ComputeMultipleInelasticStress": ("stress_model", "multiple_inelastic"),
    "ComputePlaneSmallStrain": ("strain_model", "plane_small_strain"),
    "ComputeSmallStrain": ("strain_model", "small_strain"),
    "ComputeThermalExpansionEigenstrain": ("eigenstrain_model", "thermal_expansion"),
    "IsotropicPlasticityStressUpdate": ("inelastic_model", "isotropic_plasticity"),
    "PowerLawCreepStressUpdate": ("inelastic_model", "power_law_creep"),
}


@dataclass
class IFSCheckpoint:
    """A single comparison checkpoint."""

    dimension: str
    description: str
    passed: bool
    severity: str
    detail: str | None = None


@dataclass
class GroundTruthAnnotation:
    """Wraps a PDERepresentation with evaluation-specific metadata."""

    pde: PDERepresentation
    value_modes: dict[str, str] = field(default_factory=dict)
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class IFSResult:
    """Result of IFS computation between reference and candidate PDE."""

    ifs_score: float
    checkpoints: list[IFSCheckpoint]
    num_checkpoints: int
    num_passed: int
    num_failed: int
    total_severity_weight: float
    ifs_term: float
    ifs_coeff: float
    ifs_bc: float
    ifs_ic: float
    ifs_time: float
    raw_loss: float
    variable_mapping: dict[str, str] = field(default_factory=dict)
    """Mapping from candidate variable names to reference variable names.
    Empty if no alignment was needed (names already matched)."""


def compute_ifs(
    reference: PDERepresentation | GroundTruthAnnotation,
    candidate: PDERepresentation,
    coeff_tolerance: float = 0.1,
) -> IFSResult:
    """Compare two PDE representations and compute IFS."""
    if isinstance(reference, GroundTruthAnnotation):
        gt = reference
        ref_pde = reference.pde
    else:
        gt = None
        ref_pde = reference

    # Align candidate variable names to reference before comparison.
    # LLMs choose reasonable but different names (e.g. "u" vs "T").
    # Physics-level comparison should not penalize naming conventions.
    aligned_candidate, var_mapping = _align_variables(ref_pde, candidate)

    checkpoints: list[IFSCheckpoint] = []

    _compare_terms(ref_pde, aligned_candidate, checkpoints)
    _compare_coefficients(ref_pde, aligned_candidate, checkpoints, coeff_tolerance, gt)
    _compare_bcs(ref_pde, aligned_candidate, checkpoints, coeff_tolerance, gt)
    _compare_ics(ref_pde, aligned_candidate, checkpoints)
    _compare_time_scheme(ref_pde, aligned_candidate, checkpoints)

    result = _build_result(checkpoints)
    result.variable_mapping = var_mapping
    return result


def _align_variables(
    ref: PDERepresentation,
    cand: PDERepresentation,
) -> tuple[PDERepresentation, dict[str, str]]:
    """Remap candidate variable names to match reference by operator structure.

    Variable naming is convention, not physics: "u" and "T" can encode the
    same equation.  This function finds the best 1:1 mapping between
    ref variables and candidate variables by comparing their operator
    signatures (the multiset of operators acting on each variable).

    When names already match, or no unambiguous mapping exists, the
    candidate is returned unchanged.

    Returns (aligned_candidate, var_mapping) where var_mapping is
    {candidate_name: ref_name}. Empty dict if no alignment was needed.
    """
    # If ALL variable names already match, skip alignment
    ref_vars = set(ref.variables)
    cand_vars = set(cand.variables)
    if ref_vars == cand_vars:
        return cand, {}  # names already identical — no remap needed

    if not ref_vars or not cand_vars:
        return cand, {}

    # Build operator signature per variable: var → sorted tuple of operators
    def _signature(pde: PDERepresentation) -> dict[str, tuple[str, ...]]:
        sig: dict[str, list[str]] = {}
        for t in pde.terms:
            sig.setdefault(t.variable, []).append(t.operator)
        return {v: tuple(sorted(ops)) for v, ops in sig.items()}

    ref_sigs = _signature(ref)
    cand_sigs = _signature(cand)

    # 1:1 matching with tie-breaking for isomorphic signatures.
    var_map: dict[str, str] = {}  # cand_name → ref_name
    used_ref: set[str] = set()
    used_cand: set[str] = set()

    # Pass 1: unique signature matches (no ambiguity)
    for cand_v, cand_sig in cand_sigs.items():
        matches = [r for r, s in ref_sigs.items() if s == cand_sig and r not in used_ref]
        if len(matches) == 1:
            var_map[cand_v] = matches[0]
            used_ref.add(matches[0])
            used_cand.add(cand_v)

    # Pass 2: tie-break ambiguous matches using BC boundary+value
    remaining_cand = {v: s for v, s in cand_sigs.items() if v not in used_cand}
    remaining_ref = {v: s for v, s in ref_sigs.items() if v not in used_ref}

    if remaining_cand and remaining_ref:
        # Build BC fingerprint: var → set of (boundary_normalized, value)
        def _bc_fingerprint(pde: PDERepresentation) -> dict[str, set[tuple]]:
            fp: dict[str, set[tuple]] = {}
            for bc in pde.boundary_conditions:
                fp.setdefault(bc.variable, set()).add(
                    (bc.boundary.strip("'\"").lower(), bc.value)
                )
            return fp

        ref_bc_fp = _bc_fingerprint(ref)
        cand_bc_fp = _bc_fingerprint(cand)

        for cand_v in list(remaining_cand):
            cand_sig = remaining_cand[cand_v]
            candidates = [r for r, s in remaining_ref.items() if s == cand_sig]
            if not candidates:
                continue

            # Score each candidate by BC overlap
            cand_bcs = cand_bc_fp.get(cand_v, set())
            best_ref, best_score = None, -1
            for ref_v in candidates:
                ref_bcs = ref_bc_fp.get(ref_v, set())
                score = len(cand_bcs & ref_bcs)
                if score > best_score:
                    best_score = score
                    best_ref = ref_v

            if best_ref is not None:
                var_map[cand_v] = best_ref
                used_ref.add(best_ref)
                used_cand.add(cand_v)
                remaining_ref.pop(best_ref, None)

    if not var_map:
        # No signature matches — try positional alignment if counts match
        if len(ref.variables) == len(cand.variables):
            for r_v, c_v in zip(ref.variables, cand.variables, strict=True):
                var_map[c_v] = r_v
        else:
            return cand, {}  # can't align

    def _remap(name: str) -> str:
        return var_map.get(name, name)

    # Build remapped candidate
    new_terms = [
        PDETerm(
            variable=_remap(t.variable),
            operator=t.operator,
            coefficient=t.coefficient,
            coupled_variable=t.coupled_variable,
            kernel_type=t.kernel_type,
            severity=t.severity,
        )
        for t in cand.terms
    ]
    new_bcs = [
        BoundaryCondition(
            variable=_remap(bc.variable),
            boundary=bc.boundary,
            bc_type=bc.bc_type,
            value=bc.value,
            moose_bc_class=bc.moose_bc_class,
            severity=bc.severity,
        )
        for bc in cand.boundary_conditions
    ]
    new_ics = [
        InitialCondition(
            variable=_remap(ic.variable),
            ic_type=ic.ic_type,
            value=ic.value,
            severity=ic.severity,
        )
        for ic in cand.initial_conditions
    ]

    return PDERepresentation(
        terms=new_terms,
        boundary_conditions=new_bcs,
        initial_conditions=new_ics,
        time_scheme=cand.time_scheme,
        variables=[_remap(v) for v in cand.variables],
        dimensions=cand.dimensions,
        unresolved_kernels=cand.unresolved_kernels,
        unresolved_coefficients=cand.unresolved_coefficients,
        warnings=cand.warnings,
    ), var_map


# Physics context for term violations — describes what the operator does
# without naming specific MOOSE kernels
_OPERATOR_PHYSICS_HINT: dict[str, str] = {
    "diffusion": "This term implements a second-order spatial operator (nabla·(D nabla u)) as a PDE residual term.",
    "time_derivative": "This term implements the time derivative (du/dt), required for transient problems.",
    "source": "This term implements a volumetric source/sink (body force, heat generation, etc.).",
    "reaction": "This term implements a zero-order reaction (proportional to the variable itself).",
    "advection": "This term implements first-order transport/convection (v·nabla u).",
    "stress_divergence": "This term implements stress equilibrium (nabla·sigma) for solid mechanics.",
    "coupled_force": "This term implements a coupling source driven by another variable.",
    "inertia": "This term implements inertial/acceleration effects (rho * d²u/dt²).",
    "curl_curl": "This term implements the curl-curl operator for electromagnetics.",
    "allen_cahn": "This term implements Allen-Cahn phase-field evolution (interfacial + bulk free energy).",
    "cahn_hilliard": "This term implements Cahn-Hilliard phase separation (chemical potential driven diffusion).",
    "ns_continuity": "This term enforces the incompressibility constraint (div v = 0) in the Navier-Stokes equations.",
    "ns_viscous": "This term implements viscous stress diffusion (mu laplacian v) in the Navier-Stokes momentum equation.",
    "ns_pressure": "This term implements the pressure gradient (-grad p) in the Navier-Stokes momentum equation.",
    "pf_darcy_flux": "This term implements Darcy's law flux for fluid flow in porous media.",
    "pf_effective_stress": "This term implements Biot effective stress coupling in poromechanics (THM coupling).",
}


def format_violations_for_code(ifs_result: IFSResult) -> str:
    """Format IFS violations using the LLM's original variable names.

    Checkpoints use reference (GT) variable names due to alignment.
    This function reverse-maps them back to the candidate's own names
    so the LLM can find and fix the relevant code.

    For missing-term violations, adds physics context describing what
    the operator does (without naming specific MOOSE kernels).
    """
    if not ifs_result.variable_mapping:
        inv = {}
    else:
        inv = {v: k for k, v in ifs_result.variable_mapping.items()}

    lines = []
    for cp in ifs_result.checkpoints:
        if cp.passed:
            continue
        desc = cp.description
        detail = cp.detail or ""
        # Replace ref variable names with code variable names
        for ref_name, code_name in inv.items():
            desc = desc.replace(f"variable {ref_name}", f"variable {code_name}")
            desc = desc.replace(f"for {ref_name}", f"for {code_name}")
            detail = detail.replace(f"variable {ref_name}", f"variable {code_name}")

        line = f"- [{cp.dimension}] {desc}" + (f": {detail}" if detail else "")

        # Add physics hint for missing term violations
        if cp.dimension == "term" and detail == "Missing in candidate":
            # Extract operator from description like "diffusion for variable T"
            op = desc.split(" for ")[0].strip() if " for " in desc else ""
            hint = _OPERATOR_PHYSICS_HINT.get(op, "")
            if hint:
                line += f"\n  Hint: {hint}"

        lines.append(line)
    return "\n".join(lines) if lines else "No violations."


def _compare_terms(
    ref: PDERepresentation,
    cand: PDERepresentation,
    cps: list[IFSCheckpoint],
) -> None:
    ref_pairs = {(t.variable, t.operator) for t in ref.terms}
    cand_pairs = {(t.variable, t.operator) for t in cand.terms}

    for var, op in ref_pairs:
        if (var, op) in cand_pairs:
            cps.append(IFSCheckpoint(
                dimension="term",
                description=f"{op} for variable {var}",
                passed=True,
                severity=_get_term_severity(ref, var, op),
            ))
        else:
            cps.append(IFSCheckpoint(
                dimension="term",
                description=f"{op} for variable {var}",
                passed=False,
                severity=_get_term_severity(ref, var, op),
                detail="Missing in candidate",
            ))

    for var, op in cand_pairs - ref_pairs:
        cps.append(IFSCheckpoint(
            dimension="term",
            description=f"{op} for variable {var}",
            passed=False,
            severity="medium",
            detail="Extra term not in reference",
        ))


def _compare_coefficients(
    ref: PDERepresentation,
    cand: PDERepresentation,
    cps: list[IFSCheckpoint],
    tolerance: float,
    gt: GroundTruthAnnotation | None,
) -> None:
    ref_by_key = {(t.variable, t.operator): t for t in ref.terms}
    cand_by_key = {(t.variable, t.operator): t for t in cand.terms}

    for key, ref_term in ref_by_key.items():
        if key not in cand_by_key:
            continue  # term mismatch already recorded
        cand_term = cand_by_key[key]

        checkpoint_id = f"{key[0]}:{key[1]}:coeff"

        if gt and gt.value_modes.get(checkpoint_id) == "unspecified":
            continue  # skip

        if ref_term.coefficient is None and cand_term.coefficient is None:
            continue  # both None, nothing to compare

        if ref_term.coefficient is None or cand_term.coefficient is None:
            cps.append(IFSCheckpoint(
                dimension="coefficient",
                description=f"coefficient of {key[1]} for {key[0]}",
                passed=False,
                severity="medium",
                detail=f"ref={ref_term.coefficient}, cand={cand_term.coefficient}",
            ))
            continue

        ref_val = ref_term.coefficient if isinstance(ref_term.coefficient, (int, float)) else None
        cand_val = cand_term.coefficient if isinstance(cand_term.coefficient, (int, float)) else None

        if ref_val is None or cand_val is None:
            # String coefficients — exact match
            passed = str(ref_term.coefficient) == str(cand_term.coefficient)
            cps.append(IFSCheckpoint(
                dimension="coefficient",
                description=f"coefficient of {key[1]} for {key[0]}",
                passed=passed,
                severity="medium",
                detail=None if passed else f"ref={ref_term.coefficient}, cand={cand_term.coefficient}",
            ))
            continue

        if gt and gt.value_modes.get(checkpoint_id) == "range":
            lo, hi = gt.ranges[checkpoint_id]
            passed = lo <= cand_val <= hi
        else:
            rel_error = abs(ref_val - cand_val) / max(abs(ref_val), _EPSILON)
            passed = rel_error <= tolerance

        cps.append(IFSCheckpoint(
            dimension="coefficient",
            description=f"coefficient of {key[1]} for {key[0]}",
            passed=passed,
            severity="medium",
            detail=None if passed else f"ref={ref_val}, cand={cand_val}",
        ))


def _compare_bcs(
    ref: PDERepresentation,
    cand: PDERepresentation,
    cps: list[IFSCheckpoint],
    tolerance: float,
    gt: GroundTruthAnnotation | None,
) -> None:
    matches = match_boundaries(ref.boundary_conditions, cand.boundary_conditions)

    for ref_bc, cand_bc, _conf in matches:
        if cand_bc is None:
            cps.append(IFSCheckpoint(
                dimension="bc",
                description=f"{ref_bc.bc_type} BC on {ref_bc.boundary} for {ref_bc.variable}",
                passed=False,
                severity=ref_bc.severity,
                detail="Missing in candidate",
            ))
            continue

        # Type match
        type_match = ref_bc.bc_type == cand_bc.bc_type
        cps.append(IFSCheckpoint(
            dimension="bc",
            description=f"BC type on {ref_bc.boundary} for {ref_bc.variable}",
            passed=type_match,
            severity=ref_bc.severity,
            detail=None if type_match else f"ref={ref_bc.bc_type}, cand={cand_bc.bc_type}",
        ))

        # Value match (if both have numeric values)
        if ref_bc.value is not None and cand_bc.value is not None:
            ref_val = ref_bc.value if isinstance(ref_bc.value, (int, float)) else None
            cand_val = cand_bc.value if isinstance(cand_bc.value, (int, float)) else None

            if ref_val is not None and cand_val is not None:
                checkpoint_id = f"{ref_bc.variable}:{ref_bc.boundary}:bc_val"

                if gt and gt.value_modes.get(checkpoint_id) == "unspecified":
                    pass  # skip
                elif gt and gt.value_modes.get(checkpoint_id) == "range":
                    lo, hi = gt.ranges[checkpoint_id]
                    passed = lo <= cand_val <= hi
                    cps.append(IFSCheckpoint(
                        dimension="bc",
                        description=f"BC value on {ref_bc.boundary} for {ref_bc.variable}",
                        passed=passed,
                        severity="medium",
                        detail=None if passed else f"ref={ref_val}, cand={cand_val}",
                    ))
                else:
                    rel_error = abs(ref_val - cand_val) / max(abs(ref_val), _EPSILON)
                    passed = rel_error <= tolerance
                    cps.append(IFSCheckpoint(
                        dimension="bc",
                        description=f"BC value on {ref_bc.boundary} for {ref_bc.variable}",
                        passed=passed,
                        severity="medium",
                        detail=None if passed else f"ref={ref_val}, cand={cand_val}",
                    ))

    # Extra BCs in candidate
    matched_cands = {id(r[1]) for r in matches if r[1] is not None}
    for cand_bc in cand.boundary_conditions:
        if id(cand_bc) not in matched_cands:
            cps.append(IFSCheckpoint(
                dimension="bc",
                description=f"Extra BC on {cand_bc.boundary} for {cand_bc.variable}",
                passed=False,
                severity="medium",
                detail="Not in reference",
            ))


def _compare_ics(
    ref: PDERepresentation,
    cand: PDERepresentation,
    cps: list[IFSCheckpoint],
) -> None:
    ref_by_var = {ic.variable: ic for ic in ref.initial_conditions}
    cand_by_var = {ic.variable: ic for ic in cand.initial_conditions}

    for var, ref_ic in ref_by_var.items():
        if var not in cand_by_var:
            cps.append(IFSCheckpoint(
                dimension="ic",
                description=f"IC for variable {var}",
                passed=False,
                severity=ref_ic.severity,
                detail="Missing in candidate",
            ))
            continue

        cand_ic = cand_by_var[var]
        type_match = ref_ic.ic_type == cand_ic.ic_type
        value_match = True
        if isinstance(ref_ic.value, (int, float)) and isinstance(cand_ic.value, (int, float)):
            value_match = abs(ref_ic.value - cand_ic.value) < _EPSILON
        elif ref_ic.value != cand_ic.value:
            value_match = False

        passed = type_match and value_match
        cps.append(IFSCheckpoint(
            dimension="ic",
            description=f"IC for variable {var}",
            passed=passed,
            severity=ref_ic.severity,
            detail=None if passed else f"ref=({ref_ic.ic_type}, {ref_ic.value}), cand=({cand_ic.ic_type}, {cand_ic.value})",
        ))


def _compare_time_scheme(
    ref: PDERepresentation,
    cand: PDERepresentation,
    cps: list[IFSCheckpoint],
) -> None:
    if ref.time_scheme == cand.time_scheme:
        return  # match — no checkpoint needed
    elif ref.time_scheme == "transient" and cand.time_scheme == "steady":
        cps.append(IFSCheckpoint(
            dimension="time",
            description="time scheme",
            passed=False,
            severity="critical",
            detail="Reference is transient but candidate is steady (drops time dependence)",
        ))
    else:
        cps.append(IFSCheckpoint(
            dimension="time",
            description="time scheme",
            passed=False,
            severity="medium",
            detail="Reference is steady but candidate is transient (adds unnecessary time dependence)",
        ))


def _get_term_severity(pde: PDERepresentation, var: str, op: str) -> str:
    for t in pde.terms:
        if t.variable == var and t.operator == op:
            return t.severity
    return "medium"


def _build_result(checkpoints: list[IFSCheckpoint]) -> IFSResult:
    if not checkpoints:
        return IFSResult(
            ifs_score=1.0,
            checkpoints=[],
            num_checkpoints=0,
            num_passed=0,
            num_failed=0,
            total_severity_weight=0.0,
            ifs_term=1.0,
            ifs_coeff=1.0,
            ifs_bc=1.0,
            ifs_ic=1.0,
            ifs_time=1.0,
            raw_loss=0.0,
        )

    total_weight = sum(SEVERITY_WEIGHTS.get(cp.severity, 1.0) for cp in checkpoints)
    failed_weight = sum(
        SEVERITY_WEIGHTS.get(cp.severity, 1.0) for cp in checkpoints if not cp.passed
    )

    ifs_score = 1.0 - failed_weight / total_weight if total_weight > 0 else 1.0

    num_passed = sum(1 for cp in checkpoints if cp.passed)
    num_failed = sum(1 for cp in checkpoints if not cp.passed)

    return IFSResult(
        ifs_score=ifs_score,
        checkpoints=checkpoints,
        num_checkpoints=len(checkpoints),
        num_passed=num_passed,
        num_failed=num_failed,
        total_severity_weight=total_weight,
        ifs_term=_dimensional_score(checkpoints, "term"),
        ifs_coeff=_dimensional_score(checkpoints, "coefficient"),
        ifs_bc=_dimensional_score(checkpoints, "bc"),
        ifs_ic=_dimensional_score(checkpoints, "ic"),
        ifs_time=_dimensional_score(checkpoints, "time"),
        raw_loss=failed_weight,
    )


def _dimensional_score(checkpoints: list[IFSCheckpoint], dimension: str) -> float:
    dim_cps = [cp for cp in checkpoints if cp.dimension == dimension]
    if not dim_cps:
        return 1.0
    total = sum(SEVERITY_WEIGHTS.get(cp.severity, 1.0) for cp in dim_cps)
    failed = sum(SEVERITY_WEIGHTS.get(cp.severity, 1.0) for cp in dim_cps if not cp.passed)
    return 1.0 - failed / total if total > 0 else 1.0


# ---------------------------------------------------------------------------
# Material Consistency Score (MCS) — supplementary to IFS
# ---------------------------------------------------------------------------

@dataclass
class MaterialConsistencyResult:
    """Result of material block comparison between reference and candidate."""

    score: float
    total_properties: int
    matched_properties: int
    mismatched: list[dict[str, object]]


@dataclass(frozen=True)
class _MaterialFact:
    """A material-relevant coefficient fact used by MCS."""

    key: str
    value: float | str
    source: str
    detail: dict[str, object]


def compute_material_consistency(
    ref_code: str,
    cand_code: str,
    tolerance: float = 0.1,
) -> MaterialConsistencyResult:
    """Compare material/constitutive facts between two MOOSE input files.

    This supplementary check catches coefficient/material mismatches without
    modifying IFS.  It compares material-backed coefficients as they are
    consumed by kernels, BC coefficient parameters, constitutive material model
    signatures, and unmatched constitutive material parameters such as Young's
    modulus or plastic hardening constants.

    Returns a MaterialConsistencyResult with a score in [0, 1].
    """
    from codmos.multiagent.validators.hit_parser import load

    ref_root = load(ref_code)
    cand_root = load(cand_code)

    ref_facts = _extract_material_facts(ref_root, include_direct_kernel_facts=False)
    cand_facts = _extract_material_facts(cand_root, include_direct_kernel_facts=True)

    if not ref_facts:
        return MaterialConsistencyResult(
            score=1.0, total_properties=0, matched_properties=0, mismatched=[]
        )

    total = 0
    matched = 0
    mismatched: list[dict[str, object]] = []
    remaining_cand = list(cand_facts)

    for ref_fact in ref_facts:
        total += 1
        cand_idx = _best_material_fact_match(ref_fact, remaining_cand, tolerance)

        if cand_idx is None:
            mismatched.append({
                "property": ref_fact.key,
                "ref": ref_fact.value,
                "cand": None,
                "reason": "missing_in_candidate",
                "ref_detail": ref_fact.detail,
            })
            continue

        cand_fact = remaining_cand.pop(cand_idx)
        if _material_values_match(ref_fact.value, cand_fact.value, tolerance):
            matched += 1
        else:
            mismatched.append({
                "property": ref_fact.key,
                "ref": ref_fact.value,
                "cand": cand_fact.value,
                "reason": "value_mismatch",
                "ref_detail": ref_fact.detail,
                "cand_detail": cand_fact.detail,
            })

    score = matched / total if total > 0 else 1.0
    return MaterialConsistencyResult(
        score=score,
        total_properties=total,
        matched_properties=matched,
        mismatched=mismatched,
    )


def extract_coefficient_contract(
    code: str,
    *,
    include_direct_kernel_facts: bool = True,
) -> list[dict[str, object]]:
    """Return the coefficient/material facts MCS can compare for one input.

    The contract is read-only and scoring-free. It is useful for analyzers and
    LLM repair prompts that need to explain exactly which coefficient, BC, or
    constitutive facts are expected.
    """
    from codmos.multiagent.validators.hit_parser import load

    root = load(code)
    facts = _extract_material_facts(
        root,
        include_direct_kernel_facts=include_direct_kernel_facts,
    )
    return [
        {
            "key": fact.key,
            "value": fact.value,
            "source": fact.source,
            "detail": fact.detail,
        }
        for fact in facts
    ]


def _extract_material_facts(
    root: object,
    *,
    include_direct_kernel_facts: bool,
) -> list[_MaterialFact]:
    """Extract material-relevant facts from a parsed MOOSE AST."""
    from codmos.multiagent.pde.material_resolver import MaterialResolver

    materials_node = root.find("Materials")  # type: ignore[union-attr]
    resolver = MaterialResolver(materials_node) if materials_node is not None else None

    kernel_facts, covered_material_names = _extract_kernel_material_facts(
        root,
        resolver,
        include_direct_kernel_facts=include_direct_kernel_facts,
    )

    facts = list(kernel_facts)
    facts.extend(_extract_bc_coefficient_facts(root))
    facts.extend(_extract_material_model_signature_facts(root))
    if resolver is None:
        return facts

    for record in resolver.records:
        normalized_name = _normalize_material_name(record.name)
        if normalized_name in covered_material_names:
            continue
        facts.append(_MaterialFact(
            key=f"material:{normalized_name}",
            value=record.value,
            source="material",
            detail={
                "property": record.name,
                "material": record.material_name,
                "material_type": record.material_type,
            },
        ))
    return facts


def _extract_kernel_material_facts(
    root: object,
    resolver: object | None,
    *,
    include_direct_kernel_facts: bool,
) -> tuple[list[_MaterialFact], set[str]]:
    """Extract coefficients that kernels consume from material properties."""
    kernels_node = root.find("Kernels")  # type: ignore[union-attr]
    if kernels_node is None:
        return [], set()

    kernel_map = _get_mcs_kernel_map()
    facts: list[_MaterialFact] = []
    covered_material_names: set[str] = set()

    for child in kernels_node.children:
        kernel_type = child.param("type")
        if kernel_type is None:
            continue
        mapping = kernel_map.get_kernel(kernel_type)
        if mapping is None or mapping.coefficient_param is None:
            continue

        resolved = _resolve_kernel_coefficient(
            child,
            mapping.coefficient_param,
            resolver,
            include_direct_kernel_facts=include_direct_kernel_facts,
        )
        if resolved is None:
            continue

        value, material_names = resolved
        covered_material_names.update(
            _normalize_material_name(name) for name in material_names
        )
        facts.append(_MaterialFact(
            key=f"kernel:{mapping.operator}",
            value=value,
            source="kernel",
            detail={
                "kernel": kernel_type,
                "operator": mapping.operator,
                "coefficient_param": mapping.coefficient_param,
                "material_properties": sorted(material_names),
            },
        ))

    return facts, covered_material_names


def _extract_bc_coefficient_facts(root: object) -> list[_MaterialFact]:
    """Extract BC coefficients that are not represented by the main IFS BC value."""
    bcs_node = root.find("BCs")  # type: ignore[union-attr]
    if bcs_node is None:
        return []

    kernel_map = _get_mcs_kernel_map()
    facts: list[_MaterialFact] = []
    for child in bcs_node.children:
        bc_class = child.param("type")
        if bc_class is None:
            continue

        mapping = kernel_map.get_bc(bc_class)
        candidate_params: list[str] = []
        if mapping is not None:
            candidate_params.extend(
                param for param in mapping.all_parameters
                if param in _BC_COEFFICIENT_PARAMS
            )
            if mapping.function_param in _BC_COEFFICIENT_PARAMS:
                candidate_params.append(str(mapping.function_param))
            if (
                mapping.value_param is not None
                and mapping.value_param not in _GENERIC_BC_VALUE_PARAMS
            ):
                candidate_params.append(mapping.value_param)
            bc_type = mapping.bc_type
        else:
            candidate_params.extend(
                param for param in child.params
                if param in _BC_COEFFICIENT_PARAMS
            )
            bc_type = bc_class

        seen_params: set[str] = set()
        for param in candidate_params:
            if param in seen_params:
                continue
            seen_params.add(param)
            raw = child.param(param)
            if raw is None:
                continue

            canonical_param = _BC_COEFFICIENT_ALIASES.get(param, param)
            variable = _normalize_material_name_literal(child.param("variable", ""))
            boundary = _normalize_material_name_literal(child.param("boundary", ""))
            value = _coerce_material_float(raw)
            if value is None:
                value = _normalize_fact_literal(raw)
            facts.append(_MaterialFact(
                key=(
                    f"bc:{_normalize_material_name(bc_type)}:"
                    f"{_normalize_material_name(variable)}:"
                    f"{_normalize_material_name(boundary)}:"
                    f"{_normalize_material_name(canonical_param)}"
                ),
                value=value,
                source="bc",
                detail={
                    "bc": bc_class,
                    "bc_type": bc_type,
                    "variable": variable,
                    "boundary": boundary,
                    "parameter": param,
                    "canonical_parameter": canonical_param,
                },
            ))

    return facts


def _extract_material_model_signature_facts(root: object) -> list[_MaterialFact]:
    """Extract constitutive model choices that can share the same PDE operator."""
    materials_node = root.find("Materials")  # type: ignore[union-attr]
    if materials_node is None:
        return []

    facts: list[_MaterialFact] = []
    for child in materials_node.children:
        material_type = child.param("type")
        if material_type is None:
            continue
        signature = _MATERIAL_MODEL_SIGNATURES.get(material_type)
        if signature is None:
            continue
        role, model = signature
        facts.append(_MaterialFact(
            key=f"material_model:{role}",
            value=model,
            source="material_model",
            detail={
                "material": child.name,
                "material_type": material_type,
                "role": role,
            },
        ))

    return facts


def _get_mcs_kernel_map() -> Any:
    """Cache KernelMap for repeated MCS calls inside benchmark runs."""
    from codmos.multiagent.pde.kernel_map import KernelMap

    global _MCS_KERNEL_MAP_CACHE
    if _MCS_KERNEL_MAP_CACHE is None:
        _MCS_KERNEL_MAP_CACHE = KernelMap()
    return _MCS_KERNEL_MAP_CACHE


def _resolve_kernel_coefficient(
    kernel_node: object,
    coefficient_param: str,
    resolver: object | None,
    *,
    include_direct_kernel_facts: bool,
) -> tuple[float | str, set[str]] | None:
    if coefficient_param == "density_specific_heat":
        return _resolve_density_specific_heat(
            kernel_node,
            resolver,
            include_direct_kernel_facts,
        )

    raw = kernel_node.param(coefficient_param)  # type: ignore[attr-defined]
    if raw is not None:
        numeric = _coerce_material_float(raw)
        if numeric is not None:
            return (numeric, set()) if include_direct_kernel_facts else None

        resolved = _resolve_material_property(resolver, str(raw))
        if resolved is not None:
            return resolved, {str(raw)}

        return (str(raw), set()) if include_direct_kernel_facts else None

    resolved = _resolve_material_property(resolver, coefficient_param)
    if resolved is not None:
        return resolved, {coefficient_param}
    return None


def _resolve_density_specific_heat(
    kernel_node: object,
    resolver: object | None,
    include_direct_kernel_facts: bool,
) -> tuple[float | str, set[str]] | None:
    density_name = kernel_node.param("density_name") or "density"  # type: ignore[attr-defined]
    density = _resolve_material_property(resolver, str(density_name))
    density_sources: set[str] = {str(density_name)} if density is not None else set()

    raw_specific_heat = kernel_node.param("specific_heat")  # type: ignore[attr-defined]
    specific_heat: float | str | None = None
    specific_sources: set[str] = set()
    if raw_specific_heat is not None:
        numeric = _coerce_material_float(raw_specific_heat)
        if numeric is not None:
            specific_heat = numeric if include_direct_kernel_facts else None
        else:
            specific_heat = _resolve_material_property(resolver, str(raw_specific_heat))
            if specific_heat is not None:
                specific_sources.add(str(raw_specific_heat))

    if specific_heat is None:
        specific_heat = _resolve_material_property(resolver, "specific_heat")
        if specific_heat is not None:
            specific_sources.add("specific_heat")

    if density is None or specific_heat is None:
        return None

    density_float = _coerce_material_float(density)
    specific_float = _coerce_material_float(specific_heat)
    if density_float is not None and specific_float is not None:
        return density_float * specific_float, density_sources | specific_sources

    return f"{density}*{specific_heat}", density_sources | specific_sources


def _resolve_material_property(
    resolver: object | None,
    property_name: str,
) -> float | str | None:
    if resolver is None:
        return None
    return resolver.resolve(_normalize_material_name_literal(property_name))  # type: ignore[attr-defined]


def _normalize_material_name(name: str) -> str:
    return _normalize_material_name_literal(name).lower().replace("-", "_")


def _normalize_material_name_literal(name: str) -> str:
    return name.strip().strip("'\"")


def _normalize_fact_literal(value: object) -> str:
    return str(value).strip().strip("'\"")


def _best_material_fact_match(
    ref_fact: _MaterialFact,
    candidates: list[_MaterialFact],
    tolerance: float,
) -> int | None:
    keyed = [
        (idx, cand)
        for idx, cand in enumerate(candidates)
        if cand.key == ref_fact.key
    ]
    if not keyed:
        return None

    keyed.sort(key=lambda item: (
        not _material_values_match(ref_fact.value, item[1].value, tolerance),
        _material_value_distance(ref_fact.value, item[1].value),
        item[1].source != ref_fact.source,
    ))
    return keyed[0][0]


def _material_values_match(
    ref_val: float | str,
    cand_val: float | str,
    tolerance: float,
) -> bool:
    ref_float = _coerce_material_float(ref_val)
    cand_float = _coerce_material_float(cand_val)
    if ref_float is not None and cand_float is not None:
        if abs(ref_float) <= _EPSILON:
            return abs(cand_float) <= tolerance
        return abs(ref_float - cand_float) / abs(ref_float) <= tolerance
    return str(ref_val).strip() == str(cand_val).strip()


def _material_value_distance(ref_val: float | str, cand_val: float | str) -> float:
    ref_float = _coerce_material_float(ref_val)
    cand_float = _coerce_material_float(cand_val)
    if ref_float is None or cand_float is None:
        return 0.0 if str(ref_val).strip() == str(cand_val).strip() else float("inf")
    if abs(ref_float) <= _EPSILON:
        return abs(cand_float)
    return abs(ref_float - cand_float) / abs(ref_float)


def _coerce_material_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if stripped and stripped[0] in ("'", '"') and stripped[-1] == stripped[0]:
        stripped = stripped[1:-1]
    try:
        return float(stripped)
    except ValueError:
        return None
