# src/codmos/multiagent/pde/reconstruction.py
"""Deterministic PDE reconstruction from MOOSE .i files.

Layer 2 — imports Layer 1 (representation, kernel_map, material_resolver)
plus the existing HIT parser.

Walks the parsed AST to extract kernels, BCs, ICs, executioner, mesh,
and materials into a PDERepresentation.
"""

from __future__ import annotations

import logging

from codmos.multiagent.pde.action_expander import expand_actions
from codmos.multiagent.pde.kernel_map import KernelMap
from codmos.multiagent.pde.material_resolver import MaterialResolver
from codmos.multiagent.pde.representation import (
    BoundaryCondition,
    InitialCondition,
    PDERepresentation,
    PDETerm,
)
from codmos.multiagent.validators.hit_parser import HITNode, load

logger = logging.getLogger(__name__)

_DEFAULT_KERNEL_MAP = KernelMap()


def reconstruct_pde(
    moose_code: str,
    kernel_map: KernelMap | None = None,
) -> PDERepresentation:
    """Parse a MOOSE ``.i`` file and build a PDE representation.

    This is fully deterministic — no LLM calls. Unknown kernels or
    BCs are recorded in ``unresolved_kernels`` / warnings and the
    result's ``is_partial()`` returns True.
    """
    km = kernel_map or _DEFAULT_KERNEL_MAP
    root = load(moose_code)

    variables = _extract_variables(root)
    mat_resolver = _build_material_resolver(root)
    terms, unresolved_kernels, unresolved_coefficients = _extract_terms(root, km, mat_resolver)

    # Expand known MOOSE Actions into synthetic kernel entries
    for sk in expand_actions(root):
        mapping = km.get_kernel(sk.kernel_type)
        if mapping is None:
            unresolved_kernels.append(sk.kernel_type)
            continue
        terms.append(PDETerm(
            variable=sk.variable,
            operator=mapping.operator,
            coefficient=sk.coefficient,
            coupled_variable=sk.coupled_variable,
            kernel_type=sk.kernel_type,
            severity=mapping.severity,
        ))

    bcs = _extract_bcs(root, km)
    ics = _extract_ics(root)
    time_scheme = _extract_time_scheme(root)
    dimensions = _extract_dimensions(root)

    warnings: list[str] = []
    if unresolved_kernels:
        warnings.append(f"Unknown kernels skipped: {', '.join(unresolved_kernels)}")

    return PDERepresentation(
        terms=terms,
        boundary_conditions=bcs,
        initial_conditions=ics,
        time_scheme=time_scheme,
        variables=variables,
        dimensions=dimensions,
        unresolved_kernels=unresolved_kernels,
        unresolved_coefficients=unresolved_coefficients,
        warnings=warnings,
    )


def _extract_variables(root: HITNode) -> list[str]:
    node = root.find("Variables")
    if node is None:
        return []
    return node.child_names()


def _extract_terms(
    root: HITNode,
    km: KernelMap,
    mat_resolver: MaterialResolver | None,
) -> tuple[list[PDETerm], list[str], list[str]]:
    node = root.find("Kernels")
    if node is None:
        return [], [], []

    terms: list[PDETerm] = []
    unresolved_kernels: list[str] = []
    unresolved_coefficients: list[str] = []

    for child in node.children:
        kernel_type = child.param("type")
        if kernel_type is None:
            continue

        mapping = km.get_kernel(kernel_type)
        if mapping is None:
            unresolved_kernels.append(kernel_type)
            continue

        variable = child.param(mapping.variable_param, "") if mapping.variable_param else child.param("variable", "")
        coupled_variable = child.param(mapping.coupled_param) if mapping.coupled_param else None

        coefficient: float | str | None = None
        if mapping.coefficient_param:
            # First try the kernel block directly (inline coefficient)
            raw_coef = child.param(mapping.coefficient_param)
            if raw_coef is not None:
                coefficient = _try_parse_float(raw_coef)
            elif mat_resolver is not None:
                # Fall back to material properties
                resolved = mat_resolver.resolve(mapping.coefficient_param)
                if resolved is not None:
                    coefficient = resolved
                else:
                    unresolved_coefficients.append(mapping.coefficient_param)


        terms.append(PDETerm(
            variable=variable,
            operator=mapping.operator,
            coefficient=coefficient,
            coupled_variable=coupled_variable,
            kernel_type=kernel_type,
            severity=mapping.severity,
        ))

    return terms, unresolved_kernels, unresolved_coefficients


def _extract_bcs(root: HITNode, km: KernelMap) -> list[BoundaryCondition]:
    node = root.find("BCs")
    if node is None:
        return []

    bcs: list[BoundaryCondition] = []
    for child in node.children:
        bc_type_str = child.param("type")
        if bc_type_str is None:
            continue

        variable = child.param("variable", "")
        boundary = child.param("boundary", "")

        mapping = km.get_bc(bc_type_str)
        if mapping is not None:
            raw_value = child.param(mapping.value_param) if mapping.value_param else None
            value = _try_parse_float(raw_value) if raw_value else None
            bcs.append(BoundaryCondition(
                variable=variable,
                boundary=boundary,
                bc_type=mapping.bc_type,
                value=value,
                moose_bc_class=bc_type_str,
                severity=mapping.severity,
            ))
        else:
            # Unknown BC — still record it with raw info
            bcs.append(BoundaryCondition(
                variable=variable,
                boundary=boundary,
                bc_type=bc_type_str,
                value=_try_parse_float(child.param("value")),
                moose_bc_class=bc_type_str,
                severity="medium",
            ))

    return bcs


def _extract_ics(root: HITNode) -> list[InitialCondition]:
    node = root.find("ICs")
    if node is None:
        return []

    ics: list[InitialCondition] = []
    for child in node.children:
        ic_type_raw = child.param("type", "")
        variable = child.param("variable", "")
        raw_value = child.param("value")

        if "Constant" in ic_type_raw:
            ic_type = "constant"
        elif "Function" in ic_type_raw:
            ic_type = "function"
        else:
            ic_type = ic_type_raw.lower()

        value = _try_parse_float(raw_value) if raw_value else raw_value

        ics.append(InitialCondition(
            variable=variable,
            ic_type=ic_type,
            value=value,
            severity="medium",
        ))

    return ics


def _extract_time_scheme(root: HITNode) -> str:
    node = root.find("Executioner")
    if node is None:
        return "steady"
    exec_type = node.param("type", "Steady")
    return "transient" if exec_type == "Transient" else "steady"


def _extract_dimensions(root: HITNode) -> int:
    mesh = root.find("Mesh")
    if mesh is None:
        return 1
    # Check sub-blocks (e.g. [gen])
    for child in mesh.children:
        dim_str = child.param("dim")
        if dim_str is not None:
            return int(dim_str)
    # Check direct params
    dim_str = mesh.param("dim")
    if dim_str is not None:
        return int(dim_str)
    return 1


def _build_material_resolver(root: HITNode) -> MaterialResolver | None:
    node = root.find("Materials")
    if node is None:
        return None
    return MaterialResolver(node)


def _try_parse_float(value: str | None) -> float | str | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return value
