"""L2 MOOSE object/type validator.

This layer runs after HIT syntax parsing.  It checks whether every
``type = ...`` is legal for the local MOOSE block context and whether
object parameters are known by the application syntax registry.
"""

from __future__ import annotations

import contextlib
import difflib
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from codmos.multiagent.moose_registry import MooseObjectSpec, MooseRegistry
from codmos.multiagent.validators.hit_parser import HITNode
from codmos.multiagent.validators.hit_syntax import HITSyntaxValidator


@dataclass(frozen=True)
class MOOSETypeIssue:
    """One L2 registry/schema validation issue."""

    severity: str
    kind: str
    path: str
    message: str
    object_type: str | None = None
    parameter: str | None = None
    suggestions: list[str] = field(default_factory=list)
    autofixable: bool = False
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class MOOSETypeResult:
    """Outcome of a MOOSE type validation pass."""

    passed: bool
    issues: list[MOOSETypeIssue]
    tree: HITNode | None = None
    final_code: str = ""


@dataclass
class MOOSETypeRepairResult:
    """Result of conservative L2 repair."""

    changed: bool
    repaired_code: str
    before: MOOSETypeResult
    after: MOOSETypeResult
    changes: list[str]


@dataclass(frozen=True)
class _ModelContext:
    variables: frozenset[str]
    boundaries: frozenset[str] | None
    material_properties: frozenset[str]


class MOOSETypeValidator:
    """Validate object names, local contexts, and known parameters."""

    def __init__(self, registry: MooseRegistry, *, strict_parameters: bool = True) -> None:
        self.registry = registry
        self.strict_parameters = strict_parameters
        self.syntax_validator = HITSyntaxValidator()

    def validate(self, code: str) -> MOOSETypeResult:
        syntax = self.syntax_validator.validate(code)
        if not syntax.passed:
            issue = MOOSETypeIssue(
                severity="error",
                kind="hit_syntax",
                path="/",
                message=syntax.error,
                context={"error_location": syntax.error_location},
            )
            return MOOSETypeResult(False, [issue], None, syntax.final_code)

        assert syntax.tree is not None
        model_context = _collect_model_context(syntax.tree)
        issues: list[MOOSETypeIssue] = []
        issues.extend(_framework_issues(syntax.tree))
        issues.extend(_missing_type_issues(syntax.tree))
        issues.extend(_duplicate_initial_condition_issues(syntax.tree))
        issues.extend(_duplicate_material_property_issues(syntax.tree))
        for node, path_parts in _walk_with_paths(syntax.tree):
            type_name = node.param("type")
            if type_name is None:
                continue
            context = _context_for_path(path_parts, self.registry)
            path = "/" + "/".join(path_parts)
            spec = self.registry.get(type_name)

            if spec is None:
                suggestions = _closest(type_name, self.registry.candidates_for_context(context))
                if not suggestions:
                    suggestions = _closest(type_name, self.registry.all_type_names())
                issues.append(MOOSETypeIssue(
                    severity="error",
                    kind="unknown_type",
                    path=path,
                    object_type=type_name,
                    message=f"Unknown MOOSE object type {type_name!r} in context {context!r}.",
                    suggestions=suggestions,
                    autofixable=False,
                    context={"expected_context": context},
                ))
                continue

            if context not in spec.contexts:
                suggestions = self.registry.candidates_for_context(context)
                issues.append(MOOSETypeIssue(
                    severity="error",
                    kind="context_mismatch",
                    path=path,
                    object_type=type_name,
                    message=(
                        f"MOOSE object type {type_name!r} is registered, but not under "
                        f"context {context!r}."
                    ),
                    suggestions=_closest(type_name, suggestions),
                    autofixable=_is_safe_context_repair(type_name, context, self.registry),
                    context={"actual_contexts": sorted(spec.contexts), "expected_context": context},
                ))
                continue

            if self.strict_parameters and spec.complete_params:
                issues.extend(_parameter_issues(node, path, type_name, spec))
                issues.extend(_reference_issues(node, path, type_name, spec, model_context))

        passed = not any(issue.severity == "error" for issue in issues)
        return MOOSETypeResult(passed, issues, syntax.tree, syntax.final_code)

    def repair(self, code: str, *, max_iterations: int = 2) -> MOOSETypeRepairResult:
        """Apply conservative local repairs and revalidate.

        This avoids semantic rewrites such as deleting diffusion coefficients.
        It fixes context-equivalent mesh objects and common one-token parameter
        aliases when the target parameter exists.
        """
        before = self.validate(code)
        if before.tree is None:
            return MOOSETypeRepairResult(False, before.final_code or code, before, before, [])

        tree = deepcopy(before.tree)
        changes: list[str] = []
        for _ in range(max_iterations):
            changed = _apply_safe_repairs(tree, self.registry, changes)
            if not changed:
                break

        repaired_code = dump_hit(tree)
        after = self.validate(repaired_code)
        return MOOSETypeRepairResult(bool(changes), repaired_code, before, after, changes)


_INSTANCE_BLOCKS = {
    "Adaptivity/Markers",
    "AuxKernels",
    "BCs",
    "Constraints",
    "DGKernels",
    "Dampers",
    "DiracKernels",
    "Functions",
    "FVBCs",
    "ICs",
    "InterfaceKernels",
    "Kernels",
    "Materials",
    "Mesh",
    "Outputs",
    "Postprocessors",
    "Preconditioning",
    "UserObjects",
    "VectorPostprocessors",
}

_TYPED_INSTANCE_BLOCKS = _INSTANCE_BLOCKS - {"Variables", "AuxVariables"}


def _walk_with_paths(root: HITNode):
    def rec(node: HITNode, prefix: list[str]):
        for child in node.children:
            path = [*prefix, child.name]
            yield child, path
            yield from rec(child, path)

    yield from rec(root, [])


def _context_for_path(path_parts: list[str], registry: MooseRegistry | None = None) -> str:
    if not path_parts:
        return ""
    joined = "/".join(path_parts)
    joined_parent = "/".join(path_parts[:-1])
    first = path_parts[0]

    if registry is not None:
        if registry.candidates_for_context(joined):
            return joined
        parent_instance = f"{joined_parent}/*" if joined_parent else ""
        if parent_instance and registry.candidates_for_context(parent_instance):
            return parent_instance

    if len(path_parts) == 1:
        return first
    if joined_parent in _INSTANCE_BLOCKS:
        return f"{joined_parent}/*"
    if first in _INSTANCE_BLOCKS and len(path_parts) == 2:
        return f"{first}/*"
    return joined_parent


def _framework_issues(root: HITNode) -> list[MOOSETypeIssue]:
    mesh = next((child for child in root.children if child.name == "Mesh"), None)
    mesh_generators = next((child for child in root.children if child.name == "MeshGenerators"), None)

    has_mesh_generation = False
    if mesh is not None:
        has_mesh_generation = (
            bool(mesh.children) or mesh.param("type") is not None or mesh.param("file") is not None
        )
    if not has_mesh_generation and mesh_generators is not None:
        has_mesh_generation = bool(mesh_generators.children)

    if has_mesh_generation:
        return []
    return [
        MOOSETypeIssue(
            severity="error",
            kind="missing_mesh_generation",
            path="/Mesh",
            message="No Mesh type/file or mesh generator block was found.",
        )
    ]


def _missing_type_issues(root: HITNode) -> list[MOOSETypeIssue]:
    issues: list[MOOSETypeIssue] = []
    for parent, path_parts in _walk_with_paths(root):
        parent_path = "/".join(path_parts)
        if parent_path not in _TYPED_INSTANCE_BLOCKS:
            continue
        for child in parent.children:
            if child.param("type") is not None:
                continue
            issues.append(MOOSETypeIssue(
                severity="error",
                kind="missing_type",
                path=f"/{parent_path}/{child.name}",
                message=f"Missing required MOOSE object type in /{parent_path}/{child.name}.",
                autofixable=False,
                context={"expected_context": f"{parent_path}/*"},
            ))
    return issues


def _collect_model_context(root: HITNode) -> _ModelContext:
    variables = set()
    for block_name in ("Variables", "AuxVariables"):
        block = root.find(f"/{block_name}")
        if block is not None:
            variables.update(child.name for child in block.children)

    return _ModelContext(
        variables=frozenset(variables),
        boundaries=_generated_mesh_boundaries(root),
        material_properties=frozenset(_material_property_names(root)),
    )


def _generated_mesh_boundaries(root: HITNode) -> frozenset[str] | None:
    mesh = root.find("/Mesh")
    if mesh is None:
        return None

    candidates: list[HITNode] = []
    if mesh.param("type") in {"GeneratedMesh", "GeneratedMeshGenerator"}:
        candidates.append(mesh)
    candidates.extend(
        child
        for child in mesh.children
        if child.param("type") in {"GeneratedMesh", "GeneratedMeshGenerator"}
    )

    mesh_generators = root.find("/MeshGenerators")
    if mesh_generators is not None:
        candidates.extend(
            child
            for child in mesh_generators.children
            if child.param("type") in {"GeneratedMesh", "GeneratedMeshGenerator"}
        )

    if not candidates:
        return None

    dim = _parse_int(candidates[0].param("dim"))
    if dim == 1:
        return frozenset({"left", "right", "0", "1"})
    if dim == 2:
        return frozenset({"left", "right", "bottom", "top", "0", "1", "2", "3"})
    if dim == 3:
        return frozenset({
            "left",
            "right",
            "bottom",
            "top",
            "front",
            "back",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
        })
    return None


def _material_property_names(root: HITNode) -> set[str]:
    materials = root.find("/Materials")
    if materials is None:
        return set()

    names: set[str] = set()
    for material in materials.children:
        prop_names = material.param("prop_names")
        if prop_names is not None:
            names.update(_split_hit_tokens(prop_names))
        property_name = material.param("property_name")
        if property_name is not None:
            names.update(_split_hit_tokens(property_name))
    return names


def _parameter_issues(
    node: HITNode,
    path: str,
    type_name: str,
    spec: MooseObjectSpec,
) -> list[MOOSETypeIssue]:
    issues: list[MOOSETypeIssue] = []
    allowed = set(spec.params)
    for param in node.params:
        if param not in allowed:
            suggestions = _closest(param, sorted(allowed))
            issues.append(MOOSETypeIssue(
                severity="error",
                kind="unknown_parameter",
                path=path,
                object_type=type_name,
                parameter=param,
                message=f"Unknown parameter {param!r} for MOOSE object type {type_name!r}.",
                suggestions=suggestions,
                autofixable=_safe_param_alias(param, allowed) is not None,
            ))

    for param_name, param_spec in spec.params.items():
        if param_name == "type":
            continue
        if param_spec.required and param_name not in node.params:
            issues.append(MOOSETypeIssue(
                severity="error",
                kind="missing_required_parameter",
                path=path,
                object_type=type_name,
                parameter=param_name,
                message=f"Missing required parameter {param_name!r} for {type_name!r}.",
            ))
    return issues


def _reference_issues(
    node: HITNode,
    path: str,
    type_name: str,
    spec: MooseObjectSpec,
    model_context: _ModelContext,
) -> list[MOOSETypeIssue]:
    issues: list[MOOSETypeIssue] = []
    issues.extend(_variable_reference_issues(node, path, type_name, model_context))
    issues.extend(_boundary_reference_issues(node, path, type_name, model_context))
    issues.extend(_material_property_reference_issues(
        node,
        path,
        type_name,
        spec,
        model_context,
    ))
    return issues


def _variable_reference_issues(
    node: HITNode,
    path: str,
    type_name: str,
    model_context: _ModelContext,
) -> list[MOOSETypeIssue]:
    if not model_context.variables:
        return []

    issues: list[MOOSETypeIssue] = []
    for parameter in ("variable", "v"):
        value = node.param(parameter)
        if value is None:
            continue
        for token in _split_hit_tokens(value):
            if token in model_context.variables:
                continue
            issues.append(MOOSETypeIssue(
                severity="error",
                kind="unknown_variable",
                path=path,
                object_type=type_name,
                parameter=parameter,
                message=f"Variable reference {token!r} is not defined in [Variables].",
                suggestions=sorted(model_context.variables),
            ))
    return issues


def _boundary_reference_issues(
    node: HITNode,
    path: str,
    type_name: str,
    model_context: _ModelContext,
) -> list[MOOSETypeIssue]:
    value = node.param("boundary")
    if value is None or model_context.boundaries is None:
        return []

    issues: list[MOOSETypeIssue] = []
    for token in _split_hit_tokens(value):
        if token in model_context.boundaries or _is_number(token):
            continue
        issues.append(MOOSETypeIssue(
            severity="error",
            kind="unknown_boundary",
            path=path,
            object_type=type_name,
            parameter="boundary",
            message=f"Boundary reference {token!r} is not available on the generated mesh.",
            suggestions=sorted(model_context.boundaries),
        ))
    return issues


def _material_property_reference_issues(
    node: HITNode,
    path: str,
    type_name: str,
    spec: MooseObjectSpec,
    model_context: _ModelContext,
) -> list[MOOSETypeIssue]:
    issues: list[MOOSETypeIssue] = []
    for parameter, param_spec in spec.params.items():
        if not _is_material_property_param(param_spec.cpp_type):
            continue
        raw_value = node.param(parameter)
        source = "parameter"
        if raw_value is None:
            raw_value = param_spec.default
            source = "default"
        if raw_value is None:
            continue
        for token in _split_hit_tokens(raw_value):
            if not token:
                continue
            if _is_number(token):
                if source == "parameter":
                    issues.append(MOOSETypeIssue(
                        severity="error",
                        kind="numeric_material_property",
                        path=path,
                        object_type=type_name,
                        parameter=parameter,
                        message=(
                            f"Material-property parameter {parameter!r} received numeric "
                            f"value {token!r}; MOOSE expects a material property name."
                        ),
                        autofixable=True,
                    ))
                continue
            if token in model_context.material_properties:
                continue
            issues.append(MOOSETypeIssue(
                severity="error",
                kind="missing_material_property",
                path=path,
                object_type=type_name,
                parameter=parameter,
                message=(
                    f"Material property {token!r} referenced by {parameter!r} "
                    f"({source}) is not defined in [Materials]."
                ),
                suggestions=sorted(model_context.material_properties),
            ))
    return issues


def _closest(value: str, candidates: list[str], *, n: int = 5, cutoff: float = 0.58) -> list[str]:
    if not candidates:
        return []
    return difflib.get_close_matches(value, candidates, n=n, cutoff=cutoff)


def _split_hit_tokens(value: str) -> list[str]:
    stripped = value.strip().strip("'\"")
    if not stripped:
        return []
    return [part.strip().strip("'\"") for part in stripped.split() if part.strip()]


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    with contextlib.suppress(ValueError):
        return int(float(value.strip().strip("'\"")))
    return None


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    with contextlib.suppress(ValueError):
        return float(value.strip().strip("'\""))
    return None


def _format_float(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.12g}"


def _is_number(value: str) -> bool:
    with contextlib.suppress(ValueError):
        float(value.strip().strip("'\""))
        return True
    return False


def _is_material_property_param(cpp_type: str | None) -> bool:
    return bool(cpp_type and "MaterialPropertyName" in cpp_type)


def _is_safe_context_repair(type_name: str, context: str, registry: MooseRegistry) -> bool:
    if type_name == "GeneratedMesh" and context == "Mesh/*":
        return registry.is_valid_in_context("GeneratedMeshGenerator", context)
    if type_name == "GeneratedMeshGenerator" and context == "Mesh":
        return registry.is_valid_in_context("GeneratedMesh", context)
    return False


def _apply_safe_repairs(
    root: HITNode,
    registry: MooseRegistry,
    changes: list[str],
) -> bool:
    changed = False
    changed = _repair_duplicate_generic_constant_material_properties(root, changes) or changed
    changed = _repair_duplicate_initial_conditions(root, changes) or changed
    changed = _repair_generated_mesh_boundary_aliases(root, registry, changes) or changed
    for node, path_parts in _walk_with_paths(root):
        type_name = node.param("type")
        if type_name is None:
            continue
        context = _context_for_path(path_parts, registry)
        path = "/" + "/".join(path_parts)

        changed = _repair_all_generated_mesh_boundary(root, node, path, context, changes) or changed

        if type_name == "GeneratedMesh" and context == "Mesh/*":
            if registry.is_valid_in_context("GeneratedMeshGenerator", context):
                node.params["type"] = "GeneratedMeshGenerator"
                changes.append(f"{path}: type GeneratedMesh -> GeneratedMeshGenerator")
                changed = True
                type_name = "GeneratedMeshGenerator"
        elif type_name == "GeneratedMeshGenerator" and context == "Mesh":
            if registry.is_valid_in_context("GeneratedMesh", context):
                node.params["type"] = "GeneratedMesh"
                changes.append(f"{path}: type GeneratedMeshGenerator -> GeneratedMesh")
                changed = True
                type_name = "GeneratedMesh"
        elif (
            type_name == "RobinBC"
            and context == "BCs/*"
            and registry.is_valid_in_context("ADRobinBC", context)
        ):
            node.params["type"] = "ADRobinBC"
            changes.append(f"{path}: type RobinBC -> ADRobinBC")
            changed = True
            type_name = "ADRobinBC"

        spec = registry.get(type_name)
        if spec is None or context not in spec.contexts or not spec.complete_params:
            changed = _repair_petsc_options(node, path, changes) or changed
            continue
        changed = _repair_petsc_options(node, path, changes) or changed
        changed = _repair_missing_params_from_global_params(
            root,
            node,
            path,
            type_name,
            spec,
            changes,
        ) or changed
        changed = _repair_solver_tolerances(node, path, type_name, spec, changes) or changed
        changed = _repair_adrobin_value_to_convective(
            root,
            node,
            path,
            type_name,
            registry,
            changes,
        ) or changed
        changed = _repair_adrobin_alpha_beta_gamma(node, path, type_name, changes) or changed
        type_name = node.param("type") or type_name
        spec = registry.get(type_name)
        if spec is None or context not in spec.contexts or not spec.complete_params:
            continue
        changed = _repair_heat_conduction_thermal_conductivity(
            root,
            node,
            path,
            type_name,
            spec,
            registry,
            changes,
        ) or changed
        changed = _repair_mat_coupled_force_material_property_alias(
            node,
            path,
            type_name,
            spec,
            changes,
        ) or changed
        changed = _repair_numeric_mobility_alias(
            root,
            node,
            path,
            spec,
            registry,
            changes,
        ) or changed
        changed = _repair_zero_vector_value_params(node, path, spec, changes) or changed
        changed = _repair_heat_source_parameter(root, node, path, type_name, changes) or changed
        changed = _repair_heat_time_density_specific_heat(
            root,
            node,
            path,
            type_name,
            registry,
            changes,
        ) or changed
        changed = _repair_numeric_material_property_params(
            root,
            node,
            path,
            spec,
            registry,
            changes,
        ) or changed
        allowed = set(spec.params)
        for param in list(node.params):
            if param in allowed:
                continue
            replacement = _safe_param_alias(param, allowed)
            if replacement is None:
                continue
            node.params[replacement] = node.params.pop(param)
            changes.append(f"{path}: param {param} -> {replacement}")
            changed = True

    return changed


def _duplicate_initial_condition_issues(root: HITNode) -> list[MOOSETypeIssue]:
    issues: list[MOOSETypeIssue] = []
    variable_ics = _variable_initial_conditions(root)
    explicit_ics = _explicit_initial_conditions(root)

    for ic_path, variable, value, _node in explicit_ics:
        variable_value = variable_ics.get(variable, (None, None))[1]
        if variable_value is None:
            continue
        issues.append(MOOSETypeIssue(
            severity="error",
            kind="duplicate_initial_condition",
            path=ic_path,
            parameter="initial_condition",
            message=(
                f"Variable {variable!r} has both [Variables] initial_condition "
                "and an explicit [ICs] object."
            ),
            autofixable=True,
            context={"variable": variable, "variable_value": variable_value, "ic_value": value},
        ))

    seen_explicit: dict[str, tuple[str, str | None]] = {}
    for ic_path, variable, value, _node in explicit_ics:
        if variable not in seen_explicit:
            seen_explicit[variable] = (ic_path, value)
            continue
        first_path, first_value = seen_explicit[variable]
        issues.append(MOOSETypeIssue(
            severity="error",
            kind="duplicate_initial_condition",
            path=ic_path,
            parameter="initial_condition",
            message=(
                f"Variable {variable!r} has multiple explicit [ICs] objects: "
                f"{first_path} and {ic_path}."
            ),
            suggestions=[first_path],
            autofixable=_values_equal(first_value, value),
            context={"variable": variable, "first_value": first_value, "ic_value": value},
        ))
    return issues


def _variable_initial_conditions(root: HITNode) -> dict[str, tuple[HITNode, str | None]]:
    variables = root.find("/Variables")
    if variables is None:
        return {}
    return {
        variable.name: (variable, variable.param("initial_condition"))
        for variable in variables.children
        if variable.param("initial_condition") is not None
    }


def _explicit_initial_conditions(root: HITNode) -> list[tuple[str, str, str | None, HITNode]]:
    ics = root.find("/ICs")
    if ics is None:
        return []
    result: list[tuple[str, str, str | None, HITNode]] = []
    for child in ics.children:
        variable = child.param("variable")
        if variable is None:
            continue
        result.append((f"/ICs/{child.name}", variable, child.param("value"), child))
    return result


def _repair_duplicate_initial_conditions(root: HITNode, changes: list[str]) -> bool:
    changed = False
    variable_ics = _variable_initial_conditions(root)
    explicit_ics = _explicit_initial_conditions(root)

    for ic_path, variable, _value, _node in explicit_ics:
        variable_node, variable_value = variable_ics.get(variable, (None, None))
        if variable_node is None or variable_value is None:
            continue
        variable_node.params.pop("initial_condition", None)
        changes.append(
            f"/Variables/{variable}: removed duplicate initial_condition already covered by {ic_path}"
        )
        changed = True

    ics = root.find("/ICs")
    if ics is None:
        return changed

    seen: dict[str, str | None] = {}
    keep_children: list[HITNode] = []
    for child in ics.children:
        variable = child.param("variable")
        value = child.param("value")
        if variable is None:
            keep_children.append(child)
            continue
        if variable in seen and _values_equal(seen[variable], value):
            changes.append(f"/ICs/{child.name}: removed duplicate IC for {variable}")
            changed = True
            continue
        seen[variable] = value
        keep_children.append(child)
    ics.children[:] = keep_children
    return changed


def _values_equal(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return left == right
    left_float = _parse_float(left)
    right_float = _parse_float(right)
    if left_float is not None and right_float is not None:
        return abs(left_float - right_float) <= 1e-12
    return left.strip().strip("'\"") == right.strip().strip("'\"")


def _global_params(root: HITNode) -> dict[str, str]:
    global_params = root.find("/GlobalParams")
    if global_params is None:
        return {}
    return dict(global_params.params)


def _repair_missing_params_from_global_params(
    root: HITNode,
    node: HITNode,
    path: str,
    type_name: str,
    spec: MooseObjectSpec,
    changes: list[str],
) -> bool:
    values = _global_params(root)
    if not values:
        return False

    changed = False
    for param_name, param_spec in spec.params.items():
        if param_name == "type" or not param_spec.required or param_name in node.params:
            continue
        if param_name not in values:
            continue
        node.params[param_name] = values[param_name]
        changes.append(f"{path}: copied required {param_name} from GlobalParams for {type_name}")
        changed = True
    return changed


def _repair_generated_mesh_boundary_aliases(
    root: HITNode,
    registry: MooseRegistry,
    changes: list[str],
) -> bool:
    mesh = root.find("/Mesh")
    if mesh is None or mesh.param("type") not in {"GeneratedMesh", "GeneratedMeshGenerator"}:
        return False

    context = _context_for_path(["Mesh"], registry)
    spec = registry.get(mesh.param("type") or "")
    if spec is None or context not in spec.contexts or not spec.complete_params:
        return False

    alias_params = {
        "left_boundary": "left",
        "right_boundary": "right",
        "bottom_boundary": "bottom",
        "top_boundary": "top",
        "front_boundary": "front",
        "back_boundary": "back",
    }
    aliases = {
        value.strip().strip("'\""): canonical
        for param, canonical in alias_params.items()
        if (value := mesh.params.get(param))
    }
    if not aliases:
        return False

    changed = False
    for block_name in ("BCs", "FVBCs"):
        block = root.find(f"/{block_name}")
        if block is None:
            continue
        for child in block.children:
            raw_boundary = child.params.get("boundary")
            if raw_boundary is None:
                continue
            tokens = _split_hit_tokens(raw_boundary)
            mapped = [aliases.get(token, token) for token in tokens]
            if mapped == tokens:
                continue
            child.params["boundary"] = "'" + " ".join(mapped) + "'"
            changes.append(
                f"/{block_name}/{child.name}: mapped GeneratedMesh boundary aliases to canonical sides"
            )
            changed = True

    allowed = set(spec.params)
    for param in ("boundary_name_prefix", *alias_params):
        if param in mesh.params and param not in allowed:
            mesh.params.pop(param)
            changes.append(f"/Mesh: removed unsupported GeneratedMesh boundary alias parameter {param}")
            changed = True
    return changed


def _repair_adrobin_value_to_convective(
    root: HITNode,
    node: HITNode,
    path: str,
    type_name: str,
    registry: MooseRegistry,
    changes: list[str],
) -> bool:
    if type_name != "ADRobinBC":
        return False
    if not _is_heat_variable_name(node.param("variable")):
        return False
    value = node.params.get("value")
    convective = _convective_robin_values(node)
    if value is None or convective is None:
        return False
    if not registry.is_valid_in_context("ConvectiveHeatFluxBC", "BCs/*"):
        return False
    t_inf_value, coefficient_value = convective

    t_inf_name = _unique_material_property_name(
        root,
        _safe_material_property_name(node.name, "T_infinity"),
    )
    htc_name = _unique_material_property_name(
        root,
        _safe_material_property_name(node.name, "heat_transfer_coefficient"),
        reserved={t_inf_name},
    )
    if not _add_generic_constant_material(
        root,
        registry,
        [t_inf_name, htc_name],
        [t_inf_value, coefficient_value],
    ):
        return False

    node.params["type"] = "ConvectiveHeatFluxBC"
    node.params["T_infinity"] = t_inf_name
    node.params["heat_transfer_coefficient"] = htc_name
    node.params.pop("value", None)
    node.params.pop("coefficient", None)
    node.params.pop("coef1", None)
    node.params.pop("coef2", None)
    node.params.pop("alpha", None)
    node.params.pop("beta", None)
    node.params.pop("gamma", None)
    changes.append(
        f"{path}: converted ADRobinBC coefficient/value to ConvectiveHeatFluxBC"
    )
    return True


def _convective_robin_values(node: HITNode) -> tuple[str, str] | None:
    value = _parse_float(node.params.get("value"))
    if value is None:
        gamma = _parse_float(node.params.get("gamma"))
        value = gamma
    if value is None:
        return None

    coefficient = _parse_float(node.params.get("coefficient"))
    if coefficient is not None:
        return _format_float(value), _format_float(coefficient)

    coef1 = _parse_float(node.params.get("coef1"))
    coef2 = _parse_float(node.params.get("coef2"))
    if coef1 is None:
        coef1 = _parse_float(node.params.get("alpha"))
    if coef2 is None:
        coef2 = _parse_float(node.params.get("beta"))
    if coef1 is not None and coef2 not in (None, 0.0):
        return _format_float(value / coef1), _format_float(coef1 / coef2)

    return _format_float(value), "1.0"


def _is_heat_variable_name(variable: str | None) -> bool:
    return (variable or "").lower() in {"t", "temp", "temperature"}


def _repair_solver_tolerances(
    node: HITNode,
    path: str,
    type_name: str,
    spec: MooseObjectSpec,
    changes: list[str],
) -> bool:
    if type_name not in {"Steady", "Transient"}:
        return False
    if "nl_abs_tol" not in spec.params or "nl_abs_tol" in node.params:
        return False

    node.params["nl_abs_tol"] = "1e-8"
    changes.append(f"{path}: added nl_abs_tol = 1e-8 solver boilerplate")
    return True


def _repair_heat_conduction_thermal_conductivity(
    root: HITNode,
    node: HITNode,
    path: str,
    type_name: str,
    spec: MooseObjectSpec,
    registry: MooseRegistry,
    changes: list[str],
) -> bool:
    if type_name != "HeatConduction" or "thermal_conductivity" not in node.params:
        return False
    if "diffusion_coefficient" not in spec.params:
        return False

    property_name = node.params.pop("thermal_conductivity")
    node.params["diffusion_coefficient"] = property_name
    changed = True

    derivative_param = "diffusion_coefficient_dT"
    if derivative_param in spec.params and derivative_param not in node.params:
        derivative_name = _safe_material_property_name(
            property_name.strip().strip("'\""),
            "dT",
        )
        if derivative_name not in _material_property_names(root):
            derivative_name = _unique_material_property_name(root, derivative_name)
            if _add_generic_constant_material(root, registry, [derivative_name], ["0.0"]):
                node.params[derivative_param] = derivative_name
                changes.append(
                    f"{path}: added zero {derivative_param} property {derivative_name}"
                )
        else:
            node.params[derivative_param] = derivative_name

    changes.append(f"{path}: param thermal_conductivity -> diffusion_coefficient")
    return changed


def _repair_all_generated_mesh_boundary(
    root: HITNode,
    node: HITNode,
    path: str,
    context: str,
    changes: list[str],
) -> bool:
    if context not in {"BCs/*", "FVBCs/*"}:
        return False
    raw_boundary = node.params.get("boundary")
    if raw_boundary is None:
        return False
    tokens = _split_hit_tokens(raw_boundary)
    if len(tokens) != 1 or tokens[0].lower() != "all":
        return False

    boundaries = _named_generated_mesh_boundaries(root)
    if not boundaries:
        return False
    node.params["boundary"] = "'" + " ".join(boundaries) + "'"
    changes.append(f"{path}: expanded boundary='all' to generated mesh boundaries")
    return True


def _named_generated_mesh_boundaries(root: HITNode) -> tuple[str, ...] | None:
    dim = None
    for path in ("/Mesh", "/Mesh/generated"):
        node = root.find(path)
        if node is not None:
            dim = _parse_int(node.param("dim"))
            break
    if dim == 1:
        return ("left", "right")
    if dim == 2:
        return ("left", "right", "bottom", "top")
    if dim == 3:
        return ("left", "right", "bottom", "top", "front", "back")
    return None


def _repair_mat_coupled_force_material_property_alias(
    node: HITNode,
    path: str,
    type_name: str,
    spec: MooseObjectSpec,
    changes: list[str],
) -> bool:
    if type_name != "MatCoupledForce" or "mat_prop_coef" not in node.params:
        return False
    if "material_properties" not in spec.params:
        return False

    node.params["material_properties"] = node.params.pop("mat_prop_coef")
    changes.append(f"{path}: param mat_prop_coef -> material_properties")
    return True


def _repair_numeric_mobility_alias(
    root: HITNode,
    node: HITNode,
    path: str,
    spec: MooseObjectSpec,
    registry: MooseRegistry,
    changes: list[str],
) -> bool:
    raw_value = node.params.get("mobility")
    if raw_value is None or "mob_name" not in spec.params:
        return False
    tokens = _split_hit_tokens(raw_value)
    if len(tokens) != 1 or not _is_number(tokens[0]):
        return False

    property_name = _unique_material_property_name(
        root,
        _safe_material_property_name(node.name, "mobility"),
    )
    if not _add_generic_constant_material(root, registry, [property_name], [tokens[0]]):
        return False

    node.params.pop("mobility", None)
    node.params["mob_name"] = property_name
    changes.append(
        f"{path}: materialized numeric mobility={tokens[0]} as mob_name {property_name}"
    )
    return True


def _repair_zero_vector_value_params(
    node: HITNode,
    path: str,
    spec: MooseObjectSpec,
    changes: list[str],
) -> bool:
    changed = False
    for parameter, param_spec in spec.params.items():
        cpp_type = param_spec.cpp_type or ""
        if "VectorValue" not in cpp_type:
            continue
        raw_value = node.params.get(parameter)
        if raw_value is None:
            continue
        tokens = _split_hit_tokens(raw_value)
        if len(tokens) != 1:
            continue
        value = _parse_float(tokens[0])
        if value not in (0.0, -0.0):
            continue
        node.params[parameter] = "'0 0 0'"
        changes.append(f"{path}: expanded zero {parameter} scalar to 3-vector")
        changed = True
    return changed


def _repair_heat_source_parameter(
    root: HITNode,
    node: HITNode,
    path: str,
    type_name: str,
    changes: list[str],
) -> bool:
    if type_name != "HeatSource" or "heat_source" not in node.params:
        return False

    raw_value = node.params.pop("heat_source")
    numeric = raw_value.strip().strip("'\"")
    if not _is_number(numeric):
        material_value = _lookup_generic_material_property_value(root, numeric)
        if material_value is None:
            node.params["heat_source"] = raw_value
            return False
        numeric = material_value
    node.params["value"] = numeric
    changes.append(f"{path}: converted heat_source to HeatSource value")
    return True


def _repair_adrobin_alpha_beta_gamma(
    node: HITNode,
    path: str,
    type_name: str,
    changes: list[str],
) -> bool:
    if type_name != "ADRobinBC":
        return False
    if not {"alpha", "beta", "gamma"}.issubset(node.params):
        return False

    alpha = _parse_float(node.params.get("alpha"))
    beta = _parse_float(node.params.get("beta"))
    gamma = _parse_float(node.params.get("gamma"))
    if alpha is None or beta in (None, 0.0) or gamma not in (0.0, -0.0):
        return False

    node.params["coefficient"] = _format_float(alpha / beta)
    node.params.pop("alpha")
    node.params.pop("beta")
    node.params.pop("gamma")
    changes.append(
        f"{path}: converted homogeneous ADRobinBC alpha/beta/gamma to coefficient"
    )
    return True


def _lookup_generic_material_property_value(root: HITNode, prop_name: str) -> str | None:
    for _material_path, name, value, _node, _index in _generic_constant_material_declarations(root):
        if name == prop_name and value is not None:
            return value
    return None


def _duplicate_material_property_issues(root: HITNode) -> list[MOOSETypeIssue]:
    declarations = _generic_constant_material_declarations(root)
    by_name: dict[str, list[tuple[str, str | None]]] = {}
    for material_path, prop_name, prop_value, _node, _index in declarations:
        by_name.setdefault(prop_name, []).append((material_path, prop_value))

    issues: list[MOOSETypeIssue] = []
    for prop_name, entries in by_name.items():
        if len(entries) < 2:
            continue
        values = {value for _path, value in entries}
        autofixable = len(values) == 1 and None not in values
        issues.append(MOOSETypeIssue(
            severity="error",
            kind="duplicate_material_property",
            path="/Materials",
            message=(
                f"Material property {prop_name!r} is declared by multiple "
                "GenericConstantMaterial blocks."
            ),
            suggestions=[path for path, _value in entries],
            autofixable=autofixable,
            context={
                "property_name": prop_name,
                "values": sorted(value for value in values if value is not None),
            },
        ))
    return issues


def _generic_constant_material_declarations(
    root: HITNode,
) -> list[tuple[str, str, str | None, HITNode, int]]:
    materials = root.find("/Materials")
    if materials is None:
        return []

    declarations: list[tuple[str, str, str | None, HITNode, int]] = []
    for material in materials.children:
        if material.param("type") not in {"GenericConstantMaterial", "ADGenericConstantMaterial"}:
            continue
        prop_names = _split_hit_tokens(material.param("prop_names") or "")
        prop_values = _split_hit_tokens(material.param("prop_values") or "")
        for index, prop_name in enumerate(prop_names):
            prop_value = prop_values[index] if index < len(prop_values) else None
            declarations.append((
                f"/Materials/{material.name}",
                prop_name,
                prop_value,
                material,
                index,
            ))
    return declarations


def _repair_duplicate_generic_constant_material_properties(
    root: HITNode,
    changes: list[str],
) -> bool:
    declarations = _generic_constant_material_declarations(root)
    first_value: dict[str, str | None] = {}
    updates: dict[int, list[str]] = {}
    reserved = set(_material_property_names(root))
    changed = False

    for material_path, prop_name, prop_value, material, index in declarations:
        if prop_name not in first_value:
            first_value[prop_name] = prop_value
            continue
        if first_value[prop_name] != prop_value or prop_value is None:
            continue

        material_id = id(material)
        names = updates.get(material_id)
        if names is None:
            names = _split_hit_tokens(material.param("prop_names") or "")
            updates[material_id] = names
        new_name = _unique_material_property_name(
            root,
            _safe_material_property_name(material.name, prop_name),
            reserved=reserved,
        )
        reserved.add(new_name)
        names[index] = new_name
        changes.append(
            f"{material_path}: duplicate material property {prop_name} -> {new_name}"
        )
        changed = True

    if not changed:
        return False

    material_by_id = {
        id(material): material
        for _path, _prop_name, _prop_value, material, _index in declarations
    }
    for material_id, names in updates.items():
        material_by_id[material_id].params["prop_names"] = "'" + " ".join(names) + "'"
    return True


def _repair_heat_time_density_specific_heat(
    root: HITNode,
    node: HITNode,
    path: str,
    type_name: str,
    registry: MooseRegistry,
    changes: list[str],
) -> bool:
    if type_name not in {"HeatConductionTimeDerivative", "ADHeatConductionTimeDerivative"}:
        return False
    raw_value = node.params.get("density_specific_heat")
    if raw_value is None or not _is_number(raw_value):
        return False

    node.params.pop("density_specific_heat")
    changed = True
    if "density_name" not in node.params and "specific_heat" not in node.params:
        density_name = _unique_material_property_name(
            root,
            _safe_material_property_name(node.name, "density"),
        )
        specific_heat_name = _unique_material_property_name(
            root,
            _safe_material_property_name(node.name, "specific_heat"),
            reserved={density_name},
        )
        if _add_generic_constant_material(
            root,
            registry,
            [density_name, specific_heat_name],
            ["1.0", raw_value.strip().strip("'\"")],
        ):
            node.params["density_name"] = density_name
            node.params["specific_heat"] = specific_heat_name
            changes.append(
                f"{path}: materialized density_specific_heat as "
                "density_name/specific_heat properties"
            )
        else:
            changes.append(f"{path}: removed unsupported density_specific_heat parameter")
        return changed

    changes.append(f"{path}: removed unsupported density_specific_heat parameter")
    return changed


def _repair_numeric_material_property_params(
    root: HITNode,
    node: HITNode,
    path: str,
    spec: MooseObjectSpec,
    registry: MooseRegistry,
    changes: list[str],
) -> bool:
    changed = False
    reserved: set[str] = set()
    for parameter, param_spec in spec.params.items():
        if not _is_material_property_param(param_spec.cpp_type):
            continue
        raw_value = node.params.get(parameter)
        if raw_value is None:
            continue
        tokens = _split_hit_tokens(raw_value)
        if len(tokens) != 1 or not _is_number(tokens[0]):
            continue

        property_name = _unique_material_property_name(
            root,
            _safe_material_property_name(node.name, parameter),
            reserved=reserved,
        )
        if not _add_generic_constant_material(root, registry, [property_name], [tokens[0]]):
            continue
        reserved.add(property_name)
        node.params[parameter] = property_name
        changes.append(
            f"{path}: materialized numeric {parameter}={tokens[0]} as property {property_name}"
        )
        changed = True
    return changed


def _generic_constant_material_type(registry: MooseRegistry) -> str | None:
    for type_name in ("GenericConstantMaterial", "ADGenericConstantMaterial"):
        if registry.is_valid_in_context(type_name, "Materials/*"):
            return type_name
    return None


def _add_generic_constant_material(
    root: HITNode,
    registry: MooseRegistry,
    prop_names: list[str],
    prop_values: list[str],
) -> bool:
    material_type = _generic_constant_material_type(registry)
    if material_type is None:
        return False

    materials = root.find("/Materials")
    if materials is None:
        materials = HITNode(name="Materials")
        root.children.append(materials)

    block_base = "__cm_" + "_".join(prop_names)
    block_name = _unique_child_name(materials, block_base)
    materials.children.append(HITNode(
        name=block_name,
        params={
            "type": material_type,
            "prop_names": "'" + " ".join(prop_names) + "'",
            "prop_values": "'" + " ".join(prop_values) + "'",
        },
    ))
    return True


def _safe_material_property_name(node_name: str, parameter: str) -> str:
    stem = f"{node_name}_{parameter}".strip("_") or parameter
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in stem)
    cleaned = cleaned.strip("_")
    return cleaned or parameter


def _unique_material_property_name(
    root: HITNode,
    base: str,
    *,
    reserved: set[str] | None = None,
) -> str:
    used = set(_material_property_names(root))
    if reserved:
        used.update(reserved)
    if base not in used:
        return base
    index = 2
    while f"{base}_{index}" in used:
        index += 1
    return f"{base}_{index}"


def _unique_child_name(parent: HITNode, base: str) -> str:
    existing = {child.name for child in parent.children}
    if base not in existing:
        return base
    index = 2
    while f"{base}_{index}" in existing:
        index += 1
    return f"{base}_{index}"


def _repair_petsc_options(
    node: HITNode,
    path: str,
    changes: list[str],
) -> bool:
    value = node.params.get("petsc_options")
    if value is None:
        return False

    tokens = _split_hit_tokens(value)
    if len(tokens) < 2 or len(tokens) % 2 != 0:
        return False
    names = tokens[0::2]
    values = tokens[1::2]
    if not all(name.startswith("-") for name in names):
        return False
    if not all(not value.startswith("-") for value in values):
        return False

    node.params.pop("petsc_options")
    node.params.setdefault("petsc_options_iname", "'" + " ".join(names) + "'")
    node.params.setdefault("petsc_options_value", "'" + " ".join(values) + "'")
    changes.append(f"{path}: split petsc_options into petsc_options_iname/value")
    return True


def _safe_param_alias(param: str, allowed: set[str]) -> str | None:
    aliases = {
        "coeff": ("coef", "coefficient", "value", "rate"),
        "coefficient": ("Coefficient", "coef", "value", "rate"),
        "coef": ("Coefficient", "coefficient", "value", "rate"),
        "diffusion_coefficient": ("diffusivity",),
        "mat_prop": ("material_property", "material_properties"),
    }
    for candidate in aliases.get(param, ()):
        if candidate in allowed:
            return candidate
    return None


def dump_hit(root: HITNode) -> str:
    """Serialize a parsed HIT tree in a stable, comment-free form."""
    lines: list[str] = []

    def write(node: HITNode, indent: int) -> None:
        pad = " " * indent
        if node.name:
            lines.append(f"{pad}[{node.name}]")
            pad = " " * (indent + 2)
        for key, value in node.params.items():
            lines.append(f"{pad}{key} = {value}")
        for child in node.children:
            write(child, indent + (2 if node.name else 0))
        if node.name:
            lines.append(" " * indent + "[]")

    for child in root.children:
        write(child, 0)
    return "\n".join(lines) + "\n"
