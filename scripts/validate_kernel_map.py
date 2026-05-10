#!/usr/bin/env python3
"""Validate artifact kernel_map.yaml consistency.

The artifact mapping contains executable reconstruction metadata: normalized
operator labels, coefficient extraction metadata, severity, equivalence groups,
and source traceability. Descriptor display annotations are presented in the
paper text.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codmos.multiagent.pde.kernel_map import KernelMap

VALID_SEVERITIES = {"highest", "very_high", "high", "medium", "medium_low", "low"}
FORBIDDEN_KEYS = {
    "operator_types",
    "valid_trial_ops",
    "valid_test_ops",
    "valid_contractions",
}
FORBIDDEN_KEYS.update({f"weak_{name}" for name in ["form_tuple", "form_latex", "form_description", "form_code", "form_inherited_from"]})
FORBIDDEN_KEYS.add("math_" + "form_latex")


def _walk_forbidden(obj: object, path: str = "root") -> list[str]:
    issues: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{path}.{key}"
            if key in FORBIDDEN_KEYS:
                issues.append(child)
            issues.extend(_walk_forbidden(value, child))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            issues.extend(_walk_forbidden(value, f"{path}[{idx}]"))
    return issues


def validate_raw_schema(path: Path) -> list[str]:
    raw = yaml.safe_load(path.read_text())
    issues = [f"forbidden display field retained: {p}" for p in _walk_forbidden(raw)]
    for section in ("kernels", "bcs", "ics"):
        if section not in raw or not isinstance(raw[section], dict) or not raw[section]:
            issues.append(f"missing or empty section: {section}")
    return issues


def validate_bidirectional_equivalence(km: KernelMap) -> list[str]:
    issues: list[str] = []
    for name, mapping in km._kernels.items():
        for equiv_name in mapping.equivalent_to:
            equiv = km.get_kernel(equiv_name)
            if equiv is None:
                issues.append(f"{name}: equivalent_to unknown kernel {equiv_name}")
                continue
            if name not in equiv.equivalent_to:
                issues.append(f"{name} -> {equiv_name} but not vice versa")
    return issues


def validate_severity_values(km: KernelMap) -> list[str]:
    issues: list[str] = []
    for mapping in km._kernels.values():
        if mapping.severity not in VALID_SEVERITIES:
            issues.append(f"{mapping.kernel_class}: invalid severity {mapping.severity}")
    for mapping in km._bcs.values():
        if mapping.severity not in VALID_SEVERITIES:
            issues.append(f"{mapping.bc_class}: invalid severity {mapping.severity}")
    for mapping in km._ics.values():
        if mapping.severity not in VALID_SEVERITIES:
            issues.append(f"{mapping.ic_class}: invalid severity {mapping.severity}")
    return issues


def validate_required_coverage(km: KernelMap) -> list[str]:
    issues: list[str] = []
    for name in ["Diffusion", "HeatConduction", "TimeDerivative", "BodyForce", "CoupledForce", "Reaction"]:
        if km.get_kernel(name) is None:
            issues.append(f"missing required kernel: {name}")
    for name in ["DirichletBC", "NeumannBC", "FunctionDirichletBC"]:
        if km.get_bc(name) is None:
            issues.append(f"missing required BC: {name}")
    for name in ["ConstantIC", "FunctionIC"]:
        if km.get_ic(name) is None:
            issues.append(f"missing required IC: {name}")
    return issues


def main() -> bool:
    path = Path(__file__).resolve().parents[1] / "data" / "pde_mapping" / "kernel_map.yaml"
    km = KernelMap(path)

    checks = {
        "raw schema": validate_raw_schema(path),
        "severity values": validate_severity_values(km),
        "required coverage": validate_required_coverage(km),
        "bidirectional equivalence": validate_bidirectional_equivalence(km),
    }

    print("kernel_map.yaml artifact validation")
    print(f"  kernels: {len(km._kernels)}")
    print(f"  bcs:     {len(km._bcs)}")
    print(f"  ics:     {len(km._ics)}")
    print(f"  operators: {len(km.list_operators())}")

    total = 0
    for label, issues in checks.items():
        total += len(issues)
        status = "PASS" if not issues else f"FAIL ({len(issues)})"
        print(f"\n[{label}] {status}")
        for issue in issues:
            print(f"  - {issue}")

    print(f"\nTotal issues: {total}")
    if total == 0:
        print("PASS: kernel_map.yaml validation passed")
    return total == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
