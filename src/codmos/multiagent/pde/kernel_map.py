# src/codmos/multiagent/pde/kernel_map.py
"""Kernel-PDE mapping table loader for the paper artifact.

The artifact mapping keeps the executable metadata used by deterministic
reconstruction and IFS: normalized operator type, coefficient extraction,
severity, equivalence groups, and source traceability. Descriptor-level display
annotations are presented in the paper text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_DEFAULT_YAML = Path(__file__).resolve().parents[4] / "data" / "pde_mapping" / "kernel_map.yaml"


@dataclass
class KernelPDEMapping:
    """Mapping from a single MOOSE kernel class to its PDE semantics."""

    kernel_class: str
    operator: str
    variable_param: str | None
    coefficient_param: str | None
    coupled_param: str | None
    severity: str
    equivalent_to: list[str] = field(default_factory=list)
    all_parameters: list[str] = field(default_factory=list)
    source_doc: str | None = None


@dataclass
class BCPDEMapping:
    """Mapping from a single MOOSE BC class to its PDE semantics."""

    bc_class: str
    bc_type: str  # normalized: Dirichlet / Neumann / Robin / Periodic / etc.
    value_param: str | None
    severity: str
    variable_param: str | None = None
    boundary_param: str | None = None
    function_param: str | None = None
    transient_only: bool = False
    equivalent_to: list[str] = field(default_factory=list)
    all_parameters: list[str] = field(default_factory=list)
    source_doc: str | None = None


@dataclass
class ICPDEMapping:
    """Mapping from a single MOOSE IC class to its PDE semantics."""

    ic_class: str
    ic_type: str  # constant / function / etc.
    variable_param: str | None
    value_param: str | None
    severity: str
    function_param: str | None = None
    transient_only: bool = True
    equivalent_to: list[str] = field(default_factory=list)
    all_parameters: list[str] = field(default_factory=list)
    source_doc: str | None = None


_BC_TYPE_NORMALIZE: dict[str, str] = {
    "dirichlet": "Dirichlet",
    "neumann": "Neumann",
    "neumann_like": "Neumann",
    "robin": "Robin",
    "robin_like": "Robin",
    "traction": "Neumann",
    "matched_value": "Dirichlet",
    "other_bc": "Other",
    "periodic": "Periodic",
}


def _capitalize_type(raw: str) -> str:
    """Normalize condition_type to standard PDE BC classification.

    Maps collaborator's fine-grained types to standard mathematical types:
      dirichlet → Dirichlet
      neumann_like → Neumann (includes flux BCs, convective BCs)
      robin_like → Robin
      traction → Neumann (stress-based Neumann)
      matched_value → Dirichlet (value-matching)
    """
    if not raw:
        return raw
    return _BC_TYPE_NORMALIZE.get(raw, raw.capitalize())


class KernelMap:
    """Loads and queries the kernel-PDE mapping table (schema v3.0).

    Usage::

        km = KernelMap()  # loads default YAML
        m = km.get_kernel("HeatConduction")
        assert m.operator == "diffusion"
    """

    def __init__(self, yaml_path: str | Path | None = None) -> None:
        path = Path(yaml_path) if yaml_path else _DEFAULT_YAML
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # --- Kernels ---
        self._kernels: dict[str, KernelPDEMapping] = {}
        for name, entry in (raw.get("kernels") or {}).items():
            self._kernels[name] = KernelPDEMapping(
                kernel_class=name,
                operator=entry["operator"],
                variable_param=entry.get("variable_param"),
                coefficient_param=entry.get("coefficient_param"),
                coupled_param=entry.get("coupled_param"),
                severity=entry["severity"],
                equivalent_to=entry.get("equivalent_to") or [],
                all_parameters=entry.get("all_parameters") or [],
                source_doc=entry.get("source_doc"),
            )

        # --- BCs (schema v3.0 uses 'bcs' key, 'condition_type' field) ---
        self._bcs: dict[str, BCPDEMapping] = {}
        for name, entry in (raw.get("bcs") or raw.get("boundary_conditions") or {}).items():
            raw_type = entry.get("condition_type") or entry.get("bc_type", "")
            self._bcs[name] = BCPDEMapping(
                bc_class=name,
                bc_type=_capitalize_type(raw_type),
                value_param=entry.get("value_param"),
                severity=entry["severity"],
                variable_param=entry.get("variable_param"),
                boundary_param=entry.get("boundary_param"),
                function_param=entry.get("function_param"),
                transient_only=entry.get("transient_only", False),
                equivalent_to=entry.get("equivalent_to") or [],
                all_parameters=entry.get("all_parameters") or [],
                source_doc=entry.get("source_doc"),
            )

        # --- ICs (new in schema v3.0) ---
        self._ics: dict[str, ICPDEMapping] = {}
        for name, entry in (raw.get("ics") or {}).items():
            self._ics[name] = ICPDEMapping(
                ic_class=name,
                ic_type=entry.get("ic_type", "constant"),
                variable_param=entry.get("variable_param"),
                value_param=entry.get("value_param"),
                severity=entry["severity"],
                function_param=entry.get("function_param"),
                transient_only=entry.get("transient_only", True),
                equivalent_to=entry.get("equivalent_to") or [],
                all_parameters=entry.get("all_parameters") or [],
                source_doc=entry.get("source_doc"),
            )

    def get_kernel(self, kernel_class: str) -> KernelPDEMapping | None:
        """Look up a MOOSE kernel class. Returns ``None`` for unknown kernels."""
        return self._kernels.get(kernel_class)

    def get_bc(self, bc_class: str) -> BCPDEMapping | None:
        """Look up a MOOSE BC class. Returns ``None`` for unknown BCs."""
        return self._bcs.get(bc_class)

    def get_ic(self, ic_class: str) -> ICPDEMapping | None:
        """Look up a MOOSE IC class. Returns ``None`` for unknown ICs."""
        return self._ics.get(ic_class)

    def list_operators(self) -> list[str]:
        """All unique PDE operators in the mapping (no duplicates)."""
        return sorted({m.operator for m in self._kernels.values()})
