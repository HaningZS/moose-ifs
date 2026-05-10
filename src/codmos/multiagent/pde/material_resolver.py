# src/codmos/multiagent/pde/material_resolver.py
"""Material property resolver for MOOSE [Materials] blocks.

Layer 1 — depends only on HITNode from the existing HIT parser.

Extracts numeric property values from multiple material types:
  - GenericConstantMaterial / ADGenericConstantMaterial (prop_names/prop_values)
  - HeatConductionMaterial (thermal_conductivity, specific_heat)
  - Any material with directly-named numeric parameters (fallback)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from codmos.multiagent.validators.hit_parser import HITNode

logger = logging.getLogger(__name__)

# Material types that use prop_names/prop_values pattern
_GENERIC_TYPES = {
    "GenericConstantMaterial",
    "ADGenericConstantMaterial",
    "GenericFunctionMaterial",
    "ADGenericFunctionMaterial",
}

# Material types with known direct-parameter properties.
# These are extracted first; any remaining numeric params are also scanned
# via the fallback strategy to avoid losing resolution.
_DIRECT_PARAM_MATERIALS: dict[str, list[str]] = {
    "HeatConductionMaterial": ["thermal_conductivity", "specific_heat"],
    "ADHeatConductionMaterial": ["thermal_conductivity", "specific_heat"],
    "ComputeIsotropicElasticityTensor": ["youngs_modulus", "poissons_ratio"],
    "Density": ["density"],
    "ADDensity": ["density"],
    "PorousFlowPermeabilityConst": ["permeability"],
    "PorousFlowConstantBiotModulus": ["biot_coefficient", "fluid_bulk_modulus"],
    "IsotropicPlasticityStressUpdate": ["yield_stress", "hardening_constant"],
    "ADIsotropicPlasticityStressUpdate": ["yield_stress", "hardening_constant"],
    "SimpleFluidProperties": ["viscosity", "density0", "thermal_expansion"],
}


@dataclass(frozen=True)
class MaterialPropertyRecord:
    """A material property with its originating block context."""

    name: str
    value: float | str
    material_name: str
    material_type: str


class MaterialResolver:
    """Resolve material property names to numeric values.

    Scans ``[Materials]`` sub-blocks and extracts properties from:
    1. Generic materials (prop_names/prop_values positional matching)
    2. Specialized materials with known direct parameters
    3. Any material block where a parameter name matches the query

    Usage::

        resolver = MaterialResolver(materials_node)
        k = resolver.resolve("thermal_conductivity")  # -> 45.0 or None
    """

    def __init__(self, materials_node: HITNode) -> None:
        self._properties: dict[str, float | str] = {}
        self._records: list[MaterialPropertyRecord] = []
        self._parse(materials_node)

    @property
    def records(self) -> tuple[MaterialPropertyRecord, ...]:
        """All extracted material properties, preserving duplicates and context."""
        return tuple(self._records)

    def _parse(self, node: HITNode) -> None:
        for child in node.children:
            mat_type = child.param("type")
            if not mat_type:
                continue

            # Strategy 1: Generic prop_names/prop_values pattern
            if mat_type in _GENERIC_TYPES:
                self._parse_generic(child, mat_type)
                continue

            # Strategy 2: Known direct-parameter materials
            if mat_type in _DIRECT_PARAM_MATERIALS:
                for param_name in _DIRECT_PARAM_MATERIALS[mat_type]:
                    raw = child.param(param_name)
                    if raw is not None:
                        self._store(param_name, raw, child.name, mat_type)
                # Also run fallback to capture any additional numeric params
                # not listed in the known set (purely additive, no regression)
                self._parse_fallback(
                    child,
                    skip_names=set(_DIRECT_PARAM_MATERIALS[mat_type]),
                )
                continue

            # Strategy 3: Fallback — scan all parameters for numeric values
            # Only extract params that look like physics properties (not control params)
            self._parse_fallback(child)

    def _parse_generic(self, child: HITNode, mat_type: str) -> None:
        """Parse GenericConstantMaterial prop_names/prop_values."""
        raw_names = child.param("prop_names", "")
        raw_values = child.param("prop_values", "")
        if not raw_names or not raw_values:
            return
        names = _strip_quotes_and_split(raw_names)
        values = _strip_quotes_and_split(raw_values)
        for name, raw_val in zip(names, values, strict=False):
            self._store(name, raw_val, child.name, mat_type)

    def _parse_fallback(
        self,
        child: HITNode,
        skip_names: set[str] | None = None,
    ) -> None:
        """Extract directly-named numeric parameters from any material block."""
        _SKIP_PARAMS = {
            "type", "block", "boundary", "displacements", "use_displaced_mesh",
            "outputs", "output_properties", "implicit", "constant_on",
            "compute", "f_name", "function", "coupled_variables",
            "base_name", "eigenstrain_name",
        }
        skip_names = skip_names or set()
        for name, raw in child.params.items():
            if name in _SKIP_PARAMS or name in skip_names or name.startswith("_"):
                continue
            try:
                float(_strip_surrounding_quotes(str(raw).strip()))
                self._store(name, raw, child.name, child.param("type") or "")
            except (ValueError, TypeError):
                pass  # skip non-numeric

    def _store(
        self,
        name: str,
        raw_val: str,
        material_name: str,
        material_type: str,
    ) -> None:
        """Store a property, converting to float if possible."""
        raw_val = _strip_surrounding_quotes(str(raw_val).strip())
        try:
            value: float | str = float(raw_val)
        except (ValueError, TypeError):
            value = raw_val
        self._properties[name] = value
        self._records.append(MaterialPropertyRecord(
            name=name,
            value=value,
            material_name=material_name,
            material_type=material_type,
        ))

    def resolve(self, property_name: str) -> float | str | None:
        """Resolve a property name to its value.

        Returns:
            ``float`` if the value is numeric,
            ``str`` if the value is a non-numeric expression,
            ``None`` if the property is not found.
        """
        return self._properties.get(property_name)


def _strip_quotes_and_split(raw: str) -> list[str]:
    """Strip surrounding quotes and split on whitespace."""
    stripped = _strip_surrounding_quotes(raw.strip())
    return stripped.split()


def _strip_surrounding_quotes(raw: str) -> str:
    """Strip one matching layer of surrounding quotes."""
    if raw and raw[0] in ("'", '"') and raw[-1] == raw[0]:
        return raw[1:-1]
    return raw
