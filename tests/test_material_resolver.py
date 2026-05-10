# tests/test_material_resolver.py
"""Tests for material property resolver."""

import pytest

from codmos.multiagent.pde.material_resolver import MaterialResolver
from codmos.multiagent.validators.hit_parser import HITNode, load


def _make_materials_node(hit_text: str) -> HITNode:
    """Parse a [Materials] block and return its HITNode."""
    root = load(hit_text)
    node = root.find("Materials")
    assert node is not None
    return node


class TestMaterialResolver:
    def test_generic_constant_single_property(self):
        node = _make_materials_node("""
[Materials]
  [steel]
    type = GenericConstantMaterial
    prop_names = 'thermal_conductivity'
    prop_values = '45.0'
  []
[]
""")
        resolver = MaterialResolver(node)
        assert resolver.resolve("thermal_conductivity") == 45.0

    def test_generic_constant_multiple_properties(self):
        node = _make_materials_node("""
[Materials]
  [mat1]
    type = GenericConstantMaterial
    prop_names = 'thermal_conductivity density specific_heat'
    prop_values = '45.0 7800 500'
  []
[]
""")
        resolver = MaterialResolver(node)
        assert resolver.resolve("thermal_conductivity") == 45.0
        assert resolver.resolve("density") == 7800.0
        assert resolver.resolve("specific_heat") == 500.0

    def test_unknown_property_returns_none(self):
        node = _make_materials_node("""
[Materials]
  [mat1]
    type = GenericConstantMaterial
    prop_names = 'thermal_conductivity'
    prop_values = '45.0'
  []
[]
""")
        resolver = MaterialResolver(node)
        assert resolver.resolve("nonexistent_prop") is None

    def test_non_numeric_value_returns_string(self):
        node = _make_materials_node("""
[Materials]
  [mat1]
    type = GenericConstantMaterial
    prop_names = 'diffusivity'
    prop_values = 'some_function'
  []
[]
""")
        resolver = MaterialResolver(node)
        assert resolver.resolve("diffusivity") == "some_function"

    def test_multiple_material_blocks(self):
        node = _make_materials_node("""
[Materials]
  [thermal]
    type = GenericConstantMaterial
    prop_names = 'thermal_conductivity'
    prop_values = '45.0'
  []
  [mechanical]
    type = GenericConstantMaterial
    prop_names = 'youngs_modulus'
    prop_values = '2e11'
  []
[]
""")
        resolver = MaterialResolver(node)
        assert resolver.resolve("thermal_conductivity") == 45.0
        assert resolver.resolve("youngs_modulus") == 2e11

    def test_empty_materials_block(self):
        node = _make_materials_node("""
[Materials]
[]
""")
        resolver = MaterialResolver(node)
        assert resolver.resolve("anything") is None

    def test_non_generic_material_skipped(self):
        node = _make_materials_node("""
[Materials]
  [elasticity]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 2e11
    poissons_ratio = 0.3
  []
[]
""")
        resolver = MaterialResolver(node)
        # ComputeIsotropicElasticityTensor is a known direct-param material
        assert resolver.resolve("youngs_modulus") == 2e11
        assert resolver.resolve("poissons_ratio") == 0.3

    def test_quoted_prop_names_values(self):
        node = _make_materials_node("""
[Materials]
  [mat1]
    type = GenericConstantMaterial
    prop_names = 'k rho cp'
    prop_values = '10 1000 500'
  []
[]
""")
        resolver = MaterialResolver(node)
        assert resolver.resolve("k") == 10.0
        assert resolver.resolve("rho") == 1000.0
        assert resolver.resolve("cp") == 500.0

    def test_records_preserve_material_context(self):
        node = _make_materials_node("""
[Materials]
  [steel]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = '2e11'
    poissons_ratio = 0.3
  []
[]
""")
        resolver = MaterialResolver(node)
        records = resolver.records
        assert len(records) == 2
        assert records[0].name == "youngs_modulus"
        assert records[0].value == 2e11
        assert records[0].material_name == "steel"
        assert records[0].material_type == "ComputeIsotropicElasticityTensor"
