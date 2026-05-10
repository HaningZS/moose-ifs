"""Tests for SGSA L2 MOOSE type registry validation."""

from __future__ import annotations

from codmos.multiagent.moose_registry import MooseRegistry
from codmos.multiagent.validators import MOOSETypeValidator


def _registry() -> MooseRegistry:
    return MooseRegistry.from_moose_json({
        "blocks": {
            "Mesh": {
                "types": {
                    "GeneratedMesh": {
                        "parent_syntax": "Mesh",
                        "label": "MooseApp",
                        "moose_base": "MooseMesh",
                        "parameters": {
                            "type": {"name": "type", "required": False},
                            "dim": {"name": "dim", "required": True},
                            "nx": {"name": "nx", "required": False},
                        },
                    },
                },
                "subblock_types": {
                    "GeneratedMeshGenerator": {
                        "parent_syntax": "Mesh/*",
                        "label": "MooseApp",
                        "moose_base": "MeshGenerator",
                        "parameters": {
                            "type": {"name": "type", "required": False},
                            "dim": {"name": "dim", "required": True},
                            "nx": {"name": "nx", "required": False},
                        },
                    },
                },
            },
            "Kernels": {
                "subblock_types": {
                    "Diffusion": {
                        "parent_syntax": "Kernels/*",
                        "label": "MooseApp",
                        "moose_base": "Kernel",
                        "parameters": {
                            "type": {"name": "type", "required": False},
                            "variable": {"name": "variable", "required": True},
                        },
                    },
                    "CoupledForce": {
                        "parent_syntax": "Kernels/*",
                        "label": "MooseApp",
                        "moose_base": "Kernel",
                        "parameters": {
                            "type": {"name": "type", "required": False},
                            "variable": {"name": "variable", "required": True},
                            "v": {"name": "v", "required": True},
                            "coef": {"name": "coef", "required": False},
                        },
                    },
                    "MatCoupledForce": {
                        "parent_syntax": "Kernels/*",
                        "label": "MooseApp",
                        "moose_base": "Kernel",
                        "parameters": {
                            "type": {"name": "type", "required": False},
                            "variable": {"name": "variable", "required": True},
                            "v": {"name": "v", "required": True},
                            "coef": {"name": "coef", "required": False},
                            "material_properties": {
                                "name": "material_properties",
                                "required": False,
                                "cpp_type": "std::vector<MaterialPropertyName>",
                            },
                        },
                    },
                    "CoefTimeDerivative": {
                        "parent_syntax": "Kernels/*",
                        "label": "MooseApp",
                        "moose_base": "Kernel",
                        "parameters": {
                            "type": {"name": "type", "required": False},
                            "variable": {"name": "variable", "required": True},
                            "Coefficient": {"name": "Coefficient", "required": False},
                        },
                    },
                    "HeatSource": {
                        "parent_syntax": "Kernels/*",
                        "label": "HeatTransferApp",
                        "moose_base": "Kernel",
                        "parameters": {
                            "type": {"name": "type", "required": False},
                            "variable": {"name": "variable", "required": True},
                            "value": {"name": "value", "required": False},
                            "function": {"name": "function", "required": False},
                        },
                    },
                    "MatBodyForce": {
                        "parent_syntax": "Kernels/*",
                        "label": "MooseApp",
                        "moose_base": "Kernel",
                        "parameters": {
                            "type": {"name": "type", "required": False},
                            "variable": {"name": "variable", "required": True},
                            "material_property": {
                                "name": "material_property",
                                "required": True,
                                "cpp_type": "MaterialPropertyName",
                            },
                            "value": {"name": "value", "required": False},
                        },
                    },
                    "ADHeatConduction": {
                        "parent_syntax": "Kernels/*",
                        "label": "HeatTransferApp",
                        "moose_base": "Kernel",
                        "parameters": {
                            "type": {"name": "type", "required": False},
                            "variable": {"name": "variable", "required": True},
                            "thermal_conductivity": {
                                "name": "thermal_conductivity",
                                "required": False,
                                "cpp_type": "MaterialPropertyName",
                                "default": "thermal_conductivity",
                            },
                        },
                    },
                    "HeatConduction": {
                        "parent_syntax": "Kernels/*",
                        "label": "HeatTransferApp",
                        "moose_base": "Kernel",
                        "parameters": {
                            "type": {"name": "type", "required": False},
                            "variable": {"name": "variable", "required": True},
                            "diffusion_coefficient": {
                                "name": "diffusion_coefficient",
                                "required": False,
                                "cpp_type": "MaterialPropertyName",
                                "default": "thermal_conductivity",
                            },
                            "diffusion_coefficient_dT": {
                                "name": "diffusion_coefficient_dT",
                                "required": False,
                                "cpp_type": "MaterialPropertyName",
                                "default": "thermal_conductivity_dT",
                            },
                        },
                    },
                    "HeatConductionTimeDerivative": {
                        "parent_syntax": "Kernels/*",
                        "label": "HeatTransferApp",
                        "moose_base": "Kernel",
                        "parameters": {
                            "type": {"name": "type", "required": False},
                            "variable": {"name": "variable", "required": True},
                            "density_name": {
                                "name": "density_name",
                                "required": False,
                                "cpp_type": "MaterialPropertyName",
                                "default": "density",
                            },
                            "specific_heat": {
                                "name": "specific_heat",
                                "required": False,
                                "cpp_type": "MaterialPropertyName",
                                "default": "specific_heat",
                            },
                        },
                    },
                    "CahnHilliard": {
                        "parent_syntax": "Kernels/*",
                        "label": "PhaseFieldApp",
                        "moose_base": "Kernel",
                        "parameters": {
                            "type": {"name": "type", "required": False},
                            "variable": {"name": "variable", "required": True},
                            "f_name": {
                                "name": "f_name",
                                "required": True,
                                "cpp_type": "MaterialPropertyName",
                            },
                            "mob_name": {
                                "name": "mob_name",
                                "required": False,
                                "cpp_type": "MaterialPropertyName",
                                "default": "M",
                            },
                        },
                    },
                    "PorousFlowAdvectiveFlux": {
                        "parent_syntax": "Kernels/*",
                        "label": "PorousFlowApp",
                        "moose_base": "Kernel",
                        "parameters": {
                            "type": {"name": "type", "required": False},
                            "variable": {"name": "variable", "required": True},
                            "gravity": {
                                "name": "gravity",
                                "required": True,
                                "cpp_type": "libMesh::VectorValue<double>",
                            },
                        },
                    },
                },
            },
            "BCs": {
                "subblock_types": {
                    "ADRobinBC": {
                        "parent_syntax": "BCs/*",
                        "label": "MooseApp",
                        "moose_base": "BoundaryCondition",
                        "parameters": {
                            "type": {"name": "type", "required": False},
                            "variable": {"name": "variable", "required": True},
                            "boundary": {"name": "boundary", "required": True},
                            "coefficient": {"name": "coefficient", "required": False},
                        },
                    },
                    "DirichletBC": {
                        "parent_syntax": "BCs/*",
                        "label": "MooseApp",
                        "moose_base": "BoundaryCondition",
                        "parameters": {
                            "type": {"name": "type", "required": False},
                            "variable": {"name": "variable", "required": True},
                            "boundary": {"name": "boundary", "required": True},
                            "value": {"name": "value", "required": False},
                        },
                    },
                    "ADConvectiveHeatFluxBC": {
                        "parent_syntax": "BCs/*",
                        "label": "HeatTransferApp",
                        "moose_base": "BoundaryCondition",
                        "parameters": {
                            "type": {"name": "type", "required": False},
                            "variable": {"name": "variable", "required": True},
                            "boundary": {"name": "boundary", "required": True},
                            "T_infinity": {
                                "name": "T_infinity",
                                "required": False,
                                "cpp_type": "MaterialPropertyName",
                            },
                            "heat_transfer_coefficient": {
                                "name": "heat_transfer_coefficient",
                                "required": False,
                                "cpp_type": "MaterialPropertyName",
                            },
                        },
                    },
                    "ConvectiveHeatFluxBC": {
                        "parent_syntax": "BCs/*",
                        "label": "HeatTransferApp",
                        "moose_base": "BoundaryCondition",
                        "parameters": {
                            "type": {"name": "type", "required": False},
                            "variable": {"name": "variable", "required": True},
                            "boundary": {"name": "boundary", "required": True},
                            "T_infinity": {
                                "name": "T_infinity",
                                "required": False,
                                "cpp_type": "MaterialPropertyName",
                            },
                            "heat_transfer_coefficient": {
                                "name": "heat_transfer_coefficient",
                                "required": False,
                                "cpp_type": "MaterialPropertyName",
                            },
                        },
                    },
                },
            },
            "ICs": {
                "subblock_types": {
                    "ConstantIC": {
                        "parent_syntax": "ICs/*",
                        "label": "MooseApp",
                        "moose_base": "InitialCondition",
                        "parameters": {
                            "type": {"name": "type", "required": False},
                            "variable": {"name": "variable", "required": True},
                            "value": {"name": "value", "required": False},
                        },
                    },
                    "FunctionIC": {
                        "parent_syntax": "ICs/*",
                        "label": "MooseApp",
                        "moose_base": "InitialCondition",
                        "parameters": {
                            "type": {"name": "type", "required": False},
                            "variable": {"name": "variable", "required": True},
                            "function": {"name": "function", "required": True},
                        },
                    },
                },
            },
            "Materials": {
                "subblock_types": {
                    "GenericConstantMaterial": {
                        "parent_syntax": "Materials/*",
                        "label": "MooseApp",
                        "moose_base": "Material",
                        "parameters": {
                            "type": {"name": "type", "required": False},
                            "prop_names": {"name": "prop_names", "required": True},
                            "prop_values": {"name": "prop_values", "required": True},
                        },
                    },
                },
            },
            "Preconditioning": {
                "subblock_types": {
                    "SMP": {
                        "parent_syntax": "Preconditioning/*",
                        "label": "MooseApp",
                        "moose_base": "MooseObject",
                        "parameters": {
                            "type": {"name": "type", "required": False},
                            "full": {"name": "full", "required": False},
                        },
                    },
                },
            },
            "Executioner": {
                "types": {
                    "Transient": {
                        "parent_syntax": "Executioner",
                        "label": "MooseApp",
                        "moose_base": "Executioner",
                        "parameters": {
                            "type": {"name": "type", "required": False},
                            "petsc_options": {"name": "petsc_options", "required": False},
                            "petsc_options_iname": {
                                "name": "petsc_options_iname",
                                "required": False,
                            },
                            "petsc_options_value": {
                                "name": "petsc_options_value",
                                "required": False,
                            },
                            "nl_abs_tol": {"name": "nl_abs_tol", "required": False},
                        },
                    },
                },
            },
        }
    })


def test_accepts_context_valid_object_and_params():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 2
[]
[Variables]
  [u]
  []
[]
[Kernels]
  [diff]
    type = Diffusion
    variable = u
  []
[]
"""
    result = MOOSETypeValidator(_registry()).validate(code)
    assert result.passed
    assert result.issues == []


def test_source_scanner_reads_registered_moose_source(tmp_path):
    source = tmp_path / "moose" / "framework" / "src" / "kernels" / "ExampleKernel.C"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
registerMooseObject("MooseApp", ExampleKernel);

InputParameters
ExampleKernel::validParams()
{
  auto params = Kernel::validParams();
  params.addRequiredParam<Real>("coef", "coefficient");
  params.addCoupledVar("v", 0, "coupled variable");
  return params;
}
""",
        encoding="utf-8",
    )
    registry = MooseRegistry.from_moose_source(tmp_path / "moose")
    spec = registry.get("ExampleKernel")

    assert spec is not None
    assert spec.contexts == {"Kernels/*"}
    assert {"coef", "v"}.issubset(spec.params)
    assert not spec.complete_params


def test_detects_context_mismatch_for_mesh_object_in_generator_slot():
    code = """
[Mesh]
  [gen]
    type = GeneratedMesh
    dim = 2
  []
[]
"""
    result = MOOSETypeValidator(_registry()).validate(code)
    assert not result.passed
    assert result.issues[0].kind == "context_mismatch"
    assert result.issues[0].autofixable


def test_detects_missing_mesh_generation():
    code = """
[Variables]
  [u]
  []
[]
"""
    result = MOOSETypeValidator(_registry()).validate(code)
    assert not result.passed
    assert result.issues[0].kind == "missing_mesh_generation"


def test_detects_missing_type_in_typed_instance_block():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Variables]
  [u]
  []
[]
[BCs]
  [left]
    variable = u
    boundary = left
    value = 0
  []
[]
"""
    result = MOOSETypeValidator(_registry()).validate(code)
    assert not result.passed
    assert any(issue.kind == "missing_type" for issue in result.issues)


def test_detects_unknown_variable_and_boundary_references():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Variables]
  [u]
  []
[]
[BCs]
  [bad]
    type = ADRobinBC
    variable = v
    boundary = top
  []
[]
"""
    result = MOOSETypeValidator(_registry()).validate(code)
    kinds = {issue.kind for issue in result.issues}
    assert "unknown_variable" in kinds
    assert "unknown_boundary" in kinds


def test_detects_missing_material_property_reference():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Variables]
  [T]
  []
[]
[Kernels]
  [diff]
    type = ADHeatConduction
    variable = T
  []
[]
"""
    result = MOOSETypeValidator(_registry()).validate(code)
    assert not result.passed
    assert any(issue.kind == "missing_material_property" for issue in result.issues)


def test_detects_and_repairs_numeric_material_property_parameter():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Variables]
  [T]
  []
[]
[Kernels]
  [diff]
    type = ADHeatConduction
    variable = T
    thermal_conductivity = 45.0
  []
[]
"""
    validator = MOOSETypeValidator(_registry())

    result = validator.validate(code)
    assert not result.passed
    assert any(issue.kind == "numeric_material_property" for issue in result.issues)

    repair = validator.repair(code)
    assert repair.after.passed
    assert "thermal_conductivity = diff_thermal_conductivity" in repair.repaired_code
    assert "prop_names = 'diff_thermal_conductivity'" in repair.repaired_code
    assert "prop_values = '45.0'" in repair.repaired_code


def test_repairs_heat_time_density_specific_heat_parameter():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Variables]
  [T]
  []
[]
[Kernels]
  [time]
    type = HeatConductionTimeDerivative
    variable = T
    density_specific_heat = 3510000.0
  []
[]
"""
    repair = MOOSETypeValidator(_registry()).repair(code)

    assert repair.after.passed
    assert "density_specific_heat" not in repair.repaired_code
    assert "density_name = time_density" in repair.repaired_code
    assert "specific_heat = time_specific_heat" in repair.repaired_code
    assert "prop_names = 'time_density time_specific_heat'" in repair.repaired_code
    assert "prop_values = '1.0 3510000.0'" in repair.repaired_code


def test_repairs_duplicate_generic_constant_material_property_with_same_value():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Materials]
  [T_diff]
    type = GenericConstantMaterial
    prop_names = 'diffusivity'
    prop_values = '1.0'
  []
  [c_diff]
    type = GenericConstantMaterial
    prop_names = 'diffusivity'
    prop_values = '1.0'
  []
[]
"""
    validator = MOOSETypeValidator(_registry())

    result = validator.validate(code)
    assert not result.passed
    issue = next(issue for issue in result.issues if issue.kind == "duplicate_material_property")
    assert issue.autofixable

    repair = validator.repair(code)
    assert repair.after.passed
    assert "prop_names = 'diffusivity'" in repair.repaired_code
    assert "prop_names = 'c_diff_diffusivity'" in repair.repaired_code


def test_repairs_duplicate_variable_and_explicit_initial_conditions():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Variables]
  [u]
    initial_condition = 0.0
  []
[]
[ICs]
  [u_ic]
    type = ConstantIC
    variable = u
    value = 0.0
  []
[]
"""
    repair = MOOSETypeValidator(_registry()).repair(code)

    assert repair.after.passed
    assert "initial_condition =" not in repair.repaired_code
    assert "[u_ic]" in repair.repaired_code


def test_repairs_conflicting_variable_initial_condition_by_preserving_explicit_ic():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 2
[]
[Variables]
  [eta]
    initial_condition = 0.5
  []
[]
[ICs]
  [eta_ic]
    type = FunctionIC
    variable = eta
    function = '0.5 + 0.01*cos(2*pi*x)'
  []
[]
"""
    repair = MOOSETypeValidator(_registry()).repair(code)

    assert repair.after.passed
    assert "initial_condition =" not in repair.repaired_code
    assert "type = FunctionIC" in repair.repaired_code
    assert "function = '0.5 + 0.01*cos(2*pi*x)'" in repair.repaired_code


def test_repairs_numeric_phase_field_mobility_to_material_property():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 2
[]
[Variables]
  [c]
  []
[]
[Kernels]
  [c_ch]
    type = CahnHilliard
    variable = c
    f_name = f_bulk
    mobility = 0.5
  []
[]
[Materials]
  [free_energy]
    type = GenericConstantMaterial
    prop_names = 'f_bulk'
    prop_values = '0.0'
  []
[]
"""
    repair = MOOSETypeValidator(_registry()).repair(code)

    assert repair.after.passed
    assert "mobility =" not in repair.repaired_code
    assert "mob_name = c_ch_mobility" in repair.repaired_code
    assert "prop_names = 'c_ch_mobility'" in repair.repaired_code
    assert "prop_values = '0.5'" in repair.repaired_code


def test_repairs_zero_vector_value_scalar_to_three_component_vector():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Variables]
  [pgas]
  []
[]
[Kernels]
  [pgas_flux]
    type = PorousFlowAdvectiveFlux
    variable = pgas
    gravity = '0'
  []
[]
"""
    repair = MOOSETypeValidator(_registry()).repair(code)

    assert repair.after.passed
    assert "gravity = '0 0 0'" in repair.repaired_code


def test_repairs_generated_mesh_boundary_aliases_to_canonical_sides():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
  left_boundary = inner
  right_boundary = outer
  boundary_name_prefix = ''
[]
[Variables]
  [u]
  []
[]
[BCs]
  [u_inner]
    type = DirichletBC
    variable = u
    boundary = inner
    value = 1.0
  []
  [u_outer]
    type = DirichletBC
    variable = u
    boundary = outer
    value = 0.0
  []
[]
"""
    repair = MOOSETypeValidator(_registry()).repair(code)

    assert repair.after.passed
    assert "boundary = 'left'" in repair.repaired_code
    assert "boundary = 'right'" in repair.repaired_code
    assert "left_boundary =" not in repair.repaired_code
    assert "right_boundary =" not in repair.repaired_code
    assert "boundary_name_prefix =" not in repair.repaired_code


def test_repairs_missing_required_parameter_from_global_params():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[GlobalParams]
  PorousFlowDictator = dictator
[]
[Variables]
  [pgas]
  []
[]
[Kernels]
  [pgas_flux]
    type = PorousFlowAdvectiveFlux
    variable = pgas
    gravity = '0'
  []
[]
"""
    repair = MOOSETypeValidator(_registry()).repair(code)

    assert repair.after.passed
    assert "PorousFlowDictator = dictator" in repair.repaired_code
    assert "gravity = '0 0 0'" in repair.repaired_code


def test_repair_maps_lowercase_coefficient_to_uppercase_when_required():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Variables]
  [T]
  []
[]
[Kernels]
  [time]
    type = CoefTimeDerivative
    variable = T
    coefficient = 3.0
  []
[]
"""
    repair = MOOSETypeValidator(_registry()).repair(code)

    assert repair.after.passed
    assert "Coefficient = 3.0" in repair.repaired_code
    assert "coefficient = 3.0" not in repair.repaired_code


def test_repairs_heat_source_material_property_to_direct_value():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Variables]
  [T]
  []
[]
[Kernels]
  [source]
    type = HeatSource
    variable = T
    heat_source = heat_source
  []
[]
[Materials]
  [source_mat]
    type = GenericConstantMaterial
    prop_names = 'heat_source'
    prop_values = '42.0'
  []
[]
"""
    repair = MOOSETypeValidator(_registry()).repair(code)

    assert repair.after.passed
    assert "value = 42.0" in repair.repaired_code
    assert "heat_source =" not in repair.repaired_code


def test_repairs_adrobin_value_to_convective_bc_with_material_properties():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Variables]
  [T]
  []
[]
[BCs]
  [left]
    type = ADRobinBC
    variable = T
    boundary = left
    coefficient = 2.0
    value = 300.0
  []
[]
"""
    repair = MOOSETypeValidator(_registry()).repair(code)

    assert repair.after.passed
    assert "type = ConvectiveHeatFluxBC" in repair.repaired_code
    assert "T_infinity = left_T_infinity" in repair.repaired_code
    assert "heat_transfer_coefficient = left_heat_transfer_coefficient" in repair.repaired_code
    assert "prop_values = '300 2'" in repair.repaired_code


def test_repairs_homogeneous_adrobin_alpha_beta_gamma_to_coefficient():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Variables]
  [T]
  []
[]
[BCs]
  [left]
    type = ADRobinBC
    variable = T
    boundary = left
    alpha = 2.0
    beta = 4.0
    gamma = 0.0
  []
[]
"""
    repair = MOOSETypeValidator(_registry()).repair(code)

    assert repair.after.passed
    assert "coefficient = 0.5" in repair.repaired_code
    assert "alpha =" not in repair.repaired_code
    assert "beta =" not in repair.repaired_code
    assert "gamma =" not in repair.repaired_code


def test_accepts_defined_material_property_reference():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Variables]
  [T]
  []
[]
[Kernels]
  [diff]
    type = ADHeatConduction
    variable = T
  []
[]
[Materials]
  [mat]
    type = GenericConstantMaterial
    prop_names = 'thermal_conductivity'
    prop_values = '1.0'
  []
[]
"""
    result = MOOSETypeValidator(_registry()).validate(code)
    assert result.passed


def test_repair_splits_petsc_options_pairs():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Executioner]
  type = Transient
  petsc_options = '-snes_type newtonls'
[]
"""
    repair = MOOSETypeValidator(_registry()).repair(code)

    assert repair.changed
    assert "petsc_options_iname = '-snes_type'" in repair.repaired_code
    assert "petsc_options_value = 'newtonls'" in repair.repaired_code


def test_repair_adds_executioner_absolute_tolerance_boilerplate():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Executioner]
  type = Transient
[]
"""
    repair = MOOSETypeValidator(_registry()).repair(code)

    assert repair.changed
    assert repair.after.passed
    assert "nl_abs_tol = 1e-8" in repair.repaired_code


def test_repairs_heat_conduction_thermal_conductivity_to_diffusion_coefficient():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Variables]
  [T]
  []
[]
[Kernels]
  [diff]
    type = HeatConduction
    variable = T
    thermal_conductivity = thermal_conductivity
  []
[]
[Materials]
  [k]
    type = GenericConstantMaterial
    prop_names = 'thermal_conductivity'
    prop_values = '3.5'
  []
[]
"""
    repair = MOOSETypeValidator(_registry()).repair(code)

    assert repair.after.passed
    assert "diffusion_coefficient = thermal_conductivity" in repair.repaired_code
    assert "diffusion_coefficient_dT = thermal_conductivity_dT" in repair.repaired_code
    assert "prop_names = 'thermal_conductivity_dT'" in repair.repaired_code
    assert "prop_values = '0.0'" in repair.repaired_code


def test_repairs_mat_coupled_force_material_property_alias():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Variables]
  [u]
  []
  [v]
  []
[]
[Kernels]
  [force]
    type = MatCoupledForce
    variable = u
    v = v
    mat_prop_coef = coupling_coef
  []
[]
[Materials]
  [coupling]
    type = GenericConstantMaterial
    prop_names = 'coupling_coef'
    prop_values = '0.3'
  []
[]
"""
    repair = MOOSETypeValidator(_registry()).repair(code)

    assert repair.after.passed
    assert "material_properties = coupling_coef" in repair.repaired_code
    assert "mat_prop_coef" not in repair.repaired_code


def test_repairs_mat_body_force_material_property_alias():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Variables]
  [u]
  []
[]
[Kernels]
  [source]
    type = MatBodyForce
    variable = u
    mat_prop = source_coef
  []
[]
[Materials]
  [source_material]
    type = GenericConstantMaterial
    prop_names = 'source_coef'
    prop_values = '4.0'
  []
[]
"""
    repair = MOOSETypeValidator(_registry()).repair(code)

    assert repair.after.passed
    assert "material_property = source_coef" in repair.repaired_code
    assert "mat_prop =" not in repair.repaired_code


def test_repairs_coef_time_derivative_lowercase_coef_alias():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Variables]
  [u]
  []
[]
[Kernels]
  [time]
    type = CoefTimeDerivative
    variable = u
    coef = 2.0
  []
[]
"""
    repair = MOOSETypeValidator(_registry()).repair(code)

    assert repair.after.passed
    assert "Coefficient = 2.0" in repair.repaired_code


def test_repairs_generated_mesh_boundary_all_to_named_sides():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Variables]
  [u]
  []
[]
[BCs]
  [all_bc]
    type = DirichletBC
    variable = u
    boundary = 'all'
    value = 0.0
  []
[]
"""
    repair = MOOSETypeValidator(_registry()).repair(code)

    assert repair.after.passed
    assert "boundary = 'left right'" in repair.repaired_code


def test_repairs_temperature_adrobin_coef1_coef2_value_to_convective_bc():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Variables]
  [T]
  []
[]
[BCs]
  [right]
    type = ADRobinBC
    variable = T
    boundary = right
    coef1 = 2.0
    coef2 = 4.0
    value = 600.0
  []
[]
"""
    repair = MOOSETypeValidator(_registry()).repair(code)

    assert repair.after.passed
    assert "type = ConvectiveHeatFluxBC" in repair.repaired_code
    assert "T_infinity = right_T_infinity" in repair.repaired_code
    assert "heat_transfer_coefficient = right_heat_transfer_coefficient" in repair.repaired_code
    assert "prop_values = '300 0.5'" in repair.repaired_code


def test_does_not_convert_non_heat_adrobin_value_to_heat_flux():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Variables]
  [u]
  []
[]
[BCs]
  [right]
    type = ADRobinBC
    variable = u
    boundary = right
    value = 1.0
  []
[]
"""
    repair = MOOSETypeValidator(_registry()).repair(code)

    assert not repair.after.passed
    assert "type = ConvectiveHeatFluxBC" not in repair.repaired_code


def test_detects_unknown_type_with_contextual_suggestions():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[BCs]
  [right]
    type = RobinBC
    variable = u
    boundary = right
  []
[]
"""
    result = MOOSETypeValidator(_registry()).validate(code)
    assert not result.passed
    assert result.issues[0].kind == "unknown_type"
    assert result.issues[0].suggestions == ["ADRobinBC"]


def test_detects_unknown_parameter_and_suggests_alias():
    code = """
[Kernels]
  [force]
    type = CoupledForce
    variable = u
    v = v
    coeff = 3.0
  []
[]
"""
    result = MOOSETypeValidator(_registry()).validate(code)
    assert not result.passed
    issue = next(i for i in result.issues if i.kind == "unknown_parameter")
    assert issue.parameter == "coeff"
    assert issue.autofixable


def test_accepts_preconditioning_subblock_context():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Preconditioning]
  [smp]
    type = SMP
    full = true
  []
[]
"""
    result = MOOSETypeValidator(_registry()).validate(code)
    assert result.passed


def test_safe_repair_fixes_mesh_context_robin_alias_and_param_alias():
    code = """
[Mesh]
  [gen]
    type = GeneratedMesh
    dim = 2
  []
[]
[Kernels]
  [force]
    type = CoupledForce
    variable = u
    v = v
    coeff = 3.0
  []
[]
[BCs]
  [right]
    type = RobinBC
    variable = u
    boundary = right
    coefficient = 2.0
  []
[]
"""
    repair = MOOSETypeValidator(_registry()).repair(code)
    assert repair.changed
    assert repair.after.passed
    assert "type = GeneratedMeshGenerator" in repair.repaired_code
    assert "type = ADRobinBC" in repair.repaired_code
    assert "coef = 3.0" in repair.repaired_code
