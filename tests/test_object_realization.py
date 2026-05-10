"""Tests for registry-grounded object realization support."""

from __future__ import annotations

from codmos.multiagent.moose_registry import MooseRegistry
from codmos.multiagent.object_realization import (
    build_object_plan,
    validate_and_repair_code,
)
from codmos.multiagent.pde.representation import (
    BoundaryCondition,
    InitialCondition,
    PDERepresentation,
    PDETerm,
)


def _registry() -> MooseRegistry:
    return MooseRegistry.from_moose_json({
        "blocks": {
            "Mesh": {
                "types": {
                    "GeneratedMesh": {
                        "parent_syntax": "Mesh",
                        "parameters": {
                            "type": {"name": "type"},
                            "dim": {"name": "dim", "required": True},
                        },
                    },
                },
            },
            "Kernels": {
                "subblock_types": {
                    "HeatConduction": {
                        "parent_syntax": "Kernels/*",
                        "parameters": {
                            "type": {"name": "type"},
                            "variable": {"name": "variable", "required": True},
                        },
                    },
                    "Diffusion": {
                        "parent_syntax": "Kernels/*",
                        "parameters": {
                            "type": {"name": "type"},
                            "variable": {"name": "variable", "required": True},
                        },
                    },
                    "CoupledForce": {
                        "parent_syntax": "Kernels/*",
                        "parameters": {
                            "type": {"name": "type"},
                            "variable": {"name": "variable", "required": True},
                            "v": {"name": "v", "required": True},
                            "coef": {"name": "coef"},
                        },
                    },
                    "HeatConductionTimeDerivative": {
                        "parent_syntax": "Kernels/*",
                        "parameters": {
                            "type": {"name": "type"},
                            "variable": {"name": "variable", "required": True},
                            "density_name": {"name": "density_name"},
                            "specific_heat": {"name": "specific_heat"},
                        },
                    },
                },
            },
            "BCs": {
                "subblock_types": {
                    "DirichletBC": {
                        "parent_syntax": "BCs/*",
                        "parameters": {
                            "type": {"name": "type"},
                            "variable": {"name": "variable", "required": True},
                            "boundary": {"name": "boundary", "required": True},
                            "value": {"name": "value"},
                        },
                    },
                    "ADRobinBC": {
                        "parent_syntax": "BCs/*",
                        "parameters": {
                            "type": {"name": "type"},
                            "variable": {"name": "variable", "required": True},
                            "boundary": {"name": "boundary", "required": True},
                            "coefficient": {"name": "coefficient"},
                        },
                    },
                    "ConvectiveHeatFluxBC": {
                        "parent_syntax": "BCs/*",
                        "parameters": {
                            "type": {"name": "type"},
                            "variable": {"name": "variable", "required": True},
                            "boundary": {"name": "boundary", "required": True},
                            "T_infinity": {"name": "T_infinity"},
                            "heat_transfer_coefficient": {"name": "heat_transfer_coefficient"},
                        },
                    },
                },
            },
            "ICs": {
                "subblock_types": {
                    "ConstantIC": {
                        "parent_syntax": "ICs/*",
                        "parameters": {
                            "type": {"name": "type"},
                            "variable": {"name": "variable", "required": True},
                            "value": {"name": "value"},
                        },
                    },
                },
            },
        }
    })


def _pde() -> PDERepresentation:
    return PDERepresentation(
        terms=[
            PDETerm(
                variable="T",
                operator="diffusion",
                coefficient=45.0,
                coupled_variable=None,
                kernel_type=None,
                severity="high",
            )
        ],
        boundary_conditions=[
            BoundaryCondition(
                variable="T",
                boundary="left",
                bc_type="Dirichlet",
                value=300.0,
                moose_bc_class=None,
                severity="high",
            )
        ],
        initial_conditions=[
            InitialCondition(
                variable="T",
                ic_type="constant",
                value=300.0,
                severity="medium",
            )
        ],
        time_scheme="steady",
        variables=["T"],
        dimensions=1,
    )


def test_object_plan_targets_registered_candidates():
    plan = build_object_plan(_pde(), _registry())
    prompt_section = plan.to_prompt_section()

    assert "Frozen MOOSE object-realization plan" in prompt_section
    assert "HeatConduction [Kernels/*]" in prompt_section
    assert "DirichletBC [BCs/*]" in prompt_section
    assert "ConstantIC [ICs/*]" in prompt_section
    assert "Do not invent" in prompt_section


def test_object_plan_prefers_generic_diffusion_for_non_heat_variables():
    pde = PDERepresentation(
        terms=[
            PDETerm(
                variable="c",
                operator="diffusion",
                coefficient=1.0,
                coupled_variable=None,
                kernel_type=None,
                severity="high",
            )
        ],
        boundary_conditions=[],
        initial_conditions=[],
        time_scheme="steady",
        variables=["c"],
        dimensions=1,
    )

    plan = build_object_plan(pde, _registry())
    first = plan.items[0].candidates[0]

    assert first.object_type == "Diffusion"


def test_object_plan_does_not_expose_density_specific_heat_as_moose_parameter():
    pde = PDERepresentation(
        terms=[
            PDETerm(
                variable="T",
                operator="time_derivative",
                coefficient=3510000.0,
                coupled_variable=None,
                kernel_type="HeatConductionTimeDerivative",
                severity="highest",
            )
        ],
        boundary_conditions=[],
        initial_conditions=[],
        time_scheme="transient",
        variables=["T"],
        dimensions=1,
    )

    plan = build_object_plan(pde, _registry())
    prompt_section = plan.to_prompt_section()

    assert "HeatConductionTimeDerivative [Kernels/*]" in prompt_section
    assert "semantic=['variable', 'density_name', 'specific_heat']" in prompt_section
    assert "coefficient_param=density_specific_heat" not in prompt_section
    assert "do not use density_specific_heat as a parameter" in prompt_section


def test_object_plan_prefers_convective_heat_flux_for_temperature_robin():
    pde = PDERepresentation(
        terms=[],
        boundary_conditions=[
            BoundaryCondition(
                variable="T",
                boundary="left",
                bc_type="Robin",
                value=300.0,
                moose_bc_class=None,
                severity="high",
            )
        ],
        initial_conditions=[],
        time_scheme="steady",
        variables=["T"],
        dimensions=1,
    )

    plan = build_object_plan(pde, _registry())

    assert plan.items[0].candidates[0].object_type == "ConvectiveHeatFluxBC"


def test_registry_repair_summarizes_safe_parameter_alias():
    code = """
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Kernels]
  [force]
    type = CoupledForce
    variable = u
    v = v
    coefficient = 2.0
  []
[]
"""

    summary = validate_and_repair_code(code, _registry())

    assert not summary.before_passed
    assert summary.after_passed
    assert summary.changed
    assert summary.issue_kinds["unknown_parameter"] == 1
    assert "coef = 2.0" in summary.repaired_code
