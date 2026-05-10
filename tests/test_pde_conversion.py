# tests/test_pde_conversion.py
"""Tests for PhysicsSpecification <-> PDERepresentation conversion."""

from codmos.multiagent.agents import PhysicsSpecification
from codmos.multiagent.pde.conversion import pde_to_physicsspec, physicsspec_to_pde
from codmos.multiagent.pde.representation import BoundaryCondition, PDERepresentation, PDETerm


def _make_heat_spec() -> PhysicsSpecification:
    """A simple steady-state heat conduction spec with structured fields."""
    return PhysicsSpecification(
        problem_description="Steady-state heat conduction in a 1D rod",
        spatial_dimensionality="1D",
        modeling_assumptions=["steady-state", "constant properties"],
        physics_modules=["heat_conduction"],
        primary_variables=[{"name": "temperature", "type": "scalar"}],
        governing_equations="nabla . (k nabla T) = 0",
        boundary_conditions=[
            {"name": "left", "type": "Dirichlet", "variable": "temperature", "value": "300"},
            {"name": "right", "type": "Dirichlet", "variable": "temperature", "value": "500"},
        ],
        initial_conditions=[],
        material_properties=[
            {"name": "thermal_conductivity", "value": 45.0},
        ],
        time_integration={"type": "Steady"},
        numerical_considerations="PJFNK solver",
        outputs_and_diagnostics=["exodus"],
    )


def _make_transient_spec() -> PhysicsSpecification:
    """A transient heat conduction spec."""
    return PhysicsSpecification(
        problem_description="Transient heat conduction in a 1D rod",
        spatial_dimensionality="1D",
        modeling_assumptions=["transient"],
        physics_modules=["heat_conduction"],
        primary_variables=[{"name": "temperature", "type": "scalar"}],
        governing_equations="rho cp dT/dt = nabla . (k nabla T)",
        boundary_conditions=[
            {"name": "left", "type": "Dirichlet", "variable": "temperature", "value": "300"},
        ],
        initial_conditions=[
            {"variable": "temperature", "type": "constant", "value": "300"},
        ],
        material_properties=[
            {"name": "thermal_conductivity", "value": 45.0},
        ],
        time_integration={"type": "Transient"},
        numerical_considerations="PJFNK solver",
        outputs_and_diagnostics=["exodus"],
    )


class TestPhysicsSpecToPDE:
    def test_basic_conversion(self):
        spec = _make_heat_spec()
        pde = physicsspec_to_pde(spec)
        assert isinstance(pde, PDERepresentation)

    def test_variables_extracted(self):
        spec = _make_heat_spec()
        pde = physicsspec_to_pde(spec)
        assert "temperature" in pde.variables

    def test_dimensions_parsed(self):
        spec = _make_heat_spec()
        pde = physicsspec_to_pde(spec)
        assert pde.dimensions == 1

    def test_time_scheme_steady(self):
        spec = _make_heat_spec()
        pde = physicsspec_to_pde(spec)
        assert pde.time_scheme == "steady"

    def test_time_scheme_transient(self):
        spec = _make_transient_spec()
        pde = physicsspec_to_pde(spec)
        assert pde.time_scheme == "transient"

    def test_boundary_conditions_converted(self):
        spec = _make_heat_spec()
        pde = physicsspec_to_pde(spec)
        assert len(pde.boundary_conditions) == 2
        left = next(bc for bc in pde.boundary_conditions if bc.boundary == "left")
        assert left.bc_type == "Dirichlet"
        assert left.variable == "temperature"
        assert left.value == 300.0

    def test_initial_conditions_converted(self):
        spec = _make_transient_spec()
        pde = physicsspec_to_pde(spec)
        assert len(pde.initial_conditions) == 1
        assert pde.initial_conditions[0].variable == "temperature"
        assert pde.initial_conditions[0].value == 300.0

    def test_heat_conduction_produces_diffusion_term(self):
        spec = _make_heat_spec()
        pde = physicsspec_to_pde(spec)
        ops = [t.operator for t in pde.terms]
        assert "diffusion" in ops

    def test_transient_produces_time_derivative(self):
        spec = _make_transient_spec()
        pde = physicsspec_to_pde(spec)
        ops = [t.operator for t in pde.terms]
        assert "time_derivative" in ops

    def test_material_properties_set_coefficients(self):
        spec = _make_heat_spec()
        pde = physicsspec_to_pde(spec)
        diffusion_terms = [t for t in pde.terms if t.operator == "diffusion"]
        assert len(diffusion_terms) >= 1
        assert diffusion_terms[0].coefficient == 45.0

    def test_structured_fields_only_no_governing_equations_parsing(self):
        """When structured fields are sufficient, governing_equations is not parsed."""
        spec = _make_heat_spec()
        spec.governing_equations = "THIS IS GARBAGE THAT SHOULD NOT BE PARSED"
        pde = physicsspec_to_pde(spec)
        # Should still produce valid PDE from structured fields alone
        assert len(pde.terms) >= 1
        assert pde.is_partial() is False


class TestPDEToPhysicsSpec:
    def test_basic_conversion(self):
        pde = PDERepresentation(
            terms=[PDETerm("temperature", "diffusion", 45.0, None, "HeatConduction", "high")],
            boundary_conditions=[
                BoundaryCondition("temperature", "left", "Dirichlet", 300.0, "DirichletBC", "high"),
            ],
            initial_conditions=[],
            time_scheme="steady",
            variables=["temperature"],
            dimensions=1,
        )
        spec = pde_to_physicsspec(pde)
        assert isinstance(spec, PhysicsSpecification)
        assert "temperature" in [v["name"] for v in spec.primary_variables]
        assert spec.time_integration["type"] == "Steady"

    def test_dimensionality_mapping(self):
        pde = PDERepresentation([], [], [], "steady", [], 2)
        spec = pde_to_physicsspec(pde)
        assert spec.spatial_dimensionality == "2D"

    def test_boundary_conditions_mapped(self):
        pde = PDERepresentation(
            terms=[],
            boundary_conditions=[
                BoundaryCondition("T", "left", "Dirichlet", 300.0, "DirichletBC", "high"),
            ],
            initial_conditions=[],
            time_scheme="steady",
            variables=["T"],
            dimensions=1,
        )
        spec = pde_to_physicsspec(pde)
        assert len(spec.boundary_conditions) == 1
        assert spec.boundary_conditions[0]["type"] == "Dirichlet"
