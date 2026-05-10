"""Tests for PDE representation dataclasses."""

from codmos.multiagent.pde.representation import (
    BoundaryCondition,
    InitialCondition,
    PDERepresentation,
    PDETerm,
)


class TestPDETerm:
    def test_basic_construction(self):
        term = PDETerm(
            variable="temperature",
            operator="diffusion",
            coefficient=10.0,
            coupled_variable=None,
            kernel_type="HeatConduction",
            severity="high",
        )
        assert term.variable == "temperature"
        assert term.operator == "diffusion"
        assert term.coefficient == 10.0
        assert term.coupled_variable is None
        assert term.kernel_type == "HeatConduction"
        assert term.severity == "high"

    def test_coupled_term(self):
        term = PDETerm(
            variable="temperature",
            operator="coupled_source",
            coefficient=1.0,
            coupled_variable="displacement",
            kernel_type="CoupledForce",
            severity="high",
        )
        assert term.coupled_variable == "displacement"

    def test_string_coefficient(self):
        term = PDETerm(
            variable="temperature",
            operator="diffusion",
            coefficient="thermal_mat",
            coupled_variable=None,
            kernel_type="HeatConduction",
            severity="high",
        )
        assert term.coefficient == "thermal_mat"

    def test_none_coefficient(self):
        term = PDETerm(
            variable="temperature",
            operator="time_derivative",
            coefficient=None,
            coupled_variable=None,
            kernel_type="TimeDerivative",
            severity="critical",
        )
        assert term.coefficient is None


class TestBoundaryCondition:
    def test_basic_construction(self):
        bc = BoundaryCondition(
            variable="temperature",
            boundary="left",
            bc_type="Dirichlet",
            value=300.0,
            moose_bc_class="DirichletBC",
            severity="high",
        )
        assert bc.variable == "temperature"
        assert bc.boundary == "left"
        assert bc.bc_type == "Dirichlet"
        assert bc.value == 300.0

    def test_neumann_bc(self):
        bc = BoundaryCondition(
            variable="temperature",
            boundary="right",
            bc_type="Neumann",
            value=0.0,
            moose_bc_class="NeumannBC",
            severity="medium",
        )
        assert bc.bc_type == "Neumann"
        assert bc.severity == "medium"


class TestInitialCondition:
    def test_constant_ic(self):
        ic = InitialCondition(
            variable="temperature",
            ic_type="constant",
            value=300.0,
            severity="medium",
        )
        assert ic.ic_type == "constant"
        assert ic.value == 300.0

    def test_function_ic(self):
        ic = InitialCondition(
            variable="temperature",
            ic_type="function",
            value="300+10*x",
            severity="medium",
        )
        assert ic.ic_type == "function"


class TestPDERepresentation:
    def _make_simple_pde(self) -> PDERepresentation:
        return PDERepresentation(
            terms=[
                PDETerm("temperature", "diffusion", 10.0, None, "HeatConduction", "high"),
            ],
            boundary_conditions=[
                BoundaryCondition("temperature", "left", "Dirichlet", 300.0, "DirichletBC", "high"),
                BoundaryCondition("temperature", "right", "Dirichlet", 500.0, "DirichletBC", "high"),
            ],
            initial_conditions=[],
            time_scheme="steady",
            variables=["temperature"],
            dimensions=1,
        )

    def test_basic_construction(self):
        pde = self._make_simple_pde()
        assert pde.time_scheme == "steady"
        assert pde.variables == ["temperature"]
        assert pde.dimensions == 1
        assert len(pde.terms) == 1
        assert len(pde.boundary_conditions) == 2

    def test_default_quality_fields(self):
        pde = self._make_simple_pde()
        assert pde.unresolved_kernels == []
        assert pde.unresolved_coefficients == []
        assert pde.warnings == []

    def test_is_partial_false_when_complete(self):
        pde = self._make_simple_pde()
        assert pde.is_partial() is False

    def test_is_partial_true_with_unresolved_kernels(self):
        pde = self._make_simple_pde()
        pde.unresolved_kernels = ["CustomKernel"]
        assert pde.is_partial() is True

    def test_is_partial_true_with_unresolved_coefficients(self):
        pde = self._make_simple_pde()
        pde.unresolved_coefficients = ["unknown_property"]
        assert pde.is_partial() is True

    def test_transient_with_ics(self):
        pde = PDERepresentation(
            terms=[
                PDETerm("temperature", "time_derivative", None, None, "TimeDerivative", "critical"),
                PDETerm("temperature", "diffusion", 10.0, None, "HeatConduction", "high"),
            ],
            boundary_conditions=[
                BoundaryCondition("temperature", "left", "Dirichlet", 300.0, "DirichletBC", "high"),
            ],
            initial_conditions=[
                InitialCondition("temperature", "constant", 300.0, "medium"),
            ],
            time_scheme="transient",
            variables=["temperature"],
            dimensions=1,
        )
        assert pde.time_scheme == "transient"
        assert len(pde.initial_conditions) == 1

    def test_multivariable_pde(self):
        pde = PDERepresentation(
            terms=[
                PDETerm("temperature", "diffusion", 10.0, None, "HeatConduction", "high"),
                PDETerm("displacement", "diffusion", 1e9, None, "StressDivergenceTensors", "high"),
                PDETerm("temperature", "coupled_source", 1.0, "displacement", "CoupledForce", "high"),
            ],
            boundary_conditions=[],
            initial_conditions=[],
            time_scheme="steady",
            variables=["temperature", "displacement"],
            dimensions=2,
        )
        assert len(pde.variables) == 2
        assert len(pde.terms) == 3
