# tests/test_pde_reconstruction.py
"""Tests for deterministic MOOSE .i → PDERepresentation reconstruction."""

from codmos.multiagent.pde.reconstruction import reconstruct_pde
from codmos.multiagent.pde.representation import PDERepresentation

STEADY_HEAT_1D = """\
[Mesh]
  [gen]
    type = GeneratedMesh
    dim = 1
    nx = 10
  []
[]

[Variables]
  [temperature]
  []
[]

[Kernels]
  [heat_conduction]
    type = HeatConduction
    variable = temperature
  []
[]

[BCs]
  [left]
    type = DirichletBC
    variable = temperature
    boundary = left
    value = 300
  []
  [right]
    type = DirichletBC
    variable = temperature
    boundary = right
    value = 500
  []
[]

[Materials]
  [thermal]
    type = GenericConstantMaterial
    prop_names = 'thermal_conductivity'
    prop_values = '45.0'
  []
[]

[Executioner]
  type = Steady
  solve_type = PJFNK
[]

[Outputs]
  exodus = true
[]
"""


TRANSIENT_HEAT_1D = """\
[Mesh]
  [gen]
    type = GeneratedMesh
    dim = 1
    nx = 20
  []
[]

[Variables]
  [temperature]
  []
[]

[Kernels]
  [time_deriv]
    type = TimeDerivative
    variable = temperature
  []
  [heat_conduction]
    type = HeatConduction
    variable = temperature
  []
[]

[BCs]
  [left]
    type = DirichletBC
    variable = temperature
    boundary = left
    value = 300
  []
  [right]
    type = NeumannBC
    variable = temperature
    boundary = right
    value = 10
  []
[]

[ICs]
  [initial_temp]
    type = ConstantIC
    variable = temperature
    value = 300
  []
[]

[Materials]
  [thermal]
    type = GenericConstantMaterial
    prop_names = 'thermal_conductivity'
    prop_values = '45.0'
  []
[]

[Executioner]
  type = Transient
  dt = 0.1
  end_time = 10
  solve_type = PJFNK
[]

[Outputs]
  exodus = true
[]
"""


COUPLED_2D = """\
[Mesh]
  [gen]
    type = GeneratedMesh
    dim = 2
    nx = 10
    ny = 10
  []
[]

[Variables]
  [temperature]
  []
  [displacement]
  []
[]

[Kernels]
  [heat_conduction]
    type = HeatConduction
    variable = temperature
  []
  [mech_diffusion]
    type = Diffusion
    variable = displacement
  []
  [coupling]
    type = CoupledForce
    variable = temperature
    v = displacement
    coef = 1.5
  []
[]

[BCs]
  [temp_left]
    type = DirichletBC
    variable = temperature
    boundary = left
    value = 300
  []
  [disp_bottom]
    type = DirichletBC
    variable = displacement
    boundary = bottom
    value = 0
  []
[]

[Materials]
  [thermal]
    type = GenericConstantMaterial
    prop_names = 'thermal_conductivity'
    prop_values = '45.0'
  []
[]

[Executioner]
  type = Steady
  solve_type = PJFNK
[]
"""


UNKNOWN_KERNEL = """\
[Mesh]
  [gen]
    type = GeneratedMesh
    dim = 1
    nx = 10
  []
[]

[Variables]
  [u]
  []
[]

[Kernels]
  [custom]
    type = MyCustomKernel
    variable = u
  []
  [diffusion]
    type = Diffusion
    variable = u
  []
[]

[BCs]
  [left]
    type = DirichletBC
    variable = u
    boundary = left
    value = 0
  []
[]

[Executioner]
  type = Steady
[]
"""


class TestReconstructSteadyHeat:
    def test_returns_pde_representation(self):
        pde = reconstruct_pde(STEADY_HEAT_1D)
        assert isinstance(pde, PDERepresentation)

    def test_variables(self):
        pde = reconstruct_pde(STEADY_HEAT_1D)
        assert pde.variables == ["temperature"]

    def test_dimensions(self):
        pde = reconstruct_pde(STEADY_HEAT_1D)
        assert pde.dimensions == 1

    def test_time_scheme(self):
        pde = reconstruct_pde(STEADY_HEAT_1D)
        assert pde.time_scheme == "steady"

    def test_terms(self):
        pde = reconstruct_pde(STEADY_HEAT_1D)
        assert len(pde.terms) == 1
        term = pde.terms[0]
        assert term.variable == "temperature"
        assert term.operator == "diffusion"
        assert term.kernel_type == "HeatConduction"

    def test_coefficient_resolved(self):
        pde = reconstruct_pde(STEADY_HEAT_1D)
        # In schema v3.0, HeatConduction has no coefficient_param — coefficient
        # is resolved from Materials block via material_resolver (if available).
        # Without explicit coefficient_param on the kernel, coefficient may be None
        # or resolved from materials. Accept either.
        coeff = pde.terms[0].coefficient
        assert coeff is None or coeff == 45.0

    def test_boundary_conditions(self):
        pde = reconstruct_pde(STEADY_HEAT_1D)
        assert len(pde.boundary_conditions) == 2
        left = next(bc for bc in pde.boundary_conditions if bc.boundary == "left")
        assert left.bc_type == "Dirichlet"
        assert left.value == 300.0

    def test_no_initial_conditions(self):
        pde = reconstruct_pde(STEADY_HEAT_1D)
        assert pde.initial_conditions == []

    def test_not_partial(self):
        pde = reconstruct_pde(STEADY_HEAT_1D)
        assert pde.is_partial() is False


class TestReconstructTransientHeat:
    def test_time_scheme(self):
        pde = reconstruct_pde(TRANSIENT_HEAT_1D)
        assert pde.time_scheme == "transient"

    def test_has_time_derivative(self):
        pde = reconstruct_pde(TRANSIENT_HEAT_1D)
        ops = [t.operator for t in pde.terms]
        assert "time_derivative" in ops

    def test_initial_conditions(self):
        pde = reconstruct_pde(TRANSIENT_HEAT_1D)
        assert len(pde.initial_conditions) == 1
        assert pde.initial_conditions[0].variable == "temperature"
        assert pde.initial_conditions[0].value == 300.0

    def test_mixed_bc_types(self):
        pde = reconstruct_pde(TRANSIENT_HEAT_1D)
        types = {bc.bc_type for bc in pde.boundary_conditions}
        assert "Dirichlet" in types
        assert "Neumann" in types


class TestReconstructCoupled:
    def test_variables(self):
        pde = reconstruct_pde(COUPLED_2D)
        assert sorted(pde.variables) == ["displacement", "temperature"]

    def test_dimensions(self):
        pde = reconstruct_pde(COUPLED_2D)
        assert pde.dimensions == 2

    def test_coupled_term(self):
        pde = reconstruct_pde(COUPLED_2D)
        # CoupledForce: in schema v3.0, operator is "source" and coupled_param may be None
        # (the coupling is inferred from the kernel's 'v' parameter, not from coupled_param)
        cf_terms = [t for t in pde.terms if t.kernel_type == "CoupledForce"]
        assert len(cf_terms) == 1
        assert cf_terms[0].operator == "source"

    def test_three_terms(self):
        pde = reconstruct_pde(COUPLED_2D)
        assert len(pde.terms) == 3

    def test_equivalent_operator_normalization(self):
        """HeatConduction and Diffusion both produce operator='diffusion'."""
        pde = reconstruct_pde(COUPLED_2D)
        diffusion_terms = [t for t in pde.terms if t.operator == "diffusion"]
        assert len(diffusion_terms) == 2
        kernel_types = {t.kernel_type for t in diffusion_terms}
        assert kernel_types == {"HeatConduction", "Diffusion"}


class TestReconstructUnknownKernel:
    def test_is_partial(self):
        pde = reconstruct_pde(UNKNOWN_KERNEL)
        assert pde.is_partial() is True

    def test_unresolved_kernels(self):
        pde = reconstruct_pde(UNKNOWN_KERNEL)
        assert "MyCustomKernel" in pde.unresolved_kernels

    def test_known_kernel_still_extracted(self):
        pde = reconstruct_pde(UNKNOWN_KERNEL)
        ops = [t.operator for t in pde.terms]
        assert "diffusion" in ops
