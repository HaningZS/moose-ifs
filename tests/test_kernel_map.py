# tests/test_kernel_map.py
"""Tests for kernel-PDE mapping table loader (schema v3.0)."""

from pathlib import Path

import pytest

from codmos.multiagent.pde.kernel_map import BCPDEMapping, ICPDEMapping, KernelMap, KernelPDEMapping


@pytest.fixture
def kernel_map() -> KernelMap:
    """Load the default kernel map from data/pde_mapping/kernel_map.yaml."""
    return KernelMap()


class TestKernelMap:
    def test_loads_without_error(self, kernel_map: KernelMap):
        assert kernel_map is not None

    def test_get_known_kernel(self, kernel_map: KernelMap):
        mapping = kernel_map.get_kernel("Diffusion")
        assert mapping is not None
        assert mapping.operator == "diffusion"
        assert mapping.severity in ("high", "very_high")

    def test_get_heat_conduction_kernel(self, kernel_map: KernelMap):
        mapping = kernel_map.get_kernel("HeatConduction")
        assert mapping is not None
        assert mapping.operator == "diffusion"
        # In schema v3.0, equivalences are per-collaborator classification
        assert isinstance(mapping.equivalent_to, list)

    def test_equivalent_kernels_share_operator(self, kernel_map: KernelMap):
        diffusion = kernel_map.get_kernel("Diffusion")
        heat = kernel_map.get_kernel("HeatConduction")
        assert diffusion is not None and heat is not None
        assert diffusion.operator == heat.operator

    def test_get_time_derivative(self, kernel_map: KernelMap):
        mapping = kernel_map.get_kernel("TimeDerivative")
        assert mapping is not None
        assert mapping.operator == "time_derivative"
        assert mapping.severity == "highest"

    def test_get_coupled_force(self, kernel_map: KernelMap):
        mapping = kernel_map.get_kernel("CoupledForce")
        assert mapping is not None
        assert mapping.operator in ("source", "coupled_source")

    def test_get_unknown_kernel_returns_none(self, kernel_map: KernelMap):
        assert kernel_map.get_kernel("NonexistentKernel") is None

    def test_get_dirichlet_bc(self, kernel_map: KernelMap):
        mapping = kernel_map.get_bc("DirichletBC")
        assert mapping is not None
        assert mapping.bc_type == "Dirichlet"
        assert mapping.severity == "high"

    def test_get_neumann_bc(self, kernel_map: KernelMap):
        mapping = kernel_map.get_bc("NeumannBC")
        assert mapping is not None
        assert mapping.bc_type.startswith("Neumann")

    def test_get_unknown_bc_returns_none(self, kernel_map: KernelMap):
        assert kernel_map.get_bc("MagicBC") is None

    def test_list_operators(self, kernel_map: KernelMap):
        ops = kernel_map.list_operators()
        assert "diffusion" in ops
        assert "time_derivative" in ops
        assert "source" in ops
        assert len(ops) == len(set(ops))  # no duplicates

    def test_custom_yaml_path(self, tmp_path: Path):
        yaml_content = """
kernels:
  FakeKernel:
    operator: "fake_op"
    variable_param: "variable"
    coefficient_param: null
    coupled_param: null
    severity: "low"
    source_doc: null
    all_parameters: []
    equivalent_to: []

bcs:
  FakeBC:
    condition_type: "fake"
    variable_param: "variable"
    boundary_param: "boundary"
    value_param: "val"
    function_param: null
    severity: "low"
    transient_only: false
    source_doc: null
    all_parameters: []
    equivalent_to: []

ics:
  FakeIC:
    ic_type: "constant"
    variable_param: "variable"
    value_param: "value"
    function_param: null
    severity: "high"
    transient_only: true
    source_doc: null
    all_parameters: []
    equivalent_to: []
"""
        yaml_file = tmp_path / "test_map.yaml"
        yaml_file.write_text(yaml_content)
        km = KernelMap(yaml_path=yaml_file)
        assert km.get_kernel("FakeKernel") is not None
        assert km.get_kernel("FakeKernel").operator == "fake_op"
        assert km.get_bc("FakeBC") is not None
        assert km.get_bc("FakeBC").bc_type == "Fake"
        assert km.get_ic("FakeIC") is not None
        assert km.get_ic("FakeIC").ic_type == "constant"

    def test_minimum_kernel_coverage(self, kernel_map: KernelMap):
        """Verify the YAML has the required minimum set of kernels."""
        required_kernels = [
            "Diffusion", "HeatConduction", "TimeDerivative",
            "BodyForce", "CoupledForce", "Reaction",
        ]
        for name in required_kernels:
            assert kernel_map.get_kernel(name) is not None, f"Missing required kernel: {name}"

    def test_minimum_bc_coverage(self, kernel_map: KernelMap):
        """Verify the YAML has the required minimum set of BCs."""
        required_bcs = ["DirichletBC", "NeumannBC", "FunctionDirichletBC"]
        for name in required_bcs:
            assert kernel_map.get_bc(name) is not None, f"Missing required BC: {name}"

    def test_schema_v3_scale(self, kernel_map: KernelMap):
        """Verify trimmed scale: 90%+ coverage set from schema v3.0."""
        assert len(kernel_map._kernels) >= 80  # ~85 after expert curation
        assert len(kernel_map._bcs) >= 5         # ~7, 91.5% BC coverage
        assert len(kernel_map._ics) >= 5          # ~11, 90.3% IC coverage

    def test_ic_lookup(self, kernel_map: KernelMap):
        """Verify IC lookups work."""
        ic = kernel_map.get_ic("ConstantIC")
        assert ic is not None
        assert ic.ic_type == "constant"
        assert kernel_map.get_ic("NonexistentIC") is None

    def test_severity_values(self, kernel_map: KernelMap):
        """All severity values should be from the 6-level scale."""
        valid = {"highest", "very_high", "high", "medium", "medium_low", "low"}
        for m in kernel_map._kernels.values():
            assert m.severity in valid, f"{m.kernel_class}: severity={m.severity}"
        for m in kernel_map._bcs.values():
            assert m.severity in valid, f"{m.bc_class}: severity={m.severity}"
        for m in kernel_map._ics.values():
            assert m.severity in valid, f"{m.ic_class}: severity={m.severity}"
