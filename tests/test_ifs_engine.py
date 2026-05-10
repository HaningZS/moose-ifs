"""Tests for IFS (Intent Fidelity Score) computation engine."""

from codmos.multiagent.pde.ifs_engine import (
    SEVERITY_WEIGHTS,
    GroundTruthAnnotation,
    compute_ifs,
    compute_material_consistency,
    extract_coefficient_contract,
)
from codmos.multiagent.pde.representation import (
    BoundaryCondition,
    InitialCondition,
    PDERepresentation,
    PDETerm,
)


def _term(var: str, op: str, coef: float | None = None, sev: str = "high") -> PDETerm:
    return PDETerm(var, op, coef, None, None, sev)


def _bc(var: str, bnd: str, typ: str = "Dirichlet", val: float = 300.0, sev: str = "high") -> BoundaryCondition:
    return BoundaryCondition(var, bnd, typ, val, None, sev)


def _pde(
    terms: list[PDETerm] | None = None,
    bcs: list[BoundaryCondition] | None = None,
    ics: list[InitialCondition] | None = None,
    time_scheme: str = "steady",
) -> PDERepresentation:
    terms = terms or []
    bcs = bcs or []
    ics = ics or []
    variables = sorted({t.variable for t in terms} | {b.variable for b in bcs})
    return PDERepresentation(terms, bcs, ics, time_scheme, variables, 1)


class TestPerfectMatch:
    def test_score_is_one(self):
        ref = _pde(
            terms=[_term("T", "diffusion", 10.0)],
            bcs=[_bc("T", "left"), _bc("T", "right", val=500.0)],
        )
        cand = _pde(
            terms=[_term("T", "diffusion", 10.0)],
            bcs=[_bc("T", "left"), _bc("T", "right", val=500.0)],
        )
        result = compute_ifs(ref, cand)
        assert result.ifs_score == 1.0

    def test_all_passed(self):
        ref = _pde(terms=[_term("T", "diffusion")])
        result = compute_ifs(ref, _pde(terms=[_term("T", "diffusion")]))
        assert result.num_passed == result.num_checkpoints
        assert result.num_failed == 0

    def test_raw_loss_zero(self):
        ref = _pde(terms=[_term("T", "diffusion")])
        result = compute_ifs(ref, _pde(terms=[_term("T", "diffusion")]))
        assert result.raw_loss == 0.0


class TestAllMissing:
    def test_score_is_zero(self):
        ref = _pde(terms=[_term("T", "diffusion", 10.0)])
        cand = _pde(terms=[])
        result = compute_ifs(ref, cand)
        assert result.ifs_score == 0.0

    def test_all_failed(self):
        ref = _pde(terms=[_term("T", "diffusion")])
        result = compute_ifs(ref, _pde(terms=[]))
        assert result.num_failed == result.num_checkpoints
        assert result.num_passed == 0


class TestSeverityWeightImpact:
    def test_critical_miss_worse_than_medium_miss(self):
        ref = _pde(terms=[
            _term("T", "time_derivative", sev="critical"),
            _term("T", "diffusion", sev="high"),
            _term("T", "body_force", sev="medium"),
        ])
        # Missing critical term only
        cand_miss_critical = _pde(terms=[
            _term("T", "diffusion"),
            _term("T", "body_force"),
        ])
        # Missing medium term only
        cand_miss_medium = _pde(terms=[
            _term("T", "time_derivative"),
            _term("T", "diffusion"),
        ])
        r_critical = compute_ifs(ref, cand_miss_critical)
        r_medium = compute_ifs(ref, cand_miss_medium)
        assert r_critical.ifs_score < r_medium.ifs_score


class TestCoefficientComparison:
    def test_within_tolerance_passes(self):
        ref = _pde(terms=[_term("T", "diffusion", 10.0)])
        cand = _pde(terms=[_term("T", "diffusion", 10.5)])
        result = compute_ifs(ref, cand, coeff_tolerance=0.1)
        coeff_cps = [cp for cp in result.checkpoints if cp.dimension == "coefficient"]
        assert all(cp.passed for cp in coeff_cps)

    def test_outside_tolerance_fails(self):
        ref = _pde(terms=[_term("T", "diffusion", 10.0)])
        cand = _pde(terms=[_term("T", "diffusion", 20.0)])
        result = compute_ifs(ref, cand, coeff_tolerance=0.1)
        coeff_cps = [cp for cp in result.checkpoints if cp.dimension == "coefficient"]
        assert any(not cp.passed for cp in coeff_cps)

    def test_both_none_coefficients_pass(self):
        ref = _pde(terms=[_term("T", "time_derivative", None)])
        cand = _pde(terms=[_term("T", "time_derivative", None)])
        result = compute_ifs(ref, cand)
        assert result.ifs_score == 1.0


class TestBCAlignment:
    def test_bc_type_mismatch_fails(self):
        ref = _pde(bcs=[_bc("T", "left", "Dirichlet")])
        cand = _pde(bcs=[_bc("T", "left", "Neumann")])
        result = compute_ifs(ref, cand)
        bc_cps = [cp for cp in result.checkpoints if cp.dimension == "bc"]
        assert any(not cp.passed for cp in bc_cps)

    def test_missing_bc_fails(self):
        ref = _pde(bcs=[_bc("T", "left"), _bc("T", "right")])
        cand = _pde(bcs=[_bc("T", "left")])
        result = compute_ifs(ref, cand)
        assert result.ifs_score < 1.0

    def test_extra_bc_in_candidate(self):
        ref = _pde(bcs=[_bc("T", "left")])
        cand = _pde(bcs=[_bc("T", "left"), _bc("T", "right")])
        result = compute_ifs(ref, cand)
        # Extra BC in candidate is a checkpoint failure (unexpected BC)
        assert result.ifs_score < 1.0


class TestTimeScheme:
    def test_transient_to_steady_is_critical(self):
        ref = _pde(time_scheme="transient")
        cand = _pde(time_scheme="steady")
        result = compute_ifs(ref, cand)
        time_cps = [cp for cp in result.checkpoints if cp.dimension == "time"]
        assert len(time_cps) == 1
        assert not time_cps[0].passed
        assert time_cps[0].severity == "critical"

    def test_steady_to_transient_is_medium(self):
        ref = _pde(time_scheme="steady")
        cand = _pde(time_scheme="transient")
        result = compute_ifs(ref, cand)
        time_cps = [cp for cp in result.checkpoints if cp.dimension == "time"]
        assert len(time_cps) == 1
        assert not time_cps[0].passed
        assert time_cps[0].severity == "medium"

    def test_matching_time_scheme_passes(self):
        ref = _pde(time_scheme="transient")
        cand = _pde(time_scheme="transient")
        result = compute_ifs(ref, cand)
        time_cps = [cp for cp in result.checkpoints if cp.dimension == "time"]
        assert all(cp.passed for cp in time_cps)


class TestICMatching:
    def test_transient_ic_match(self):
        ic = InitialCondition("T", "constant", 300.0, "medium")
        ref = _pde(ics=[ic], time_scheme="transient")
        cand = _pde(ics=[ic], time_scheme="transient")
        result = compute_ifs(ref, cand)
        ic_cps = [cp for cp in result.checkpoints if cp.dimension == "ic"]
        assert all(cp.passed for cp in ic_cps)

    def test_missing_ic_fails(self):
        ic = InitialCondition("T", "constant", 300.0, "medium")
        ref = _pde(ics=[ic], time_scheme="transient")
        cand = _pde(ics=[], time_scheme="transient")
        result = compute_ifs(ref, cand)
        assert result.ifs_ic < 1.0


class TestDimensionalBreakdown:
    def test_ifs_term_computed(self):
        ref = _pde(terms=[_term("T", "diffusion")])
        cand = _pde(terms=[_term("T", "diffusion")])
        result = compute_ifs(ref, cand)
        assert result.ifs_term == 1.0

    def test_ifs_bc_computed(self):
        ref = _pde(bcs=[_bc("T", "left")])
        cand = _pde(bcs=[_bc("T", "left")])
        result = compute_ifs(ref, cand)
        assert result.ifs_bc == 1.0


class TestSummaryStatistics:
    def test_num_checkpoints_equals_passed_plus_failed(self):
        ref = _pde(
            terms=[_term("T", "diffusion", 10.0), _term("T", "time_derivative")],
            bcs=[_bc("T", "left")],
        )
        cand = _pde(
            terms=[_term("T", "diffusion", 10.0)],
            bcs=[_bc("T", "left")],
        )
        result = compute_ifs(ref, cand)
        assert result.num_checkpoints == result.num_passed + result.num_failed

    def test_total_severity_weight(self):
        ref = _pde(terms=[_term("T", "diffusion", sev="high")])
        cand = _pde(terms=[_term("T", "diffusion")])
        result = compute_ifs(ref, cand)
        assert result.total_severity_weight > 0

    def test_raw_loss_is_severity_weighted(self):
        """raw_loss accumulates severity-weighted losses, not just failure count."""
        ref = _pde(terms=[
            _term("T", "time_derivative", sev="critical"),
            _term("T", "diffusion", sev="high"),
        ])
        cand = _pde(terms=[])  # both missing
        result = compute_ifs(ref, cand)
        expected_loss = SEVERITY_WEIGHTS["critical"] + SEVERITY_WEIGHTS["high"]
        assert result.raw_loss == expected_loss


class TestGroundTruthAnnotation:
    def test_range_mode_in_range_passes(self):
        ref_pde = _pde(terms=[_term("T", "diffusion", 10.0)])
        annotation = GroundTruthAnnotation(
            pde=ref_pde,
            value_modes={"T:diffusion:coeff": "range"},
            ranges={"T:diffusion:coeff": (5.0, 20.0)},
        )
        cand = _pde(terms=[_term("T", "diffusion", 15.0)])
        result = compute_ifs(annotation, cand)
        coeff_cps = [cp for cp in result.checkpoints if cp.dimension == "coefficient"]
        assert all(cp.passed for cp in coeff_cps)

    def test_range_mode_out_of_range_fails(self):
        ref_pde = _pde(terms=[_term("T", "diffusion", 10.0)])
        annotation = GroundTruthAnnotation(
            pde=ref_pde,
            value_modes={"T:diffusion:coeff": "range"},
            ranges={"T:diffusion:coeff": (5.0, 8.0)},
        )
        cand = _pde(terms=[_term("T", "diffusion", 15.0)])
        result = compute_ifs(annotation, cand)
        coeff_cps = [cp for cp in result.checkpoints if cp.dimension == "coefficient"]
        assert any(not cp.passed for cp in coeff_cps)

    def test_unspecified_mode_skips_checkpoint(self):
        ref_pde = _pde(terms=[_term("T", "diffusion", 10.0)])
        annotation = GroundTruthAnnotation(
            pde=ref_pde,
            value_modes={"T:diffusion:coeff": "unspecified"},
            ranges={},
        )
        cand = _pde(terms=[_term("T", "diffusion", 999.0)])
        result = compute_ifs(annotation, cand)
        # The coefficient checkpoint should be skipped
        coeff_cps = [cp for cp in result.checkpoints if cp.dimension == "coefficient"]
        assert len(coeff_cps) == 0


class TestMaterialConsistency:
    def test_material_backed_kernel_coefficient_matches_inline_candidate(self):
        ref = """
[Variables]
  [c]
  []
[]
[Kernels]
  [diff]
    type = MatDiffusion
    variable = c
    diffusivity = D
  []
[]
[Materials]
  [diff_coef]
    type = GenericConstantMaterial
    prop_names = 'D'
    prop_values = '1e-5'
  []
[]
"""
        cand = """
[Variables]
  [c]
  []
[]
[Kernels]
  [diff]
    type = MatDiffusion
    variable = c
    diffusivity = 1e-5
  []
[]
"""
        result = compute_material_consistency(ref, cand)
        assert result.score == 1.0
        assert result.total_properties == 1

    def test_unused_candidate_material_property_does_not_mask_missing_coefficient(self):
        ref = """
[Variables]
  [c]
  []
[]
[Kernels]
  [diff]
    type = MatDiffusion
    variable = c
    diffusivity = D
  []
[]
[Materials]
  [diff_coef]
    type = GenericConstantMaterial
    prop_names = 'D'
    prop_values = '1e-5'
  []
[]
"""
        cand = """
[Variables]
  [c]
  []
[]
[Kernels]
  [diff]
    type = Diffusion
    variable = c
  []
[]
[Materials]
  [unused]
    type = GenericConstantMaterial
    prop_names = 'diffusivity'
    prop_values = '1e-5'
  []
[]
"""
        result = compute_material_consistency(ref, cand)
        assert result.score == 0.0
        assert result.mismatched[0]["property"] == "kernel:diffusion"

    def test_quoted_candidate_material_reference_is_resolved(self):
        ref = """
[Variables]
  [c]
  []
[]
[Kernels]
  [diff]
    type = MatDiffusion
    variable = c
    diffusivity = D
  []
[]
[Materials]
  [diff_coef]
    type = GenericConstantMaterial
    prop_names = 'D'
    prop_values = '1e-5'
  []
[]
"""
        cand = """
[Variables]
  [c]
  []
[]
[Kernels]
  [diff]
    type = MatDiffusion
    variable = c
    diffusivity = 'diffusivity'
  []
[]
[Materials]
  [diff_coef]
    type = GenericConstantMaterial
    prop_names = 'diffusivity'
    prop_values = '1e-5'
  []
[]
"""
        result = compute_material_consistency(ref, cand)
        assert result.score == 1.0
        assert result.total_properties == 1

    def test_density_specific_heat_product_is_compared_as_effective_coefficient(self):
        ref = """
[Variables]
  [T]
  []
[]
[Kernels]
  [time]
    type = HeatConductionTimeDerivative
    variable = T
  []
[]
[Materials]
  [thermal]
    type = HeatConductionMaterial
    specific_heat = 385
  []
  [density]
    type = GenericConstantMaterial
    prop_names = 'density'
    prop_values = '8960'
  []
[]
"""
        cand = """
[Variables]
  [T]
  []
[]
[Kernels]
  [time]
    type = HeatConductionTimeDerivative
    variable = T
  []
[]
[Materials]
  [thermal]
    type = HeatConductionMaterial
    density = 1
    specific_heat = 3449600
  []
[]
"""
        result = compute_material_consistency(ref, cand)
        assert result.score == 1.0
        assert result.total_properties == 1

    def test_uncovered_constitutive_material_parameters_remain_checked(self):
        ref = """
[Materials]
  [elasticity]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 110e9
    poissons_ratio = 0.34
  []
[]
"""
        cand = """
[Materials]
  [elasticity]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 1e9
    poissons_ratio = 0.3
  []
[]
"""
        result = compute_material_consistency(ref, cand)
        assert result.score < 1.0
        assert result.total_properties >= 2
        assert any(item["property"] == "material:youngs_modulus" for item in result.mismatched)
        assert any(item["property"] == "material:poissons_ratio" for item in result.mismatched)

    def test_convective_heat_flux_bc_coefficients_are_checked(self):
        ref = """
[Variables]
  [T]
  []
[]
[BCs]
  [convective]
    type = ConvectiveHeatFluxBC
    variable = T
    boundary = right
    T_infinity = 300
    heat_transfer_coefficient = 50
  []
[]
"""
        cand = """
[Variables]
  [T]
  []
[]
[BCs]
  [convective]
    type = ConvectiveHeatFluxBC
    variable = T
    boundary = right
    T_infinity = 300
    heat_transfer_coefficient = 5
  []
[]
"""
        result = compute_material_consistency(ref, cand)
        assert result.score == 0.5
        assert result.total_properties == 2
        assert any(
            item["property"] == "bc:neumann:t:right:heat_transfer_coefficient"
            for item in result.mismatched
        )

    def test_material_model_signature_catches_plasticity_model_omission(self):
        ref = """
[Materials]
  [elasticity]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 110e9
    poissons_ratio = 0.34
  []
  [stress]
    type = ComputeMultipleInelasticStress
    inelastic_models = plasticity
  []
  [plasticity]
    type = IsotropicPlasticityStressUpdate
    yield_stress = 250e6
    hardening_constant = 1e9
  []
[]
"""
        cand = """
[Materials]
  [elasticity]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 110e9
    poissons_ratio = 0.34
  []
  [stress]
    type = ComputeLinearElasticStress
  []
[]
"""
        result = compute_material_consistency(ref, cand)
        assert result.score < 1.0
        assert any(
            item["property"] == "material_model:stress_model"
            and item["reason"] == "value_mismatch"
            for item in result.mismatched
        )
        assert any(
            item["property"] == "material_model:inelastic_model"
            and item["reason"] == "missing_in_candidate"
            for item in result.mismatched
        )

    def test_extract_coefficient_contract_lists_bc_and_model_facts(self):
        code = """
[BCs]
  [convective]
    type = ADConvectiveHeatFluxBC
    variable = T
    boundary = right
    T_infinity_functor = ambient
    heat_transfer_coefficient_functor = h_wall
  []
[]
[Materials]
  [strain]
    type = ComputeSmallStrain
  []
[]
"""
        contract = extract_coefficient_contract(code)
        keys = {item["key"] for item in contract}
        assert "bc:neumann:t:right:t_infinity" in keys
        assert "bc:neumann:t:right:heat_transfer_coefficient" in keys
        assert "material_model:strain_model" in keys
