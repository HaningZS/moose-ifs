"""Tests for paper experiment pipeline (Phase 4)."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from codmos.multiagent.pde.representation import BoundaryCondition, PDERepresentation, PDETerm
from experiments.moosebench_loader import BenchmarkPrompt, load_benchmark
from experiments.llm import MockLLM
from experiments.moose_env import detect_moose
from experiments.prompts import (
    format_codegen_prompt,
    format_refinement_prompt,
    format_spec_guided_prompt,
)
from experiments.runner import run_single_prompt
from experiments.variants.ablations import (
    run_full_cross_llm,
    run_full_exec_only,
    run_full_generic_feedback,
    run_full_no_refine,
)
from experiments.variants.base import VariantResult
from experiments.variants.direct import run_direct
from experiments.variants.full import run_full
from experiments.variants.spec_guided import run_spec_guided
from experiments.variants.verif_only import run_verif_only


class TestLLMBackend:
    def test_mock_llm_returns_configured_response(self):
        llm = MockLLM(responses=["hello world"])
        result = llm.generate("system", "user")
        assert result == "hello world"

    def test_mock_llm_cycles_responses(self):
        llm = MockLLM(responses=["first", "second"])
        assert llm.generate("s", "u") == "first"
        assert llm.generate("s", "u") == "second"
        # Cycles back
        assert llm.generate("s", "u") == "first"

    def test_mock_llm_records_calls(self):
        llm = MockLLM(responses=["r"])
        llm.generate("sys_prompt", "user_prompt", temperature=0.5)
        assert len(llm.call_log) == 1
        assert llm.call_log[0]["system_prompt"] == "sys_prompt"
        assert llm.call_log[0]["user_prompt"] == "user_prompt"
        assert llm.call_log[0]["temperature"] == 0.5


class TestMOOSEEnv:
    def test_detect_moose_not_found(self):
        with patch("shutil.which", return_value=None):
            env = detect_moose(explicit_path=None, conda_env="nonexistent_env_xyz")
            assert env.detected_method == "not_found"

    def test_detect_moose_explicit_path(self, tmp_path):
        fake_moose = tmp_path / "moose-opt"
        fake_moose.touch()
        env = detect_moose(explicit_path=str(fake_moose))
        assert env.detected_method == "explicit"
        assert env.moose_opt_path == fake_moose


class TestBenchmarkLoader:
    def test_load_prompt_from_json(self, tmp_path):
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        gt_dir = tmp_path / "ground_truth"
        gt_dir.mkdir()

        prompt_data = {
            "id": "thermal_001",
            "nl_description": "Simulate steady-state heat conduction in a 1D rod.",
            "physics_family": "heat_transfer",
            "complexity": "simple",
            "expected_kernels": ["HeatConduction"],
            "notes": "Basic thermal problem",
        }
        (prompt_dir / "thermal_001.json").write_text(json.dumps(prompt_data))

        gt_data = {
            "terms": [
                {"operator": "diffusion", "variable": "T", "coefficient": 45.0,
                 "kernel_type": "HeatConduction", "severity": "high"}
            ],
            "boundary_conditions": [
                {"bc_type": "dirichlet", "boundary": "left", "variable": "T",
                 "value": 300.0, "moose_type": "DirichletBC", "severity": "high"}
            ],
            "initial_conditions": [],
            "time_scheme": "steady",
            "variables": ["T"],
            "domain_dim": 1,
        }
        (gt_dir / "thermal_001.json").write_text(json.dumps(gt_data))

        prompts = load_benchmark(prompt_dir, gt_dir)
        assert len(prompts) == 1
        assert prompts[0].id == "thermal_001"
        assert prompts[0].nl_description == "Simulate steady-state heat conduction in a 1D rod."
        assert prompts[0].pde_gt is not None
        assert prompts[0].pde_gt.variables == ["T"]

    def test_load_prompt_missing_gt_still_loads(self, tmp_path):
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        gt_dir = tmp_path / "ground_truth"
        gt_dir.mkdir()

        prompt_data = {
            "id": "mech_001",
            "nl_description": "A problem with no GT yet.",
            "physics_family": "mechanics",
            "complexity": "medium",
            "expected_kernels": [],
            "notes": "",
        }
        (prompt_dir / "mech_001.json").write_text(json.dumps(prompt_data))

        prompts = load_benchmark(prompt_dir, gt_dir)
        assert len(prompts) == 1
        assert prompts[0].pde_gt is None

    def test_load_empty_dir_returns_empty(self, tmp_path):
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        gt_dir = tmp_path / "ground_truth"
        gt_dir.mkdir()
        prompts = load_benchmark(prompt_dir, gt_dir)
        assert prompts == []


class TestVariantResult:
    def test_construction(self):
        result = VariantResult(
            variant="direct",
            code="[Mesh]\n  type = GeneratedMesh\n[]",
            iterations=1,
            internal_ifs=None,
            pde_llm=None,
            parse_success=True,
            moose_result=None,
        )
        assert result.variant == "direct"
        assert result.iterations == 1

    def test_to_dict(self):
        result = VariantResult(
            variant="full",
            code="code",
            iterations=2,
            internal_ifs=None,
            pde_llm=None,
            parse_success=True,
            moose_result=None,
        )
        d = result.to_dict()
        assert d["variant"] == "full"
        assert d["iterations"] == 2


class TestPromptTemplates:
    def test_codegen_prompt_contains_nl(self):
        prompt = format_codegen_prompt("Simulate heat conduction in a rod")
        assert "heat conduction" in prompt.lower()

    def test_spec_guided_prompt_contains_pde_terms(self):
        pde = PDERepresentation(
            terms=[PDETerm(
                variable="T",
                operator="diffusion",
                coefficient=45.0,
                coupled_variable=None,
                kernel_type="HeatConduction",
                severity="high",
            )],
            boundary_conditions=[],
            initial_conditions=[],
            time_scheme="steady",
            variables=["T"],
            dimensions=1,
            unresolved_kernels=[],
            unresolved_coefficients=[],
            warnings=[],
        )
        prompt = format_spec_guided_prompt(pde)
        assert "diffusion" in prompt
        assert "T" in prompt

    def test_refinement_prompt_contains_violations(self):
        prompt = format_refinement_prompt(
            nl_description="Heat conduction",
            previous_code="[Mesh]...",
            violation_report="Missing BC on left boundary",
        )
        assert "Missing BC" in prompt


MOCK_MOOSE_CODE = """\
[Mesh]
  type = GeneratedMesh
  dim = 1
  nx = 10
[]

[Variables]
  [T]
  []
[]

[Kernels]
  [heat]
    type = HeatConduction
    variable = T
  []
[]

[BCs]
  [left]
    type = DirichletBC
    variable = T
    boundary = left
    value = 300
  []
  [right]
    type = DirichletBC
    variable = T
    boundary = right
    value = 500
  []
[]

[Materials]
  [thermal]
    type = GenericConstantMaterial
    prop_names = 'thermal_conductivity'
    prop_values = '45'
  []
[]

[Executioner]
  type = Steady
[]

[Outputs]
  exodus = true
[]
"""


def _make_pde_llm():
    """Create a PDERepresentation matching MOCK_MOOSE_CODE for testing."""
    return PDERepresentation(
        terms=[PDETerm(
            variable="T",
            operator="diffusion",
            coefficient=45.0,
            coupled_variable=None,
            kernel_type="HeatConduction",
            severity="high",
        )],
        boundary_conditions=[
            BoundaryCondition(variable="T", boundary="left", bc_type="dirichlet",
                            value=300.0, moose_bc_class="DirichletBC", severity="high"),
            BoundaryCondition(variable="T", boundary="right", bc_type="dirichlet",
                            value=500.0, moose_bc_class="DirichletBC", severity="high"),
        ],
        initial_conditions=[],
        time_scheme="steady",
        variables=["T"],
        dimensions=1,
        unresolved_kernels=[],
        unresolved_coefficients=[],
        warnings=[],
    )


def _moose_registry_for_runner():
    from codmos.multiagent.moose_registry import MooseRegistry

    return MooseRegistry.from_moose_json({
        "blocks": {
            "Mesh": {
                "types": {
                    "GeneratedMesh": {
                        "parent_syntax": "Mesh",
                        "parameters": {
                            "type": {"name": "type"},
                            "dim": {"name": "dim", "required": True},
                            "nx": {"name": "nx"},
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
                },
            },
            "Materials": {
                "subblock_types": {
                    "GenericConstantMaterial": {
                        "parent_syntax": "Materials/*",
                        "parameters": {
                            "type": {"name": "type"},
                            "prop_names": {"name": "prop_names", "required": True},
                            "prop_values": {"name": "prop_values", "required": True},
                        },
                    },
                },
            },
            "Executioner": {
                "types": {
                    "Steady": {
                        "parent_syntax": "Executioner",
                        "parameters": {"type": {"name": "type"}},
                    },
                },
            },
        }
    })


class TestVariantDirect:
    def test_direct_produces_variant_result(self):
        llm = MockLLM(responses=[MOCK_MOOSE_CODE])
        result = run_direct(nl_description="Heat conduction in a 1D rod", llm=llm)
        assert result.variant == "direct"
        assert result.iterations == 1
        assert result.code == MOCK_MOOSE_CODE
        assert result.pde_llm is None
        assert result.internal_ifs is None

    def test_direct_records_parse_success(self):
        llm = MockLLM(responses=[MOCK_MOOSE_CODE])
        result = run_direct(nl_description="Heat conduction", llm=llm)
        assert result.parse_success is True

    def test_direct_handles_unparseable_code(self):
        llm = MockLLM(responses=["this is not valid MOOSE code at all"])
        result = run_direct(nl_description="Simulate something", llm=llm)
        assert result.parse_success is False


class TestVariantSpecGuided:
    def test_spec_guided_uses_pde_llm(self):
        llm = MockLLM(responses=[MOCK_MOOSE_CODE])
        pde_llm = _make_pde_llm()
        result = run_spec_guided(pde_llm=pde_llm, llm=llm)
        assert result.variant == "spec_guided"
        assert result.pde_llm is pde_llm
        assert result.iterations == 1
        assert "diffusion" in llm.call_log[0]["user_prompt"]

    def test_spec_guided_no_internal_ifs(self):
        llm = MockLLM(responses=[MOCK_MOOSE_CODE])
        result = run_spec_guided(pde_llm=_make_pde_llm(), llm=llm)
        assert result.internal_ifs is None


class TestVariantVerifOnly:
    def test_verif_only_stops_when_ifs_high(self):
        llm = MockLLM(responses=[MOCK_MOOSE_CODE])
        pde_llm = _make_pde_llm()
        result = run_verif_only(
            nl_description="Heat conduction in a rod",
            pde_llm=pde_llm, llm=llm,
            ifs_threshold=0.4, max_iterations=3,
        )
        assert result.variant == "verif_only"
        assert result.iterations == 1
        assert result.internal_ifs is not None
        assert result.internal_ifs.ifs_score >= 0.4

    def test_verif_only_respects_max_iterations(self):
        # Code that parses but has low IFS (missing kernels)
        bad = "[Mesh]\n  type = GeneratedMesh\n  dim = 1\n[]\n[Variables]\n  [T]\n  []\n[]\n[Executioner]\n  type = Steady\n[]\n[Outputs]\n  exodus = true\n[]"
        llm = MockLLM(responses=[bad])
        pde_llm = _make_pde_llm()
        result = run_verif_only(
            nl_description="Heat conduction",
            pde_llm=pde_llm, llm=llm,
            ifs_threshold=0.99, max_iterations=2,
        )
        assert result.iterations <= 2


class TestVariantFull:
    def test_full_uses_spec_guided_generation(self):
        llm = MockLLM(responses=[MOCK_MOOSE_CODE])
        pde_llm = _make_pde_llm()
        result = run_full(
            nl_description="Heat conduction",
            pde_llm=pde_llm, llm=llm,
            ifs_threshold=0.5, max_iterations=3,
        )
        assert result.variant == "full"
        assert result.pde_llm is pde_llm
        assert "diffusion" in llm.call_log[0]["user_prompt"]

    def test_full_has_internal_ifs(self):
        llm = MockLLM(responses=[MOCK_MOOSE_CODE])
        pde_llm = _make_pde_llm()
        result = run_full(
            nl_description="Heat conduction",
            pde_llm=pde_llm, llm=llm,
            ifs_threshold=0.5, max_iterations=3,
        )
        assert result.internal_ifs is not None
        assert 0.0 <= result.internal_ifs.ifs_score <= 1.0

    def test_full_refines_on_low_ifs(self):
        bad = "[Mesh]\n  type = GeneratedMesh\n  dim = 1\n[]\n[Variables]\n  [T]\n  []\n[]\n[Executioner]\n  type = Steady\n[]\n[Outputs]\n  exodus = true\n[]"
        llm = MockLLM(responses=[bad, MOCK_MOOSE_CODE])
        pde_llm = _make_pde_llm()
        result = run_full(
            nl_description="Heat conduction",
            pde_llm=pde_llm, llm=llm,
            ifs_threshold=0.8, max_iterations=3,
        )
        assert result.iterations >= 1


class TestAblations:
    def test_no_refine_single_iteration(self):
        llm = MockLLM(responses=[MOCK_MOOSE_CODE])
        pde_llm = _make_pde_llm()
        result = run_full_no_refine(
            nl_description="Heat conduction",
            pde_llm=pde_llm, llm=llm,
        )
        assert result.variant == "D-no-refine"
        assert result.iterations == 1
        assert result.internal_ifs is not None

    def test_generic_feedback_no_violation_details(self):
        bad_code = "[Mesh]\n  type = GeneratedMesh\n  dim = 1\n[]\n[Variables]\n  [T]\n  []\n[]\n[Executioner]\n  type = Steady\n[]\n[Outputs]\n  exodus = true\n[]"
        llm = MockLLM(responses=[bad_code, MOCK_MOOSE_CODE])
        pde_llm = _make_pde_llm()
        result = run_full_generic_feedback(
            nl_description="Heat conduction",
            pde_llm=pde_llm, llm=llm,
            ifs_threshold=0.8, max_iterations=3,
        )
        assert result.variant == "D-generic"
        if len(llm.call_log) > 1:
            # Generic feedback should say "errors" but NOT specific violations
            assert "errors" in llm.call_log[1]["user_prompt"].lower()

    def test_exec_only_stops_on_parse_success(self):
        llm = MockLLM(responses=[MOCK_MOOSE_CODE])
        pde_llm = _make_pde_llm()
        result = run_full_exec_only(
            nl_description="Heat conduction",
            pde_llm=pde_llm, llm=llm, max_iterations=3,
        )
        assert result.variant == "D-exec-only"
        assert result.iterations == 1  # Stops immediately on parse success

    def test_cross_llm_uses_generate_backend(self):
        llm_generate = MockLLM(responses=[MOCK_MOOSE_CODE])
        pde_llm = _make_pde_llm()
        result = run_full_cross_llm(
            nl_description="Heat conduction",
            pde_llm=pde_llm,
            llm_generate=llm_generate,
            ifs_threshold=0.5, max_iterations=3,
        )
        assert result.variant == "D-cross"
        assert len(llm_generate.call_log) >= 1


class TestRunner:
    def test_run_single_prompt_direct(self):
        pde_gt = _make_pde_llm()
        prompt = BenchmarkPrompt(
            id="test_001",
            nl_description="Heat conduction in a 1D rod",
            physics_family="heat_transfer",
            complexity="simple",
            expected_kernels=["HeatConduction"],
            notes="",
            pde_gt=pde_gt,
        )
        llm = MockLLM(responses=[MOCK_MOOSE_CODE])
        result = run_single_prompt(prompt=prompt, variant="direct", llm=llm)
        assert result.prompt_id == "test_001"
        assert result.variant == "direct"
        assert result.eval_ifs is not None
        assert 0.0 <= result.eval_ifs.ifs_score <= 1.0

    def test_run_single_prompt_full(self):
        pde_gt = _make_pde_llm()
        prompt = BenchmarkPrompt(
            id="test_002",
            nl_description="Heat conduction",
            physics_family="heat_transfer",
            complexity="simple",
            expected_kernels=["HeatConduction"],
            notes="",
            pde_gt=pde_gt,
        )
        llm = MockLLM(responses=[MOCK_MOOSE_CODE])
        result = run_single_prompt(
            prompt=prompt, variant="full", llm=llm,
            pde_llm=_make_pde_llm(), ifs_threshold=0.5,
        )
        assert result.variant == "full"
        assert result.eval_ifs is not None

    def test_run_returns_none_eval_when_no_gt(self):
        prompt = BenchmarkPrompt(
            id="test_003",
            nl_description="Something",
            physics_family="mechanics",
            complexity="medium",
            expected_kernels=[],
            notes="",
            pde_gt=None,
        )
        llm = MockLLM(responses=[MOCK_MOOSE_CODE])
        result = run_single_prompt(prompt=prompt, variant="direct", llm=llm)
        assert result.eval_ifs is None


class TestIntegration:
    """End-to-end: benchmark loading → all 4 variants → evaluation."""

    def test_full_pipeline_with_benchmark_data(self, tmp_path):
        import json

        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        gt_dir = tmp_path / "ground_truth"
        gt_dir.mkdir()

        prompt_data = {
            "id": "integration_001",
            "nl_description": "Simulate steady-state heat conduction in a 1D rod with T=300 on left and T=500 on right.",
            "physics_family": "heat_transfer",
            "complexity": "simple",
            "expected_kernels": ["HeatConduction"],
            "notes": "Integration test",
        }
        (prompt_dir / "integration_001.json").write_text(json.dumps(prompt_data))

        gt_data = {
            "terms": [
                {"operator": "diffusion", "variable": "T", "coefficient": 45.0,
                 "kernel_type": "HeatConduction", "severity": "high"}
            ],
            "boundary_conditions": [
                {"bc_type": "dirichlet", "boundary": "left", "variable": "T",
                 "value": 300.0, "moose_type": "DirichletBC", "severity": "high"},
                {"bc_type": "dirichlet", "boundary": "right", "variable": "T",
                 "value": 500.0, "moose_type": "DirichletBC", "severity": "high"},
            ],
            "initial_conditions": [],
            "time_scheme": "steady",
            "variables": ["T"],
            "domain_dim": 1,
        }
        (gt_dir / "integration_001.json").write_text(json.dumps(gt_data))

        from experiments.moosebench_loader import load_benchmark
        prompts = load_benchmark(prompt_dir, gt_dir)
        assert len(prompts) == 1

        pde_llm = _make_pde_llm()
        results = []
        for variant in ["direct", "spec_guided", "verif_only", "full"]:
            llm_fresh = MockLLM(responses=[MOCK_MOOSE_CODE])
            result = run_single_prompt(
                prompt=prompts[0], variant=variant, llm=llm_fresh,
                pde_llm=pde_llm, ifs_threshold=0.5,
            )
            results.append(result)
            assert result.eval_ifs is not None, f"{variant} should have eval_ifs"
            assert result.eval_ifs.ifs_score > 0, f"{variant} IFS should be > 0"

        assert len(results) == 4


class TestMooseBenchRunnerParallel:
    def test_parse_args_accepts_workers(self, monkeypatch):
        import sys

        from scripts import run_moosebench

        monkeypatch.setattr(
            sys,
            "argv",
            ["run_moosebench.py", "--llm", "deepseek-flash", "--workers", "3"],
        )

        args = run_moosebench.parse_args()

        assert args.workers == 3

    def test_parse_args_accepts_registry_methods(self, monkeypatch):
        import sys

        from scripts import run_moosebench

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_moosebench.py",
                "--llm",
                "deepseek-flash",
                "--methods",
                "AReg",
                "DReg",
                "--registry-json",
                "combined_syntax_full.txt",
            ],
        )

        args = run_moosebench.parse_args()

        assert args.methods == ["AReg", "DReg"]
        assert str(args.registry_json) == "combined_syntax_full.txt"

    def test_parse_args_accepts_exec_repair_reg(self, monkeypatch):
        import sys

        from scripts import run_moosebench

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_moosebench.py",
                "--llm",
                "deepseek-flash",
                "--methods",
                "ExecRepairReg",
                "--registry-json",
                "combined_syntax_full.txt",
                "--moose-app",
                "combined-opt",
                "--smoke-timeout",
                "5",
            ],
        )

        args = run_moosebench.parse_args()

        assert args.methods == ["ExecRepairReg"]
        assert str(args.registry_json) == "combined_syntax_full.txt"
        assert str(args.moose_app) == "combined-opt"
        assert args.smoke_timeout == 5

    def test_parse_args_default_smoke_timeout_is_two(self, monkeypatch):
        import sys

        from scripts import run_moosebench

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_moosebench.py",
                "--llm",
                "deepseek-flash",
                "--methods",
                "ExecRepairReg",
                "--registry-json",
                "combined_syntax_full.txt",
                "--moose-app",
                "combined-opt",
            ],
        )

        args = run_moosebench.parse_args()

        assert args.smoke_timeout == 2

    def test_run_evaluation_parallel_uses_multiple_worker_threads(
        self, tmp_path, monkeypatch
    ):
        import contextlib
        import json
        import threading

        from scripts import run_moosebench

        prompt_dir = tmp_path / "prompts"
        gt_dir = tmp_path / "ground_truth"
        prompt_dir.mkdir()
        gt_dir.mkdir()

        for idx in range(4):
            case_id = f"parallel_{idx:03d}"
            (prompt_dir / f"{case_id}.json").write_text(
                json.dumps({
                    "id": case_id,
                    "nl_description": f"Parallel test case {idx}",
                    "physics_family": "heat_transfer",
                    "complexity": "simple",
                })
            )
            (gt_dir / f"{case_id}.json").write_text("{}")

        monkeypatch.setattr(run_moosebench, "_PROMPTS_DIR", prompt_dir)
        monkeypatch.setattr(run_moosebench, "_GT_DIR", gt_dir)
        monkeypatch.setattr(run_moosebench, "make_llm", lambda _name: object())
        monkeypatch.setattr(run_moosebench, "load_gt", lambda _path: _make_pde_llm())

        thread_ids: set[int] = set()
        thread_lock = threading.Lock()
        gate = threading.Barrier(2)

        def fake_run_method_a(_llm, _nl, _gt):
            with thread_lock:
                thread_ids.add(threading.get_ident())
            with contextlib.suppress(threading.BrokenBarrierError):
                gate.wait(timeout=2)
            return {
                "parse": True,
                "ifs": 1.0,
                "term": 1.0,
                "coeff": 1.0,
                "bc": 1.0,
                "ic": 1.0,
                "time_dim": 1.0,
                "passed": 1,
                "total": 1,
                "internal_ifs": None,
                "extraction_ifs": None,
                "error": None,
            }

        monkeypatch.setattr(run_moosebench, "run_method_a", fake_run_method_a)

        output_path = tmp_path / "results.jsonl"
        run_moosebench.run_evaluation(
            llm_name="deepseek-flash",
            methods=["A"],
            limit=0,
            output_path=output_path,
            workers=2,
        )

        records = [
            json.loads(line)
            for line in output_path.read_text().splitlines()
            if line.strip()
        ]

        assert len(records) == 4
        assert len(thread_ids) >= 2


class TestMooseBenchExtractionRetry:
    @staticmethod
    def _pde_json(*, terms: list[dict] | None = None, variables: list[str] | None = None) -> str:
        return json.dumps({
            "variables": variables if variables is not None else ["T"],
            "terms": terms if terms is not None else [
                {"variable": "T", "operator": "diffusion", "coefficient": 45.0}
            ],
            "boundary_conditions": [
                {
                    "variable": "T",
                    "boundary": "left",
                    "bc_type": "Dirichlet",
                    "value": 300.0,
                },
                {
                    "variable": "T",
                    "boundary": "right",
                    "bc_type": "Dirichlet",
                    "value": 500.0,
                },
            ],
            "initial_conditions": [],
            "time_scheme": "steady",
            "dimensions": 1,
        })

    def test_low_confidence_self_check_retries_and_uses_second_extraction(self):
        from scripts import run_moosebench

        first_spec = self._pde_json(terms=[])
        retry_spec = self._pde_json()
        self_check = json.dumps({
            "confidence": 0.2,
            "needs_retry": True,
            "flags": ["possible missing diffusion term"],
            "checklist": ["recheck variables"],
        })
        llm = MockLLM(responses=[first_spec, self_check, retry_spec, MOCK_MOOSE_CODE])

        result = run_moosebench.run_method_b(
            llm,
            "Simulate steady heat conduction in a 1D rod.",
            _make_pde_llm(),
            fallback_threshold=0.0,
        )

        assert result["extraction_retried"] is True
        assert "self-check" in result["extraction_retry_reason"]
        assert len(llm.call_log) == 4
        assert "diffusion" in llm.call_log[3]["user_prompt"]

    def test_retry_keeps_first_extraction_when_second_is_lower_quality(self):
        from scripts import run_moosebench

        first_spec = self._pde_json(
            variables=["T", "pp", "disp_x"],
            terms=[
                {"variable": "pp", "operator": "pf_darcy_flux", "coefficient": 1.0},
                {
                    "variable": "disp_x",
                    "operator": "pf_effective_stress",
                    "coefficient": None,
                },
            ],
        )
        retry_spec = self._pde_json(
            variables=["pp"],
            terms=[
                {"variable": "pp", "operator": "diffusion", "coefficient": 1.0},
            ],
        )
        self_check = json.dumps({
            "confidence": 0.2,
            "needs_retry": True,
            "flags": ["possible missing THM coupling"],
            "checklist": ["recheck coupling"],
        })
        llm = MockLLM(responses=[first_spec, self_check, retry_spec])

        attempt = run_moosebench._extract_pde_llm(
            llm,
            "Model thermo-hydro-mechanical coupling with pore pressure, "
            "Darcy flow, and effective stress.",
        )

        assert attempt.retried is True
        assert attempt.retry_accepted is False
        assert {term.operator for term in attempt.pde.terms} == {
            "pf_darcy_flux",
            "pf_effective_stress",
        }

    def test_parse_pde_llm_extracts_json_object_from_prose(self):
        from scripts import run_moosebench

        response = f"Here is the extracted PDE JSON:\n{self._pde_json()}\nDone."

        pde = run_moosebench.parse_pde_llm(response)

        assert pde is not None
        assert pde.variables == ["T"]
        assert pde.terms[0].operator == "diffusion"

    def test_parse_pde_llm_repairs_numeric_coefficient_expression(self):
        from scripts import run_moosebench

        response = """
        {
          "variables": ["T"],
          "terms": [
            {"variable": "T", "operator": "time_derivative", "coefficient": 7200.0 * 500.0},
            {"variable": "T", "operator": "diffusion", "coefficient": 25.0 / 5.0}
          ],
          "boundary_conditions": [
            {"variable": "T", "boundary": "left", "bc_type": "Dirichlet", "value": 300.0 + 5.0}
          ],
          "initial_conditions": [],
          "time_scheme": "transient",
          "dimensions": 2
        }
        """

        pde = run_moosebench.parse_pde_llm(response)

        assert pde is not None
        assert pde.terms[0].coefficient == 3600000.0
        assert pde.terms[1].coefficient == 5.0
        assert pde.boundary_conditions[0].value == 305.0

    def test_parse_pde_llm_does_not_repair_non_numeric_coefficient(self):
        from scripts import run_moosebench

        response = """
        {
          "variables": ["T"],
          "terms": [
            {"variable": "T", "operator": "diffusion", "coefficient": rho_cp * 500.0}
          ],
          "boundary_conditions": [],
          "initial_conditions": [],
          "time_scheme": "steady",
          "dimensions": 2
        }
        """

        assert run_moosebench.parse_pde_llm(response) is None

    def test_parse_pde_llm_repairs_commented_structured_coefficient(self):
        from scripts import run_moosebench

        response = """
        {
          "variables": ["c"],
          "terms": [
            {
              "variable": "c",
              "operator": "diffusion",
              "coefficient": {
                "type": "tensor",
                "values": [[/* D_x */, /* D_xy */], [/* D_yx */, /* D_y */]]
              }
            },
            {"variable": "c", "operator": "reaction", "coefficient": -0.5}
          ],
          "boundary_conditions": [],
          "initial_conditions": [],
          "time_scheme": "transient",
          "dimensions": 2
        }
        """

        pde = run_moosebench.parse_pde_llm(response)

        assert pde is not None
        assert pde.terms[0].operator == "diffusion"
        assert pde.terms[0].coefficient is None
        assert pde.terms[1].coefficient == -0.5

    def test_code_fence_repair_only_after_parse_failure(self):
        from scripts import run_moosebench

        llm = MockLLM(responses=[f"```moose\n{MOCK_MOOSE_CODE}\n```"])

        result = run_moosebench.run_method_a(
            llm,
            "Simulate steady heat conduction in a 1D rod.",
            _make_pde_llm(),
        )

        assert result["parse"] is True
        assert result["code_repair_applied"] is True
        assert result["_code"].strip() == MOCK_MOOSE_CODE.strip()

    def test_parse_success_keeps_original_code_unchanged(self):
        from scripts import run_moosebench

        llm = MockLLM(responses=[MOCK_MOOSE_CODE])

        result = run_moosebench.run_method_a(
            llm,
            "Simulate steady heat conduction in a 1D rod.",
            _make_pde_llm(),
        )

        assert result["parse"] is True
        assert result["code_repair_applied"] is False
        assert result["_code"] == MOCK_MOOSE_CODE

    def test_heat_prompt_does_not_add_targeted_pf_thm_retry_hint(self):
        from scripts import run_moosebench

        self_check = json.dumps({
            "confidence": 0.95,
            "needs_retry": False,
            "flags": [],
            "checklist": [],
        })
        llm = MockLLM(responses=[self._pde_json(), self_check, MOCK_MOOSE_CODE])

        result = run_moosebench.run_method_b(
            llm,
            "Simulate steady heat conduction in a 1D rod.",
            _make_pde_llm(),
            fallback_threshold=0.0,
        )

        assert result["extraction_retried"] is False
        combined_user_prompts = "\n".join(call["user_prompt"] for call in llm.call_log)
        assert "Phase-field retry context" not in combined_user_prompts
        assert "Porous-flow / THM retry context" not in combined_user_prompts

    def test_thm_targeted_hint_only_appears_in_retry_prompt(self):
        from scripts import run_moosebench

        generic_spec = self._pde_json(
            variables=["T", "pp", "disp_x"],
            terms=[
                {"variable": "T", "operator": "diffusion", "coefficient": 1.0},
                {"variable": "disp_x", "operator": "stress_divergence", "coefficient": None},
            ],
        )
        retry_spec = self._pde_json()
        self_check = json.dumps({
            "confidence": 0.9,
            "needs_retry": False,
            "flags": [],
            "checklist": [],
        })
        llm = MockLLM(responses=[generic_spec, self_check, retry_spec, MOCK_MOOSE_CODE])

        result = run_moosebench.run_method_b(
            llm,
            "Model thermo-hydro-mechanical coupling with pore pressure, Darcy flow, and effective stress.",
            _make_pde_llm(),
            fallback_threshold=0.0,
        )

        assert result["extraction_retried"] is True
        assert "Porous-flow / THM retry context" not in llm.call_log[0]["user_prompt"]
        assert "Porous-flow / THM retry context" in llm.call_log[2]["user_prompt"]

    def test_self_check_parse_failure_skips_retry_without_crashing(self):
        from scripts import run_moosebench

        llm = MockLLM(responses=[self._pde_json(), "not valid json", MOCK_MOOSE_CODE])

        result = run_moosebench.run_method_b(
            llm,
            "Simulate steady heat conduction in a 1D rod.",
            _make_pde_llm(),
            fallback_threshold=0.0,
        )

        assert result["parse"] is True
        assert result["extraction_retried"] is False
        assert result["extraction_retry_reason"] is None
        assert len(llm.call_log) == 3

    def test_parallel_runner_records_retry_metadata_per_task(
        self, tmp_path, monkeypatch
    ):
        from scripts import run_moosebench

        prompt_dir = tmp_path / "prompts"
        gt_dir = tmp_path / "ground_truth"
        prompt_dir.mkdir()
        gt_dir.mkdir()

        for idx in range(4):
            case_id = f"retry_parallel_{idx:03d}"
            (prompt_dir / f"{case_id}.json").write_text(json.dumps({
                "id": case_id,
                "nl_description": "Simulate heat conduction with a low-confidence first extraction.",
                "physics_family": "heat_transfer",
                "complexity": "simple",
            }))
            (gt_dir / f"{case_id}.json").write_text("{}")

        first_spec = self._pde_json(terms=[])
        retry_spec = self._pde_json()
        self_check = json.dumps({
            "confidence": 0.1,
            "needs_retry": True,
            "flags": ["missing terms"],
            "checklist": ["recheck variables"],
        })

        def make_fake_llm(_name):
            return MockLLM(responses=[first_spec, self_check, retry_spec, MOCK_MOOSE_CODE])

        monkeypatch.setattr(run_moosebench, "_PROMPTS_DIR", prompt_dir)
        monkeypatch.setattr(run_moosebench, "_GT_DIR", gt_dir)
        monkeypatch.setattr(run_moosebench, "make_llm", make_fake_llm)
        monkeypatch.setattr(run_moosebench, "load_gt", lambda _path: _make_pde_llm())

        output_path = tmp_path / "results.jsonl"
        run_moosebench.run_evaluation(
            llm_name="mock",
            methods=["B"],
            limit=0,
            output_path=output_path,
            workers=2,
            fallback_threshold=0.0,
        )

        records = [
            json.loads(line)
            for line in output_path.read_text().splitlines()
            if line.strip()
        ]

        assert len(records) == 4
        assert all(record["extraction_retried"] is True for record in records)
        assert all("self-check" in record["extraction_retry_reason"] for record in records)

    def test_runner_records_artifacts_provenance_and_mcs(
        self, tmp_path, monkeypatch
    ):
        from scripts import run_moosebench

        prompt_dir = tmp_path / "prompts"
        gt_dir = tmp_path / "ground_truth"
        source_dir = tmp_path / "source_files"
        prompt_dir.mkdir()
        gt_dir.mkdir()
        source_dir.mkdir()

        case_id = "artifact_001"
        (prompt_dir / f"{case_id}.json").write_text(json.dumps({
            "id": case_id,
            "nl_description": "Simulate steady heat conduction in a 1D rod.",
            "physics_family": "heat_transfer",
            "complexity": "simple",
        }))
        (gt_dir / f"{case_id}.json").write_text("{}")
        (source_dir / f"{case_id}.i").write_text(MOCK_MOOSE_CODE)

        spec_json = self._pde_json()
        self_check = json.dumps({
            "confidence": 0.95,
            "needs_retry": False,
            "flags": [],
            "checklist": [],
        })

        monkeypatch.setattr(run_moosebench, "_PROMPTS_DIR", prompt_dir)
        monkeypatch.setattr(run_moosebench, "_GT_DIR", gt_dir)
        monkeypatch.setattr(run_moosebench, "_SOURCE_FILES_DIR", source_dir)
        monkeypatch.setattr(
            run_moosebench,
            "make_llm",
            lambda _name: MockLLM(responses=[spec_json, self_check, MOCK_MOOSE_CODE]),
        )
        monkeypatch.setattr(run_moosebench, "load_gt", lambda _path: _make_pde_llm())

        output_path = tmp_path / "results.jsonl"
        artifact_dir = tmp_path / "artifacts"
        run_moosebench.run_evaluation(
            llm_name="mock",
            methods=["B"],
            limit=0,
            output_path=output_path,
            workers=1,
            fallback_threshold=0.0,
            artifact_dir=artifact_dir,
        )

        record = json.loads(output_path.read_text().strip())
        assert Path(record["code_path"]).exists()
        assert Path(record["pde_llm_path"]).exists()
        assert json.loads(Path(record["pde_llm_path"]).read_text())["variables"] == ["T"]
        assert record["prompt_version"] == run_moosebench._PROMPT_VERSION
        assert record["runner_commit"]
        assert record["generator_llm"] == "mock"
        assert record["extractor_llm"] == "mock"
        assert record["retry_enabled"] is True
        assert record["fallback_threshold"] == 0.0
        assert record["mcs"] == pytest.approx(1.0)
        assert record["mcs_applicable"] is True
        assert record["mcs_total"] == 1
        assert record["mcs_matched"] == 1


class TestMooseBenchRegistryMethods:
    def test_areg_applies_registry_without_preplan(self):
        from scripts import run_moosebench

        llm = MockLLM(responses=[MOCK_MOOSE_CODE])

        result = run_moosebench.run_method_areg(
            llm,
            "Simulate steady heat conduction in a 1D rod.",
            _make_pde_llm(),
            registry=_moose_registry_for_runner(),
        )

        assert result["parse"] is True
        assert result["registry_enabled"] is True
        assert result["registry_preplan_applied"] is False
        assert result["registry_l2_pass_after"] is True

    def test_execrepairreg_skips_llm_repair_when_smoke_passes(self, monkeypatch):
        from scripts import run_moosebench

        llm = MockLLM(responses=[MOCK_MOOSE_CODE])

        monkeypatch.setattr(
            run_moosebench,
            "_smoke_exec_code",
            lambda *_args, **_kwargs: run_moosebench.SmokeExecResult(
                passed=True,
                status="timeout_no_error",
                returncode="timeout",
                error_excerpt=None,
                timeout_s=5,
            ),
        )

        result = run_moosebench.run_method_execrepairreg(
            llm,
            "Simulate steady heat conduction in a 1D rod.",
            _make_pde_llm(),
            registry=_moose_registry_for_runner(),
            moose_app=Path("fake-moose"),
        )

        assert result["parse"] is True
        assert result["exec_repair_enabled"] is True
        assert result["exec_repair_attempted"] is False
        assert result["exec_repair_smoke_before_pass"] is True
        assert len(llm.call_log) == 1

    def test_execrepairreg_uses_one_runtime_log_repair(self, monkeypatch):
        from scripts import run_moosebench

        llm = MockLLM(responses=[MOCK_MOOSE_CODE, MOCK_MOOSE_CODE])
        smoke_results = iter([
            run_moosebench.SmokeExecResult(
                passed=False,
                status="failed_fast",
                returncode=1,
                error_excerpt="missing required parameter variable",
                timeout_s=5,
            ),
            run_moosebench.SmokeExecResult(
                passed=True,
                status="timeout_no_error",
                returncode="timeout",
                error_excerpt=None,
                timeout_s=5,
            ),
        ])

        monkeypatch.setattr(
            run_moosebench,
            "_smoke_exec_code",
            lambda *_args, **_kwargs: next(smoke_results),
        )

        result = run_moosebench.run_method_execrepairreg(
            llm,
            "Simulate steady heat conduction in a 1D rod.",
            _make_pde_llm(),
            registry=_moose_registry_for_runner(),
            moose_app=Path("fake-moose"),
        )

        assert result["parse"] is True
        assert result["exec_repair_attempted"] is True
        assert result["exec_repair_accepted"] is True
        assert result["exec_repair_rounds"] == 1
        assert result["exec_repair_smoke_before_pass"] is False
        assert result["exec_repair_smoke_after_pass"] is True
        assert len(llm.call_log) == 2
        assert "missing required parameter variable" in llm.call_log[1]["user_prompt"]
        assert "PDE/IFS feedback" in llm.call_log[1]["user_prompt"]

    def test_dreg_includes_object_plan_and_records_artifact(self, tmp_path):
        from scripts import run_moosebench

        spec_json = TestMooseBenchExtractionRetry._pde_json()
        self_check = json.dumps({
            "confidence": 0.95,
            "needs_retry": False,
            "flags": [],
            "checklist": [],
        })
        llm = MockLLM(responses=[spec_json, self_check, MOCK_MOOSE_CODE])

        result = run_moosebench.run_method_dreg(
            llm,
            "Simulate steady heat conduction in a 1D rod.",
            _make_pde_llm(),
            fallback_threshold=0.0,
            registry=_moose_registry_for_runner(),
        )

        assert result["parse"] is True
        assert result["registry_preplan_applied"] is True
        assert "Frozen MOOSE object-realization plan" in result["_object_plan"]
        assert "Frozen MOOSE object-realization plan" in llm.call_log[2]["user_prompt"]

        task = {
            "case_id": "registry_001",
            "method": "DReg",
            "source_path": None,
        }
        paths = run_moosebench._write_artifacts(
            task,
            "mock",
            result,
            tmp_path,
        )

        assert Path(paths["object_plan_path"]).exists()


class TestMooseBenchAnalyzer:
    def test_build_summary_reports_dc_decomposition_and_retry_groups(self, tmp_path):
        from scripts import analyze_moosebench

        records = [
            {
                "id": "case_1", "method": "A", "llm": "mock",
                "family": "heat", "complexity": "simple",
                "parse": True, "ifs": 0.5,
            },
            {
                "id": "case_1", "method": "B", "llm": "mock",
                "family": "heat", "complexity": "simple",
                "parse": True, "ifs": 0.7, "extraction_ifs": 0.9,
                "extraction_retried": False,
            },
            {
                "id": "case_1", "method": "C", "llm": "mock",
                "family": "heat", "complexity": "simple",
                "parse": True, "ifs": 0.95,
            },
            {
                "id": "case_1", "method": "D", "llm": "mock",
                "family": "heat", "complexity": "simple",
                "parse": True, "ifs": 0.75, "internal_ifs": 0.8,
                "extraction_ifs": 0.9, "extraction_retried": True,
            },
        ]
        path = tmp_path / "results.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records))

        loaded = analyze_moosebench.load_records(path)
        summary = analyze_moosebench.build_summary(loaded)
        tidy_rows = analyze_moosebench.build_tidy_rows(loaded)

        assert summary["method_means"]["D"]["mean_ifs"] == pytest.approx(0.75)
        assert summary["gap_means"]["C-D"] == pytest.approx(0.20)
        assert summary["gap_means"]["D-B"] == pytest.approx(0.05)
        assert summary["dc_decomposition"]["n"] == 1
        assert summary["dc_decomposition"]["mean_extraction_loss"] == pytest.approx(0.10)
        assert summary["dc_decomposition"]["mean_generation_loss"] == pytest.approx(0.20)
        assert summary["retry"]["retried"]["n"] == 1
        assert summary["extraction_bins"][">=0.9"]["n"] == 2
        assert any(row["metric"] == "C-D" for row in tidy_rows)
