"""Benchmark data model and loading for paper experiments.

Defines BenchmarkPrompt (prompt metadata + optional PDE_gt) and
a loader that reads JSON files from prompts/ and ground_truth/ dirs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from codmos.multiagent.pde.representation import (
    BoundaryCondition,
    InitialCondition,
    PDERepresentation,
    PDETerm,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkPrompt:
    """A single benchmark prompt with optional ground truth PDE."""

    id: str
    nl_description: str
    physics_family: str
    complexity: str  # "simple" | "medium" | "complex"
    expected_kernels: list[str]
    notes: str
    pde_gt: PDERepresentation | None = None


def load_benchmark(
    prompts_dir: Path,
    ground_truth_dir: Path,
) -> list[BenchmarkPrompt]:
    """Load all benchmark prompts from JSON files.

    For each prompt, looks for a matching ground truth file (same stem).
    If found, deserializes it into a PDERepresentation.
    """
    prompts: list[BenchmarkPrompt] = []

    for prompt_file in sorted(prompts_dir.glob("*.json")):
        data = json.loads(prompt_file.read_text())
        gt_file = ground_truth_dir / f"{prompt_file.stem}.json"
        pde_gt = _load_ground_truth(gt_file) if gt_file.exists() else None

        prompts.append(BenchmarkPrompt(
            id=data["id"],
            nl_description=data["nl_description"],
            physics_family=data["physics_family"],
            complexity=data["complexity"],
            expected_kernels=data.get("expected_kernels", []),
            notes=data.get("notes", ""),
            pde_gt=pde_gt,
        ))

    return prompts


def _load_ground_truth(path: Path) -> PDERepresentation:
    """Deserialize a ground truth JSON into PDERepresentation."""
    data = json.loads(path.read_text())

    terms = [
        PDETerm(
            variable=t["variable"],
            operator=t["operator"],
            coefficient=t.get("coefficient", 1.0),
            coupled_variable=t.get("coupled_variable", None),
            kernel_type=t.get("kernel_type", None),
            severity=t.get("severity", "medium"),
        )
        for t in data.get("terms", [])
    ]

    bcs = [
        BoundaryCondition(
            variable=bc["variable"],
            boundary=bc["boundary"],
            bc_type=bc["bc_type"],
            value=bc.get("value", 0.0),
            moose_bc_class=bc.get("moose_type", None),
            severity=bc.get("severity", "medium"),
        )
        for bc in data.get("boundary_conditions", [])
    ]

    ics = [
        InitialCondition(
            variable=ic["variable"],
            ic_type=ic.get("ic_type", "constant"),
            value=ic.get("value", 0.0),
            severity=ic.get("severity", "medium"),
        )
        for ic in data.get("initial_conditions", [])
    ]

    return PDERepresentation(
        terms=terms,
        boundary_conditions=bcs,
        initial_conditions=ics,
        time_scheme=data.get("time_scheme", "steady"),
        variables=data.get("variables", []),
        dimensions=data.get("domain_dim", data.get("dimensions", 1)),
        unresolved_kernels=[],
        unresolved_coefficients=[],
        warnings=[],
    )
