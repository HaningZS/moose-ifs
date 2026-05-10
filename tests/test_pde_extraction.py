# tests/test_pde_extraction.py
"""Tests for PDE extraction from natural language (Strategy A and B)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codmos.multiagent.agents import PhysicsSpecification
from codmos.multiagent.pde.extraction import extract_pde
from codmos.multiagent.pde.representation import PDERepresentation


def _mock_physics_spec() -> PhysicsSpecification:
    return PhysicsSpecification(
        problem_description="Heat conduction in 1D rod",
        spatial_dimensionality="1D",
        modeling_assumptions=["steady-state"],
        physics_modules=["heat_conduction"],
        primary_variables=[{"name": "temperature", "type": "scalar"}],
        governing_equations="nabla . (k nabla T) = 0",
        boundary_conditions=[
            {"name": "left", "type": "Dirichlet", "variable": "temperature", "value": "300"},
        ],
        initial_conditions=[],
        material_properties=[{"name": "thermal_conductivity", "value": 45.0}],
        time_integration={"type": "Steady"},
        numerical_considerations="",
        outputs_and_diagnostics=[],
    )


class TestExtractPDEStrategyA:
    @pytest.mark.asyncio
    async def test_returns_pde_representation(self):
        mock_spec = _mock_physics_spec()
        with patch(
            "codmos.multiagent.pde.extraction._run_modeling_agent",
            new_callable=AsyncMock,
            return_value=mock_spec,
        ):
            result = await extract_pde("Heat conduction in a 1D rod", strategy="modelingagent")
            assert isinstance(result, PDERepresentation)

    @pytest.mark.asyncio
    async def test_has_diffusion_term(self):
        mock_spec = _mock_physics_spec()
        with patch(
            "codmos.multiagent.pde.extraction._run_modeling_agent",
            new_callable=AsyncMock,
            return_value=mock_spec,
        ):
            result = await extract_pde("Heat conduction in a 1D rod", strategy="modelingagent")
            ops = [t.operator for t in result.terms]
            assert "diffusion" in ops

    @pytest.mark.asyncio
    async def test_has_boundary_conditions(self):
        mock_spec = _mock_physics_spec()
        with patch(
            "codmos.multiagent.pde.extraction._run_modeling_agent",
            new_callable=AsyncMock,
            return_value=mock_spec,
        ):
            result = await extract_pde("Heat conduction in a 1D rod", strategy="modelingagent")
            assert len(result.boundary_conditions) == 1


class TestExtractPDEStrategyB:
    @pytest.mark.asyncio
    async def test_raises_not_implemented(self):
        """Strategy B is a stub for now -- should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await extract_pde("Heat conduction in a 1D rod", strategy="direct")
