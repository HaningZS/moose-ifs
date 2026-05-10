# src/codmos/multiagent/pde/extraction.py
"""PDE intent extraction from natural language.

Layer 2 — imports Layer 1 (representation) and Layer 3 (conversion)
plus existing ModelingAgent.

Strategy A (default): ModelingAgent -> PhysicsSpec -> conversion -> PDERepr
Strategy B (fallback): Direct LLM -> PDERepr (not yet implemented)
"""

from __future__ import annotations

import logging

from codmos.multiagent.agents import PhysicsSpecification
from codmos.multiagent.pde.conversion import physicsspec_to_pde
from codmos.multiagent.pde.representation import PDERepresentation

logger = logging.getLogger(__name__)


async def extract_pde(
    description: str,
    strategy: str = "modelingagent",
    llm_backend: str = "claude",
) -> PDERepresentation:
    """Extract PDE intent from a natural language description.

    Args:
        description: Natural language physics problem description.
        strategy: ``"modelingagent"`` (default) uses existing ModelingAgent
            then converts via ``physicsspec_to_pde()``.
            ``"direct"`` is a fallback (not yet implemented).
        llm_backend: LLM backend identifier (for Strategy A, passed to
            ModelingAgent).

    Returns:
        A PDERepresentation of the user's physics intent.

    Raises:
        NotImplementedError: If ``strategy="direct"`` (Strategy B stub).
    """
    if strategy == "modelingagent":
        spec = await _run_modeling_agent(description, llm_backend)
        return physicsspec_to_pde(spec)
    elif strategy == "direct":
        raise NotImplementedError(
            "Strategy B (direct LLM → PDERepresentation) is not yet implemented. "
            "Use strategy='modelingagent' (Strategy A)."
        )
    else:
        raise ValueError(f"Unknown extraction strategy: {strategy!r}")


async def _run_modeling_agent(
    description: str,
    llm_backend: str,
) -> PhysicsSpecification:
    """Run the existing ModelingAgent to produce a PhysicsSpecification.

    This is the integration point with the existing multi-agent system.
    The actual LLM call and prompt construction are handled by
    ModelingAgent.analyze().
    """
    # Import here to avoid circular dependency at module level
    from codmos.multiagent.agents import ModelingAgent

    agent = ModelingAgent()
    return await agent.analyze(description, backend=llm_backend)
