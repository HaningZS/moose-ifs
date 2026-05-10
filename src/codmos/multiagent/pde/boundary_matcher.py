"""Conservative boundary condition matching.

Layer 1 — no external dependencies beyond representation.py.

Matches reference BCs to candidate BCs using exact and normalized
name comparison only. No physics-context-dependent alias tables
(e.g. "left" != "inlet") to avoid false positive matches.
"""

from __future__ import annotations

import re

from codmos.multiagent.pde.representation import BoundaryCondition

MatchResult = tuple[BoundaryCondition, BoundaryCondition | None, float | None]


def match_boundaries(
    ref_bcs: list[BoundaryCondition],
    cand_bcs: list[BoundaryCondition],
) -> list[MatchResult]:
    """Match reference BCs to candidate BCs by (variable, boundary).

    Returns a list with one entry per ``ref_bcs`` element:
    ``(ref_bc, matched_cand_bc_or_None, confidence)``.

    Confidence: 1.0 for exact match, 0.9 for normalized match,
    ``None`` if unmatched.
    """
    used: set[int] = set()
    results: list[MatchResult] = []

    for ref in ref_bcs:
        match, conf = _find_match(ref, cand_bcs, used)
        results.append((ref, match, conf))

    return results


def _find_match(
    ref: BoundaryCondition,
    candidates: list[BoundaryCondition],
    used: set[int],
) -> tuple[BoundaryCondition | None, float | None]:
    # Pass 1: exact match
    for i, cand in enumerate(candidates):
        if i in used:
            continue
        if cand.variable == ref.variable and cand.boundary == ref.boundary:
            used.add(i)
            return cand, 1.0

    # Pass 2: normalized match
    ref_norm = _normalize(ref.boundary)
    for i, cand in enumerate(candidates):
        if i in used:
            continue
        if cand.variable == ref.variable and _normalize(cand.boundary) == ref_norm:
            used.add(i)
            return cand, 0.9

    return None, None


def _normalize(name: str) -> str:
    """Lowercase, strip quotes, remove underscores, hyphens, and whitespace."""
    return re.sub(r"[\s_\-]", "", name.strip("'\"").lower())
