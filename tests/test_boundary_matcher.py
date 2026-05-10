# tests/test_boundary_matcher.py
"""Tests for conservative boundary condition matching."""

from codmos.multiagent.pde.boundary_matcher import match_boundaries
from codmos.multiagent.pde.representation import BoundaryCondition


def _bc(variable: str, boundary: str, bc_type: str = "Dirichlet") -> BoundaryCondition:
    return BoundaryCondition(variable, boundary, bc_type, 300.0, None, "high")


class TestMatchBoundaries:
    def test_exact_match(self):
        ref = [_bc("T", "left"), _bc("T", "right")]
        cand = [_bc("T", "left"), _bc("T", "right")]
        results = match_boundaries(ref, cand)
        assert len(results) == 2
        assert all(r[1] is not None for r in results)
        assert all(r[2] == 1.0 for r in results)

    def test_normalized_match_case(self):
        ref = [_bc("T", "Left")]
        cand = [_bc("T", "left")]
        results = match_boundaries(ref, cand)
        assert results[0][1] is not None
        assert results[0][2] == 0.9

    def test_normalized_match_underscores(self):
        ref = [_bc("T", "top_surface")]
        cand = [_bc("T", "topsurface")]
        results = match_boundaries(ref, cand)
        assert results[0][1] is not None
        assert results[0][2] == 0.9

    def test_normalized_match_hyphens(self):
        ref = [_bc("T", "left-wall")]
        cand = [_bc("T", "leftwall")]
        results = match_boundaries(ref, cand)
        assert results[0][1] is not None

    def test_unmatched_bc(self):
        ref = [_bc("T", "left")]
        cand = [_bc("T", "bottom")]
        results = match_boundaries(ref, cand)
        assert results[0][1] is None
        assert results[0][2] is None

    def test_no_alias_matching(self):
        """left and inlet should NOT match — context-dependent aliases are unsafe."""
        ref = [_bc("T", "left")]
        cand = [_bc("T", "inlet")]
        results = match_boundaries(ref, cand)
        assert results[0][1] is None

    def test_different_variables_dont_match(self):
        ref = [_bc("T", "left")]
        cand = [_bc("u", "left")]
        results = match_boundaries(ref, cand)
        assert results[0][1] is None

    def test_empty_ref(self):
        results = match_boundaries([], [_bc("T", "left")])
        assert results == []

    def test_empty_cand(self):
        results = match_boundaries([_bc("T", "left")], [])
        assert len(results) == 1
        assert results[0][1] is None

    def test_multiple_variables(self):
        ref = [_bc("T", "left"), _bc("u", "left")]
        cand = [_bc("T", "left"), _bc("u", "left")]
        results = match_boundaries(ref, cand)
        assert len(results) == 2
        assert all(r[1] is not None for r in results)

    def test_candidate_not_reused(self):
        """Each candidate BC can only match one reference BC."""
        ref = [_bc("T", "left"), _bc("T", "left")]
        cand = [_bc("T", "left")]
        results = match_boundaries(ref, cand)
        matched = [r for r in results if r[1] is not None]
        assert len(matched) == 1
