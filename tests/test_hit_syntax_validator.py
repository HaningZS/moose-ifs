"""Tests for the L1 HIT syntax validation layer with a pure-Python parser."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "hit_samples"


class TestPackageImport:
    """Public API surface is importable from the package root."""

    def test_public_api_imports(self):
        from codmos.multiagent.validators import (
            HITGrammarRepair,
            HITNode,
            HITParseError,
            HITSanitizer,
            HITSyntaxValidator,
            SyntaxResult,
            load,
        )
        assert HITGrammarRepair is not None
        assert HITNode is not None
        assert HITParseError is not None
        assert HITSanitizer is not None
        assert HITSyntaxValidator is not None
        assert SyntaxResult is not None
        assert load is not None


class TestHITNode:
    def test_root_is_empty_name(self):
        from codmos.multiagent.validators import HITNode
        root = HITNode()
        assert root.name == ""
        assert root.params == {}
        assert root.children == []

    def test_find_returns_self_for_empty_or_slash(self):
        from codmos.multiagent.validators import HITNode
        root = HITNode()
        assert root.find("") is root
        assert root.find("/") is root

    def test_find_navigates_root_relative_path(self):
        from codmos.multiagent.validators import HITNode
        leaf = HITNode(name="k1", params={"type": "Diffusion"})
        kernels = HITNode(name="Kernels", children=[leaf])
        root = HITNode(children=[kernels])
        assert root.find("/Kernels") is kernels
        assert root.find("/Kernels/k1") is leaf

    def test_find_returns_none_for_missing_path(self):
        from codmos.multiagent.validators import HITNode
        root = HITNode(children=[HITNode(name="Mesh")])
        assert root.find("/Variables") is None
        assert root.find("/Mesh/missing") is None

    def test_param_returns_value_or_default(self):
        from codmos.multiagent.validators import HITNode
        node = HITNode(name="Mesh", params={"type": "GeneratedMesh", "dim": "2"})
        assert node.param("type") == "GeneratedMesh"
        assert node.param("dim") == "2"
        assert node.param("nx") is None
        assert node.param("nx", default="10") == "10"

    def test_child_names_preserves_source_order(self):
        from codmos.multiagent.validators import HITNode
        root = HITNode(children=[
            HITNode(name="Mesh"),
            HITNode(name="Variables"),
            HITNode(name="Kernels"),
        ])
        assert root.child_names() == ["Mesh", "Variables", "Kernels"]


class TestHITParseError:
    def test_carries_line_col_and_raw_message(self):
        from codmos.multiagent.validators import HITParseError
        err = HITParseError("unexpected token", line=3, col=5)
        assert err.line == 3
        assert err.col == 5
        assert err.raw_message == "unexpected token"
        assert "line 3" in str(err)

    def test_default_line_col_zero(self):
        from codmos.multiagent.validators import HITParseError
        err = HITParseError("generic")
        assert err.line == 0
        assert err.col == 0


class TestHITTokenizer:
    def setup_method(self):
        from codmos.multiagent.validators.hit_parser import _tokenize
        self._tokenize = _tokenize

    def test_empty_input_yields_eof_only(self):
        tokens = list(self._tokenize(""))
        assert len(tokens) == 1
        assert tokens[0].kind == "EOF"

    def test_simple_block_open_close(self):
        tokens = list(self._tokenize("[Mesh]\n[]\n"))
        kinds = [t.kind for t in tokens]
        assert kinds == ["BLOCK_OPEN", "NEWLINE", "BLOCK_CLOSE", "NEWLINE", "EOF"]
        assert tokens[0].value == "Mesh"

    def test_legacy_block_style_tokenized(self):
        tokens = list(self._tokenize("[./k1]\n[../]\n"))
        kinds = [t.kind for t in tokens]
        assert kinds == ["BLOCK_OPEN", "NEWLINE", "BLOCK_CLOSE", "NEWLINE", "EOF"]
        assert tokens[0].value == "k1"

    def test_simple_assignment(self):
        tokens = list(self._tokenize("type = GeneratedMesh\n"))
        kinds = [t.kind for t in tokens]
        assert kinds == ["KEY", "EQ", "VALUE", "NEWLINE", "EOF"]
        assert tokens[0].value == "type"
        assert tokens[2].value == "GeneratedMesh"

    def test_quoted_string_value(self):
        tokens = list(self._tokenize('name = "hello world"\n'))
        kinds = [t.kind for t in tokens]
        assert kinds == ["KEY", "EQ", "VALUE", "NEWLINE", "EOF"]
        assert tokens[2].value == '"hello world"'

    def test_single_quoted_list_value(self):
        tokens = list(self._tokenize("active = 'foo bar baz'\n"))
        assert tokens[2].kind == "VALUE"
        assert tokens[2].value == "'foo bar baz'"

    def test_comment_silently_consumed_but_newline_emitted(self):
        tokens = list(self._tokenize("# this is a comment\ntype = X\n"))
        kinds = [t.kind for t in tokens]
        # Comment line: only NEWLINE survives; then KEY EQ VALUE NEWLINE EOF
        assert kinds == ["NEWLINE", "KEY", "EQ", "VALUE", "NEWLINE", "EOF"]

    def test_inline_comment_after_assignment(self):
        tokens = list(self._tokenize("type = X  # comment\n"))
        kinds = [t.kind for t in tokens]
        assert kinds == ["KEY", "EQ", "VALUE", "NEWLINE", "EOF"]
        assert tokens[2].value == "X"

    def test_line_col_tracking(self):
        tokens = list(self._tokenize("[Mesh]\n  type = X\n[]\n"))
        # First [Mesh] at line 1
        assert tokens[0].line == 1
        # `type` at line 2, col 3 (after 2 spaces)
        type_tok = next(t for t in tokens if t.kind == "KEY" and t.value == "type")
        assert type_tok.line == 2
        assert type_tok.col == 3

    def test_blank_lines_yield_only_newlines(self):
        tokens = list(self._tokenize("\n\n[Mesh]\n[]\n"))
        kinds = [t.kind for t in tokens]
        assert kinds.count("NEWLINE") == 4  # 2 blank + 2 after blocks
        assert kinds.count("BLOCK_OPEN") == 1
        assert kinds.count("BLOCK_CLOSE") == 1

    def test_unclosed_quote_raises(self):
        from codmos.multiagent.validators import HITParseError
        with pytest.raises(HITParseError) as exc_info:
            list(self._tokenize('name = "unterminated\n'))
        assert exc_info.value.line >= 1


class TestHITParser:
    def setup_method(self):
        from codmos.multiagent.validators import load
        self.load = load

    def test_empty_input_yields_empty_root(self):
        root = self.load("")
        assert root.name == ""
        assert root.children == []
        assert root.params == {}

    def test_single_block_no_params(self):
        root = self.load("[Mesh]\n[]\n")
        assert root.child_names() == ["Mesh"]
        assert root.find("/Mesh").params == {}

    def test_block_with_params(self):
        code = "[Mesh]\n  type = GeneratedMesh\n  dim = 2\n[]\n"
        root = self.load(code)
        mesh = root.find("/Mesh")
        assert mesh is not None
        assert mesh.param("type") == "GeneratedMesh"
        assert mesh.param("dim") == "2"

    def test_nested_blocks(self):
        code = (
            "[Variables]\n"
            "  [T]\n"
            "    order = FIRST\n"
            "  []\n"
            "[]\n"
        )
        root = self.load(code)
        t_var = root.find("/Variables/T")
        assert t_var is not None
        assert t_var.param("order") == "FIRST"

    def test_legacy_style_blocks_parse(self):
        code = (
            "[Kernels]\n"
            "  [./k1]\n"
            "    type = Diffusion\n"
            "  [../]\n"
            "[]\n"
        )
        root = self.load(code)
        k1 = root.find("/Kernels/k1")
        assert k1 is not None
        assert k1.param("type") == "Diffusion"

    def test_top_level_assignment_attaches_to_root(self):
        # HIT allows top-level params (e.g., [GlobalParams] body) — minimum
        # behavior is they attach to the root node.
        root = self.load("global_param = X\n[Mesh]\n[]\n")
        assert root.param("global_param") == "X"

    def test_mixed_quoted_and_list_values(self):
        code = (
            "[Postprocessors]\n"
            '  msg = "hello world"\n'
            "  active = 'p1 p2 p3'\n"
            "[]\n"
        )
        root = self.load(code)
        pp = root.find("/Postprocessors")
        assert pp.param("msg") == '"hello world"'
        assert pp.param("active") == "'p1 p2 p3'"

    def test_unclosed_block_raises(self):
        from codmos.multiagent.validators import HITParseError
        with pytest.raises(HITParseError):
            self.load("[Mesh]\n  type = X\n")  # no [] closer

    def test_extra_close_raises(self):
        from codmos.multiagent.validators import HITParseError
        with pytest.raises(HITParseError):
            self.load("[]\n")  # close without matching open

    def test_three_levels_of_nesting(self):
        code = (
            "[A]\n"
            "  [B]\n"
            "    [C]\n"
            "      x = 1\n"
            "    []\n"
            "  []\n"
            "[]\n"
        )
        root = self.load(code)
        assert root.find("/A/B/C").param("x") == "1"

    def test_two_sibling_blocks_at_root(self):
        code = "[Mesh]\n[]\n[Variables]\n[]\n"
        root = self.load(code)
        assert root.child_names() == ["Mesh", "Variables"]

    def test_load_is_callable_via_module_attr(self):
        # Verify it's also reachable as hit_parser.load.
        from codmos.multiagent.validators import hit_parser
        assert hit_parser.load("") is not None


class TestHITSanitizer:
    def setup_method(self):
        from codmos.multiagent.validators import HITSanitizer
        self.sanitizer = HITSanitizer()

    def test_nfkc_fullwidth_to_halfwidth(self):
        assert self.sanitizer.sanitize("ｔｙｐｅ = Ｍｅｓｈ") == "type = Mesh"

    def test_curly_double_quotes_to_ascii(self):
        assert self.sanitizer.sanitize('name = \u201cfoo\u201d') == 'name = "foo"'

    def test_curly_single_quotes_to_ascii(self):
        assert self.sanitizer.sanitize("name = \u2018bar\u2019") == "name = 'bar'"

    def test_zero_width_chars_removed(self):
        raw = "type\u200B = \u200CMesh\u200D\uFEFF"
        assert self.sanitizer.sanitize(raw) == "type = Mesh"

    def test_non_ascii_whitespace_to_space(self):
        raw = "key\u00A0=\u3000value"
        assert self.sanitizer.sanitize(raw) == "key = value"

    def test_line_endings_normalized(self):
        assert self.sanitizer.sanitize("a\r\nb\r\nc") == "a\nb\nc"

    def test_idempotent(self):
        messy = "ｔｙｐｅ\u00A0=\u3000\u201cMesh\u201d\r\n"
        once = self.sanitizer.sanitize(messy)
        assert self.sanitizer.sanitize(once) == once

    def test_composite_real_world_input(self):
        raw = "\uFEFF[Mesh]\r\n  type\u00A0＝\u3000\u201cGeneratedMesh\u201d\r\n[]\r\n"
        expected = '[Mesh]\n  type = "GeneratedMesh"\n[]\n'
        assert self.sanitizer.sanitize(raw) == expected

    def test_ascii_only_unchanged(self):
        clean = "[Mesh]\n  type = GeneratedMesh\n[]\n"
        assert self.sanitizer.sanitize(clean) == clean


class TestHITGrammarRepair_Brackets:
    def setup_method(self):
        from codmos.multiagent.validators import HITGrammarRepair
        self.repair = HITGrammarRepair()

    def test_unclosed_top_level_block(self):
        raw = "[Mesh]\n  type = GeneratedMesh\n"
        result = self.repair.apply(raw, Exception("unclosed"))
        assert result.rstrip().endswith("[]")
        assert "type = GeneratedMesh" in result

    def test_closed_block_unchanged(self):
        raw = "[Mesh]\n  type = GeneratedMesh\n[]\n"
        assert self.repair.apply(raw, Exception()) == raw

    def test_two_unclosed_blocks_get_two_closers(self):
        raw = "[Mesh]\n  type = X\n[Variables]\n  [u]\n"
        result = self.repair.apply(raw, Exception())
        assert result.count("\n[]") >= 3  # [Mesh], [Variables], [u]

    def test_extra_closer_balanced(self):
        raw = "[Mesh]\n  type = X\n[]\n[]\n"
        result = self.repair.apply(raw, Exception())
        opens = sum(1 for l in result.splitlines() if l.strip().startswith("[") and not l.strip().startswith("[]") and not l.strip().startswith("[../]"))
        closes = sum(1 for l in result.splitlines() if l.strip() in ("[]", "[../]"))
        assert opens == closes


class TestHITGrammarRepair_Assignments:
    def setup_method(self):
        from codmos.multiagent.validators import HITGrammarRepair
        self.repair = HITGrammarRepair()

    def test_double_equals_collapsed(self):
        assert "type = GeneratedMesh" in self.repair.apply("  type==GeneratedMesh\n", Exception())

    def test_no_space_around_equals(self):
        assert "type = GeneratedMesh" in self.repair.apply("  type=GeneratedMesh\n", Exception())

    def test_space_only_before_equals(self):
        assert "type = GeneratedMesh" in self.repair.apply("  type =GeneratedMesh\n", Exception())

    def test_space_only_after_equals(self):
        assert "type = GeneratedMesh" in self.repair.apply("  type= GeneratedMesh\n", Exception())

    def test_orphan_equals_line_removed(self):
        raw = "[Mesh]\n  type =\n  nx = 10\n[]\n"
        result = self.repair.apply(raw, Exception())
        assert "nx = 10" in result
        assert not any(l.strip() == "type =" for l in result.splitlines())

    def test_normal_assignment_unchanged(self):
        assert "type = GeneratedMesh" in self.repair.apply("  type = GeneratedMesh\n", Exception())

    def test_string_with_equals_preserved(self):
        raw = '  expression = "a == b"\n'
        assert 'expression = "a == b"' in self.repair.apply(raw, Exception())


class TestHITGrammarRepair_ListSeparators:
    def setup_method(self):
        from codmos.multiagent.validators import HITGrammarRepair
        self.repair = HITGrammarRepair()

    def test_comma_to_space(self):
        assert "active = 'foo bar baz'" in self.repair.apply("  active = 'foo, bar, baz'\n", Exception())

    def test_semicolon_to_space(self):
        assert "variables = 'u v w'" in self.repair.apply("  variables = 'u; v; w'\n", Exception())

    def test_mixed_separators_collapsed(self):
        assert "names = 'a b c d'" in self.repair.apply("  names = 'a, b; c  d'\n", Exception())

    def test_double_quoted_string_unchanged(self):
        raw = '  message = "hello, world"\n'
        assert 'message = "hello, world"' in self.repair.apply(raw, Exception())

    def test_empty_list_unchanged(self):
        assert "active = ''" in self.repair.apply("  active = ''\n", Exception())


class TestHITGrammarRepair_BlockStyle:
    def setup_method(self):
        from codmos.multiagent.validators import HITGrammarRepair
        self.repair = HITGrammarRepair()

    def test_legacy_open_unified(self):
        result = self.repair.apply("[Kernels]\n  [./k1]\n    type = X\n  [../]\n[]\n", Exception())
        assert "[./k1]" not in result
        assert "[k1]" in result

    def test_legacy_close_unified(self):
        result = self.repair.apply("[Kernels]\n  [./k1]\n    type = X\n  [../]\n[]\n", Exception())
        assert "[../]" not in result

    def test_mixed_styles_normalized(self):
        code = (
            "[Kernels]\n"
            "  [./k1]\n    type = Diffusion\n  [../]\n"
            "  [k2]\n    type = TimeDerivative\n  []\n"
            "[]\n"
        )
        result = self.repair.apply(code, Exception())
        assert "[./k1]" not in result
        assert "[../]" not in result

    def test_new_style_unchanged(self):
        raw = "[Kernels]\n  [k1]\n    type = X\n  []\n[]\n"
        result = self.repair.apply(raw, Exception())
        assert "[k1]" in result
        assert "type = X" in result


class TestHITSyntaxValidator_RepairLoop:
    def setup_method(self):
        from codmos.multiagent.validators import HITSyntaxValidator
        self.v = HITSyntaxValidator()

    def test_input_needing_repair_succeeds(self):
        code = (FIXTURES / "needs_repair.i").read_text()
        result = self.v.validate(code)
        assert result.passed is True, f"Expected success after repair: {result.error}"
        assert "[./k1]" not in result.final_code
        assert "[../]" not in result.final_code

    def test_unrepairable_input_fails(self):
        result = self.v.validate("]]] ===nonsense=== [[[\n")
        assert result.passed is False
        assert result.error != ""

    def test_already_valid_does_not_invoke_repair(self, monkeypatch):
        from codmos.multiagent.validators import hit_syntax
        calls = {"n": 0}
        orig = hit_syntax.HITGrammarRepair.apply

        def spy(self, code, error):
            calls["n"] += 1
            return orig(self, code, error)

        monkeypatch.setattr(hit_syntax.HITGrammarRepair, "apply", spy)
        result = self.v.validate("[Mesh]\n  type = X\n[]\n")
        assert result.passed is True
        assert calls["n"] == 0

    def test_repair_attempts_capped_at_3(self, monkeypatch):
        from codmos.multiagent.validators import hit_syntax
        calls = {"n": 0}

        def always_broken(self, code, error):
            calls["n"] += 1
            return "still ]]] broken"

        monkeypatch.setattr(hit_syntax.HITGrammarRepair, "apply", always_broken)
        result = self.v.validate("]]] broken")
        assert result.passed is False
        assert calls["n"] == 3


class TestHITSyntaxValidator_HappyPath:
    def setup_method(self):
        from codmos.multiagent.validators import HITSyntaxValidator
        self.v = HITSyntaxValidator()

    def test_valid_input_passes(self):
        code = (FIXTURES / "valid_simple.i").read_text()
        result = self.v.validate(code)
        assert result.passed is True
        assert result.tree is not None
        assert result.error == ""
        # tree is the actual HITNode root.
        assert "Mesh" in result.tree.child_names()

    def test_final_code_is_sanitized(self):
        raw = "\uFEFF[Mesh]\r\n  type\u00A0=\u3000GeneratedMesh\r\n[]\r\n"
        result = self.v.validate(raw)
        assert result.passed is True
        for bad in ("\r", "\u00A0", "\uFEFF"):
            assert bad not in result.final_code

    def test_returns_syntax_result_dataclass(self):
        from codmos.multiagent.validators import SyntaxResult
        result = self.v.validate("[Mesh]\n  type = X\n[]\n")
        assert isinstance(result, SyntaxResult)


class TestHITSyntaxValidator_ErrorLocation:
    def setup_method(self):
        from codmos.multiagent.validators import HITSyntaxValidator
        self.v = HITSyntaxValidator()

    def test_irrecoverable_sets_error_location(self):
        code = (FIXTURES / "irrecoverable.i").read_text()
        result = self.v.validate(code)
        assert result.passed is False
        assert result.error_location is not None
        assert "raw_message" in result.error_location

    def test_error_location_carries_line_col_from_HITParseError(self):
        # Force a parse error at a known line.
        # `=` with no key on line 1 → tokenizer raises with line 1.
        result = self.v.validate("= no_key\n")
        assert result.passed is False
        assert result.error_location is not None
        assert result.error_location.get("line") == 1
        assert result.error_location.get("col", 0) >= 1

    def test_non_HITParseError_falls_through_with_None_line(self, monkeypatch):
        # If hit_load somehow raises a different Exception, line is None.
        from codmos.multiagent.validators import hit_syntax

        def bad_load(code):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(hit_syntax, "hit_load", bad_load)
        result = self.v.validate("[Mesh]\n[]\n")
        assert result.passed is False
        assert result.error_location is not None
        assert result.error_location.get("line") is None


class TestHITSyntaxValidator_Integration:
    """End-to-end integration over crafted fixtures.

    Note on `mixed_block_style.i`: it omits the outer closing `[]` for
    `[Kernels]`. `HITSyntaxValidator.validate()` only invokes the
    legacy-to-new style canonicalizer (`_unify_block_style`) via the repair
    loop, which fires only when the initial parse fails.
    A fully-valid mixed-style file would pass through with `final_code`
    retaining `[./k1]`/`[../]` verbatim, breaking
    `test_mixed_block_style_canonicalized`. The unbalanced brackets
    force an initial parse error so canonicalization is exercised.
    """

    def setup_method(self):
        from codmos.multiagent.validators import HITSyntaxValidator
        self.v = HITSyntaxValidator()

    @pytest.mark.parametrize("fixture", [
        "valid_simple.i",
        "mixed_block_style.i",
        "needs_repair.i",
        "unicode_heavy.i",
    ])
    def test_recoverable_fixtures_pass(self, fixture):
        code = (FIXTURES / fixture).read_text()
        result = self.v.validate(code)
        assert result.passed is True, f"{fixture}: {result.error}"
        assert result.tree is not None

    def test_irrecoverable_fails_with_location(self):
        code = (FIXTURES / "irrecoverable.i").read_text()
        result = self.v.validate(code)
        assert result.passed is False
        assert result.error_location is not None

    def test_mixed_block_style_canonicalized(self):
        code = (FIXTURES / "mixed_block_style.i").read_text()
        result = self.v.validate(code)
        assert result.passed is True
        assert "[./k1]" not in result.final_code
        assert "[../]" not in result.final_code

    def test_unicode_heavy_fully_sanitized(self):
        code = (FIXTURES / "unicode_heavy.i").read_text()
        result = self.v.validate(code)
        assert result.passed is True
        for bad in ("\u00A0", "\u3000", "\uFEFF", "\u200B", "\u201C", "\u201D", "\r"):
            assert bad not in result.final_code

    def test_tree_navigable_after_validate(self):
        code = (FIXTURES / "valid_simple.i").read_text()
        result = self.v.validate(code)
        assert result.tree is not None
        assert result.tree.find("/Mesh").param("type") == "GeneratedMesh"
        assert result.tree.find("/Variables/T").param("order") == "FIRST"
        assert result.tree.find("/Kernels/diff").param("type") == "Diffusion"
        assert result.tree.find("/Executioner").param("type") == "Steady"
