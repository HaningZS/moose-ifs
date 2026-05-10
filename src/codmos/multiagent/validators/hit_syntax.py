"""L1 HIT syntax validator — sanitization, grammar repair, parse orchestration.

Components:
- HITSanitizer (Task 4): idempotent Unicode / whitespace normalization.
- HITGrammarRepair (Tasks 5-8): deterministic local fixes (brackets,
  assignments, list separators, legacy/new block style).
- HITSyntaxValidator (Tasks 9-11): sanitize → hit_parser.load →
  3-shot repair loop on failure.

This artifact ships the parser and repair logic used by the experiments; the
development-design notes are omitted from the anonymous release.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from codmos.multiagent.validators.hit_parser import HITNode, HITParseError
from codmos.multiagent.validators.hit_parser import load as hit_load


@dataclass
class SyntaxResult:
    """Outcome of an L1 validation attempt.

    - passed=True: final_code parses; tree is the HITNode root.
    - passed=False: error is the exception message; error_location is a
      structured pointer (line/col, 0 if unknown).
    """
    passed: bool
    tree: HITNode | None = None
    final_code: str = ""
    error: str = ""
    error_location: dict[str, Any] | None = None


_CURLY_QUOTES = str.maketrans({"\u201C": '"', "\u201D": '"', "\u2018": "'", "\u2019": "'"})
_ZW_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF]")
_NONASCII_WS_RE = re.compile(r"[\u00A0\u202F\u2000-\u200A\u3000]")


class HITSanitizer:
    """Idempotent Unicode / whitespace normalization. Spec §5.1 rules 1-5."""

    def sanitize(self, code: str) -> str:
        code = unicodedata.normalize("NFKC", code)               # rule 1
        code = code.replace("\r\n", "\n").replace("\r", "\n")    # rule 5 (before WS)
        code = code.translate(_CURLY_QUOTES)                     # rule 2
        code = _ZW_RE.sub("", code)                              # rule 3
        code = _NONASCII_WS_RE.sub(" ", code)                    # rule 4
        return code


_BLOCK_OPEN_LINE_RE = re.compile(r"^\s*\[(?:\./)?[^\]\s]+\]\s*(#.*)?$")
_BLOCK_CLOSE_LINE_RE = re.compile(r"^\s*\[(?:\.\./)?\]\s*(#.*)?$")
_ORPHAN_EQ_RE = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_./\-]*\s*=+\s*$")
_LEGACY_OPEN_RE = re.compile(r"^(?P<indent>\s*)\[\./(?P<name>[^\]]+)\]\s*$")
_LEGACY_CLOSE_RE = re.compile(r"^(?P<indent>\s*)\[\.\./\]\s*$")
_SQ_LIST_RE = re.compile(r"'([^'\n]*)'")


def _norm_list_content(content: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[,;]+", " ", content)).strip()


def _col_inside_quotes(line: str, col: int) -> bool:
    in_d, in_s = False, False
    for ch in line[:col]:
        if ch == '"' and not in_s:
            in_d = not in_d
        elif ch == "'" and not in_d:
            in_s = not in_s
    return in_d or in_s


def _block_balance(code: str) -> int:
    opens = sum(1 for ln in code.splitlines() if _BLOCK_OPEN_LINE_RE.match(ln))
    closes = sum(1 for ln in code.splitlines() if _BLOCK_CLOSE_LINE_RE.match(ln))
    return opens - closes


class HITGrammarRepair:
    """Deterministic local grammar repair (spec §5.1 rules 1-5).

    apply() runs all rules unconditionally; each is idempotent so
    over-application is safe.
    """

    def apply(self, code: str, error: Exception) -> str:
        code = self._unify_block_style(code)
        code = self._balance_brackets(code)
        code = self._drop_orphan_equals(code)
        code = self._normalize_assignments(code)
        code = self._normalize_list_separators(code)
        return code

    def _unify_block_style(self, code: str) -> str:
        out = []
        for line in code.splitlines():
            mo = _LEGACY_OPEN_RE.match(line)
            if mo:
                out.append(f"{mo['indent']}[{mo['name']}]")
                continue
            mc = _LEGACY_CLOSE_RE.match(line)
            if mc:
                out.append(f"{mc['indent']}[]")
                continue
            out.append(line)
        return "\n".join(out) + "\n"

    def _drop_orphan_equals(self, code: str) -> str:
        out = [line for line in code.splitlines() if not _ORPHAN_EQ_RE.match(line)]
        return "\n".join(out) + "\n"

    def _normalize_assignments(self, code: str) -> str:
        out = []
        for line in code.splitlines():
            if _BLOCK_OPEN_LINE_RE.match(line) or _BLOCK_CLOSE_LINE_RE.match(line):
                out.append(line)
                continue

            eq_pos = -1
            for i, ch in enumerate(line):
                if ch == "=" and not _col_inside_quotes(line, i):
                    eq_pos = i
                    break
            if eq_pos < 0:
                out.append(line)
                continue

            left = line[:eq_pos].rstrip()
            j = eq_pos
            while j < len(line) and line[j] == "=":
                j += 1
            right = line[j:].lstrip()

            stripped_left = left.lstrip()
            indent = left[: len(left) - len(stripped_left)]

            if right == "":
                continue  # would be orphan after split — drop
            out.append(f"{indent}{stripped_left} = {right}")
        return "\n".join(out) + "\n"

    def _normalize_list_separators(self, code: str) -> str:
        def per_line(line: str) -> str:
            eq_pos = -1
            for i, ch in enumerate(line):
                if ch == "=" and not _col_inside_quotes(line, i):
                    eq_pos = i
                    break
            if eq_pos < 0:
                return line
            head, tail = line[: eq_pos + 1], line[eq_pos + 1 :]
            tail = _SQ_LIST_RE.sub(lambda m: "'" + _norm_list_content(m.group(1)) + "'", tail)
            return head + tail

        return "\n".join(per_line(line) for line in code.splitlines()) + "\n"

    def _balance_brackets(self, code: str) -> str:
        balance = _block_balance(code)
        lines = code.rstrip("\n").splitlines()
        if balance > 0:
            lines.extend(["[]"] * balance)
        elif balance < 0:
            to_drop = -balance
            i = len(lines) - 1
            while to_drop > 0 and i >= 0:
                if _BLOCK_CLOSE_LINE_RE.match(lines[i]):
                    del lines[i]
                    to_drop -= 1
                i -= 1
        return "\n".join(lines) + "\n"


def _extract_error_location(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, HITParseError):
        return {"raw_message": exc.raw_message, "line": exc.line, "col": exc.col}
    return {"raw_message": str(exc), "line": None, "col": None}


class HITSyntaxValidator:
    """Sanitize → parse → 3-shot repair (Tasks 9-11 build incrementally).

    Task 9: happy path only. Task 10 adds the repair loop. Task 11 adds
    error_location extraction from HITParseError.
    """

    def __init__(self) -> None:
        self.sanitizer = HITSanitizer()
        self.repair = HITGrammarRepair()

    def validate(self, code: str) -> SyntaxResult:
        """Sanitize → parse; on failure run up to 3 repair-and-retry cycles.

        Already-valid input bypasses repair entirely (initial parse succeeds).
        After 3 failed repair attempts, returns passed=False with
        error_location populated from HITParseError (line/col) or None for
        non-HITParseError exceptions.
        """
        clean = self.sanitizer.sanitize(code)
        try:
            tree = hit_load(clean)
            return SyntaxResult(passed=True, tree=tree, final_code=clean)
        except Exception as exc:  # widened from HITParseError
            last_error = exc

        for _ in range(3):
            repaired = self.repair.apply(clean, last_error)
            repaired = self.sanitizer.sanitize(repaired)  # idempotent re-sanitize
            try:
                tree = hit_load(repaired)
                return SyntaxResult(passed=True, tree=tree, final_code=repaired)
            except Exception as exc2:  # widened from HITParseError
                clean = repaired
                last_error = exc2

        return SyntaxResult(
            passed=False,
            final_code=clean,
            error=str(last_error),
            error_location=_extract_error_location(last_error),
        )
