"""Pure-Python HIT parser for L1 syntax validation.

HIT (Hierarchical Input Text) is the MOOSE input file format. This
parser is a minimal recursive-descent implementation covering the
subset used by the MOOSEnger 7-family benchmark: blocks, key=value
assignments, single-quoted list values, double-quoted strings, bare
values, comments, and nested blocks (both new-style [name]/[] and
legacy-style [./name]/[../]).

OUT OF SCOPE (would raise HITParseError or pass through unchanged):
- !include directives
- ${var} interpolation
- complex expression evaluation
- file-path resolution

The parser provides the stable API used by the artifact validators.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class HITNode:
    """A node in the parsed HIT tree.

    Root has empty `name` and only `children`. Each block has its own
    `name`, scalar `params` (raw string values — consumer converts as
    needed), and nested `children`.
    """

    name: str = ""                              # "" for root
    params: dict[str, str] = field(default_factory=dict)
    children: list[HITNode] = field(default_factory=list)

    def find(self, path: str) -> HITNode | None:
        """Navigate by path. '/Mesh' or '/Kernels/k1' (root-relative if
        leading '/'); 'k1' (relative to self if no leading '/'). Empty
        string or '/' returns self/root."""
        if not path or path == "/":
            return self
        parts = [p for p in path.split("/") if p]
        node = self
        for part in parts:
            match = next((c for c in node.children if c.name == part), None)
            if match is None:
                return None
            node = match
        return node

    def param(self, key: str, default: str | None = None) -> str | None:
        """Get a parameter as raw string. Returns default if absent."""
        return self.params.get(key, default)

    def child_names(self) -> list[str]:
        """Names of immediate children, in source order."""
        return [c.name for c in self.children]


class HITParseError(Exception):
    """Raised on malformed input. Carries 1-based line and col (0 if unknown)."""

    def __init__(self, message: str, line: int = 0, col: int = 0):
        self.raw_message = message
        self.line = line
        self.col = col
        super().__init__(f"line {line}, col {col}: {message}" if line else message)


def load(code: str) -> HITNode:
    """Parse a HIT string into a tree. Raises HITParseError on malformed input.

    Recursive-descent over the tokenizer output. Maintains an open-block
    stack: BLOCK_OPEN pushes a new HITNode; BLOCK_CLOSE pops and attaches
    to its parent; KEY EQ VALUE stores into the top-of-stack node's params.
    """
    tokens = list(_tokenize(code))
    root = HITNode()
    stack: list[HITNode] = [root]

    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]

        if tok.kind == "EOF":
            break

        if tok.kind == "NEWLINE":
            i += 1
            continue

        if tok.kind == "BLOCK_OPEN":
            new_node = HITNode(name=tok.value)
            stack[-1].children.append(new_node)
            stack.append(new_node)
            i += 1
            continue

        if tok.kind == "BLOCK_CLOSE":
            if len(stack) == 1:
                raise HITParseError(
                    "unmatched block close",
                    line=tok.line, col=tok.col,
                )
            stack.pop()
            i += 1
            continue

        if tok.kind == "KEY":
            # Expect KEY EQ VALUE in sequence.
            if i + 2 >= n or tokens[i + 1].kind != "EQ" or tokens[i + 2].kind != "VALUE":
                raise HITParseError(
                    f"malformed assignment near key {tok.value!r}",
                    line=tok.line, col=tok.col,
                )
            stack[-1].params[tok.value] = tokens[i + 2].value
            i += 3
            continue

        # Unknown token kind — defensive.
        raise HITParseError(
            f"unexpected token kind {tok.kind!r} ({tok.value!r})",
            line=tok.line, col=tok.col,
        )

    if len(stack) > 1:
        unclosed = stack[-1].name
        raise HITParseError(
            f"unclosed block: [{unclosed}]",
            line=tokens[-1].line if tokens else 0,
            col=0,
        )

    return root



@dataclass
class _Token:
    """Internal lexer token. kind ∈ {BLOCK_OPEN, BLOCK_CLOSE, KEY, EQ, VALUE, NEWLINE, EOF}."""
    kind: str
    value: str
    line: int   # 1-based
    col: int    # 1-based


# Block headers — match either new-style [name] or legacy [./name].
_BLOCK_OPEN_RE = re.compile(r"\[(?:\./)?([^\]\s]+)\]")
# Block closers — [] or [../]; nothing else may sit on the line content.
_BLOCK_CLOSE_RE = re.compile(r"\[(?:\.\./)?\]")
# Identifier (key, bare value): letters, digits, dot, slash, underscore, hyphen.
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./\-]*")


def _tokenize(code: str):
    """Yield _Token objects with 1-based line/col tracking.

    Handles: block open/close, key=value assignments, quoted strings
    ('single' and "double"), bare values, # comments, blank lines.
    Raises HITParseError on unterminated quotes.
    """
    line_no = 0
    for raw_line in code.splitlines():
        line_no += 1
        i = 0
        n = len(raw_line)

        # Skip leading whitespace for the first non-WS classification.
        while i < n and raw_line[i] in " \t":
            i += 1

        # Blank line or pure-comment line → emit only NEWLINE.
        if i >= n or raw_line[i] == "#":
            yield _Token("NEWLINE", "\n", line_no, i + 1)
            continue

        # Block close BEFORE block open (because [] is a prefix-match concern).
        m_close = _BLOCK_CLOSE_RE.match(raw_line, i)
        if m_close and _is_only_whitespace_or_comment_after(raw_line, m_close.end()):
            yield _Token("BLOCK_CLOSE", m_close.group(0), line_no, i + 1)
            yield _Token("NEWLINE", "\n", line_no, n + 1)
            continue

        # Block open.
        m_open = _BLOCK_OPEN_RE.match(raw_line, i)
        if m_open and _is_only_whitespace_or_comment_after(raw_line, m_open.end()):
            yield _Token("BLOCK_OPEN", m_open.group(1), line_no, i + 1)
            yield _Token("NEWLINE", "\n", line_no, n + 1)
            continue

        # Assignment line: KEY EQ VALUE.
        m_key = _IDENT_RE.match(raw_line, i)
        if not m_key:
            raise HITParseError(
                f"unexpected token: {raw_line[i:i+20]!r}",
                line=line_no, col=i + 1,
            )
        yield _Token("KEY", m_key.group(0), line_no, i + 1)
        i = m_key.end()

        while i < n and raw_line[i] in " \t":
            i += 1
        if i >= n or raw_line[i] != "=":
            raise HITParseError(
                f"expected '=' after key {m_key.group(0)!r}",
                line=line_no, col=i + 1,
            )
        yield _Token("EQ", "=", line_no, i + 1)
        i += 1

        while i < n and raw_line[i] in " \t":
            i += 1
        if i >= n or raw_line[i] == "#":
            raise HITParseError(
                f"missing value after '=' for key {m_key.group(0)!r}",
                line=line_no, col=i + 1,
            )

        # Value: quoted string, single-quoted list, or bare token sequence
        # (until comment-# or end-of-line). Whitespace inside quotes is preserved.
        val_start_col = i + 1
        if raw_line[i] in ("'", '"'):
            quote = raw_line[i]
            j = i + 1
            while j < n and raw_line[j] != quote:
                j += 1
            if j >= n:
                raise HITParseError(
                    f"unterminated {quote} quote",
                    line=line_no, col=i + 1,
                )
            value = raw_line[i:j + 1]
            i = j + 1
        else:
            # Bare value: read until inline comment (#) or end-of-line.
            # Per HIT, bare values are typically single tokens but we accept
            # whitespace-separated continuations on the same line (rare).
            j = i
            while j < n:
                if raw_line[j] == "#":
                    break
                j += 1
            value = raw_line[i:j].rstrip()
            i = j

        yield _Token("VALUE", value, line_no, val_start_col)
        yield _Token("NEWLINE", "\n", line_no, n + 1)

    yield _Token("EOF", "", line_no + 1, 1)


def _is_only_whitespace_or_comment_after(line: str, start: int) -> bool:
    """True if everything after `start` in `line` is whitespace or a comment."""
    rest = line[start:]
    stripped = rest.lstrip()
    return stripped == "" or stripped.startswith("#")
