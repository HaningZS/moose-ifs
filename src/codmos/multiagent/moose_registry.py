"""Context-aware MOOSE object registry for L2 validation.

The preferred source is the official MOOSE application syntax dump
(`app-opt --json`), because it includes inherited parameters and the exact
parent syntax path for each object.  A source-tree scanner is provided as a
fallback for local development when only a MOOSE checkout is available.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

START_JSON_MARKER = "**START JSON DATA**"
END_JSON_MARKER = "**END JSON DATA**"


@dataclass(frozen=True)
class MooseParamSpec:
    """Single input parameter definition from a MOOSE syntax registry."""

    name: str
    required: bool = False
    default: str | None = None
    cpp_type: str | None = None
    basic_type: str | None = None
    options: str | None = None
    deprecated: bool = False


@dataclass
class MooseObjectSpec:
    """Registry entry for one concrete MOOSE object type."""

    type_name: str
    contexts: set[str] = field(default_factory=set)
    params: dict[str, MooseParamSpec] = field(default_factory=dict)
    labels: set[str] = field(default_factory=set)
    moose_bases: set[str] = field(default_factory=set)
    source_files: set[str] = field(default_factory=set)
    complete_params: bool = False

    def merge_params(self, params: dict[str, MooseParamSpec], *, complete: bool) -> None:
        self.params.update(params)
        self.complete_params = self.complete_params or complete


class MooseRegistry:
    """Query object validity by local HIT context, e.g. ``Kernels/*``."""

    def __init__(self) -> None:
        self._objects: dict[str, MooseObjectSpec] = {}
        self._by_context: dict[str, set[str]] = {}

    def add_object(
        self,
        type_name: str,
        *,
        context: str,
        params: dict[str, MooseParamSpec] | None = None,
        label: str | None = None,
        moose_base: str | None = None,
        source_file: str | None = None,
        complete_params: bool = False,
    ) -> MooseObjectSpec:
        spec = self._objects.get(type_name)
        if spec is None:
            spec = MooseObjectSpec(type_name=type_name)
            self._objects[type_name] = spec

        spec.contexts.add(context)
        self._by_context.setdefault(context, set()).add(type_name)
        if params:
            spec.merge_params(params, complete=complete_params)
        if label:
            spec.labels.add(label)
        if moose_base:
            spec.moose_bases.add(moose_base)
        if source_file:
            spec.source_files.add(source_file)
        return spec

    def get(self, type_name: str) -> MooseObjectSpec | None:
        return self._objects.get(type_name)

    def has_type(self, type_name: str) -> bool:
        return type_name in self._objects

    def is_valid_in_context(self, type_name: str, context: str) -> bool:
        spec = self.get(type_name)
        return spec is not None and context in spec.contexts

    def candidates_for_context(self, context: str) -> list[str]:
        return sorted(self._by_context.get(context, set()))

    def all_type_names(self) -> list[str]:
        return sorted(self._objects)

    @classmethod
    def from_moose_json_text(cls, text: str) -> MooseRegistry:
        """Build from raw MOOSE ``--json`` output, including warning wrappers."""
        return cls.from_moose_json(_extract_moose_json(text))

    @classmethod
    def from_moose_json_file(cls, path: str | Path) -> MooseRegistry:
        return cls.from_moose_json_text(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def from_moose_json(cls, data: dict[str, Any]) -> MooseRegistry:
        registry = cls()

        def add_json_entry(type_name: str, entry: dict[str, Any]) -> None:
            has_object_metadata = "parameters" in entry and (
                "parent_syntax" in entry or "syntax_path" in entry
            )
            if not has_object_metadata:
                return

            params = _params_from_json(entry.get("parameters") or {})
            context = entry.get("parent_syntax") or _context_from_syntax_path(
                entry.get("syntax_path")
            )
            if not context:
                return

            registry.add_object(
                type_name,
                context=context,
                params=params,
                label=entry.get("label"),
                moose_base=entry.get("moose_base"),
                source_file=entry.get("register_file"),
                complete_params=True,
            )
            for source_file in entry.get("file_info") or {}:
                spec = registry.get(type_name)
                if spec is not None:
                    spec.source_files.add(source_file)

        def visit(node: Any) -> None:
            if not isinstance(node, dict):
                return
            for object_bucket in ("types", "subblock_types", "subblocks"):
                entries = node.get(object_bucket)
                if isinstance(entries, dict):
                    for type_name, entry in entries.items():
                        if not isinstance(entry, dict):
                            continue
                        add_json_entry(type_name, entry)
                        visit(entry)
            for key, value in node.items():
                if key in {"types", "subblock_types", "subblocks"}:
                    continue
                visit(value)

        visit(data.get("blocks", data))
        return registry

    @classmethod
    def from_moose_source(
        cls,
        moose_root: str | Path,
        *,
        app_names: set[str] | None = None,
    ) -> MooseRegistry:
        """Best-effort fallback scanner over a local MOOSE checkout.

        This scanner is conservative: it trusts registered object names and
        source-path-derived contexts, but parameter lists are marked incomplete
        because inherited ``validParams()`` entries are hard to recover from C++
        without running the application syntax dump.
        """
        root = Path(moose_root)
        registry = cls()
        for path in _iter_source_files(root):
            context = _context_from_source_path(path)
            if context is None:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            params = _params_from_source(text)
            for app, type_name in _registered_types_from_source(text):
                if app_names is not None and app not in app_names:
                    continue
                registry.add_object(
                    type_name,
                    context=context,
                    params=params,
                    label=app,
                    source_file=str(path),
                    complete_params=False,
                )
        return registry


def _extract_moose_json(text: str) -> dict[str, Any]:
    start = text.find(START_JSON_MARKER)
    end = text.find(END_JSON_MARKER)
    if start >= 0 and end > start:
        text = text[start + len(START_JSON_MARKER) : end]
    text = text.strip()
    if not text:
        return {}
    return json.loads(text)


def _params_from_json(raw: dict[str, Any]) -> dict[str, MooseParamSpec]:
    params: dict[str, MooseParamSpec] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        param_name = entry.get("name") or name
        params[param_name] = MooseParamSpec(
            name=param_name,
            required=bool(entry.get("required", False)),
            default=str(entry["default"]) if "default" in entry else None,
            cpp_type=entry.get("cpp_type"),
            basic_type=entry.get("basic_type"),
            options=entry.get("options"),
            deprecated=bool(entry.get("deprecated", False)),
        )
    return params


def _context_from_syntax_path(path: str | None) -> str | None:
    if not path:
        return None
    parts = path.split("/")
    if "<type>" in parts:
        return "/".join(parts[: parts.index("<type>")])
    if len(parts) > 1:
        return "/".join(parts[:-1])
    return parts[0]


_REGISTER_RE = re.compile(
    r"registerMooseObject(?:Deprecated|Renamed)?\(\s*\"([^\"]+)\"\s*,\s*([A-Za-z_][A-Za-z0-9_]*)",
)
_REGISTER_ALIASED_RE = re.compile(
    r"registerMooseObjectAliased\(\s*\"([^\"]+)\"\s*,\s*[A-Za-z_][A-Za-z0-9_]*\s*,\s*\"([^\"]+)\"",
)
_PARAM_RE = re.compile(
    r"params\.add(?:Required|RangeChecked|RequiredRangeChecked|CustomType|RequiredCustomType)?"
    r"Param(?:<[^>]+>)?\s*\(\s*\"([^\"]+)\"",
    re.DOTALL,
)
_COUPLED_PARAM_RE = re.compile(
    r"params\.add(?:Required)?CoupledVar\s*\(\s*\"([^\"]+)\"",
    re.DOTALL,
)


def _iter_source_files(root: Path):
    search_roots = [root / "framework" / "src", root / "modules", root / "test" / "src"]
    for search_root in search_roots:
        if not search_root.exists():
            continue
        yield from search_root.rglob("*.C")


def _registered_types_from_source(text: str) -> list[tuple[str, str]]:
    found = [(app, type_name) for app, type_name in _REGISTER_RE.findall(text)]
    found.extend((app, alias) for app, alias in _REGISTER_ALIASED_RE.findall(text))
    return found


def _params_from_source(text: str) -> dict[str, MooseParamSpec]:
    names = set(_PARAM_RE.findall(text)) | set(_COUPLED_PARAM_RE.findall(text))
    return {name: MooseParamSpec(name=name) for name in names}


_SOURCE_CONTEXTS: tuple[tuple[str, str], ...] = (
    ("/src/kernels/", "Kernels/*"),
    ("/src/dgkernels/", "DGKernels/*"),
    ("/src/dirackernels/", "DiracKernels/*"),
    ("/src/interfacekernels/", "InterfaceKernels/*"),
    ("/src/bcs/", "BCs/*"),
    ("/src/fvbcs/", "FVBCs/*"),
    ("/src/linearfvbcs/", "FVBCs/*"),
    ("/src/ics/", "ICs/*"),
    ("/src/fvics/", "ICs/*"),
    ("/src/materials/", "Materials/*"),
    ("/src/functormaterials/", "Materials/*"),
    ("/src/meshgenerators/", "Mesh/*"),
    ("/src/mesh/", "Mesh"),
    ("/src/executioners/", "Executioner"),
    ("/src/timesteppers/", "Executioner/TimeStepper"),
    ("/src/functions/", "Functions/*"),
    ("/src/postprocessors/", "Postprocessors/*"),
    ("/src/vectorpostprocessors/", "VectorPostprocessors/*"),
    ("/src/auxkernels/", "AuxKernels/*"),
    ("/src/userobjects/", "UserObjects/*"),
    ("/src/constraints/", "Constraints/*"),
    ("/src/dampers/", "Dampers/*"),
    ("/src/markers/", "Adaptivity/Markers/*"),
)


def _context_from_source_path(path: Path) -> str | None:
    normalized = "/" + path.as_posix()
    for marker, context in _SOURCE_CONTEXTS:
        if marker in normalized:
            return context
    return None
