"""Module-introspection-based API reference generator.

The roadmap pins the API reference as one of the documentation
deliverables: "Sphinx + autoapi over `domain/`, `core/`, every
public module. Versioned with the codebase." This module is the
**pure-Python introspection core** — given a Python module
object, extract the public symbols (classes / functions /
dataclasses / enums / constants), their docstrings, and their
signatures into a structured `ApiReference` dataclass that the
operator's docs build (Sphinx, MkDocs, plain markdown) consumes.

Picked module-introspection over Sphinx-autoapi for the core
because (a) the operator's docs build pipeline can plug in any
renderer (markdown, ReST, JSON, OpenAPI), (b) the introspection
result is testable + regression-pinnable in a way that Sphinx
output isn't, (c) the existing wave-pattern is "ship the data
structure, defer the rendering" — Sphinx integration is an
operator-side build script, not a Round-4 module.

Pinned semantics:
- **Public-by-default.** Symbols whose names start with `_` are
  skipped unless `include_private=True`. The rule applies to
  modules + symbols + classes' methods.
- **Deterministic ordering.** Symbols within a module are sorted
  alphabetically; modules are sorted by qualified name. Operators
  diffing two reference outputs see real changes, not iteration-
  order noise.
- **`__all__` honoured when present.** A module that defines
  `__all__` exports only the listed symbols; the introspector
  respects this even when `include_private=False` would include
  more.
- **Symbol kind detected via inspection.** Functions, classes,
  dataclasses, enums, constants are all recognised; the engine
  classifies them deterministically so the renderer can group.
- **Render output never includes operator secret values.** The
  introspector reads class/function definitions, not module-
  level mutable state; render-time defenses (secret-name
  denylist) further prevent leakage of `password` / `api_key` /
  `secret` / `token` constants. Mirrors the no-secret patterns
  of Wave 8.D OTLP + Wave 3.B vault + Wave 12.G co-pilot.
"""

from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass, field
from enum import Enum
from types import ModuleType
from typing import Any


class SymbolKind(str, Enum):
    """The category of a documented symbol.

    Pinned string values for JSON / markdown rendering stability.
    """

    CLASS = "class"
    DATACLASS = "dataclass"
    FUNCTION = "function"
    ENUM = "enum"
    CONSTANT = "constant"


# Constant names that look like secrets and should be redacted at
# render time. Mirrors Wave 8.D OTLP redacted-attribute denylist.
_SECRET_NAME_FRAGMENTS: frozenset[str] = frozenset(
    {
        "password",
        "secret",
        "api_key",
        "token",
        "private_key",
        "session_id",
    }
)


def _is_secret_name(name: str) -> bool:
    lower = name.lower()
    return any(fragment in lower for fragment in _SECRET_NAME_FRAGMENTS)


@dataclass(frozen=True)
class ApiSymbol:
    """One documented symbol (class, function, etc.).

    `qualified_name` is the dotted import path
    (e.g., `halal_trader.web.kyc.KYCLevel`); `summary` is the first
    line of the docstring; `description` is the full docstring;
    `signature` is the human-readable parameter list (functions)
    or `dataclass(...)` shape (dataclasses) or `Enum(...)` member
    list (enums) or empty for plain constants.
    """

    qualified_name: str
    short_name: str
    kind: SymbolKind
    summary: str
    description: str
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.qualified_name or not self.qualified_name.strip():
            raise ValueError("qualified_name must be non-empty")
        if not self.short_name or not self.short_name.strip():
            raise ValueError("short_name must be non-empty")


@dataclass(frozen=True)
class ApiModule:
    """One documented module."""

    qualified_name: str
    summary: str
    description: str
    symbols: tuple[ApiSymbol, ...]

    def __post_init__(self) -> None:
        if not self.qualified_name or not self.qualified_name.strip():
            raise ValueError("qualified_name must be non-empty")


@dataclass(frozen=True)
class ApiReference:
    """The full extracted API surface across a set of modules."""

    package_name: str
    modules: tuple[ApiModule, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.package_name or not self.package_name.strip():
            raise ValueError("package_name must be non-empty")


def _split_docstring(docstring: str | None) -> tuple[str, str]:
    """Split a docstring into (summary, description).

    Summary is the first non-empty line; description is the full
    cleaned docstring (or summary if no body).
    """

    if not docstring:
        return ("", "")
    cleaned = inspect.cleandoc(docstring)
    lines = cleaned.split("\n", 1)
    summary = lines[0].strip()
    description = cleaned
    return (summary, description)


def _function_signature(func: Any) -> str:
    """Return a human-readable function signature."""

    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return ""
    return f"{func.__name__}{sig}"


def _dataclass_signature(cls: type) -> str:
    """Return a 'Name(field1: type, field2: type)' signature for a dataclass."""

    fields = dataclasses.fields(cls)
    parts = []
    for f in fields:
        type_str = f.type.__name__ if hasattr(f.type, "__name__") else str(f.type)
        parts.append(f"{f.name}: {type_str}")
    return f"{cls.__name__}({', '.join(parts)})"


def _enum_signature(cls: type) -> str:
    """Return 'Name(MEMBER_A, MEMBER_B, ...)' for an Enum."""

    members = sorted(cls.__members__.keys())
    return f"{cls.__name__}({', '.join(members)})"


def _classify_symbol(name: str, value: Any) -> SymbolKind | None:
    """Classify an attribute into a SymbolKind, or None to skip."""

    if inspect.isclass(value):
        if issubclass(value, Enum):
            return SymbolKind.ENUM
        if dataclasses.is_dataclass(value):
            return SymbolKind.DATACLASS
        return SymbolKind.CLASS
    if inspect.isfunction(value) or inspect.isbuiltin(value):
        return SymbolKind.FUNCTION
    # Skip modules, classmethods, staticmethods, and instances of
    # mutable-state-like containers.
    if inspect.ismodule(value):
        return None
    # Treat anything else as a constant — but only if it's a simple
    # value (not a class instance with side effects).
    if isinstance(value, (int, float, str, bool, tuple, frozenset)):
        return SymbolKind.CONSTANT
    return None


def _is_module_local(name: str, value: Any, module: ModuleType) -> bool:
    """True if the value is defined in this module (vs imported from elsewhere).

    Imported symbols (e.g., a frozenset imported from another module)
    should not be re-documented in this module's reference.
    """

    if inspect.ismodule(value):
        return False
    defined_module = getattr(value, "__module__", None)
    if defined_module is None:
        # Constants like ints/strings have no __module__; assume local
        # when the module-level dict actually contains them.
        return True
    return defined_module == module.__name__


def extract_module_reference(
    module: ModuleType,
    *,
    include_private: bool = False,
) -> ApiModule:
    """Extract a single module's public API into an `ApiModule`.

    Honours the module's `__all__` when present; otherwise filters
    by leading-underscore. Symbols sorted alphabetically by name.
    """

    declared_all: tuple[str, ...] | None = None
    raw_all = getattr(module, "__all__", None)
    if raw_all is not None:
        declared_all = tuple(raw_all)

    summary, description = _split_docstring(module.__doc__)

    symbols: list[ApiSymbol] = []
    for name in sorted(dir(module)):
        if name.startswith("_") and not include_private:
            continue
        if declared_all is not None and name not in declared_all:
            continue

        try:
            value = getattr(module, name)
        except AttributeError:
            continue

        if not _is_module_local(name, value, module):
            continue

        kind = _classify_symbol(name, value)
        if kind is None:
            continue

        sym_summary, sym_description = _split_docstring(getattr(value, "__doc__", None))
        signature = ""
        if kind is SymbolKind.FUNCTION:
            signature = _function_signature(value)
        elif kind is SymbolKind.DATACLASS:
            signature = _dataclass_signature(value)
        elif kind is SymbolKind.ENUM:
            signature = _enum_signature(value)

        symbols.append(
            ApiSymbol(
                qualified_name=f"{module.__name__}.{name}",
                short_name=name,
                kind=kind,
                summary=sym_summary,
                description=sym_description,
                signature=signature,
            )
        )

    return ApiModule(
        qualified_name=module.__name__,
        summary=summary,
        description=description,
        symbols=tuple(symbols),
    )


def extract_api_reference(
    *,
    package_name: str,
    modules: tuple[ModuleType, ...],
    include_private: bool = False,
) -> ApiReference:
    """Extract API reference for a set of modules.

    Returns an `ApiReference` with the modules sorted by qualified
    name. The persistence layer (Sphinx / MkDocs / plain markdown
    build script) renders this structure into the final docs.
    """

    if not package_name or not package_name.strip():
        raise ValueError("package_name must be non-empty")

    extracted = tuple(
        sorted(
            (extract_module_reference(m, include_private=include_private) for m in modules),
            key=lambda m: m.qualified_name,
        )
    )
    return ApiReference(package_name=package_name, modules=extracted)


_KIND_HEADER: dict[SymbolKind, str] = {
    SymbolKind.CLASS: "Classes",
    SymbolKind.DATACLASS: "Dataclasses",
    SymbolKind.FUNCTION: "Functions",
    SymbolKind.ENUM: "Enums",
    SymbolKind.CONSTANT: "Constants",
}


def render_markdown(reference: ApiReference) -> str:
    """Render the reference as Markdown.

    Pinned no-secret-leak: constants whose name matches the
    secret-name denylist render with their value redacted as
    `<redacted>`. Operators audit the source for full values.
    """

    lines: list[str] = [f"# {reference.package_name} API Reference", ""]

    if not reference.modules:
        lines.append("_(no modules)_")
        return "\n".join(lines)

    for module in reference.modules:
        lines.append(f"## `{module.qualified_name}`")
        if module.summary:
            lines.append("")
            lines.append(module.summary)
        lines.append("")

        if not module.symbols:
            lines.append("_(no public symbols)_")
            lines.append("")
            continue

        # Group symbols by kind for readability.
        for kind in SymbolKind:
            kind_symbols = [s for s in module.symbols if s.kind is kind]
            if not kind_symbols:
                continue
            lines.append(f"### {_KIND_HEADER[kind]}")
            lines.append("")
            for sym in kind_symbols:
                lines.append(f"#### `{sym.short_name}`")
                if kind is SymbolKind.CONSTANT and _is_secret_name(sym.short_name):
                    lines.append("```")
                    lines.append(f"{sym.short_name} = <redacted>")
                    lines.append("```")
                elif sym.signature:
                    lines.append("```python")
                    lines.append(sym.signature)
                    lines.append("```")
                if sym.summary:
                    lines.append("")
                    lines.append(sym.summary)
                lines.append("")

    return "\n".join(lines)


__all__ = [
    "ApiModule",
    "ApiReference",
    "ApiSymbol",
    "SymbolKind",
    "extract_api_reference",
    "extract_module_reference",
    "render_markdown",
]
