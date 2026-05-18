"""Tests for the API reference auto-generator."""

from __future__ import annotations

import dataclasses
import sys
import types
from enum import Enum

import pytest

from halal_trader.core.api_reference import (
    ApiModule,
    ApiReference,
    ApiSymbol,
    SymbolKind,
    extract_api_reference,
    extract_module_reference,
    render_markdown,
)


def _make_test_module(name: str = "test_mod_intro") -> types.ModuleType:
    """Build a synthetic module with public + private symbols."""

    mod = types.ModuleType(name)
    mod.__doc__ = "Synthetic test module.\n\nLong description spans multiple lines."

    # Public function
    def public_fn(a: int, b: str = "x") -> bool:
        """A public function.

        With a longer description.
        """

        return True

    public_fn.__module__ = name

    # Private function
    def _private_fn(c: int) -> int:
        """A private helper."""

        return c

    _private_fn.__module__ = name

    # Public class
    class PublicClass:
        """A public regular class."""

        pass

    PublicClass.__module__ = name

    # Public dataclass
    @dataclasses.dataclass(frozen=True)
    class PublicDataclass:
        """A public dataclass."""

        x: int
        y: str

    PublicDataclass.__module__ = name

    # Public enum
    class PublicEnum(str, Enum):
        """A public enum."""

        FOO = "foo"
        BAR = "bar"

    PublicEnum.__module__ = name

    # Constants
    PUBLIC_CONST = 42
    _PRIVATE_CONST = 99
    API_KEY = "sk-secret-leaked"  # name matches secret denylist

    mod.public_fn = public_fn
    mod._private_fn = _private_fn
    mod.PublicClass = PublicClass
    mod.PublicDataclass = PublicDataclass
    mod.PublicEnum = PublicEnum
    mod.PUBLIC_CONST = PUBLIC_CONST
    mod._PRIVATE_CONST = _PRIVATE_CONST
    mod.API_KEY = API_KEY

    sys.modules[name] = mod
    return mod


@pytest.fixture
def synthetic_module() -> types.ModuleType:
    mod = _make_test_module()
    yield mod
    sys.modules.pop(mod.__name__, None)


# ---------------------------------------------------------------------------
# Symbol classification
# ---------------------------------------------------------------------------


def test_extracts_public_function(synthetic_module: types.ModuleType) -> None:
    result = extract_module_reference(synthetic_module)
    fns = [s for s in result.symbols if s.kind is SymbolKind.FUNCTION]
    names = {s.short_name for s in fns}
    assert "public_fn" in names


def test_extracts_public_class(synthetic_module: types.ModuleType) -> None:
    result = extract_module_reference(synthetic_module)
    classes = [s for s in result.symbols if s.kind is SymbolKind.CLASS]
    names = {s.short_name for s in classes}
    assert "PublicClass" in names


def test_extracts_public_dataclass(synthetic_module: types.ModuleType) -> None:
    result = extract_module_reference(synthetic_module)
    dcs = [s for s in result.symbols if s.kind is SymbolKind.DATACLASS]
    names = {s.short_name for s in dcs}
    assert "PublicDataclass" in names


def test_extracts_public_enum(synthetic_module: types.ModuleType) -> None:
    result = extract_module_reference(synthetic_module)
    enums = [s for s in result.symbols if s.kind is SymbolKind.ENUM]
    names = {s.short_name for s in enums}
    assert "PublicEnum" in names


def test_extracts_public_constant(synthetic_module: types.ModuleType) -> None:
    result = extract_module_reference(synthetic_module)
    consts = [s for s in result.symbols if s.kind is SymbolKind.CONSTANT]
    names = {s.short_name for s in consts}
    assert "PUBLIC_CONST" in names


# ---------------------------------------------------------------------------
# Public-by-default
# ---------------------------------------------------------------------------


def test_excludes_private_function_by_default(synthetic_module: types.ModuleType) -> None:
    result = extract_module_reference(synthetic_module)
    names = {s.short_name for s in result.symbols}
    assert "_private_fn" not in names


def test_excludes_private_constant_by_default(synthetic_module: types.ModuleType) -> None:
    result = extract_module_reference(synthetic_module)
    names = {s.short_name for s in result.symbols}
    assert "_PRIVATE_CONST" not in names


def test_includes_private_when_flag_set(synthetic_module: types.ModuleType) -> None:
    result = extract_module_reference(synthetic_module, include_private=True)
    names = {s.short_name for s in result.symbols}
    assert "_private_fn" in names
    assert "_PRIVATE_CONST" in names


# ---------------------------------------------------------------------------
# Imported symbols not re-documented
# ---------------------------------------------------------------------------


def test_does_not_include_imported_symbols() -> None:
    """Pin: a symbol imported from another module is not documented here."""

    # Use the actual halal_trader.web.kyc module which imports stuff
    from halal_trader.web import kyc

    result = extract_module_reference(kyc)
    # KYCLevel is defined in this module
    names = {s.short_name for s in result.symbols}
    assert "KYCLevel" in names
    # `dataclass` (imported from `dataclasses`) should NOT appear
    assert "dataclass" not in names


# ---------------------------------------------------------------------------
# __all__ honoured
# ---------------------------------------------------------------------------


def test_respects_module_all_when_present() -> None:
    """Pin: __all__ explicit list overrides public-by-default."""

    mod = types.ModuleType("test_all_mod")
    mod.__doc__ = "Test."

    def fn_a():
        """A."""

    def fn_b():
        """B."""

    fn_a.__module__ = "test_all_mod"
    fn_b.__module__ = "test_all_mod"
    mod.fn_a = fn_a
    mod.fn_b = fn_b
    mod.__all__ = ["fn_a"]  # only export fn_a

    sys.modules["test_all_mod"] = mod
    try:
        result = extract_module_reference(mod)
        names = {s.short_name for s in result.symbols}
        assert "fn_a" in names
        assert "fn_b" not in names
    finally:
        sys.modules.pop("test_all_mod", None)


# ---------------------------------------------------------------------------
# Docstring extraction
# ---------------------------------------------------------------------------


def test_module_summary_is_first_line(synthetic_module: types.ModuleType) -> None:
    result = extract_module_reference(synthetic_module)
    assert result.summary == "Synthetic test module."


def test_function_summary_is_first_line(synthetic_module: types.ModuleType) -> None:
    result = extract_module_reference(synthetic_module)
    fn = next(s for s in result.symbols if s.short_name == "public_fn")
    assert fn.summary == "A public function."


def test_function_description_includes_full_docstring(
    synthetic_module: types.ModuleType,
) -> None:
    result = extract_module_reference(synthetic_module)
    fn = next(s for s in result.symbols if s.short_name == "public_fn")
    assert "longer description" in fn.description


def test_no_docstring_returns_empty_summary() -> None:
    """A function with no docstring → empty summary, not crash."""

    mod = types.ModuleType("test_nodoc")
    mod.__doc__ = "No doc test."

    def naked_fn():
        pass  # no docstring

    naked_fn.__module__ = "test_nodoc"
    mod.naked_fn = naked_fn
    sys.modules["test_nodoc"] = mod
    try:
        result = extract_module_reference(mod)
        fn = next(s for s in result.symbols if s.short_name == "naked_fn")
        assert fn.summary == ""
        assert fn.description == ""
    finally:
        sys.modules.pop("test_nodoc", None)


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


def test_function_signature_includes_params(synthetic_module: types.ModuleType) -> None:
    result = extract_module_reference(synthetic_module)
    fn = next(s for s in result.symbols if s.short_name == "public_fn")
    assert "public_fn" in fn.signature
    assert "a:" in fn.signature
    assert "b:" in fn.signature


def test_dataclass_signature_includes_fields(
    synthetic_module: types.ModuleType,
) -> None:
    result = extract_module_reference(synthetic_module)
    dc = next(s for s in result.symbols if s.short_name == "PublicDataclass")
    assert "PublicDataclass" in dc.signature
    assert "x" in dc.signature
    assert "y" in dc.signature


def test_enum_signature_includes_members(synthetic_module: types.ModuleType) -> None:
    result = extract_module_reference(synthetic_module)
    en = next(s for s in result.symbols if s.short_name == "PublicEnum")
    assert "PublicEnum" in en.signature
    assert "FOO" in en.signature
    assert "BAR" in en.signature


def test_class_signature_is_empty(synthetic_module: types.ModuleType) -> None:
    """Plain (non-dataclass non-enum) classes have no signature in this engine."""

    result = extract_module_reference(synthetic_module)
    cls = next(s for s in result.symbols if s.short_name == "PublicClass")
    assert cls.signature == ""


# ---------------------------------------------------------------------------
# Deterministic ordering
# ---------------------------------------------------------------------------


def test_symbols_sorted_alphabetically(synthetic_module: types.ModuleType) -> None:
    """Pin: symbols within a module are alphabetically sorted by short_name."""

    result = extract_module_reference(synthetic_module)
    names = [s.short_name for s in result.symbols]
    assert names == sorted(names)


def test_modules_sorted_by_qualified_name() -> None:
    from halal_trader.web import kyc, privacy

    ref = extract_api_reference(
        package_name="halal_trader",
        modules=(privacy, kyc),  # deliberately unsorted input
    )
    qnames = [m.qualified_name for m in ref.modules]
    assert qnames == sorted(qnames)


# ---------------------------------------------------------------------------
# ApiReference + ApiModule + ApiSymbol validation
# ---------------------------------------------------------------------------


def test_api_reference_rejects_empty_package_name() -> None:
    with pytest.raises(ValueError, match="package_name"):
        ApiReference(package_name="")


def test_api_module_rejects_empty_qualified_name() -> None:
    with pytest.raises(ValueError, match="qualified_name"):
        ApiModule(
            qualified_name="",
            summary="",
            description="",
            symbols=(),
        )


def test_api_symbol_rejects_empty_qualified_name() -> None:
    with pytest.raises(ValueError, match="qualified_name"):
        ApiSymbol(
            qualified_name="",
            short_name="x",
            kind=SymbolKind.FUNCTION,
            summary="",
            description="",
        )


def test_api_symbol_rejects_empty_short_name() -> None:
    with pytest.raises(ValueError, match="short_name"):
        ApiSymbol(
            qualified_name="x.y",
            short_name="",
            kind=SymbolKind.FUNCTION,
            summary="",
            description="",
        )


def test_extract_api_reference_rejects_empty_package_name() -> None:
    with pytest.raises(ValueError, match="package_name"):
        extract_api_reference(package_name="", modules=())


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_api_reference_is_frozen() -> None:
    ref = ApiReference(package_name="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.package_name = "y"  # type: ignore[misc]


def test_api_module_is_frozen() -> None:
    m = ApiModule(qualified_name="x", summary="", description="", symbols=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.qualified_name = "y"  # type: ignore[misc]


def test_api_symbol_is_frozen() -> None:
    s = ApiSymbol(
        qualified_name="x.y",
        short_name="y",
        kind=SymbolKind.FUNCTION,
        summary="",
        description="",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.short_name = "z"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum string values pinned for JSON / DB stability
# ---------------------------------------------------------------------------


def test_symbol_kind_string_values() -> None:
    assert SymbolKind.CLASS.value == "class"
    assert SymbolKind.DATACLASS.value == "dataclass"
    assert SymbolKind.FUNCTION.value == "function"
    assert SymbolKind.ENUM.value == "enum"
    assert SymbolKind.CONSTANT.value == "constant"


# ---------------------------------------------------------------------------
# Render output — pinned no-secret-leak contract
# ---------------------------------------------------------------------------


def test_render_includes_module_header(synthetic_module: types.ModuleType) -> None:
    ref = extract_api_reference(package_name="halal_trader", modules=(synthetic_module,))
    text = render_markdown(ref)
    assert "halal_trader API Reference" in text
    assert synthetic_module.__name__ in text


def test_render_groups_by_kind(synthetic_module: types.ModuleType) -> None:
    ref = extract_api_reference(package_name="halal_trader", modules=(synthetic_module,))
    text = render_markdown(ref)
    assert "Classes" in text
    assert "Dataclasses" in text
    assert "Functions" in text
    assert "Enums" in text
    assert "Constants" in text


def test_render_secret_constant_redacted(synthetic_module: types.ModuleType) -> None:
    """Pin: constants whose name matches the secret denylist render `<redacted>`."""

    ref = extract_api_reference(package_name="halal_trader", modules=(synthetic_module,))
    text = render_markdown(ref)
    # API_KEY (in name denylist) shows the placeholder, not its value
    assert "API_KEY" in text
    assert "<redacted>" in text
    assert "sk-secret-leaked" not in text


def test_render_handles_empty_reference() -> None:
    ref = ApiReference(package_name="empty_pkg")
    text = render_markdown(ref)
    assert "empty_pkg" in text
    assert "no modules" in text


def test_render_handles_module_with_no_symbols() -> None:
    """A module with no public symbols renders cleanly."""

    mod = types.ModuleType("empty_mod")
    mod.__doc__ = "Empty module."
    sys.modules["empty_mod"] = mod
    try:
        ref = extract_api_reference(package_name="halal_trader", modules=(mod,))
        text = render_markdown(ref)
        assert "no public symbols" in text
    finally:
        sys.modules.pop("empty_mod", None)


def test_render_includes_function_signature(synthetic_module: types.ModuleType) -> None:
    ref = extract_api_reference(package_name="halal_trader", modules=(synthetic_module,))
    text = render_markdown(ref)
    assert "public_fn(a:" in text


def test_render_includes_summary(synthetic_module: types.ModuleType) -> None:
    ref = extract_api_reference(package_name="halal_trader", modules=(synthetic_module,))
    text = render_markdown(ref)
    assert "A public function." in text


# ---------------------------------------------------------------------------
# End-to-end: extract a real project module
# ---------------------------------------------------------------------------


def test_extract_kyc_module() -> None:
    """End-to-end: extract reference for the actual web/kyc module."""

    from halal_trader.web import kyc

    result = extract_module_reference(kyc)
    names = {s.short_name for s in result.symbols}

    # KYCLevel and KYCStatus are in __all__
    assert "KYCLevel" in names
    assert "KYCStatus" in names
    # `permits` function is in __all__
    assert "permits" in names


def test_extract_multi_module_reference() -> None:
    from halal_trader.web import kyc, privacy

    ref = extract_api_reference(
        package_name="halal_trader",
        modules=(kyc, privacy),
    )
    assert ref.package_name == "halal_trader"
    assert len(ref.modules) == 2
    qnames = [m.qualified_name for m in ref.modules]
    assert any("kyc" in q for q in qnames)
    assert any("privacy" in q for q in qnames)


def test_extract_real_module_summary_extracted() -> None:
    """Real modules should have non-empty summaries."""

    from halal_trader.web import kyc

    result = extract_module_reference(kyc)
    assert result.summary != ""


def test_extract_real_module_symbols_have_summaries() -> None:
    """Most documented Round-4 symbols have summaries."""

    from halal_trader.web import kyc

    result = extract_module_reference(kyc)
    documented = [s for s in result.symbols if s.summary]
    # at least a few public symbols should have docstrings
    assert len(documented) >= 3


# ---------------------------------------------------------------------------
# Extract reference handles empty modules tuple
# ---------------------------------------------------------------------------


def test_extract_with_empty_modules_returns_empty() -> None:
    ref = extract_api_reference(package_name="halal_trader", modules=())
    assert ref.modules == ()
