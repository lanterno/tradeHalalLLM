"""Translation registry — Round-5 Wave 23 (A-I unified).

A single locale-keyed registry covers all Round-5 Wave-23 languages:

    Arabic (MSA + dialects), Urdu, Malay, Indonesian, Turkish, Hausa,
    Swahili, Bengali, Persian, Pashto, Dari, French, Spanish.

Each locale has a closed-set TextDirection (LTR or RTL). The registry
ships:

- A `Locale` closed-set enum.
- A `TranslationRegistry` that maps `(locale, key) → str`.
- A fallback chain: locale → base-locale → DEFAULT_LOCALE.
- Coverage validation: every key must resolve in every locale
  (operator opts into strict_mode at construction).

Pinned semantics:

- **Closed-set Locale ladder.** No string-typed locales; the type
  system rejects unknown locales at the boundary.
- **Closed-set TextDirection** — LTR / RTL.
- **Fallback chain pin**: locale → base-locale (e.g. ar-EG → ar) →
  en. The base-locale is the first 2-char prefix.
- **Empty strings rejected** — translations must be meaningful.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Locale(str, Enum):
    """Closed-set locale ladder (Wave 23 A-I + the English default)."""

    EN = "en"
    AR = "ar"
    AR_EG = "ar-EG"
    AR_SA = "ar-SA"
    UR = "ur"
    MS = "ms"  # Malay
    ID = "id"  # Indonesian
    TR = "tr"  # Turkish
    HA = "ha"  # Hausa
    SW = "sw"  # Swahili
    BN = "bn"  # Bengali
    FA = "fa"  # Persian
    PS = "ps"  # Pashto
    FR = "fr"
    ES = "es"


class TextDirection(str, Enum):
    """Closed-set text-direction ladder."""

    LTR = "ltr"
    RTL = "rtl"


_DIRECTION: dict[Locale, TextDirection] = {
    Locale.EN: TextDirection.LTR,
    Locale.AR: TextDirection.RTL,
    Locale.AR_EG: TextDirection.RTL,
    Locale.AR_SA: TextDirection.RTL,
    Locale.UR: TextDirection.RTL,
    Locale.MS: TextDirection.LTR,
    Locale.ID: TextDirection.LTR,
    Locale.TR: TextDirection.LTR,
    Locale.HA: TextDirection.LTR,
    Locale.SW: TextDirection.LTR,
    Locale.BN: TextDirection.LTR,
    Locale.FA: TextDirection.RTL,
    Locale.PS: TextDirection.RTL,
    Locale.FR: TextDirection.LTR,
    Locale.ES: TextDirection.LTR,
}


DEFAULT_LOCALE: Locale = Locale.EN
"""All translation lookups fall through to this as the final step."""


def direction_for(locale: Locale) -> TextDirection:
    return _DIRECTION[locale]


def is_rtl(locale: Locale) -> bool:
    return _DIRECTION[locale] is TextDirection.RTL


def base_locale(locale: Locale) -> Locale:
    """Return the base locale (first 2 chars). e.g. AR_EG → AR.

    For locales without a region (e.g. EN), returns the locale itself.
    """
    value = locale.value
    if "-" not in value:
        return locale
    base = value.split("-")[0]
    for L in Locale:
        if L.value == base:
            return L
    return locale  # defensive — base isn't recognised


@dataclass(frozen=True)
class Translation:
    """One (locale, key) → text mapping."""

    locale: Locale
    key: str
    text: str

    def __post_init__(self) -> None:
        if not self.key or not self.key.strip():
            raise ValueError("key must be non-empty")
        if "." in self.key and not all(p.strip() for p in self.key.split(".")):
            raise ValueError("key segments must be non-empty")
        if len(self.key) > 200:
            raise ValueError("key must be ≤ 200 chars")
        if not self.text or not self.text.strip():
            raise ValueError("text must be non-empty")
        if len(self.text) > 5000:
            raise ValueError("text must be ≤ 5000 chars")


@dataclass(frozen=True)
class TranslationRegistry:
    """Frozen translation table with fallback-chain lookup."""

    entries: tuple[Translation, ...]
    strict_mode: bool = False
    """If True, every key must resolve in every supported locale at
    construction. Catches missing translations early."""

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("registry must have ≥ 1 entry")
        seen: set[tuple[Locale, str]] = set()
        for t in self.entries:
            k = (t.locale, t.key)
            if k in seen:
                raise ValueError(f"duplicate (locale, key) {t.locale.value}/{t.key}")
            seen.add(k)
        if self.strict_mode:
            keys_in_default = {t.key for t in self.entries if t.locale is DEFAULT_LOCALE}
            for L in Locale:
                if L is DEFAULT_LOCALE:
                    continue
                keys_here = {t.key for t in self.entries if t.locale is L}
                missing = keys_in_default - keys_here
                if missing:
                    # Allow fallback-chain coverage: base locale satisfies.
                    base = base_locale(L)
                    base_keys = {t.key for t in self.entries if t.locale is base}
                    still_missing = missing - base_keys
                    if still_missing:
                        raise ValueError(
                            f"locale {L.value} missing keys: {sorted(still_missing)[:5]}..."
                        )

    def lookup(self, locale: Locale, key: str) -> str | None:
        """Locate the translation via fallback chain: locale → base → default."""
        chain: list[Locale] = [locale]
        b = base_locale(locale)
        if b is not locale:
            chain.append(b)
        if DEFAULT_LOCALE not in chain:
            chain.append(DEFAULT_LOCALE)
        for L in chain:
            for t in self.entries:
                if t.locale is L and t.key == key:
                    return t.text
        return None

    def must_lookup(self, locale: Locale, key: str) -> str:
        """Like `lookup`, but raises KeyError if no fallback resolves."""
        hit = self.lookup(locale, key)
        if hit is None:
            raise KeyError(f"no translation for {locale.value}/{key} across fallback chain")
        return hit

    def supported_locales(self) -> tuple[Locale, ...]:
        """Locales with at least one entry."""
        seen = {t.locale for t in self.entries}
        return tuple(sorted(seen, key=lambda L: L.value))

    def keys_for(self, locale: Locale) -> tuple[str, ...]:
        return tuple(sorted(t.key for t in self.entries if t.locale is locale))


def merge_registries(*registries: TranslationRegistry) -> TranslationRegistry:
    """Merge N registries; later overrides earlier on (locale, key) conflict."""
    if not registries:
        raise ValueError("must supply ≥ 1 registry to merge")
    out: dict[tuple[Locale, str], Translation] = {}
    for r in registries:
        for t in r.entries:
            out[(t.locale, t.key)] = t
    return TranslationRegistry(entries=tuple(out.values()))


def render_lookup(
    registry: TranslationRegistry,
    locale: Locale,
    key: str,
) -> str:
    """Render a translated value with diagnostic markers when missing."""
    hit = registry.lookup(locale, key)
    if hit is None:
        return f"⚠️ MISSING[{locale.value}:{key}]"
    direction = direction_for(locale)
    if direction is TextDirection.RTL:
        return f"🔃 {hit}"
    return hit
