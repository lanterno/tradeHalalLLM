"""US Form 8949 / 1099-B row generator — Round-5 Wave 18.B.

US taxpayers report capital-gain transactions on Form 8949 ("Sales
and Other Dispositions of Capital Assets"), which feeds Schedule D.
The IRS distinguishes:

- **Short-term** (held ≤ 1 year) — taxed as ordinary income.
- **Long-term** (held > 1 year) — taxed at capital-gains rates.

This module ships the **row generator** that converts the bot's
``RealisedSlice`` records into Form-8949-formatted rows. Persistence
+ submission to a tax-prep service live above.

Pinned semantics:

- **Closed-set FormBox ladder** (BOX_A / BOX_D — basis reported,
  short / long; BOX_B / BOX_E — basis NOT reported).
- **Wash-sale flag** is operator-supplied; the generator surfaces it
  as the W code in column (f).
- **Per-row gain/loss is signed** — IRS expects positive losses
  recorded with parentheses; this module emits the signed number
  and leaves formatting to the renderer.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from enum import Enum

from halal_trader.core.tax_lots import RealisedSlice


class FormBox(str, Enum):
    """Closed-set Form 8949 boxes."""

    BOX_A = "box_a"  # short-term, basis reported to IRS
    BOX_B = "box_b"  # short-term, basis NOT reported
    BOX_C = "box_c"  # short-term, not reported on 1099-B
    BOX_D = "box_d"  # long-term, basis reported to IRS
    BOX_E = "box_e"  # long-term, basis NOT reported
    BOX_F = "box_f"  # long-term, not reported on 1099-B


@dataclass(frozen=True)
class Form8949Row:
    """A single Form 8949 row."""

    description: str  # Column (a) — usually symbol + share count
    acquired_date: date  # Column (b)
    sold_date: date  # Column (c)
    proceeds: float  # Column (d)
    cost_basis: float  # Column (e)
    code: str  # Column (f) — adjustment code, "" if none
    adjustment: float  # Column (g) — adjustment amount, 0 if none
    gain_loss: float  # Column (h)
    box: FormBox  # Which box the row is filed under

    def __post_init__(self) -> None:
        if not self.description or not self.description.strip():
            raise ValueError("description must be non-empty")
        if self.proceeds < 0:
            raise ValueError("proceeds must be non-negative")
        if self.cost_basis < 0:
            raise ValueError("cost_basis must be non-negative")
        # IRS-consistent: gain_loss = proceeds - cost_basis + adjustment
        expected = self.proceeds - self.cost_basis + self.adjustment
        if abs(expected - self.gain_loss) > 0.005:
            raise ValueError(
                "gain_loss must equal proceeds - cost_basis + adjustment "
                f"(got {self.gain_loss}, expected {expected})"
            )


def slice_to_8949_row(
    s: RealisedSlice,
    *,
    symbol: str,
    basis_reported_to_irs: bool = True,
    is_wash_sale: bool = False,
    wash_sale_disallowed: float = 0.0,
) -> Form8949Row:
    """Convert a RealisedSlice into a Form 8949 row.

    ``basis_reported_to_irs`` chooses Box A/D (yes) or Box B/E (no).
    The C/F boxes (not reported on 1099-B at all) are operator-set;
    this helper assumes the standard 1099-B path.
    """
    if s.is_long_term:
        box = FormBox.BOX_D if basis_reported_to_irs else FormBox.BOX_E
    else:
        box = FormBox.BOX_A if basis_reported_to_irs else FormBox.BOX_B

    proceeds = s.quantity * s.proceeds_per_share
    cost_basis = s.quantity * s.cost_basis_per_share
    code = "W" if is_wash_sale else ""
    adjustment = wash_sale_disallowed if is_wash_sale else 0.0
    gain_loss = proceeds - cost_basis + adjustment

    return Form8949Row(
        description=f"{s.quantity:.4f} sh {symbol}",
        acquired_date=s.acquisition_date,
        sold_date=s.sale_date,
        proceeds=proceeds,
        cost_basis=cost_basis,
        code=code,
        adjustment=adjustment,
        gain_loss=gain_loss,
        box=box,
    )


def slices_to_8949_rows(
    slices: Iterable[RealisedSlice],
    *,
    symbol: str,
    basis_reported_to_irs: bool = True,
) -> tuple[Form8949Row, ...]:
    return tuple(
        slice_to_8949_row(
            s, symbol=symbol, basis_reported_to_irs=basis_reported_to_irs
        )
        for s in slices
    )


@dataclass(frozen=True)
class BoxTotals:
    """Totals for a single Form 8949 box."""

    box: FormBox
    n_rows: int
    proceeds: float
    cost_basis: float
    adjustment: float
    gain_loss: float

    def __post_init__(self) -> None:
        if self.n_rows < 0:
            raise ValueError("n_rows must be non-negative")
        if self.proceeds < 0 or self.cost_basis < 0:
            raise ValueError("proceeds + cost_basis must be non-negative")


def box_totals(rows: Iterable[Form8949Row]) -> dict[FormBox, BoxTotals]:
    """Aggregate rows by box."""
    out: dict[FormBox, BoxTotals] = {}
    by_box: dict[FormBox, list[Form8949Row]] = {}
    for row in rows:
        by_box.setdefault(row.box, []).append(row)
    for box, rs in by_box.items():
        out[box] = BoxTotals(
            box=box,
            n_rows=len(rs),
            proceeds=sum(r.proceeds for r in rs),
            cost_basis=sum(r.cost_basis for r in rs),
            adjustment=sum(r.adjustment for r in rs),
            gain_loss=sum(r.gain_loss for r in rs),
        )
    return out


_FORBIDDEN_RENDER_TOKENS: tuple[str, ...] = (
    "@",
    "zoom.us",
    "meet.google",
    "private_email",
    "+1-",
    "Authorization",
    "SSN",
    "TaxID",
    "DOB",
)


def _scrub(text: str) -> str:
    for token in _FORBIDDEN_RENDER_TOKENS:
        if token in text:
            text = text.replace(token, "[redacted]")
    return text


def render_row(row: Form8949Row) -> str:
    sign = "" if row.gain_loss >= 0 else "(loss)"
    return _scrub(
        f"{row.box.value:6s} | {row.description} | "
        f"{row.acquired_date.isoformat()}→{row.sold_date.isoformat()} | "
        f"proc=${row.proceeds:.2f} basis=${row.cost_basis:.2f} "
        f"adj=${row.adjustment:.2f} gl=${row.gain_loss:+.2f} {sign}".rstrip()
    )


def render_summary(rows: Iterable[Form8949Row]) -> str:
    rows_t = tuple(rows)
    if not rows_t:
        return "Form 8949: no rows"
    totals = box_totals(rows_t)
    lines = [f"Form 8949: {len(rows_t)} rows across {len(totals)} boxes"]
    for box, t in sorted(totals.items(), key=lambda kv: kv[0].value):
        lines.append(
            f"  {box.value}: n={t.n_rows} proceeds=${t.proceeds:.2f} "
            f"basis=${t.cost_basis:.2f} gain/loss=${t.gain_loss:+.2f}"
        )
    return _scrub("\n".join(lines))
