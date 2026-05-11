"""FIX protocol primitives — Round-5 Wave 21.F.

Institutional FIX-4.4 message constructor + parser. This module is
**protocol-only**: it builds and parses the wire-format byte strings.
The TCP transport, sequence-number management, and session reset live
in the deployment layer.

Pinned semantics:

- **Closed-set MsgType ladder** — LOGON / LOGOUT / HEARTBEAT /
  TEST_REQUEST / NEW_ORDER_SINGLE / EXECUTION_REPORT / ORDER_CANCEL_REQUEST /
  ORDER_CANCEL_REJECT / REJECT.
- **SOH (0x01) byte** is the canonical field separator. The parser
  rejects messages whose separators are corrupted.
- **BodyLength (tag 9)** validation pinned: must equal the byte count
  from the byte after BodyLength's SOH up to (but not including) the
  CheckSum SOH.
- **CheckSum (tag 10)** validation: sum of all bytes through
  CheckSum's preceding SOH, modulo 256, zero-padded to 3 digits.
- **Pure-Python deterministic.**
- **No-secret-leak pin** — sender/target comp IDs masked in render.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

SOH = "\x01"


class MsgType(str, Enum):
    """Closed-set FIX-4.4 message-type ladder (subset)."""

    LOGON = "A"
    LOGOUT = "5"
    HEARTBEAT = "0"
    TEST_REQUEST = "1"
    NEW_ORDER_SINGLE = "D"
    EXECUTION_REPORT = "8"
    ORDER_CANCEL_REQUEST = "F"
    ORDER_CANCEL_REJECT = "9"
    REJECT = "3"


# Common FIX tag numbers as readable constants.
TAG_BEGIN_STRING = 8
TAG_BODY_LENGTH = 9
TAG_CHECKSUM = 10
TAG_MSG_TYPE = 35
TAG_SENDER_COMP_ID = 49
TAG_TARGET_COMP_ID = 56
TAG_MSG_SEQ_NUM = 34
TAG_SENDING_TIME = 52


class FixError(ValueError):
    """Raised when message construction or parsing fails."""


@dataclass(frozen=True)
class SessionConfig:
    """Per-session counterparty configuration."""

    sender_comp_id: str
    target_comp_id: str
    begin_string: str = "FIX.4.4"
    heart_beat_secs: int = 30

    def __post_init__(self) -> None:
        if not self.sender_comp_id or not self.sender_comp_id.strip():
            raise ValueError("sender_comp_id must be non-empty")
        if not self.target_comp_id or not self.target_comp_id.strip():
            raise ValueError("target_comp_id must be non-empty")
        if self.sender_comp_id == self.target_comp_id:
            raise ValueError("sender and target comp IDs must differ")
        if self.begin_string not in ("FIX.4.2", "FIX.4.4", "FIXT.1.1"):
            raise ValueError("begin_string must be FIX.4.2 / FIX.4.4 / FIXT.1.1")
        if not 1 <= self.heart_beat_secs <= 300:
            raise ValueError("heart_beat_secs must be in [1, 300]")


def _format_sending_time(ts: datetime) -> str:
    """FIX UTCTimestamp format: YYYYMMDD-HH:MM:SS.sss."""
    if ts.tzinfo is None:
        raise ValueError("sending_time must be tz-aware")
    ts_utc = ts.astimezone(timezone.utc)
    return ts_utc.strftime("%Y%m%d-%H:%M:%S.") + f"{ts_utc.microsecond // 1000:03d}"


def _compute_body_length(body: str) -> int:
    """Count the bytes between BodyLength's SOH and CheckSum's preceding SOH."""
    return len(body.encode("ascii"))


def _compute_checksum(prefix: str) -> str:
    """sum-of-bytes mod 256, zero-padded to 3 digits."""
    total = sum(prefix.encode("ascii")) % 256
    return f"{total:03d}"


def build_message(
    msg_type: MsgType,
    config: SessionConfig,
    *,
    msg_seq_num: int,
    sending_time: datetime,
    extra_fields: Sequence[tuple[int, str]] = (),
) -> str:
    """Construct a wire-format FIX message string with header, body, and trailer.

    Pinned: the trailer's CheckSum is computed last; the header tags
    8/9 always come first; tag 35 (MsgType) always comes immediately
    after tag 9.
    """
    if msg_seq_num <= 0:
        raise ValueError("msg_seq_num must be positive")
    for tag, value in extra_fields:
        if tag <= 0:
            raise ValueError(f"tag {tag} must be positive")
        if SOH in value:
            raise ValueError(f"field value for tag {tag} contains SOH byte")
        if "=" in value:
            raise ValueError(f"field value for tag {tag} contains '=' (illegal)")
    body_fields: list[tuple[int, str]] = [
        (TAG_MSG_TYPE, msg_type.value),
        (TAG_SENDER_COMP_ID, config.sender_comp_id),
        (TAG_TARGET_COMP_ID, config.target_comp_id),
        (TAG_MSG_SEQ_NUM, str(msg_seq_num)),
        (TAG_SENDING_TIME, _format_sending_time(sending_time)),
    ]
    body_fields.extend(extra_fields)
    body = "".join(f"{tag}={value}{SOH}" for tag, value in body_fields)
    body_length = _compute_body_length(body)
    prefix = (
        f"{TAG_BEGIN_STRING}={config.begin_string}{SOH}{TAG_BODY_LENGTH}={body_length}{SOH}{body}"
    )
    checksum = _compute_checksum(prefix)
    return f"{prefix}{TAG_CHECKSUM}={checksum}{SOH}"


@dataclass(frozen=True)
class ParsedMessage:
    """Parsed FIX message — ordered list of (tag, value) pairs."""

    fields: tuple[tuple[int, str], ...]

    def get(self, tag: int) -> str | None:
        for t, v in self.fields:
            if t == tag:
                return v
        return None

    def get_all(self, tag: int) -> tuple[str, ...]:
        return tuple(v for t, v in self.fields if t == tag)

    def msg_type(self) -> MsgType:
        raw = self.get(TAG_MSG_TYPE)
        if raw is None:
            raise FixError("missing tag 35 (MsgType)")
        try:
            return MsgType(raw)
        except ValueError as exc:
            raise FixError(f"unknown MsgType {raw!r}") from exc

    def sender_comp_id(self) -> str:
        v = self.get(TAG_SENDER_COMP_ID)
        if v is None:
            raise FixError("missing tag 49 (SenderCompID)")
        return v

    def target_comp_id(self) -> str:
        v = self.get(TAG_TARGET_COMP_ID)
        if v is None:
            raise FixError("missing tag 56 (TargetCompID)")
        return v

    def msg_seq_num(self) -> int:
        v = self.get(TAG_MSG_SEQ_NUM)
        if v is None:
            raise FixError("missing tag 34 (MsgSeqNum)")
        try:
            return int(v)
        except ValueError as exc:
            raise FixError(f"tag 34 (MsgSeqNum) not an integer: {v!r}") from exc


def parse_message(raw: str) -> ParsedMessage:
    """Parse a wire-format FIX message; verify BodyLength + CheckSum.

    Raises FixError on any malformation.
    """
    if not raw or not raw.endswith(SOH):
        raise FixError("message must end with SOH")
    parts = [p for p in raw.split(SOH) if p]
    if not parts:
        raise FixError("empty message")
    fields: list[tuple[int, str]] = []
    for part in parts:
        if "=" not in part:
            raise FixError(f"malformed field: {part!r}")
        tag_str, _, value = part.partition("=")
        try:
            tag = int(tag_str)
        except ValueError as exc:
            raise FixError(f"non-numeric tag: {tag_str!r}") from exc
        fields.append((tag, value))
    if len(fields) < 3:
        raise FixError("message must have at least 3 fields (8/9/10)")
    if fields[0][0] != TAG_BEGIN_STRING:
        raise FixError("first field must be tag 8 (BeginString)")
    if fields[1][0] != TAG_BODY_LENGTH:
        raise FixError("second field must be tag 9 (BodyLength)")
    if fields[-1][0] != TAG_CHECKSUM:
        raise FixError("last field must be tag 10 (CheckSum)")
    # Verify body length.
    body_start_idx = raw.index(SOH, raw.index(SOH) + 1) + 1
    checksum_field_start = raw.rindex(f"{TAG_CHECKSUM}=")
    body = raw[body_start_idx:checksum_field_start]
    declared_length = int(fields[1][1])
    if len(body.encode("ascii")) != declared_length:
        raise FixError(
            f"BodyLength mismatch: declared {declared_length}, observed {len(body.encode('ascii'))}"
        )
    # Verify checksum.
    prefix = raw[:checksum_field_start]
    expected = _compute_checksum(prefix)
    if expected != fields[-1][1]:
        raise FixError(f"CheckSum mismatch: declared {fields[-1][1]}, computed {expected}")
    return ParsedMessage(fields=tuple(fields))


def build_logon(
    config: SessionConfig,
    *,
    msg_seq_num: int,
    sending_time: datetime,
) -> str:
    """Convenience for the LOGON message (108=heart_beat_secs)."""
    return build_message(
        MsgType.LOGON,
        config,
        msg_seq_num=msg_seq_num,
        sending_time=sending_time,
        extra_fields=((108, str(config.heart_beat_secs)),),
    )


def build_heartbeat(
    config: SessionConfig,
    *,
    msg_seq_num: int,
    sending_time: datetime,
    test_req_id: str = "",
) -> str:
    extra: list[tuple[int, str]] = []
    if test_req_id:
        extra.append((112, test_req_id))
    return build_message(
        MsgType.HEARTBEAT,
        config,
        msg_seq_num=msg_seq_num,
        sending_time=sending_time,
        extra_fields=tuple(extra),
    )


def build_logout(
    config: SessionConfig,
    *,
    msg_seq_num: int,
    sending_time: datetime,
    reason: str = "",
) -> str:
    extra: list[tuple[int, str]] = []
    if reason:
        extra.append((58, reason))
    return build_message(
        MsgType.LOGOUT,
        config,
        msg_seq_num=msg_seq_num,
        sending_time=sending_time,
        extra_fields=tuple(extra),
    )


@dataclass(frozen=True)
class NewOrderSingle:
    """High-level new-order-single fields. Caller-friendly wrapper."""

    cl_ord_id: str
    """Client-assigned unique order ID. Tag 11."""
    symbol: str
    """Tag 55."""
    side: str
    """'1'=Buy, '2'=Sell. Tag 54."""
    order_qty: float
    """Tag 38."""
    ord_type: str
    """'1'=Market, '2'=Limit. Tag 40."""
    price: float | None = None
    """Required for Limit. Tag 44."""
    transact_time: datetime | None = None
    """Tag 60. Defaults to sending_time."""

    def __post_init__(self) -> None:
        if not self.cl_ord_id or not self.cl_ord_id.strip():
            raise ValueError("cl_ord_id must be non-empty")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if self.side not in ("1", "2"):
            raise ValueError("side must be '1' (Buy) or '2' (Sell)")
        if self.ord_type not in ("1", "2"):
            raise ValueError("ord_type must be '1' (Market) or '2' (Limit)")
        if self.order_qty <= 0:
            raise ValueError("order_qty must be positive")
        if self.ord_type == "2" and (self.price is None or self.price <= 0):
            raise ValueError("Limit order requires positive price")
        if self.ord_type == "1" and self.price is not None:
            raise ValueError("Market order must not set a price")


def build_new_order_single(
    config: SessionConfig,
    order: NewOrderSingle,
    *,
    msg_seq_num: int,
    sending_time: datetime,
) -> str:
    extra: list[tuple[int, str]] = [
        (11, order.cl_ord_id),
        (55, order.symbol),
        (54, order.side),
        (38, f"{order.order_qty:g}"),
        (40, order.ord_type),
    ]
    if order.price is not None:
        extra.append((44, f"{order.price:g}"))
    extra.append((60, _format_sending_time(order.transact_time or sending_time)))
    return build_message(
        MsgType.NEW_ORDER_SINGLE,
        config,
        msg_seq_num=msg_seq_num,
        sending_time=sending_time,
        extra_fields=tuple(extra),
    )


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


def render_message(msg: ParsedMessage) -> str:
    """Operator-readable summary — comp IDs masked."""
    try:
        mt = msg.msg_type().value
    except FixError:
        mt = "?"
    try:
        sender = _mask(msg.sender_comp_id())
    except FixError:
        sender = "?"
    try:
        target = _mask(msg.target_comp_id())
    except FixError:
        target = "?"
    try:
        seq = msg.msg_seq_num()
    except FixError:
        seq = "?"
    return f"📨 FIX type={mt} {sender} → {target} seq={seq} ({len(msg.fields)} fields)"
