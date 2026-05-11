"""Tests for marketplace/fix_adapter.py — Round-5 Wave 21.F."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from halal_trader.marketplace.fix_adapter import (
    SOH,
    FixError,
    MsgType,
    NewOrderSingle,
    ParsedMessage,
    SessionConfig,
    build_heartbeat,
    build_logon,
    build_logout,
    build_message,
    build_new_order_single,
    parse_message,
    render_message,
)


def _config(
    sender: str = "HALAL_PLATFORM",
    target: str = "INSTITUTION_X",
    heart_beat_secs: int = 30,
) -> SessionConfig:
    return SessionConfig(
        sender_comp_id=sender,
        target_comp_id=target,
        heart_beat_secs=heart_beat_secs,
    )


def _ts() -> datetime:
    return datetime(2026, 5, 11, 12, 0, 0, tzinfo=timezone.utc)


# --- SessionConfig validation ----------------------


def test_config_valid():
    c = _config()
    assert c.begin_string == "FIX.4.4"


def test_config_empty_sender_rejected():
    with pytest.raises(ValueError):
        _config(sender="")


def test_config_same_comp_ids_rejected():
    with pytest.raises(ValueError):
        _config(sender="A", target="A")


def test_config_invalid_begin_string_rejected():
    with pytest.raises(ValueError):
        SessionConfig(
            sender_comp_id="A",
            target_comp_id="B",
            begin_string="FIX.5.0",
        )


def test_config_invalid_heartbeat_rejected():
    with pytest.raises(ValueError):
        _config(heart_beat_secs=0)


# --- build_message + parse_message round-trip ------


def test_build_logon_roundtrip():
    config = _config()
    raw = build_logon(config, msg_seq_num=1, sending_time=_ts())
    parsed = parse_message(raw)
    assert parsed.msg_type() is MsgType.LOGON
    assert parsed.sender_comp_id() == "HALAL_PLATFORM"
    assert parsed.target_comp_id() == "INSTITUTION_X"
    assert parsed.msg_seq_num() == 1
    # 108 = heart_beat_secs.
    assert parsed.get(108) == "30"


def test_build_heartbeat_no_test_id():
    config = _config()
    raw = build_heartbeat(config, msg_seq_num=2, sending_time=_ts())
    parsed = parse_message(raw)
    assert parsed.msg_type() is MsgType.HEARTBEAT
    assert parsed.get(112) is None


def test_build_heartbeat_with_test_id():
    config = _config()
    raw = build_heartbeat(config, msg_seq_num=2, sending_time=_ts(), test_req_id="TR-1")
    parsed = parse_message(raw)
    assert parsed.get(112) == "TR-1"


def test_build_logout_with_reason():
    config = _config()
    raw = build_logout(config, msg_seq_num=3, sending_time=_ts(), reason="end of session")
    parsed = parse_message(raw)
    assert parsed.msg_type() is MsgType.LOGOUT
    assert parsed.get(58) == "end of session"


def test_build_invalid_seq_num_rejected():
    config = _config()
    with pytest.raises(ValueError):
        build_message(
            MsgType.HEARTBEAT,
            config,
            msg_seq_num=0,
            sending_time=_ts(),
        )


def test_build_negative_tag_rejected():
    config = _config()
    with pytest.raises(ValueError):
        build_message(
            MsgType.HEARTBEAT,
            config,
            msg_seq_num=1,
            sending_time=_ts(),
            extra_fields=((-1, "x"),),
        )


def test_build_field_value_with_soh_rejected():
    config = _config()
    with pytest.raises(ValueError):
        build_message(
            MsgType.HEARTBEAT,
            config,
            msg_seq_num=1,
            sending_time=_ts(),
            extra_fields=((100, f"bad{SOH}value"),),
        )


def test_build_field_value_with_equals_rejected():
    config = _config()
    with pytest.raises(ValueError):
        build_message(
            MsgType.HEARTBEAT,
            config,
            msg_seq_num=1,
            sending_time=_ts(),
            extra_fields=((100, "k=v"),),
        )


def test_build_naive_sending_time_rejected():
    config = _config()
    with pytest.raises(ValueError):
        build_message(
            MsgType.HEARTBEAT,
            config,
            msg_seq_num=1,
            sending_time=datetime(2026, 5, 11, 12, 0),
        )


# --- Body length + checksum verification ----------


def test_parse_body_length_mismatch_rejected():
    """Pin: parser rejects messages whose BodyLength is wrong."""
    config = _config()
    raw = build_heartbeat(config, msg_seq_num=1, sending_time=_ts())
    # Tamper: replace BodyLength with a wrong value.
    # Find 9=NNN<SOH>.
    parts = raw.split(SOH)
    # parts[1] is "9=NNN"
    tag9 = parts[1]
    wrong = tag9.split("=")[0] + "=" + str(int(tag9.split("=")[1]) + 5)
    parts[1] = wrong
    tampered = SOH.join(parts)
    with pytest.raises(FixError):
        parse_message(tampered)


def test_parse_checksum_mismatch_rejected():
    config = _config()
    raw = build_heartbeat(config, msg_seq_num=1, sending_time=_ts())
    # Tamper: replace the checksum value.
    cs_idx = raw.rindex("10=")
    bad = raw[:cs_idx] + "10=999" + SOH
    with pytest.raises(FixError):
        parse_message(bad)


def test_parse_empty_rejected():
    with pytest.raises(FixError):
        parse_message("")


def test_parse_missing_trailing_soh_rejected():
    with pytest.raises(FixError):
        parse_message("8=FIX.4.4\x019=10\x0135=A\x0110=000")


def test_parse_malformed_field_rejected():
    raw = f"8=FIX.4.4{SOH}9=20{SOH}35=A{SOH}NOTAFIELD{SOH}10=000{SOH}"
    with pytest.raises(FixError):
        parse_message(raw)


def test_parse_non_numeric_tag_rejected():
    raw = f"8=FIX.4.4{SOH}9=15{SOH}35=A{SOH}abc=def{SOH}10=000{SOH}"
    with pytest.raises(FixError):
        parse_message(raw)


def test_parse_too_few_fields_rejected():
    raw = f"8=FIX.4.4{SOH}9=0{SOH}"
    with pytest.raises(FixError):
        parse_message(raw)


def test_parse_first_field_must_be_begin_string():
    raw = f"9=10{SOH}35=A{SOH}10=000{SOH}"
    with pytest.raises(FixError):
        parse_message(raw)


# --- ParsedMessage helpers ----------------------


def test_parsed_get_returns_value():
    config = _config()
    raw = build_logon(config, msg_seq_num=1, sending_time=_ts())
    parsed = parse_message(raw)
    assert parsed.get(49) == "HALAL_PLATFORM"  # tag 49 = SenderCompID


def test_parsed_get_missing_returns_none():
    config = _config()
    raw = build_heartbeat(config, msg_seq_num=1, sending_time=_ts())
    parsed = parse_message(raw)
    assert parsed.get(112) is None


def test_parsed_get_all_returns_tuple():
    config = _config()
    raw = build_message(
        MsgType.NEW_ORDER_SINGLE,
        config,
        msg_seq_num=1,
        sending_time=_ts(),
        extra_fields=((100, "a"), (100, "b")),
    )
    parsed = parse_message(raw)
    values = parsed.get_all(100)
    assert values == ("a", "b")


def test_parsed_msg_type_unknown_rejected():
    # Construct a ParsedMessage by hand to exercise the MsgType branch
    # in isolation — a manual tamper of MsgType in a built message
    # would invalidate BodyLength and fail at parse time.
    bad = ParsedMessage(
        fields=(
            (8, "FIX.4.4"),
            (9, "5"),
            (35, "Z"),
        )
    )
    with pytest.raises(FixError):
        bad.msg_type()


def test_parsed_missing_required_tags_raise():
    bad = ParsedMessage(fields=((8, "FIX.4.4"),))
    with pytest.raises(FixError):
        bad.msg_type()
    with pytest.raises(FixError):
        bad.sender_comp_id()
    with pytest.raises(FixError):
        bad.msg_seq_num()


# --- NewOrderSingle ----------------------------


def test_new_order_valid_market():
    o = NewOrderSingle(
        cl_ord_id="O1",
        symbol="AAPL",
        side="1",
        order_qty=100.0,
        ord_type="1",
    )
    assert o.price is None


def test_new_order_valid_limit():
    o = NewOrderSingle(
        cl_ord_id="O1",
        symbol="AAPL",
        side="1",
        order_qty=100.0,
        ord_type="2",
        price=150.0,
    )
    assert o.price == 150.0


def test_new_order_market_with_price_rejected():
    with pytest.raises(ValueError):
        NewOrderSingle(
            cl_ord_id="O1",
            symbol="AAPL",
            side="1",
            order_qty=100.0,
            ord_type="1",
            price=150.0,
        )


def test_new_order_limit_without_price_rejected():
    with pytest.raises(ValueError):
        NewOrderSingle(
            cl_ord_id="O1",
            symbol="AAPL",
            side="1",
            order_qty=100.0,
            ord_type="2",
        )


def test_new_order_invalid_side_rejected():
    with pytest.raises(ValueError):
        NewOrderSingle(
            cl_ord_id="O1",
            symbol="AAPL",
            side="3",
            order_qty=100.0,
            ord_type="1",
        )


def test_new_order_zero_qty_rejected():
    with pytest.raises(ValueError):
        NewOrderSingle(
            cl_ord_id="O1",
            symbol="AAPL",
            side="1",
            order_qty=0,
            ord_type="1",
        )


def test_build_new_order_single_roundtrip():
    config = _config()
    order = NewOrderSingle(
        cl_ord_id="ORDER-42",
        symbol="AAPL",
        side="1",
        order_qty=100.0,
        ord_type="2",
        price=150.0,
    )
    raw = build_new_order_single(config, order, msg_seq_num=5, sending_time=_ts())
    parsed = parse_message(raw)
    assert parsed.msg_type() is MsgType.NEW_ORDER_SINGLE
    assert parsed.get(11) == "ORDER-42"
    assert parsed.get(55) == "AAPL"
    assert parsed.get(54) == "1"
    assert parsed.get(40) == "2"
    assert parsed.get(44) == "150"


# --- Render ----------------------------------


def test_render_no_secret_leak():
    config = _config(sender="SENDER@example.com", target="TARGET@example.com")
    raw = build_heartbeat(config, msg_seq_num=1, sending_time=_ts())
    parsed = parse_message(raw)
    out = render_message(parsed)
    assert "SENDER@example.com" not in out
    assert "TARGET@example.com" not in out


def test_render_format():
    config = _config()
    raw = build_heartbeat(config, msg_seq_num=42, sending_time=_ts())
    parsed = parse_message(raw)
    out = render_message(parsed)
    assert "📨" in out
    assert "type=0" in out  # HEARTBEAT
    assert "seq=42" in out


def test_render_missing_fields_renders_question_marks():
    bad = ParsedMessage(fields=((8, "FIX.4.4"),))
    out = render_message(bad)
    # Don't crash; render '?' placeholders.
    assert "type=?" in out
