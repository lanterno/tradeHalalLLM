"""Tests for `TradingBot._acquire_lock` / `_release_lock`.

The PID file lock prevents two stocks bot instances from running
simultaneously (which would race on broker calls and audit-log
writes). A regression here would either silently allow duplicates
or break the operator's restart workflow with confusing errors.

We use `monkeypatch` to redirect `_PID_FILE` to a tmp_path so tests
don't pollute the repo root.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from halal_trader.trading import scheduler as scheduler_mod
from halal_trader.trading.scheduler import TradingBot


@pytest.fixture
def tmp_pid_file(monkeypatch, tmp_path: Path) -> Path:
    """Redirect the module-level `_PID_FILE` to a tmp_path."""
    pid_path = tmp_path / "halal_trader.pid"
    monkeypatch.setattr(scheduler_mod, "_PID_FILE", pid_path)
    return pid_path


# ── Acquire ────────────────────────────────────────────────


def test_acquire_lock_creates_pid_file_with_current_pid(tmp_pid_file: Path):
    """First acquire writes the current process's PID to the file."""
    bot = TradingBot()
    bot._acquire_lock()
    try:
        assert tmp_pid_file.exists()
        assert tmp_pid_file.read_text().strip() == str(os.getpid())
    finally:
        bot._release_lock()


def test_acquire_lock_sets_lock_file_descriptor(tmp_pid_file: Path):
    """The bot stores the file descriptor for later `_release_lock`."""
    bot = TradingBot()
    assert bot._lock_file is None  # initial state
    bot._acquire_lock()
    try:
        assert bot._lock_file is not None
        assert isinstance(bot._lock_file, int)
    finally:
        bot._release_lock()


def test_acquire_lock_when_already_held_raises_with_pid(tmp_pid_file: Path):
    """A second acquire on the same file raises with a message
    pointing to the existing PID — so the operator knows what to
    kill (or that the old instance is still alive)."""
    a = TradingBot()
    a._acquire_lock()
    try:
        b = TradingBot()
        with pytest.raises(RuntimeError) as exc_info:
            b._acquire_lock()
        msg = str(exc_info.value)
        assert "already running" in msg
        assert str(os.getpid()) in msg  # the holding PID
    finally:
        a._release_lock()


def test_acquire_lock_error_message_includes_remediation(tmp_pid_file: Path):
    """The error message tells the operator how to recover —
    "Remove <path> if the previous instance crashed". Pin so the
    actionable hint stays in place."""
    a = TradingBot()
    a._acquire_lock()
    try:
        b = TradingBot()
        with pytest.raises(RuntimeError, match=r"Remove .* if the previous"):
            b._acquire_lock()
    finally:
        a._release_lock()


def test_acquire_lock_with_unreadable_existing_pid_uses_unknown(tmp_pid_file: Path, monkeypatch):
    """If the existing PID file can't be read (truncated / missing
    after lock acquired), the error reports `pid=unknown` rather
    than crashing the second bot — operator still gets a helpful
    error."""
    # Acquire from `a` first.
    a = TradingBot()
    a._acquire_lock()
    try:
        # Make the PID file read raise to simulate a partial write
        # / truncation. The lock is held by `a` so `b` will hit the
        # `OSError` branch trying to acquire fcntl.
        real_open = open

        def boom(path, *args, **kwargs):
            if Path(path) == tmp_pid_file and "r" in (args[0] if args else kwargs.get("mode", "r")):
                raise PermissionError("simulated unreadable PID")
            return real_open(path, *args, **kwargs)

        # Instead, simpler: just empty the file → read returns "" → the
        # `f.read().strip()` returns "" but doesn't raise. So forge an
        # actual exception path: replace the existing PID file with a
        # symlink or directory. Easier: monkeypatch `open` itself.
        import builtins

        def patched_open(p, *args, **kwargs):
            if Path(p) == tmp_pid_file:
                raise OSError("simulated read failure")
            return real_open(p, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", patched_open)

        b = TradingBot()
        with pytest.raises(RuntimeError) as exc_info:
            b._acquire_lock()
        assert "pid=unknown" in str(exc_info.value)
    finally:
        a._release_lock()


# ── Release ────────────────────────────────────────────────


def test_release_lock_removes_pid_file(tmp_pid_file: Path):
    """Clean release deletes the PID file so the next start can
    acquire fresh."""
    bot = TradingBot()
    bot._acquire_lock()
    bot._release_lock()
    assert not tmp_pid_file.exists()


def test_release_lock_clears_file_descriptor(tmp_pid_file: Path):
    """After release, `_lock_file` is set back to None — so a
    subsequent `_acquire_lock` doesn't see stale state."""
    bot = TradingBot()
    bot._acquire_lock()
    bot._release_lock()
    assert bot._lock_file is None


def test_release_lock_when_not_held_is_noop(tmp_pid_file: Path):
    """Calling `_release_lock` without a prior acquire must NOT
    raise. Used in shutdown paths where the lock might never have
    been taken (e.g. an init-time crash before scheduler.start)."""
    bot = TradingBot()
    assert bot._lock_file is None  # never acquired
    bot._release_lock()  # must not raise
    assert bot._lock_file is None


def test_release_lock_swallows_oserror_on_close(tmp_pid_file: Path, monkeypatch):
    """If `os.close` raises (already-closed fd), release continues —
    the file unlink still runs and `_lock_file` is cleared."""
    bot = TradingBot()
    bot._acquire_lock()

    real_close = os.close

    def boom_close(fd):
        if fd == bot._lock_file:
            raise OSError("already closed")
        real_close(fd)

    monkeypatch.setattr(os, "close", boom_close)

    bot._release_lock()
    assert bot._lock_file is None
    # PID file unlink may have run anyway, but at minimum the bot
    # state is clean.


def test_acquire_release_acquire_cycle_works(tmp_pid_file: Path):
    """Operator restart: stop → release → start → acquire must work
    cleanly. Pin the round-trip so a refactor that leaves stale
    state breaks here."""
    bot = TradingBot()
    bot._acquire_lock()
    bot._release_lock()
    # Second bot acquires fresh.
    bot2 = TradingBot()
    bot2._acquire_lock()
    try:
        assert tmp_pid_file.exists()
    finally:
        bot2._release_lock()


def test_release_lock_swallows_unlink_failure(tmp_pid_file: Path, monkeypatch):
    """If unlink fails (file already gone, race condition), release
    still cleans up the descriptor."""
    bot = TradingBot()
    bot._acquire_lock()

    real_unlink = Path.unlink

    def boom_unlink(self, missing_ok=False):
        if self == tmp_pid_file:
            raise OSError("simulated unlink failure")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", boom_unlink)

    bot._release_lock()  # must not raise
    assert bot._lock_file is None
