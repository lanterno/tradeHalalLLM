"""Centralized logging configuration: Rich console + JSON file output."""

import logging
from logging.handlers import RotatingFileHandler

from pythonjsonlogger.json import JsonFormatter  # type: ignore[import-untyped]
from rich.console import Console
from rich.logging import RichHandler

from halal_trader.config import Settings

console = Console()


class SafeRichHandler(RichHandler):
    """RichHandler that silently degrades on broken pipes.

    When the bot runs headless (e.g. via nohup / systemd), stdout may be
    closed, causing Rich to raise ``BrokenPipeError`` → ``SystemExit(1)``.
    This wrapper catches those exceptions so the bot keeps running and
    file logging continues uninterrupted.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        except BrokenPipeError, SystemExit, OSError:
            # Console output is gone — nothing we can do, but don't
            # let it crash the process.  File handlers still work.
            pass


# Third-party loggers whose INFO chatter we suppress on the console.
# They still appear in the JSON log file at whatever level the file handler allows.
_NOISY_LOGGERS = frozenset(
    {
        "apscheduler",
        "httpcore",
        "httpx",
        "aiosqlite",
        "asyncio",
        "mcp",
    }
)


class ThirdPartyConsoleFilter(logging.Filter):
    """Drop INFO (and below) records from noisy third-party loggers.

    Only WARNING and above from these libraries reach the terminal, keeping
    the console output focused on application messages.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Allow WARNING+ from any logger
        if record.levelno >= logging.WARNING:
            return True
        # Block INFO/DEBUG from noisy third-party libraries
        top_level = record.name.split(".")[0]
        if top_level in _NOISY_LOGGERS:
            return False
        return True


def setup_logging(settings: Settings, *, cli_log_level: str | None = None) -> None:
    """Configure dual-output logging: Rich console + JSON rotating log files.

    Args:
        settings: Application settings (provides log_dir, levels, rotation config).
        cli_log_level: Optional CLI override for the console log level.
    """
    # Create log directory
    settings.log_dir.mkdir(parents=True, exist_ok=True)

    # Resolve console level (CLI flag takes priority)
    console_level = (cli_log_level or settings.log_level).upper()

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # Capture everything; handlers filter by level

    # Remove any previously attached handlers (prevents duplicates on re-init)
    root.handlers.clear()

    # ── Console handler (Safe Rich wrapper — tolerates broken pipes) ──
    console_handler = SafeRichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
    )
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    console_handler.addFilter(ThirdPartyConsoleFilter())

    # ── JSON file handler – all logs ──────────────────────────
    json_formatter = JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s %(funcName)s %(lineno)d",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
    )

    file_handler = RotatingFileHandler(
        settings.log_dir / "halal_trader.log",
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(settings.log_file_level.upper())
    file_handler.setFormatter(json_formatter)
    file_handler.addFilter(ThirdPartyConsoleFilter())

    # ── Error-only file handler ───────────────────────────────
    error_handler = RotatingFileHandler(
        settings.log_dir / "error.log",
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(json_formatter)

    root.addHandler(console_handler)
    root.addHandler(file_handler)
    root.addHandler(error_handler)

    # Silence extremely noisy third-party loggers at source level.
    # These produce thousands of DEBUG messages per minute (WebSocket frames)
    # that drown out useful application logs even in the JSON file.
    for name in ("binance", "websockets", "aiosqlite"):
        logging.getLogger(name).setLevel(logging.WARNING)
