"""Centralized logging configuration: Rich console + JSON file output."""

import logging
from logging.handlers import RotatingFileHandler

from pythonjsonlogger.json import JsonFormatter  # type: ignore[import-untyped]
from rich.console import Console
from rich.logging import RichHandler

from halal_trader.config import Settings

console = Console()


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

    # ── Console handler (existing Rich behaviour) ──────────────
    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
    )
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))

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
