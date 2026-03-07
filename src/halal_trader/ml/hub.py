"""Model hub — lazy-loading registry for HuggingFace and local ML models."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelHub:
    """Central registry for ML models with lazy loading and caching."""

    def __init__(self, *, device: str = "cpu", models_dir: Path | None = None) -> None:
        self._device = device
        self._models_dir = models_dir or Path("models")
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._loaded: dict[str, object] = {}

    @property
    def device(self) -> str:
        return self._device

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    def get_model(self, name: str) -> object | None:
        """Get a loaded model by name, or None if not loaded."""
        return self._loaded.get(name)

    def register(self, name: str, model: object) -> None:
        """Register an already-loaded model."""
        self._loaded[name] = model
        logger.info("Model registered: %s", name)

    def is_loaded(self, name: str) -> bool:
        return name in self._loaded

    def unload(self, name: str) -> None:
        """Unload a model to free memory."""
        if name in self._loaded:
            del self._loaded[name]
            logger.info("Model unloaded: %s", name)
