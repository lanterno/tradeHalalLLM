"""Interpreter contract — observation → evidence (REARCHITECTURE L2)."""

from __future__ import annotations

from typing import Protocol

from halabot.belief.schema import EvidenceItem
from halabot.platform.events import Event, EventType


class Interpreter(Protocol):
    """Turns one observation into zero or more :class:`EvidenceItem`s.

    A failing interpreter returns an empty list (never a fabricated signal) —
    the router logs the failure and carries on (INV-1)."""

    consumes: frozenset[EventType]

    async def interpret(self, observation: Event) -> list[EvidenceItem]: ...
