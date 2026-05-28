"""perception — turn the outside world into typed observation.* events (L1).

Perception reports *facts*; it does not interpret (that's cognition). Each
:class:`Source` is a long-lived adapter (wrapping a feed/collector) supervised
with restart-on-error: a source failure emits nothing and self-restarts, never a
degraded/placeholder observation downstream could mistake for signal (INV-2).
Read-only: perception never trades.
"""
