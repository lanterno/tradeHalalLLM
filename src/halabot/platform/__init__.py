"""platform — the engine's foundation: clock, events, durable bus, supervision.

Everything in ``halabot`` depends on ``platform`` and ``platform`` depends on
nothing else in the engine. It provides the spine (the durable event log +
bus), the injectable clock (so tests and replay control time — INV-6), and
the task supervisor.
"""
