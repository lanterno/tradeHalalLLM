"""cognition — interpret observations into evidence (REARCHITECTURE L2).

Where *understanding* is manufactured. Split into cheap/continuous deterministic
interpreters (indicators, regime, news lexicon) that run on every observation
with NO LLM — these carry the system when the LLM is down (INV-1) — and the
sparse LLM thesis writer invoked only on a material shift (in the belief
updater). The :class:`CognitionRouter` wires interpreters to the bus and feeds
their evidence to the belief updater, so beliefs form continuously from live
events with no fixed cycle.
"""
