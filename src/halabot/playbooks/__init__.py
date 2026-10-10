"""Playbooks and the minute simulator that tests them (sim-1).

A playbook is a pure state machine over minute bars and news
(``playbook.Playbook``); the simulator (``sim.py``) runs it against stored
SIP minute bars with the live constraints: bars reach it only once visible
on the chosen feed (``clock.py``), orders pass the same admission rules as
the bot (``rules.py``: halal at the order boundary, fail closed; long-only)
and fill by a fixed market-order rule (``exchange.py``). Paths load behind a
window guard that keeps pre-registered windows closed until the ledger says
they may open (``loader.py``); results go to ``hb_playbook_*``
(``records.py``). ``legacy.py`` holds the gate-only fill models that
replicate the legacy studies (Phase 0 gates R1 and S1); no trial uses them.
``lookahead.py`` is the look-ahead harness (spec §E.1) that gate G1 and the
simulator's tests run.
What a playbook must do is the contract at the top of ``playbook.py``.

Research only: nothing here places an order, and nothing imports
``halabot.execution``.
"""
