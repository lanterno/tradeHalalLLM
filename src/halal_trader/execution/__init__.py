"""Broker-facing code: the read-only Alpaca REST client and the broker ledger.

This package is where the plan's execution layer grows (typed broker port,
order gateway, alpaca-py adapter). It starts with what the books need: the
broker's own record of fills and equity, which is the ledger of truth.
"""
