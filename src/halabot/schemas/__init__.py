"""Typed event payload schemas (REARCHITECTURE Appendix A).

Payloads travel as plain dicts on :class:`Event`; these TypedDicts document and
type the shape each ``EventType`` carries. They are validated at the perception
boundary (a malformed payload is logged + dropped, never dispatched — INV-4).
"""
