# `<alert.type>` — `<one-line summary>`

**Severity:** INFO / WARN / PAGE
**Triggers when:** … (the *exact* condition the bot raises this on)
**Acknowledgement window:** N minutes — if the alert is still active
after N minutes, escalate.

## Likely causes

Most-common-first list. Keep concrete: "broker returned 503 for ≥5
consecutive cycles", not "the network is bad".

1. …
2. …
3. …

## Diagnose

Concrete commands the on-call runs to confirm which cause it is.
Each step should produce a clear go/no-go for the next.

```bash
# Example
uv run halal-trader halt-status
```

## Mitigate

In strict order. Each step links back to a cause from the diagnose
list — operator picks the matching one.

1. **If cause 1 …** — `command + expected output`.
2. **If cause 2 …** — …

## Escalate

When mitigation fails or the cause is unclear:

* @-handle on-call rotation
* Slack `#halal-trader-oncall`
* Phone of last resort: …

## Postmortem

For PAGE-severity alerts, file a postmortem in
`docs/postmortems/<date>-<alert>.md` within 48h.

---

_Last reviewed: …_
