# Dashboard mockups: the next two pages to build

Design proposals made on 2026-10-05/06, recovered from that WSL session's
transcript on 2026-10-08 (the HTML that generated them was lost; these are the
screenshots it viewed). Their figures are from the paper accounts on the day
each was made.

| Mockup | Replaces | Build notes |
|---|---|---|
| `core-v2-desktop.jpg`, `core-v2-phone.png` | The Core portfolio page | Holdings grouped by sector with drift against target and each band; every run with its orders and fill quality (vs arrival, vs close); the live-money gate as a dated timeline. The data is all served already: `/api/core`, `/api/risk/core`, `/api/positions`, `core_orders`, `core_runs`. |
| `operations-v2-desktop.jpg` | Observability and System, as one page | Daily jobs on a timeline judged by the trading calendar, processes by heartbeat age, data freshness, LLM spend by pool, database and backups. Drawn before 2026-10-08: the day-trader is no longer retired, `DAY_TRADER_ENABLED` and the other strategy switches are gone, and every daily job now beats (the "not watched" warnings are fixed). Keep the layout, not those rows. |

The Home and Belief Board mockups were built (2026-10) and removed from here.
The day-trader archive was removed: it presented the day-trader as retired,
which the operator reversed on 2026-10-08.
