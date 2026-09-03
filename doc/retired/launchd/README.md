# Retired launchd jobs

These were installed in `~/Library/LaunchAgents/` and are kept here as a record
of what ran, not as something to reinstall.

## `com.hifi.mlx-technical` / `com.hifi.mlx-fundamental`

Removed 2026-09-03 (DJ-139). Both started `mlx_lm server` at login against
adapters the project's own evaluation had rejected:

| job | port | adapter | rejected at |
|---|---|---|---|
| `com.hifi.mlx-technical` | 1235 | `data/adapters/technical_v2` | DJ-124 — Buy @ 0.70 on 98/98 tickers, ensemble entropy 0.367 → 0.000 |
| `com.hifi.mlx-fundamental` | 1236 | `data/adapters/fundamental_v1` | DJ-058 — grounding rate 1.000 → 0.000 |

They were not idle. At the time of removal both had been up for two days and
**were listening on 127.0.0.1:1235 and :1236**. Their memory footprint was ~8 MB
each, which looks like a failed start but is not: `mlx_lm.server` loads the
model on the first request, and nothing had made one. So a health probe would
have reported the port READY and the first caller would have got the retired
adapter, fully functional.

That mattered because `scripts/run_phase15_smoke.py` probed exactly that port —
and Phase 15 is the run that has to be repeated on repaired data before any
Page-theorem claim. The rejected adapter was one health check away from voting
in the replacement for the result it had already contaminated.

Removal is safe: `hifi.live.models._AGENT_CONFIG` has had no route to 1235 or
1236 since DJ-135, and the walk-forward sweep goes through the same config.
`tests/integration/test_repo_integrity.py::TestRetiredAdaptersStayRetired`
fails if any non-archived script reintroduces one.

To reproduce the Phase 11 negative result the paper reports, the adapters are
still on disk and still servable, deliberately and explicitly:

    make finetune-serve     # advertised in `make help` as RETIRED
