# OSF Pre-registration Amendment 001 — circuit-breaker rescaling

**Study:** HiFi Phase 16 — four-arm live paper-trading ablation of LLM ensemble
architectures (DJ-111).
**Filed:** 2026-07-28, before the 2026-07-29 decision cycle.
**Author:** Alberto Espinosa
**Commit implementing the change:** see `DJ-119` in the repository history.

Paste-ready for the OSF registration amendment field. It is written to be read
by someone who has not seen the code.

---

## 1. What changed

The live experiment's risk circuit breaker had two independent halt conditions.
One is unchanged: a **2% single-day portfolio loss** halts the account.

The second was: **any single position down more than 10% against cost basis
halts the entire account.** That condition is replaced by a two-part test which
halts only when a position is both

1. down more than 10% against cost basis (unchanged), **and**
2. costing the portfolio more than 2% of equity, where the cost is

   `impact = |pnl_pct| × (position market value / account equity)`.

A position that satisfies (1) but not (2) is now recorded to the breaker log
with `action="flag"` and does not stop the arm. No observation is discarded.

## 2. Why — a structural argument that does not reference outcomes

A per-position threshold has halt probability `1 − (1 − p)^N` in the number of
open positions `N`, where `p` is the per-position probability of breaching the
threshold on a given day. For a wide book this tends to 1 regardless of how the
strategy is performing.

The four arms differ in book width by roughly two orders of magnitude, and
width is **downstream of the treatment under test**: the hypothesis (Page's
diversity-prediction theorem applied to LLM ensembles) predicts that diverse
parallel ensembles spread conviction across many names while herding sequential
ones concentrate it. The rule therefore imposed a constraint monotone in the
very quantity the experiment manipulates. It was not a risk control; it was an
uncontrolled treatment-correlated covariate.

The replacement statistic is **extensive** rather than marginal: contribution to
portfolio return is additive across positions, so the test is invariant to `N`
by construction. For an equal-weight `N`-name book the effective per-position
threshold is `2% × N`; at full concentration the weight is 1 and the rule
collapses to the same 2% bound as the daily portfolio limit, so the two halt
conditions agree at the boundary instead of contradicting each other.

This argument is derived entirely from the *cardinality* of each arm's book. It
does not reference any arm's realised returns, and the change was not chosen
after inspecting performance.

## 3. Observed effect before the change (full disclosure)

The old rule halted arm C — the non-LLM equal-weight control, ~98 positions —
on five consecutive decision cycles: 2026-07-22 (DHR), 07-24 (TSLA), 07-25
(META), and twice on 07-28 (META, TSLA). Arms A, B and D were never halted.

**The halts prevented no trades.** Arm C is a buy-once-and-hold null model: it
had already established all 98 positions on 2026-07-16 and its rule emits
orders only for tickers with no existing position. It placed zero orders on
07-16 (second cycle) and 07-20 while un-halted, and would have placed zero on
the halted days. Its equity curve is unaffected.

The halts did suppress **telemetry**: the halt returned before the daily
equity/positions capture, so arm C's stored portfolio history froze at
2026-07-17 while the other arms ran to 07-27. Because that history is fetched
from the broker, which retains it server-side, it was fully recovered on
2026-07-28 with no loss. The halt path now captures financial state before
returning.

## 4. Uniformity and timing

- The new rule is applied identically to all four arms; no arm-specific
  parameters exist.
- The change is filed before the next decision cycle (2026-07-29) and before
  any forward-return labels exist for the affected period.
- Halted days remain in the record as a covariate; they are not deleted. The
  reporting layer exposes them via `halted_days()` and treats them as censored
  intervals rather than missing data.

## 5. Related disclosure — exposure heterogeneity

Independently of the breaker, the arms differ substantially in capital
deployment. As of 2026-07-28: A 5.0% invested (1 position), B 4.9% (1), D 40.6%
(9), C 99.2% (98).

At this spread, raw return, Sharpe ratio and maximum drawdown are not
comparable across arms — they are dominated by exposure rather than signal
quality. **The primary outcome measures for the diversity hypothesis are
therefore the Information Coefficient and the herding metrics (unanimity,
cross-agent disagreement entropy), which are computed upstream of position
sizing and execution and are invariant to deployment.** Equity curves are
reported as a secondary, illustrative exhibit accompanied by an exposure column
and an exposure-adjusted return series.

This is a clarification of emphasis, not a change of hypothesis: the
pre-registered claim was always about signal quality, and the Phase 15
walk-forward result it rests on (parallel IC = +0.0642, p = 0.0019, herding
0.000; full IC = +0.0232, herding 0.361; homogeneous IC = −0.0428, p = 0.038,
herding 0.862) is an IC/herding result throughout.

## 6. Other protocol deviations in the same period

- **2026-07-27 (Mon):** no decision cycle was run (operator omission). Missing
  day, not a zero-order day.
- **2026-07-28:** two cycles ran. The first was launched at 09:34 ET, after the
  market open, so it decided on a partial daily bar and its orders filled
  intraday rather than at the next open. The second, that evening, reused the
  cached ensemble (all agents skipped, LLM arms reproduced their signals) but
  arm D re-derived against updated portfolio state and placed 4 additional
  sell orders on top of the morning's 2. Arms A and B placed 0 in the second
  cycle. Arms A, B and D each hold two decision records for this date.

Both failure modes are now blocked in code: launches inside 09:30–16:00 ET are
refused, and a second cycle for an account that already decided on a date is
skipped unless explicitly forced.
