# Phase 21 Plan: Allocation Remediation and Endpoint Re-specification

**Created:** 2026-08-27 · **Status:** PLANNED
**Execution mode:** T1–T3 are pre-cycle critical path (must land before the next
live run); T4–T6 follow in the same phase, not the same hour.
See `PHASE_21_CONTEXT.md` for the analysis, the evidence, and the design decision.

**Standing rule for this phase:** every task changes a number that has already
been wrong once. No task is complete without a test that would have failed
before it, and no fitted knob is accepted without printing its value against its
bracket (`datasaurus` G3). Refer to defects by ID in commit messages.

---

## CRITICAL PATH — must land before the next live cycle

### T1 — DJ-131: deployment must not depend on the Buy count (`src/hifi/portfolio/policy.py`)

- [ ] T1.1 Add `target_deployment: float = 1.0` to `PortfolioPolicy`. Document it
      as *the deployment decision*, distinct from the concentration decision, and
      state in the docstring that any cash held must trace to this field.
- [ ] T1.2 Redefine the ceiling so full deployment is always reachable:
      `ceil = max(ceil_max_single, target_deployment / n_candidates)`.
      Keep `ceil_max_single` as the *concentration floor-of-the-ceiling*, not as
      the deployment limit it has been acting as.
- [ ] T1.3 Update the class docstring: line 83's constant contradicted the design
      note at lines 26–32. Record that inline, where the next reader hits it —
      not only in the bitácora. (This is the DJ-124 lesson applied to ourselves.)
- [ ] T1.4 **Invariance test, written first and binding:**
      for `n` in 1..97, `max_single_stock * n >= target_deployment - tol`.
      This is the property whose violation defined the defect.
- [ ] T1.5 Knob-bracket test: assert `concentration` is the binding constraint
      across the operating range, i.e. the cap is NOT pinned to `ceil_max_single`
      for every realistic `n`. Pin the boundary `n` explicitly so a future edit
      that re-pins the knob fails loudly.
- [ ] T1.6 Regression fixture from the real event: `n=7`, convictions
      `[0.6800, 0.6623, 0.4459, 0.4416, 0.4359, 0.4286, 0.4217]` must produce a
      dollar spread > 1.0 and deployment >= 99%. The 2026-08-24 arm-A basket is
      the test case; it must never reproduce.

### T2 — C17: the cap chain must reach a fixed point (`src/hifi/mcp/portfolio_composer.py`)

- [ ] T2.1 Replace the one-pass `stock cap -> sector cap -> min position` chain
      (steps 3–5, lines ~296–303) with an iterate-to-fixed-point loop over the
      three constraints, bounded iterations, converging on max weight change.
- [ ] T2.2 Post-condition assertion on the returned weights: no weight exceeds
      `max_single_stock`; no sector exceeds `max_sector`; the residual is either
      allocated or attributable to `target_deployment`. Raise, do not log — a
      silent breach here is what produced the audit finding.
- [ ] T2.3 Non-convergence must fail closed (raise), never return a partially
      capped book.
- [ ] T2.4 Property test by fuzzing: random conviction vectors x sector maps;
      assert the post-conditions hold for every draw. The original C17 finding
      was found by fuzzing and must stay found.

### T3 — DJ-132: endpoint re-specification, recorded before further inspection

- [ ] T3.1 Write `doc/OSF_AMENDMENT_002.md`: primary endpoint moves from
      arm-level economic performance to the diversity-mechanism test (κ,
      disagreement entropy, IC); economic performance demoted to secondary and
      labelled exploratory. State the power argument and the 4.64% vs 30.10%
      simulation as the justification.
- [ ] T3.2 Fold in the amendment items already accrued and unfiled: DJ-120/121/
      122/123/124 defect cluster and the A/B/D restart; DJ-128 null ablation;
      DJ-129 hardening and the `client_order_id` schema boundary; the Genesis-II
      void period 2026-08-14 -> 2026-08-24 for **all** arms; DJ-130 context
      injection as a protocol boundary dated 2026-08-25; DJ-131 allocation
      change dated by this phase's landing.
- [ ] T3.3 Declare the primary analysis specification concretely enough to be
      falsifiable: statistic, unit of observation, how effective n is computed
      given cross-sectional correlation, and the multiple-comparison correction
      across arms.
- [ ] T3.4 **Alberto files the amendment on OSF.** Cannot be delegated; blocks
      nothing in code but blocks any claim from the live record.

### T4 — T6.6: the regime label has been a constant (`src/hifi/agents/ensemble_runner.py`)

- [ ] T4.1 Repoint `_get_regime_label` (lines 660–663) at the post-DJ-120 layout.
      Verified today: `data/market/SPY.parquet` and `data/macro/macro.parquet`
      do not exist; the live layout is `data/market/SPY/`. Prefer reusing
      `engines/market_summary.py`'s already-corrected regime snapshot over a
      second path-resolution implementation.
- [ ] T4.2 Delete the bare `except Exception: return "neutral"`. Distinguish
      *"regime genuinely indeterminate"* from *"the data path is broken"*, and
      log a warning naming the path for the second — the exact remedy applied to
      `forward_return_from_ohlcv` after DJ-120.
- [ ] T4.3 Quantify the blast radius before deciding whether it is disclosable:
      how many stored decisions carry `regime="neutral"` because of this, and
      does any debate prompt or episodic-retrieval filter consume the label? If
      the label reached prompts, it is a protocol disclosure, not a bug fix.
- [ ] T4.4 Test that pins a non-neutral regime resolving from the real layout,
      and a second that pins a broken path producing a warning rather than a
      silent `"neutral"`.

---

## FOLLOW-ON — same phase, after the critical path lands

### T5 — Deterministic-layer corrections

- [ ] T5.1 **C15** (`risk_manager.compute_portfolio_var`): replace positional
      truncation (`r[-min_len:]`) with date-intersection alignment, and
      renormalise weights over the tickers that survive alignment. Both current
      behaviours — positional splice, silent weight drop — are already pinned as
      forbidden in `engines/market_summary.py`'s tests (Phase 20 T6.3); reuse
      that logic rather than writing a third variant.
- [ ] T5.2 **C18**: rename `var_95_20d` to name the horizon it actually has (one
      day). Grep every consumer; a rename that misses a call site converts a
      naming defect into a runtime defect.
- [ ] T5.3 **D22**: `model_config = ConfigDict(extra="forbid")` in
      `src/hifi/config/loader.py`, so a typo'd key fails loudly instead of
      silently defaulting. Run the full config set afterwards — this change is
      expected to surface existing typos, and each one found is the point.
- [ ] T5.4 **D23**: remove `SafetyConfig` dead code, or wire it. Decide and
      record which; leaving it is what made it dead.
- [ ] T5.5 Tighten `scripts/simulate_next_run.py`'s drift tolerance now that the
      caps are correct (Phase 19 deferred queue item), and add a deployment
      check that would have caught DJ-131: exposure must reach
      `target_deployment` or name the rule that stopped it.

### T6 — Verification, numbering, landing

- [ ] T6.1 `pytest -q --tb=no` green; `ruff check --output-format=concise` clean
      on touched files.
- [ ] T6.2 `uv run python scripts/simulate_next_run.py` — read-only preview
      against live state. Required evidence before the next cycle: arm A's
      projected exposure >= 99% and its projected dollar spread > 1.0.
- [ ] T6.3 Update `plans/STATUS.md`: refresh from 2026-07-13, record phases 19
      and 20 as COMPLETE, record the 17/18 -> 22/23 remap and why 19/20 keep
      their numbers (tag and bitácora immutability).
- [ ] T6.4 Update `doc/HIFI_PROTOCOL_V1.md` phase table to match the remap, with
      a dated note that the divergence is recorded rather than rewritten.
- [ ] T6.5 Bitácora: `doc/bitacora/PHASE_21_ALLOCATION_REMEDIATION.md` — DJ-131
      with the knob-bracket table, DJ-132 with the false-discovery simulation,
      and the C-series closures. Include §1.4 of the context: this is the second
      habitat of the DJ-124 pattern, and that generalisation is the finding
      worth more than any individual fix.
- [ ] T6.6 Commit citing DJ-131/DJ-132/C15/C17/C18/D22/D23/T6.6. Tag
      `phase21-allocation-remediation`. PR to main.

---

## Acceptance

1. `max_single_stock * n >= target_deployment` holds for every `n` in 1..97 —
   deployment is no longer a function of the treatment.
2. The 2026-08-24 arm-A basket, replayed, produces a dollar spread > 1.0 and
   >= 99% deployment.
3. `concentration` is the binding constraint across the operating range, printed
   with its bracket in the bitácora.
4. Composer post-conditions hold under fuzzing; non-convergence raises.
5. `_get_regime_label` resolves a real regime from the live layout, and a broken
   path produces a named warning instead of `"neutral"`.
6. OSF amendment 002 filed, with the primary endpoint re-specified **before** any
   further inspection of live outcomes.
7. The next live cycle runs on this code. No cycle runs on the old allocator.

## Explicitly deferred (Phase 19 order preserved)

E2 fill/slippage reconciliation -> B7–B10 leakage cluster -> Phase 15 re-run
(also blocked on the DJ-128 redesign) -> intent-row resume semantics -> M3
fail-open branch.
