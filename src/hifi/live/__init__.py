"""The live experiment: one nightly cycle per arm, and the walk-forward sweep.

This package holds the code that runs the Phase 16 experiment. It used to live
in ``scripts/run_phase16_live.py`` and ``scripts/run_phase15_orchestrator.py``,
with the first reaching the second through a ``sys.path`` insert — so the
production ensemble ran out of a file named after an evaluation phase that had
been retracted, and none of it could be imported by a test without the same
trick (DJ-135). The scripts remain as thin CLIs over this package.

Reading order:

    paths       where everything lives
    accounts    the four arms, their credentials and per-arm state
    market      OHLCV refresh, the session date, last closes
    models      which model each agent runs on
    ensemble    the six agent passes and their aggregation
    guards      everything that can stop or qualify a cycle
    strategies  arm C, the equal-weight null model
    cycle       one arm, one evening, end to end
    walkforward the offline multi-date sweep
"""

from __future__ import annotations

__all__ = ["cycle", "guards", "market", "models", "paths", "walkforward"]
