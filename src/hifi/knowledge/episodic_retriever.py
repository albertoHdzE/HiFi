"""
EpisodicRetriever: formats episodic memory for injection into agent prompts (E5-T3, DJ-092).

Wraps EpisodicStore and produces a human-readable prefix block of past successful
decisions in similar market conditions.  The block is prepended to the agent's
memory_prefix in run_sequential_ensemble().

Temporal discipline: no episode with decision_date >= as_of_date is returned,
enforcing look-ahead-free retrieval for walk-forward evaluation (Phase 15).
"""

from __future__ import annotations

import logging

from hifi.knowledge.episodic_store import EpisodeRecord, EpisodicStore

logger = logging.getLogger(__name__)


class EpisodicRetriever:
    """
    Retrieves formatted episodic memory for injection into agent prompts.

    Parameters
    ----------
    store : EpisodicStore
        Connected EpisodicStore instance.
    """

    def __init__(self, store: EpisodicStore) -> None:
        self._store = store

    def retrieve(
        self,
        ticker: str,
        date: str,
        agent_type: str,
        regime: str,
        sector: str,
        n: int = 3,
    ) -> str:
        """
        Return a formatted episodic memory prefix for injection.

        Parameters
        ----------
        ticker : str
            Current ticker being analyzed.
        date : str
            As-of date for the current analysis (ISO 8601).
            No episode with decision_date >= date is returned.
        agent_type : str
            Current agent type (for logging; not used as filter).
        regime : str
            Current market regime label.
        sector : str
            GICS sector of the ticker.
        n : int
            Maximum number of episodes to include.

        Returns
        -------
        str
            Formatted memory prefix string, or "" if no episodes found.
        """
        episodes = self._store.search(
            ticker=ticker,
            regime=regime,
            sector=sector,
            outcome_correct=True,
            n=n * 2,  # fetch extra for temporal filtering
        )

        # Temporal discipline: exclude future-date episodes
        episodes = [e for e in episodes if e.decision_date < date]

        if not episodes:
            logger.debug(
                "No episodic memory for %s %s agent=%s regime=%s sector=%s",
                ticker, date, agent_type, regime, sector,
            )
            return ""

        episodes = episodes[:n]
        return _format_episodes(episodes, n=len(episodes))


def _format_episodes(episodes: list[EpisodeRecord], n: int) -> str:
    """
    Format a list of EpisodeRecords as a structured memory prefix block.

    Example output::

        [Episodic Memory — 3 successful past decisions in similar conditions]
        Date: 2022-06-30, Ticker: AAPL, Regime: bull_low_vol
        Decision: Buy (conf=0.75) — Strong earnings growth and solid margins.
        Outcome: CORRECT (60d return: +8.0%)

    """
    lines = [
        f"[Episodic Memory — {n} successful past decision{'s' if n != 1 else ''}"
        f" in similar conditions]"
    ]
    for ep in episodes:
        lines.append(
            f"Date: {ep.decision_date}, Ticker: {ep.ticker}, Regime: {ep.regime_label}"
        )
        summary = (ep.reasoning_summary or "").replace("\n", " ")[:200]
        lines.append(
            f"Decision: {ep.decision} (conf={ep.confidence:.2f}) — {summary}"
        )
        if ep.forward_return is not None:
            lines.append(f"Outcome: CORRECT (60d return: {ep.forward_return:+.1%})")
        else:
            lines.append("Outcome: CORRECT")
    return "\n".join(lines)
