"""Daily news headlines for the sentiment agent (DJ-133b).

Why this exists
---------------
The sentiment agent's only information source was SEC filings, which update
quarterly, while it votes daily. Measured on the live record: the context
retrieved for AAPL on 2026-08-24 and on 2026-08-27 differed by exactly one
character -- the echoed ``as_of=`` in the header -- and the 8,265-character
filing body was byte-identical. For roughly sixty trading days out of every
quarter the agent's input does not change at all, so it cannot produce a
time-varying signal. It voted Hold on 97/97 tickers on 2026-08-27.

That is not a bug in the retrieval code; the retrieval was working. It is a
mismatch between an information source's update frequency and the decision
frequency it is being asked to support. The repair is to give the agent
something that actually varies daily.

Point-in-time discipline
------------------------
Articles are filtered on ``created_at``, the publication timestamp, against the
end of the decision date in US/Eastern. Nothing published after the decision
was made can enter the context. This is enforced here rather than trusted to
the API's own ``end`` parameter, because a silent timezone or inclusivity
difference at the vendor would be indistinguishable from working code -- the
failure mode that produced DJ-133a.

Reproducibility
---------------
Every fetch is cached to ``data/news/<TICKER>/<DATE>.json``. A re-run of a past
decision date replays the cached articles rather than re-querying, so a stored
decision can be audited against the exact text the agent saw. Without this the
research record would not be reconstructible: the vendor's index changes.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["NewsArticle", "fetch_news", "format_news_block"]

#: Calendar days of history to include.
#:
#: There is a real tension here and it is worth stating: a longer window raises
#: coverage but increases the overlap between consecutive decisions, and
#: day-to-day variation is the entire reason this source exists. Measured on
#: 2026-08-30 across the live universe, a 6-day window left 38 of 97 tickers
#: with no news at all, which would have kept the sentiment agent constant for
#: 39% of the book -- a half-fix. Ten days materially improves coverage while
#: the newest-first cap below keeps the block turning over as stories land.
DEFAULT_LOOKBACK_DAYS = 10

#: Headlines per decision. The block shares a context window with the filing
#: excerpt and the market block, so this is a budget, not a preference.
DEFAULT_MAX_ARTICLES = 12


class NewsArticle:
    """One headline with the timestamp that makes point-in-time filtering possible."""

    __slots__ = ("created_at", "headline", "source", "summary", "symbols")

    def __init__(
        self,
        created_at: str,
        headline: str,
        source: str = "",
        summary: str = "",
        symbols: list[str] | None = None,
    ) -> None:
        self.created_at = created_at
        self.headline = headline
        self.source = source
        self.summary = summary
        self.symbols = symbols or []

    def to_dict(self) -> dict:
        return {
            "created_at": self.created_at,
            "headline": self.headline,
            "source": self.source,
            "summary": self.summary,
            "symbols": self.symbols,
        }

    @classmethod
    def from_dict(cls, d: dict) -> NewsArticle:
        return cls(
            created_at=d.get("created_at", ""),
            headline=d.get("headline", ""),
            source=d.get("source", ""),
            summary=d.get("summary", ""),
            symbols=d.get("symbols") or [],
        )


def _cache_path(ticker: str, as_of_date: str, data_dir: str | Path) -> Path:
    return Path(data_dir) / "news" / ticker.upper() / f"{as_of_date}.json"


def _credentials() -> tuple[str, str] | None:
    """Alpaca market-data credentials.

    News is market data, not trading, so any of the four arm key pairs works
    and none of them is privileged. The first configured pair is used.
    """
    for suffix in ("FIRST", "SECOND", "THIRD", "FOURTH"):
        key = os.environ.get(f"ALPACA_API_KEY_{suffix}")
        secret = os.environ.get(f"ALPACA_SECRET_{suffix}")
        if key and secret:
            return key, secret
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if key and secret:
        return key, secret
    return None


#: Regular-session close in US/Eastern. The information cutoff, not midnight.
_MARKET_CLOSE_HOUR = 16


def _cutoff(as_of_date: str) -> datetime:
    """Regular-session close on the decision date, as an aware UTC datetime.

    The cutoff is 16:00 US/Eastern, not the end of the calendar day. The arm
    values its book on that day's closing price, so anything published after
    the close is information the decision could not have had.

    This is not hypothetical. The first version of this module cut off at
    midnight Eastern and immediately admitted an after-hours story published
    20:30 ET into that same day's context. It looked correct in every summary
    -- the article really was published on the decision date -- and would have
    been a lookahead finding in review.
    """
    from zoneinfo import ZoneInfo

    eastern = ZoneInfo("America/New_York")
    day = datetime.strptime(as_of_date, "%Y-%m-%d")
    close = day.replace(hour=_MARKET_CLOSE_HOUR, tzinfo=eastern)
    return close.astimezone(ZoneInfo("UTC"))


def _eastern_day(iso_ts: str) -> str:
    """Calendar date of a timestamp in US/Eastern, for display.

    Rendering a UTC timestamp's date makes a 20:30 ET story read as the
    following day, which looks exactly like leakage in the audit record even
    when the filtering is right.
    """
    from zoneinfo import ZoneInfo

    try:
        return (
            datetime.fromisoformat(iso_ts)
            .astimezone(ZoneInfo("America/New_York"))
            .strftime("%Y-%m-%d")
        )
    except Exception:
        return iso_ts[:10]


def fetch_news(
    ticker: str,
    as_of_date: str,
    data_dir: str | Path = "data",
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    max_articles: int = DEFAULT_MAX_ARTICLES,
    use_cache: bool = True,
) -> list[NewsArticle]:
    """Headlines for ``ticker`` published on or before ``as_of_date``.

    Returns [] on any failure, and logs it. The sentiment agent already has a
    documented fail-open path; a news outage must degrade it, not crash a
    trading night (DJ-123).
    """
    cache = _cache_path(ticker, as_of_date, data_dir)
    if use_cache and cache.exists():
        try:
            raw = json.loads(cache.read_text())
            cached = [NewsArticle.from_dict(a) for a in raw.get("articles", [])]
            # Sort here too: the ordering guarantee must not depend on whether
            # the caller hit the cache or the vendor.
            cached.sort(key=lambda a: a.created_at, reverse=True)
            return cached[:max_articles]
        except Exception as exc:
            logger.warning("News cache unreadable for %s %s: %s", ticker, as_of_date, exc)

    # Cache-only mode. Two uses: replaying a past decision date exactly as it
    # was recorded, and keeping the test suite hermetic. Without it a unit test
    # that passes an empty tmp_path falls through to a live vendor call
    # whenever ambient credentials happen to be set -- which is how this was
    # found.
    if os.environ.get("HIFI_NEWS_OFFLINE"):
        logger.debug("HIFI_NEWS_OFFLINE set; no news for %s @ %s", ticker, as_of_date)
        return []

    creds = _credentials()
    if creds is None:
        logger.warning("No Alpaca credentials in environment; news unavailable")
        return []

    try:
        from alpaca.data.historical.news import NewsClient
        from alpaca.data.requests import NewsRequest

        from hifi.execution.alpaca_types import model

        cutoff = _cutoff(as_of_date)
        start = cutoff - timedelta(days=lookback_days + 1)
        client = NewsClient(api_key=creds[0], secret_key=creds[1])
        response = client.get_news(
            NewsRequest(
                symbols=ticker.upper(),
                start=start,
                end=cutoff,
                limit=max_articles * 4,  # headroom for the cutoff filter below
                include_content=False,
                # A headline with no body is still information -- often the
                # most timely kind. Excluding these silently zeroed tickers
                # that did have coverage (Disney and PepsiCo both lost their
                # only story in a 6-day window to this flag).
                exclude_contentless=False,
            )
        )
        raw = model(response).data.get("news", [])
    except Exception as exc:
        logger.warning("News fetch failed for %s @ %s: %s", ticker, as_of_date, exc)
        return []

    articles: list[NewsArticle] = []
    for item in raw:
        created = getattr(item, "created_at", None)
        if created is None:
            continue
        # Enforce the cutoff here rather than trusting the vendor's `end`.
        if created > cutoff:
            continue
        articles.append(
            NewsArticle(
                created_at=created.isoformat(),
                headline=(getattr(item, "headline", "") or "").strip(),
                source=getattr(item, "source", "") or "",
                summary=(getattr(item, "summary", "") or "").strip(),
                symbols=list(getattr(item, "symbols", []) or []),
            )
        )

    articles.sort(key=lambda a: a.created_at, reverse=True)
    articles = articles[:max_articles]

    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({
            "ticker": ticker.upper(),
            "as_of_date": as_of_date,
            "cutoff_utc": cutoff.isoformat(),
            "lookback_days": lookback_days,
            "fetched_at": datetime.now().astimezone().isoformat(),
            "articles": [a.to_dict() for a in articles],
        }, indent=1))
    except Exception as exc:
        logger.warning("Could not cache news for %s %s: %s", ticker, as_of_date, exc)

    return articles


def format_news_block(
    ticker: str,
    as_of_date: str,
    articles: list[NewsArticle],
) -> str:
    """Render headlines for the prompt, or "" when there are none.

    Returning "" rather than a "no news" placeholder is deliberate: an empty
    string leaves the agent's existing insufficient-data path intact, whereas a
    placeholder would be one more constant string for it to vote on.
    """
    if not articles:
        return ""
    lines = [
        f"[NEWS — {ticker} — {len(articles)} item(s) published on or before "
        f"the {as_of_date} close]"
    ]
    for a in articles:
        day = _eastern_day(a.created_at)
        lines.append(f"- ({day}) {a.headline}" + (f" [{a.source}]" if a.source else ""))
        if a.summary:
            lines.append(f"    {a.summary[:300]}")
    return "\n".join(lines)
