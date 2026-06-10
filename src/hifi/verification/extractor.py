"""
Claim extractor for the HiFi verification layer (P5-E2).

Extracts all numerical claims of the form "<field_reference> of <value>"
from an agent rationale string and maps each field reference to a canonical
MCP field name via FIELD_ALIAS_TABLE. Claims whose alias is not in the table
are returned with canonical_field=None (status becomes "unresolvable" in
the verifier -- not a hallucination signal, a coverage gap in the table).

Design decision (DJ-019)
------------------------
Regex + alias table was chosen over an LLM-based extractor because the
agents were designed against a specific citation format: prompts instruct
them to write "RSI of 42.1", "P/E of 28.3", etc. Regex is the correct tool
when the text was generated against a known format specification. Using a
second LLM to verify the first introduces a second source of inference error
in the verification chain.

If the unresolvable rate on the Phase 3/4 baselines exceeds 10%, the LLM
extractor option is revisited (DJ-019, measured at P5-E6-T3).

FIELD_ALIAS_TABLE coverage goal
---------------------------------
Unresolvable rate < 10% on Phase 3 and Phase 4 baselines. The table is a
living artefact: every new unresolvable pattern observed in a baseline run
is a candidate for extension.

Primary regex pattern
---------------------
The pattern captures "<field_alias> of <numeric_value>". Field aliases may
be one or more words (up to five tokens), where each word may contain
letters, digits, underscores, slashes, and hyphens. The value may be
negative and may include a decimal component.

False positives (e.g. "one of 5 factors") produce NumericalClaim objects
with canonical_field=None (unresolvable), which do not affect the
hallucination rate. Coverage measurement in P5-E6 tracks whether the false-
positive rate is excessive.
"""

from __future__ import annotations

import re

from hifi.verification.schemas import NumericalClaim

# ---------------------------------------------------------------------------
# Field alias table
# ---------------------------------------------------------------------------

# Maps normalised alias strings (lowercase, stripped) to canonical MCP field
# names as returned by the Phase 2 engine tools.
FIELD_ALIAS_TABLE: dict[str, str] = {
    # ---- FinancialRatioResult (get_financial_ratios) -----------------------
    "p/e": "pe",
    "pe": "pe",
    "p/e ratio": "pe",
    "price-to-earnings": "pe",
    "price to earnings": "pe",
    "p/b": "pb",
    "pb": "pb",
    "price-to-book": "pb",
    "price to book": "pb",
    "p/s": "ps",
    "ps": "ps",
    "price-to-sales": "ps",
    "price to sales": "ps",
    "ev/ebitda": "ev_ebitda",
    "ev ebitda": "ev_ebitda",
    "roe": "roe",
    "return on equity": "roe",
    "roa": "roa",
    "return on assets": "roa",
    "debt/equity": "debt_equity",
    "debt-to-equity": "debt_equity",
    "debt to equity": "debt_equity",
    "d/e": "debt_equity",
    "current ratio": "current_ratio",
    # ---- GrowthMetricsResult (get_growth_metrics) --------------------------
    "revenue growth": "revenue_growth_yoy",
    "revenue growth yoy": "revenue_growth_yoy",
    "earnings growth": "earnings_growth_yoy",
    "earnings growth yoy": "earnings_growth_yoy",
    "gross margin": "gross_margin",
    "operating margin": "operating_margin",
    "net margin": "net_margin",
    # ---- ValuationResult (get_valuation_context) ---------------------------
    "current p/e": "current_pe",
    "current pe": "current_pe",
    "pe 1y percentile": "pe_1y_percentile",
    "pe percentile": "pe_1y_percentile",
    "p/e percentile": "pe_1y_percentile",
    "price to 52w high": "price_to_52w_high",
    "price to 52w low": "price_to_52w_low",
    # ---- MacroSnapshotResult (get_macro_snapshot) --------------------------
    "fed funds": "fed_funds_rate",
    "fed funds rate": "fed_funds_rate",
    "federal funds": "fed_funds_rate",
    "federal funds rate": "fed_funds_rate",
    "fedfunds": "fed_funds_rate",
    "cpi": "cpi_yoy",
    "cpi yoy": "cpi_yoy",
    "inflation": "cpi_yoy",
    "unemployment": "unemployment_rate",
    "unemployment rate": "unemployment_rate",
    "yield 10y": "yield_10y",
    "10-year yield": "yield_10y",
    "10y yield": "yield_10y",
    "yield 2y": "yield_2y",
    "2-year yield": "yield_2y",
    "2y yield": "yield_2y",
    "yield curve slope": "yield_curve_slope",
    "yield curve": "yield_curve_slope",
    "vix": "vix",
    "gdp growth": "gdp_growth",
    # ---- TechnicalIndicatorsResult (get_technical_indicators) --------------
    "rsi": "rsi",
    "sma": "sma",
    "ema": "ema",
    "macd": "macd",
    "macd signal": "macd_signal",
    "macd_signal": "macd_signal",
    "signal line": "macd_signal",
    "macd histogram": "macd_hist",
    "histogram": "macd_hist",
    "bollinger upper": "bb_upper",
    "bb upper": "bb_upper",
    "bb_upper": "bb_upper",
    "bollinger lower": "bb_lower",
    "bb lower": "bb_lower",
    "bb_lower": "bb_lower",
    "atr": "atr",
    "average true range": "atr",
    # ---- RiskMetricsResult (get_risk_metrics) ------------------------------
    "hist vol 20": "hist_vol_20d",
    "hist vol 20d": "hist_vol_20d",
    "hist_vol_20d": "hist_vol_20d",
    "20-day vol": "hist_vol_20d",
    "20d vol": "hist_vol_20d",
    "20-day volatility": "hist_vol_20d",
    "hist vol 60": "hist_vol_60d",
    "hist vol 60d": "hist_vol_60d",
    "60-day vol": "hist_vol_60d",
    "60d vol": "hist_vol_60d",
    "60-day volatility": "hist_vol_60d",
    "hist vol 252": "hist_vol_252d",
    "hist vol 252d": "hist_vol_252d",
    "annual vol": "hist_vol_252d",
    "252d vol": "hist_vol_252d",
    "annualised volatility": "hist_vol_252d",
    "annualized volatility": "hist_vol_252d",
    "beta": "beta",
    "max drawdown": "max_drawdown_252d",
    "drawdown": "max_drawdown_252d",
    "maximum drawdown": "max_drawdown_252d",
    "sharpe": "sharpe_252d",
    "sharpe ratio": "sharpe_252d",
    "sharpe_252d": "sharpe_252d",
    "var": "var_95_20d",
    "value at risk": "var_95_20d",
    "var 95": "var_95_20d",
}

# ---------------------------------------------------------------------------
# Regex pattern
# ---------------------------------------------------------------------------

# Captures "<field_alias> of <numeric_value>".
# Field alias: one word starting with a letter, optionally followed by up to
# four more tokens (each alphanumeric/slash/hyphen/underscore); inter-word
# whitespace is allowed. Numeric value may be negative and have a decimal.
_CLAIM_PATTERN = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_/\-]*(?:\s+(?!of\b)[A-Za-z0-9_/\-]+){0,4})\s+of\s+([-]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Context window in characters on each side of a match for the snippet.
_SNIPPET_RADIUS = 40


# ---------------------------------------------------------------------------
# Alias resolution helpers
# ---------------------------------------------------------------------------


def _resolve_alias(raw_alias: str) -> tuple[str | None, str]:
    """
    Return (canonical_field, effective_alias) for a captured alias string.

    The regex may over-capture leading context words (e.g. "The P/E" instead
    of "P/E", or "Strong ROE" instead of "ROE"). This function tries the
    full alias first, then progressively strips the leading word until either
    a match is found or all words are exhausted.

    The returned effective_alias is the shortest suffix that matched (or the
    full raw_alias when no match exists). It is stored in NumericalClaim
    .field_alias so the human-readable name reflects the financial term, not
    the surrounding sentence context.

    Examples
    --------
    "The P/E"          -> ("pe",          "P/E")
    "Strong ROE"       -> ("roe",         "ROE")
    "fed funds rate"   -> ("fed_funds_rate", "fed funds rate")
    "The Sharpe ratio" -> ("sharpe_252d", "Sharpe ratio")
    "mystery_metric"   -> (None,          "mystery_metric")
    """
    words = raw_alias.strip().split()
    for i in range(len(words)):
        candidate = " ".join(words[i:])
        normalised = re.sub(r"\s+", " ", candidate.strip().lower())
        result = FIELD_ALIAS_TABLE.get(normalised)
        if result is not None:
            return result, candidate  # preserve original casing for readability
    return None, raw_alias  # no suffix matched; keep full alias as-is


# ---------------------------------------------------------------------------
# Public extraction function
# ---------------------------------------------------------------------------


def extract_numerical_claims(rationale: str) -> list[NumericalClaim]:
    """
    Extract all numerical claims from a rationale string.

    Applies the primary regex pattern to find "<field_alias> of <value>"
    constructions and maps each field alias through FIELD_ALIAS_TABLE via
    _resolve_alias (which handles leading context words the regex may
    over-capture).

    Parameters
    ----------
    rationale : str
        The agent rationale text (from AgentSignal.rationale).

    Returns
    -------
    list[NumericalClaim]
        One entry per regex match. Aliases not found in the table produce a
        NumericalClaim with canonical_field=None (unresolvable in the
        verifier). An empty rationale or a rationale with no matching
        patterns returns an empty list without raising.
    """
    if not rationale:
        return []

    claims: list[NumericalClaim] = []
    for match in _CLAIM_PATTERN.finditer(rationale):
        raw_alias = match.group(1).strip()
        raw_value = match.group(2)

        try:
            value = float(raw_value)
        except ValueError:
            continue

        # Context snippet: ±SNIPPET_RADIUS chars around the full match.
        start = max(0, match.start() - _SNIPPET_RADIUS)
        end = min(len(rationale), match.end() + _SNIPPET_RADIUS)
        snippet = rationale[start:end]

        canonical_field, effective_alias = _resolve_alias(raw_alias)

        claims.append(
            NumericalClaim(
                field_alias=effective_alias,
                canonical_field=canonical_field,
                value=value,
                context_snippet=snippet,
            )
        )

    return claims
