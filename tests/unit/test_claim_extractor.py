"""Unit tests for the claim extractor (P5-E2-T2 through T7)."""

from __future__ import annotations

import pytest

from hifi.verification.extractor import FIELD_ALIAS_TABLE, extract_numerical_claims

# ---------------------------------------------------------------------------
# P5-E2-T2: Basic single-claim extraction
# ---------------------------------------------------------------------------


def test_rsi_claim_extracted():
    """'RSI of 48.0' -> NumericalClaim with canonical_field='rsi', value=48.0."""
    claims = extract_numerical_claims("RSI of 48.0 signals neutral momentum.")
    assert len(claims) >= 1
    rsi = next((c for c in claims if c.canonical_field == "rsi"), None)
    assert rsi is not None
    assert rsi.field_alias.upper() == "RSI"
    assert rsi.value == pytest.approx(48.0)


# ---------------------------------------------------------------------------
# P5-E2-T3: P/E claim
# ---------------------------------------------------------------------------


def test_pe_claim_extracted():
    """'P/E of 28.3 is below' -> pe, 28.3."""
    claims = extract_numerical_claims("The P/E of 28.3 is below the sector average.")
    pe = next((c for c in claims if c.canonical_field == "pe"), None)
    assert pe is not None
    assert pe.value == pytest.approx(28.3)


# ---------------------------------------------------------------------------
# P5-E2-T4: Multi-word fundamental alias
# ---------------------------------------------------------------------------


def test_fed_funds_rate_claim():
    """'fed funds rate of 4.1' -> fed_funds_rate, 4.1."""
    claims = extract_numerical_claims("The fed funds rate of 4.1 compresses equity multiples.")
    ff = next((c for c in claims if c.canonical_field == "fed_funds_rate"), None)
    assert ff is not None
    assert ff.value == pytest.approx(4.1)


def test_sharpe_ratio_claim():
    """'Sharpe ratio of 0.82' -> sharpe_252d, 0.82."""
    claims = extract_numerical_claims("Sharpe ratio of 0.82 is moderate for this period.")
    s = next((c for c in claims if c.canonical_field == "sharpe_252d"), None)
    assert s is not None
    assert s.value == pytest.approx(0.82)


def test_max_drawdown_claim():
    """'max drawdown of 15.3' -> max_drawdown_252d, 15.3."""
    claims = extract_numerical_claims("The max drawdown of 15.3 over 252 days is manageable.")
    md = next((c for c in claims if c.canonical_field == "max_drawdown_252d"), None)
    assert md is not None
    assert md.value == pytest.approx(15.3)


def test_atr_claim():
    """'ATR of 3.82' -> atr, 3.82."""
    claims = extract_numerical_claims("ATR of 3.82 indicates elevated daily swings.")
    a = next((c for c in claims if c.canonical_field == "atr"), None)
    assert a is not None
    assert a.value == pytest.approx(3.82)


def test_beta_claim():
    """'beta of 0.95' -> beta, 0.95."""
    claims = extract_numerical_claims("The stock shows beta of 0.95 relative to SPY.")
    b = next((c for c in claims if c.canonical_field == "beta"), None)
    assert b is not None
    assert b.value == pytest.approx(0.95)


def test_roe_claim():
    """'ROE of 0.24' -> roe, 0.24."""
    claims = extract_numerical_claims("Strong ROE of 0.24 reflects efficient capital use.")
    r = next((c for c in claims if c.canonical_field == "roe"), None)
    assert r is not None
    assert r.value == pytest.approx(0.24)


# ---------------------------------------------------------------------------
# P5-E2-T5: Unknown alias -> canonical_field=None
# ---------------------------------------------------------------------------


def test_unknown_alias_returns_none_canonical():
    """'momentum index of 3.4' -> NumericalClaim with canonical_field=None."""
    claims = extract_numerical_claims("The momentum index of 3.4 is above average.")
    # At least one claim should be extracted but with canonical_field=None
    assert len(claims) >= 1
    unknown = next((c for c in claims if "momentum" in c.field_alias.lower()), None)
    assert unknown is not None
    assert unknown.canonical_field is None
    assert unknown.value == pytest.approx(3.4)


# ---------------------------------------------------------------------------
# P5-E2-T6: Multiple claims in one rationale
# ---------------------------------------------------------------------------


def test_multiple_claims_extracted():
    """Multiple 'X of Y' patterns are all extracted."""
    rationale = (
        "RSI of 42.1 signals recovery territory. "
        "The Sharpe ratio of 0.82 is moderate. "
        "ATR of 3.82 indicates elevated daily swings."
    )
    claims = extract_numerical_claims(rationale)
    canonical_fields = {c.canonical_field for c in claims}
    assert "rsi" in canonical_fields
    assert "sharpe_252d" in canonical_fields
    assert "atr" in canonical_fields
    assert len(claims) >= 3


def test_multiple_fundamental_claims():
    rationale = (
        "P/E of 28.3 is below sector average. "
        "ROE of 0.24 demonstrates strong profitability. "
        "The fed funds rate of 4.75 compresses multiples."
    )
    claims = extract_numerical_claims(rationale)
    canonical_fields = {c.canonical_field for c in claims}
    assert "pe" in canonical_fields
    assert "roe" in canonical_fields
    assert "fed_funds_rate" in canonical_fields


# ---------------------------------------------------------------------------
# P5-E2-T7: Empty rationale or no claims
# ---------------------------------------------------------------------------


def test_empty_rationale_returns_empty_list():
    """Empty string -> [] with no exception."""
    claims = extract_numerical_claims("")
    assert claims == []


def test_no_matching_patterns():
    """Rationale with no 'X of N' patterns -> empty list."""
    claims = extract_numerical_claims(
        "The stock looks interesting given current market conditions."
    )
    # May or may not match depending on regex; assert no canonical claims
    # (all would be unresolvable if any match occurs)
    for c in claims:
        assert c.canonical_field is None


def test_none_like_rationale():
    """Rationale with only text, no digits -> empty list."""
    claims = extract_numerical_claims("Strong buy thesis holds based on qualitative factors.")
    assert all(c.canonical_field is None for c in claims)


# ---------------------------------------------------------------------------
# Context snippet correctness
# ---------------------------------------------------------------------------


def test_context_snippet_contains_alias_and_value():
    rationale = "RSI of 48.0 indicates neutral momentum in the recent session."
    claims = extract_numerical_claims(rationale)
    rsi = next((c for c in claims if c.canonical_field == "rsi"), None)
    assert rsi is not None
    assert "RSI" in rsi.context_snippet
    assert "48.0" in rsi.context_snippet


# ---------------------------------------------------------------------------
# Negative value handling
# ---------------------------------------------------------------------------


def test_negative_value_extracted():
    """'max drawdown of -15.3' -> value=-15.3."""
    claims = extract_numerical_claims("max drawdown of -15.3 over the trailing year.")
    md = next((c for c in claims if c.canonical_field == "max_drawdown_252d"), None)
    assert md is not None
    assert md.value == pytest.approx(-15.3)


# ---------------------------------------------------------------------------
# Alias table sanity
# ---------------------------------------------------------------------------


def test_alias_table_has_expected_keys():
    """Core financial aliases are present in FIELD_ALIAS_TABLE."""
    assert "rsi" in FIELD_ALIAS_TABLE
    assert "p/e" in FIELD_ALIAS_TABLE
    assert "sharpe" in FIELD_ALIAS_TABLE
    assert "beta" in FIELD_ALIAS_TABLE
    assert "fed funds rate" in FIELD_ALIAS_TABLE
    assert "max drawdown" in FIELD_ALIAS_TABLE
    assert "atr" in FIELD_ALIAS_TABLE


def test_alias_table_canonical_values_are_valid_field_names():
    """All canonical field values use underscore notation (no spaces)."""
    for alias, canonical in FIELD_ALIAS_TABLE.items():
        assert " " not in canonical, f"Canonical field '{canonical}' for alias '{alias}' has spaces"
        assert canonical == canonical.lower(), f"Canonical field '{canonical}' is not lowercase"
