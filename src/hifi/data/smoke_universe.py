"""
SMOKE_UNIVERSE: 22-ticker stratified sample for Phase 14.1+ smoke tests (E2-T1, DJ-107).

Exactly 2 tickers per GICS sector (11 sectors x 2 = 22 tickers).
All tickers are members of PHASE14_UNIVERSE (verified by unit test).
Deterministic — same list every run, suitable for CI and reproducibility.

Sector -> tickers mapping:
  Information Technology : AAPL, CRM
  Health Care            : UNH, ABT
  Financials             : JPM, BLK
  Consumer Discretionary : AMZN, NKE
  Communication Services : GOOGL, DIS
  Industrials            : HON, CAT
  Consumer Staples       : PG, COST
  Energy                 : XOM, COP
  Materials              : LIN, FCX
  Real Estate            : PLD, AMT
  Utilities              : NEE, DUK
"""

from __future__ import annotations

SMOKE_UNIVERSE: list[dict[str, str]] = [
    # ------------------------------------------------------------------
    # Information Technology (2)
    # ------------------------------------------------------------------
    {
        "ticker": "AAPL",
        "sector": "Information Technology",
        "sub_industry": "Technology Hardware, Storage & Peripherals",
    },
    {
        "ticker": "CRM",
        "sector": "Information Technology",
        "sub_industry": "Application Software",
    },
    # ------------------------------------------------------------------
    # Health Care (2)
    # ------------------------------------------------------------------
    {
        "ticker": "UNH",
        "sector": "Health Care",
        "sub_industry": "Managed Health Care",
    },
    {
        "ticker": "ABT",
        "sector": "Health Care",
        "sub_industry": "Health Care Equipment",
    },
    # ------------------------------------------------------------------
    # Financials (2)
    # ------------------------------------------------------------------
    {
        "ticker": "JPM",
        "sector": "Financials",
        "sub_industry": "Diversified Banks",
    },
    {
        "ticker": "BLK",
        "sector": "Financials",
        "sub_industry": "Asset Management & Custody Banks",
    },
    # ------------------------------------------------------------------
    # Consumer Discretionary (2)
    # ------------------------------------------------------------------
    {
        "ticker": "AMZN",
        "sector": "Consumer Discretionary",
        "sub_industry": "Broadline Retail",
    },
    {
        "ticker": "NKE",
        "sector": "Consumer Discretionary",
        "sub_industry": "Footwear",
    },
    # ------------------------------------------------------------------
    # Communication Services (2)
    # ------------------------------------------------------------------
    {
        "ticker": "GOOGL",
        "sector": "Communication Services",
        "sub_industry": "Interactive Media & Services",
    },
    {
        "ticker": "DIS",
        "sector": "Communication Services",
        "sub_industry": "Movies & Entertainment",
    },
    # ------------------------------------------------------------------
    # Industrials (2)
    # ------------------------------------------------------------------
    {
        "ticker": "HON",
        "sector": "Industrials",
        "sub_industry": "Industrial Conglomerates",
    },
    {
        "ticker": "CAT",
        "sector": "Industrials",
        "sub_industry": "Construction Machinery & Heavy Transportation Equipment",
    },
    # ------------------------------------------------------------------
    # Consumer Staples (2)
    # ------------------------------------------------------------------
    {
        "ticker": "PG",
        "sector": "Consumer Staples",
        "sub_industry": "Household Products",
    },
    {
        "ticker": "COST",
        "sector": "Consumer Staples",
        "sub_industry": "Consumer Staples Merchandise Retail",
    },
    # ------------------------------------------------------------------
    # Energy (2)
    # ------------------------------------------------------------------
    {
        "ticker": "XOM",
        "sector": "Energy",
        "sub_industry": "Integrated Oil & Gas",
    },
    {
        "ticker": "COP",
        "sector": "Energy",
        "sub_industry": "Oil & Gas Exploration & Production",
    },
    # ------------------------------------------------------------------
    # Materials (2)
    # ------------------------------------------------------------------
    {
        "ticker": "LIN",
        "sector": "Materials",
        "sub_industry": "Industrial Gases",
    },
    {
        "ticker": "FCX",
        "sector": "Materials",
        "sub_industry": "Copper",
    },
    # ------------------------------------------------------------------
    # Real Estate (2)
    # ------------------------------------------------------------------
    {
        "ticker": "PLD",
        "sector": "Real Estate",
        "sub_industry": "Industrial REITs",
    },
    {
        "ticker": "AMT",
        "sector": "Real Estate",
        "sub_industry": "Telecom Tower REITs",
    },
    # ------------------------------------------------------------------
    # Utilities (2)
    # ------------------------------------------------------------------
    {
        "ticker": "NEE",
        "sector": "Utilities",
        "sub_industry": "Electric Utilities",
    },
    {
        "ticker": "DUK",
        "sector": "Utilities",
        "sub_industry": "Electric Utilities",
    },
]

_EXPECTED_SECTORS: frozenset[str] = frozenset({
    "Information Technology",
    "Health Care",
    "Financials",
    "Consumer Discretionary",
    "Communication Services",
    "Industrials",
    "Consumer Staples",
    "Energy",
    "Materials",
    "Real Estate",
    "Utilities",
})
