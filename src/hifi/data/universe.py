"""
PHASE14_UNIVERSE: ~100-stock ticker universe for Phase 14+ (DJ-090).

~98 large-cap US equities across all 11 GICS sectors (8-10 per sector).
All tickers are S&P 500 components or former components with EDGAR filing
coverage. Used as the base universe for:
  - Bulk OHLCV + fundamentals acquisition (E2-T2)
  - EDGAR MD&A targeted ingestion (E2-T3)
  - Phase 15 walk-forward simulation (21 years, 2004-2025)
  - Phase 16 live paper trading (IBKR)

Constraint: every ticker must have EDGAR filings (all large-cap US equities qualify).
Tickers that IPO'd after 2004 have partial history; the acquisition scripts handle
missing early periods gracefully.
"""

from __future__ import annotations

PHASE14_UNIVERSE: list[dict[str, str]] = [
    # ------------------------------------------------------------------
    # Information Technology (9)
    # ------------------------------------------------------------------
    {
        "ticker": "AAPL",
        "sector": "Information Technology",
        "sub_industry": "Technology Hardware, Storage & Peripherals",
    },
    {
        "ticker": "MSFT",
        "sector": "Information Technology",
        "sub_industry": "Systems Software",
    },
    {
        "ticker": "NVDA",
        "sector": "Information Technology",
        "sub_industry": "Semiconductors",
    },
    {
        "ticker": "AVGO",
        "sector": "Information Technology",
        "sub_industry": "Semiconductors",
    },
    {
        "ticker": "ORCL",
        "sector": "Information Technology",
        "sub_industry": "Application Software",
    },
    {
        "ticker": "CRM",
        "sector": "Information Technology",
        "sub_industry": "Application Software",
    },
    {
        "ticker": "CSCO",
        "sector": "Information Technology",
        "sub_industry": "Communications Equipment",
    },
    {
        "ticker": "ACN",
        "sector": "Information Technology",
        "sub_industry": "IT Consulting & Other Services",
    },
    {
        "ticker": "IBM",
        "sector": "Information Technology",
        "sub_industry": "IT Consulting & Other Services",
    },
    # ------------------------------------------------------------------
    # Health Care (9)
    # ------------------------------------------------------------------
    {
        "ticker": "JNJ",
        "sector": "Health Care",
        "sub_industry": "Pharmaceuticals",
    },
    {
        "ticker": "LLY",
        "sector": "Health Care",
        "sub_industry": "Pharmaceuticals",
    },
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
    {
        "ticker": "MRK",
        "sector": "Health Care",
        "sub_industry": "Pharmaceuticals",
    },
    {
        "ticker": "TMO",
        "sector": "Health Care",
        "sub_industry": "Life Sciences Tools & Services",
    },
    {
        "ticker": "DHR",
        "sector": "Health Care",
        "sub_industry": "Life Sciences Tools & Services",
    },
    {
        "ticker": "MDT",
        "sector": "Health Care",
        "sub_industry": "Health Care Equipment",
    },
    {
        "ticker": "AMGN",
        "sector": "Health Care",
        "sub_industry": "Biotechnology",
    },
    # ------------------------------------------------------------------
    # Financials (9)
    # ------------------------------------------------------------------
    {
        "ticker": "JPM",
        "sector": "Financials",
        "sub_industry": "Diversified Banks",
    },
    {
        "ticker": "BAC",
        "sector": "Financials",
        "sub_industry": "Diversified Banks",
    },
    {
        "ticker": "WFC",
        "sector": "Financials",
        "sub_industry": "Diversified Banks",
    },
    {
        "ticker": "GS",
        "sector": "Financials",
        "sub_industry": "Investment Banking & Brokerage",
    },
    {
        "ticker": "MS",
        "sector": "Financials",
        "sub_industry": "Investment Banking & Brokerage",
    },
    {
        "ticker": "BLK",
        "sector": "Financials",
        "sub_industry": "Asset Management & Custody Banks",
    },
    {
        "ticker": "CB",
        "sector": "Financials",
        "sub_industry": "Property & Casualty Insurance",
    },
    {
        "ticker": "AXP",
        "sector": "Financials",
        "sub_industry": "Consumer Finance",
    },
    {
        "ticker": "SPGI",
        "sector": "Financials",
        "sub_industry": "Financial Exchanges & Data",
    },
    # ------------------------------------------------------------------
    # Consumer Discretionary (9)
    # ------------------------------------------------------------------
    {
        "ticker": "AMZN",
        "sector": "Consumer Discretionary",
        "sub_industry": "Broadline Retail",
    },
    {
        "ticker": "TSLA",
        "sector": "Consumer Discretionary",
        "sub_industry": "Automobile Manufacturers",
    },
    {
        "ticker": "HD",
        "sector": "Consumer Discretionary",
        "sub_industry": "Home Improvement Retail",
    },
    {
        "ticker": "MCD",
        "sector": "Consumer Discretionary",
        "sub_industry": "Restaurants",
    },
    {
        "ticker": "NKE",
        "sector": "Consumer Discretionary",
        "sub_industry": "Footwear",
    },
    {
        "ticker": "SBUX",
        "sector": "Consumer Discretionary",
        "sub_industry": "Restaurants",
    },
    {
        "ticker": "TGT",
        "sector": "Consumer Discretionary",
        "sub_industry": "General Merchandise Stores",
    },
    {
        "ticker": "LOW",
        "sector": "Consumer Discretionary",
        "sub_industry": "Home Improvement Retail",
    },
    {
        "ticker": "BKNG",
        "sector": "Consumer Discretionary",
        "sub_industry": "Hotels, Resorts & Cruise Lines",
    },
    # ------------------------------------------------------------------
    # Communication Services (8)
    # ------------------------------------------------------------------
    {
        "ticker": "GOOGL",
        "sector": "Communication Services",
        "sub_industry": "Interactive Media & Services",
    },
    {
        "ticker": "META",
        "sector": "Communication Services",
        "sub_industry": "Interactive Media & Services",
    },
    {
        "ticker": "NFLX",
        "sector": "Communication Services",
        "sub_industry": "Movies & Entertainment",
    },
    {
        "ticker": "DIS",
        "sector": "Communication Services",
        "sub_industry": "Movies & Entertainment",
    },
    {
        "ticker": "T",
        "sector": "Communication Services",
        "sub_industry": "Integrated Telecommunication Services",
    },
    {
        "ticker": "VZ",
        "sector": "Communication Services",
        "sub_industry": "Integrated Telecommunication Services",
    },
    {
        "ticker": "CMCSA",
        "sector": "Communication Services",
        "sub_industry": "Cable & Satellite",
    },
    {
        "ticker": "CHTR",
        "sector": "Communication Services",
        "sub_industry": "Cable & Satellite",
    },
    # ------------------------------------------------------------------
    # Industrials (9)
    # ------------------------------------------------------------------
    {
        "ticker": "RTX",
        "sector": "Industrials",
        "sub_industry": "Aerospace & Defense",
    },
    {
        "ticker": "HON",
        "sector": "Industrials",
        "sub_industry": "Industrial Conglomerates",
    },
    {
        "ticker": "UPS",
        "sector": "Industrials",
        "sub_industry": "Air Freight & Logistics",
    },
    {
        "ticker": "CAT",
        "sector": "Industrials",
        "sub_industry": "Construction Machinery & Heavy Transportation Equipment",
    },
    {
        "ticker": "DE",
        "sector": "Industrials",
        "sub_industry": "Agricultural & Farm Machinery",
    },
    {
        "ticker": "LMT",
        "sector": "Industrials",
        "sub_industry": "Aerospace & Defense",
    },
    {
        "ticker": "GE",
        "sector": "Industrials",
        "sub_industry": "Aerospace & Defense",
    },
    {
        "ticker": "BA",
        "sector": "Industrials",
        "sub_industry": "Aerospace & Defense",
    },
    {
        "ticker": "UNP",
        "sector": "Industrials",
        "sub_industry": "Railroads",
    },
    # ------------------------------------------------------------------
    # Consumer Staples (9)
    # ------------------------------------------------------------------
    {
        "ticker": "PG",
        "sector": "Consumer Staples",
        "sub_industry": "Household Products",
    },
    {
        "ticker": "KO",
        "sector": "Consumer Staples",
        "sub_industry": "Soft Drinks & Non-Alcoholic Beverages",
    },
    {
        "ticker": "PEP",
        "sector": "Consumer Staples",
        "sub_industry": "Soft Drinks & Non-Alcoholic Beverages",
    },
    {
        "ticker": "WMT",
        "sector": "Consumer Staples",
        "sub_industry": "Consumer Staples Merchandise Retail",
    },
    {
        "ticker": "COST",
        "sector": "Consumer Staples",
        "sub_industry": "Consumer Staples Merchandise Retail",
    },
    {
        "ticker": "PM",
        "sector": "Consumer Staples",
        "sub_industry": "Tobacco",
    },
    {
        "ticker": "CL",
        "sector": "Consumer Staples",
        "sub_industry": "Household Products",
    },
    {
        "ticker": "MO",
        "sector": "Consumer Staples",
        "sub_industry": "Tobacco",
    },
    {
        "ticker": "MDLZ",
        "sector": "Consumer Staples",
        "sub_industry": "Packaged Foods & Meats",
    },
    # ------------------------------------------------------------------
    # Energy (9)
    # ------------------------------------------------------------------
    {
        "ticker": "XOM",
        "sector": "Energy",
        "sub_industry": "Integrated Oil & Gas",
    },
    {
        "ticker": "CVX",
        "sector": "Energy",
        "sub_industry": "Integrated Oil & Gas",
    },
    {
        "ticker": "COP",
        "sector": "Energy",
        "sub_industry": "Oil & Gas Exploration & Production",
    },
    {
        "ticker": "EOG",
        "sector": "Energy",
        "sub_industry": "Oil & Gas Exploration & Production",
    },
    {
        "ticker": "SLB",
        "sector": "Energy",
        "sub_industry": "Oil & Gas Equipment & Services",
    },
    {
        "ticker": "MPC",
        "sector": "Energy",
        "sub_industry": "Oil & Gas Refining & Marketing & Transportation",
    },
    {
        "ticker": "PSX",
        "sector": "Energy",
        "sub_industry": "Oil & Gas Refining & Marketing & Transportation",
    },
    {
        "ticker": "OXY",
        "sector": "Energy",
        "sub_industry": "Oil & Gas Exploration & Production",
    },
    {
        "ticker": "DVN",
        "sector": "Energy",
        "sub_industry": "Oil & Gas Exploration & Production",
    },
    # ------------------------------------------------------------------
    # Materials (9)
    # ------------------------------------------------------------------
    {
        "ticker": "LIN",
        "sector": "Materials",
        "sub_industry": "Industrial Gases",
    },
    {
        "ticker": "APD",
        "sector": "Materials",
        "sub_industry": "Industrial Gases",
    },
    {
        "ticker": "FCX",
        "sector": "Materials",
        "sub_industry": "Copper",
    },
    {
        "ticker": "NEM",
        "sector": "Materials",
        "sub_industry": "Gold Mining",
    },
    {
        "ticker": "ECL",
        "sector": "Materials",
        "sub_industry": "Specialty Chemicals",
    },
    {
        "ticker": "SHW",
        "sector": "Materials",
        "sub_industry": "Specialty Chemicals",
    },
    {
        "ticker": "PPG",
        "sector": "Materials",
        "sub_industry": "Specialty Chemicals",
    },
    {
        "ticker": "NUE",
        "sector": "Materials",
        "sub_industry": "Steel",
    },
    {
        "ticker": "MLM",
        "sector": "Materials",
        "sub_industry": "Construction Materials",
    },
    # ------------------------------------------------------------------
    # Real Estate (9)
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
    {
        "ticker": "CCI",
        "sector": "Real Estate",
        "sub_industry": "Telecom Tower REITs",
    },
    {
        "ticker": "EQIX",
        "sector": "Real Estate",
        "sub_industry": "Data Center REITs",
    },
    {
        "ticker": "O",
        "sector": "Real Estate",
        "sub_industry": "Net Lease REITs",
    },
    {
        "ticker": "SPG",
        "sector": "Real Estate",
        "sub_industry": "Retail REITs",
    },
    {
        "ticker": "PSA",
        "sector": "Real Estate",
        "sub_industry": "Self-Storage REITs",
    },
    {
        "ticker": "WELL",
        "sector": "Real Estate",
        "sub_industry": "Health Care REITs",
    },
    {
        "ticker": "EQR",
        "sector": "Real Estate",
        "sub_industry": "Residential REITs",
    },
    # ------------------------------------------------------------------
    # Utilities (9)
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
    {
        "ticker": "SO",
        "sector": "Utilities",
        "sub_industry": "Electric Utilities",
    },
    {
        "ticker": "D",
        "sector": "Utilities",
        "sub_industry": "Electric Utilities",
    },
    {
        "ticker": "SRE",
        "sector": "Utilities",
        "sub_industry": "Multi-Utilities",
    },
    {
        "ticker": "AEP",
        "sector": "Utilities",
        "sub_industry": "Electric Utilities",
    },
    {
        "ticker": "EXC",
        "sector": "Utilities",
        "sub_industry": "Electric Utilities",
    },
    {
        "ticker": "XEL",
        "sector": "Utilities",
        "sub_industry": "Electric Utilities",
    },
    {
        "ticker": "WEC",
        "sector": "Utilities",
        "sub_industry": "Electric Utilities",
    },
]

_GICS_SECTORS: frozenset[str] = frozenset(
    {
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
    }
)


def get_tickers() -> list[str]:
    """Return all ticker symbols in the Phase 14 universe."""
    return [entry["ticker"] for entry in PHASE14_UNIVERSE]


def get_sectors() -> set[str]:
    """Return the set of GICS sectors represented in the universe."""
    return {entry["sector"] for entry in PHASE14_UNIVERSE}


def get_tickers_by_sector(sector: str) -> list[str]:
    """Return all tickers in the given GICS sector."""
    return [entry["ticker"] for entry in PHASE14_UNIVERSE if entry["sector"] == sector]


def get_sector(ticker: str) -> str | None:
    """Return the GICS sector for a ticker, or None if not in the universe."""
    for entry in PHASE14_UNIVERSE:
        if entry["ticker"] == ticker:
            return entry["sector"]
    return None


def get_sub_industry(ticker: str) -> str | None:
    """Return the GICS sub-industry for a ticker, or None if not in the universe."""
    for entry in PHASE14_UNIVERSE:
        if entry["ticker"] == ticker:
            return entry["sub_industry"]
    return None
