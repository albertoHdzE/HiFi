"""Portfolio construction policy (DJ-122).

Position limits derived from the number of investable candidates rather than
hardcoded as absolute percentages. See ``policy.PortfolioPolicy``.
"""

from hifi.portfolio.policy import PortfolioPolicy

__all__ = ["PortfolioPolicy"]
