"""Real Central Bank of Ireland (CBI) aggregate figures used to calibrate the
portfolio-level scenario.

These are *aggregate, public* statistics. They are used only to set realistic scale
(arrears rate, interest-rate band, portfolio size). No Irish loan-level data is used.
See ``data/PROVENANCE.md`` for full citations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IrelandCalibration:
    """Headline CBI figures, each carrying its own source string."""

    # Residential mortgage arrears, PDH, end-June 2025 (Q2 2025).
    pdh_accounts_total: int = 702_343
    pdh_accounts_90d_arrears: int = 24_583
    pdh_share_90d_arrears: float = 0.035  # 3.5% of all PDH accounts
    pdh_balance_90d_arrears_share: float = 0.048  # 4.8% of total PDH balance
    arrears_source: str = (
        "Central Bank of Ireland, Residential Mortgage Arrears and Repossessions "
        "Statistics, Q2 2025 (end-June 2025)."
    )

    # Average interest rate on new mortgage agreements, Ireland, Sep 2025.
    new_mortgage_rate_avg: float = 0.0359  # 3.59%
    new_mortgage_rate_fixed: float = 0.0351  # 3.51%
    new_mortgage_rate_variable: float = 0.0408  # 4.08%
    euro_area_avg_rate: float = 0.0334  # 3.34%
    rate_source: str = (
        "Central Bank of Ireland, Retail Interest Rates, September 2025."
    )

    def realistic_rate_band(self) -> tuple[float, float]:
        """A plausible Irish offered-rate band around the real average.

        Lower bound near the fixed-rate average, upper bound near the variable
        average, widened modestly to give the lender meaningful pricing choices.
        """
        return (0.030, 0.055)


CBI = IrelandCalibration()
