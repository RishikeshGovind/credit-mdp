"""Real Central Bank of Ireland (CBI) figures, used as real-world context.

These are aggregate, public statistics. They ground the exercise in a real lending
market and give a sanity check on the scale of arrears and interest rates. They do
**not** supply any loan-level data, and (because the loan-level data here is US
Lending Club consumer credit) they are context, not a tight calibration. See
``data/PROVENANCE.md`` for full citations and dates.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IrelandContext:
    """Headline CBI figures, each carrying its own source string."""

    # Residential mortgage arrears, principal dwelling houses (PDH), end-2025.
    pdh_accounts_total: int = 704_290
    pdh_accounts_90d_arrears: int = 21_833
    pdh_share_90d_arrears: float = 0.031          # 3.1% of all PDH accounts
    long_term_arrears_accounts: int = 16_115       # more than 1 year behind
    long_term_arrears_share: float = 0.022         # 2.2% of all PDH accounts
    arrears_source: str = (
        "Central Bank of Ireland, Residential Mortgage Arrears and Repossessions "
        "Statistics, Q4 2025 (end-December 2025); reported by the Irish Times, "
        "13 March 2026."
    )

    # Average interest rate on new mortgage agreements, Ireland, April 2026.
    new_mortgage_rate_avg: float = 0.035           # 3.5%
    euro_area_avg_rate: float = 0.0345             # 3.45%
    rate_source: str = (
        "Central Bank of Ireland, Retail Interest Rates, April 2026; reported by "
        "RTÉ, 13 May 2026."
    )


CBI = IrelandContext()
