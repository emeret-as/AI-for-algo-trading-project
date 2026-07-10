"""
Date utilities for the volatility infrastructure platform.

Provides day-count conventions, date-to-year-fraction conversion,
and business-day calendar helpers. All analytics use year fractions
computed through this module to ensure consistency.
"""

from datetime import date, datetime, timedelta
from enum import Enum
from typing import Optional


class DayCountConvention(Enum):
    """Supported day-count conventions."""
    ACT_365 = "ACT/365"
    ACT_360 = "ACT/360"
    ACT_ACT = "ACT/ACT"
    BUS_252 = "BUS/252"


def year_fraction(
    start: date,
    end: date,
    convention: DayCountConvention = DayCountConvention.ACT_365,
    calendar: Optional["TradingCalendar"] = None,
) -> float:
    """
    Compute the year fraction between two dates using the specified convention.

    Args:
        start: Start date (typically valuation date).
        end: End date (typically expiry date).
        convention: Day-count convention to use.
        calendar: Trading calendar for BUS/252 convention.

    Returns:
        Year fraction as a float. Always non-negative if end >= start.
    """
    if end < start:
        return -year_fraction(end, start, convention, calendar)

    days = (end - start).days

    if convention == DayCountConvention.ACT_365:
        return days / 365.0
    elif convention == DayCountConvention.ACT_360:
        return days / 360.0
    elif convention == DayCountConvention.ACT_ACT:
        # Simplified: use actual days in year
        year_days = 366.0 if _is_leap_year(start.year) else 365.0
        return days / year_days
    elif convention == DayCountConvention.BUS_252:
        if calendar is None:
            calendar = TradingCalendar()
        bus_days = calendar.business_days_between(start, end)
        return bus_days / 252.0
    else:
        raise ValueError(f"Unsupported day-count convention: {convention}")


def _is_leap_year(year: int) -> bool:
    """Check if a year is a leap year."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


class TradingCalendar:
    """
    Simple trading calendar supporting US equity market holidays.

    Provides business-day logic for date calculations. Can be extended
    with exchange-specific holiday schedules.
    """

    # US market holidays (fixed-date approximation for standard holidays)
    STANDARD_HOLIDAYS_MONTH_DAY = {
        (1, 1),    # New Year's Day
        (7, 4),    # Independence Day
        (12, 25),  # Christmas Day
    }

    def __init__(self, additional_holidays: Optional[set] = None):
        """
        Args:
            additional_holidays: Set of date objects for additional holidays.
        """
        self._additional_holidays = additional_holidays or set()

    def is_business_day(self, d: date) -> bool:
        """Check if a date is a business day (not weekend, not holiday)."""
        if d.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        if (d.month, d.day) in self.STANDARD_HOLIDAYS_MONTH_DAY:
            return False
        if d in self._additional_holidays:
            return False
        return True

    def business_days_between(self, start: date, end: date) -> int:
        """Count business days between two dates (exclusive of start, inclusive of end)."""
        if end <= start:
            return 0
        count = 0
        current = start + timedelta(days=1)
        while current <= end:
            if self.is_business_day(current):
                count += 1
            current += timedelta(days=1)
        return count

    def next_business_day(self, d: date) -> date:
        """Return the next business day after d."""
        candidate = d + timedelta(days=1)
        while not self.is_business_day(candidate):
            candidate += timedelta(days=1)
        return candidate

    def previous_business_day(self, d: date) -> date:
        """Return the previous business day before d."""
        candidate = d - timedelta(days=1)
        while not self.is_business_day(candidate):
            candidate -= timedelta(days=1)
        return candidate


def expiry_to_maturity_years(
    valuation_date: date,
    expiry_date: date,
    convention: DayCountConvention = DayCountConvention.ACT_365,
) -> float:
    """
    Convert an expiry date into a maturity in years.

    This is the canonical entry point for converting calendar dates
    into the year-fraction coordinate used by all analytics modules.

    Args:
        valuation_date: Current valuation date.
        expiry_date: Option expiry date.
        convention: Day-count convention.

    Returns:
        Maturity in years. Returns 0.0 if expiry is today or past.
    """
    t = year_fraction(valuation_date, expiry_date, convention)
    return max(t, 0.0)


def parse_expiry_string(expiry_str: str) -> date:
    """
    Parse an expiry string in YYYYMMDD or YYYY-MM-DD format.

    Args:
        expiry_str: Expiry date string.

    Returns:
        Parsed date object.
    """
    expiry_str = expiry_str.strip().replace("-", "")
    if len(expiry_str) != 8:
        raise ValueError(f"Invalid expiry string format: {expiry_str}")
    return datetime.strptime(expiry_str, "%Y%m%d").date()
