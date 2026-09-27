"""
us_calendar.py  —  US Trading Days Calendar
"""

import pandas as pd
from pandas_market_calendars import get_calendar


def get_us_trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    nyse = get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    return schedule.index


def get_last_n_trading_days(n: int, end_date: str = None) -> pd.DatetimeIndex:
    if end_date is None:
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    start_date = pd.Timestamp(end_date) - pd.Timedelta(days=n * 2)
    all_days = get_us_trading_days(start_date.strftime('%Y-%m-%d'), end_date)
    return all_days[-n:]


def is_trading_day(date: str) -> bool:
    try:
        trading_days = get_us_trading_days(date, date)
        return len(trading_days) > 0
    except:
        return False


def get_next_trading_day(date: str) -> str:
    d = pd.Timestamp(date)
    for i in range(1, 11):
        check_date = (d + pd.Timedelta(days=i)).strftime('%Y-%m-%d')
        if is_trading_day(check_date):
            return check_date
    days = get_us_trading_days(date, (d + pd.Timedelta(days=15)).strftime('%Y-%m-%d'))
    if len(days) > 0:
        return days[0].strftime('%Y-%m-%d')
    return date
