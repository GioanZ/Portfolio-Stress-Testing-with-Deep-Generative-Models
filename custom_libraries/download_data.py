# custom_libraries/download_data.py

"""
Copyright:
    Portfolio Stress Testing with Deep Generative Models
    github.com/GioanZ
Disclaimer:
    This software is for research and educational purposes only.
    It is NOT intended for actual financial decision-making or investment strategies.
    The authors assume no liability for any losses or damages arising from the use
    of this code. Users should conduct their own due diligence before making financial
    decisions.

    This project utilizes deep generative models to simulate financial stress testing.
    The models are trained on historical market and macroeconomic data, but all results
    should be interpreted with caution.
"""

import yfinance as yf
from fredapi import Fred
import pandas as pd

from .utils import (
    rename_col_yf,
)


def yf_download(tickers, start_date, end_date):
    return yf.download(tickers, start=start_date, end=end_date, progress=False)


def download_market_data(tickers, start_date, end_date):
    market_data = yf_download(tickers, start_date=start_date, end_date=end_date)
    return market_data


def download_fred_data(api_key, start_date, end_date):
    fred = Fred(api_key=api_key)
    indicators = {
        "inflation": "FPCPITOTLZGUSA",
        "fed_funds": "FEDFUNDS",
        "unemployment": "UNRATE",
        "short_term": "DGS3MO",
        "long_term": "DGS10",
    }
    macro_df = pd.concat(
        {
            name: fred.get_series(series_id, start_date, end_date)
            for name, series_id in indicators.items()
        },
        axis=1,
    )
    macro_df.ffill(inplace=True)
    return macro_df


def download_other_data(start_date, end_date):
    ticker = "^VIX"
    vix_df = yf_download([ticker], start_date=start_date, end_date=end_date)
    vix_data = rename_col_yf(vix_df, ticker, "vix")

    ticker = "EURUSD=X"
    fx_df = yf_download([ticker], start_date=start_date, end_date=end_date)
    fx_data = rename_col_yf(fx_df, ticker, "eurusd")

    ticker = "^GSPC"
    sp500_df = yf_download([ticker], start_date=start_date, end_date=end_date)
    sp500_data = rename_col_yf(sp500_df, ticker, "sp500")

    return vix_data, fx_data, sp500_data
