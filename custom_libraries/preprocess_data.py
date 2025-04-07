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

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from custom_libraries.utils_plot import (
    plot_missing_values,
)

SEED_RANDOM = 29
np.random.seed(SEED_RANDOM)


def preprocess_market_data(market_data, start_date, backtest_start, backtest_end):
    close_data = market_data["Close"]
    returns_all = np.log(close_data / close_data.shift(1))

    # Use returns only from start_date to backtest_date for training
    returns_train = returns_all.loc[start_date:backtest_start]
    returns_test = returns_all.loc[backtest_start:backtest_end]

    # Scale returns
    returns_scaler = StandardScaler()
    returns_train_scaled = returns_train.copy()
    returns_train_scaled.iloc[:] = returns_scaler.fit_transform(returns_train)

    return (
        returns_all,
        returns_train,
        returns_test,
        returns_train_scaled,
        returns_scaler,
    )


def preprocess_macro_data_delete(
    macro_df,
    returns_train,
    returns_test,
    start_date,
    backtest_start,
    backtest_end,
):
    macro_df_shifted = macro_df.shift(1)
    macro_df_shifted = macro_df_shifted.fillna(method="ffill")
    macro_df_shifted = macro_df_shifted.loc[start_date:backtest_end]

    macro_train = macro_df_shifted.loc[start_date:backtest_start]
    macro_test = macro_df_shifted.loc[backtest_start:backtest_end]

    macro_train = macro_train.reindex(returns_train.index)
    macro_test = macro_test.reindex(returns_test.index)

    # Scale macro indicators
    macro_scaler = StandardScaler()
    macro_train_scaled = macro_scaler.fit_transform(macro_train)
    macro_test_scaled = macro_scaler.transform(macro_test)

    return (
        macro_df,
        macro_train,
        macro_test,
        macro_train_scaled,
        macro_test_scaled,
        macro_scaler,
    )


def preprocess_macro_data(
    macro_df, vix_data, fx_data, sp500_data, start_date, end_date, desired_index
):
    """
    Preprocesses macroeconomic and market data for portfolio stress testing by integrating
    key financial indicators, computing SP500-based risk-sensitive metrics, and ensuring
    proper data alignment.

    Key Features:
    1. Merging Macroeconomic Indicators
       - Combines `macro_df` (FRED data) with VIX and FX rates (if available).

    2. SP500 Return Computation:
       - Computes log returns of SP500 as:
         returns_sp500 = log(SP500_t / SP500_t-1)

    3. Rolling Worst-Case SP500 Return (5-Day Window):
       - Constructs a drawdown-sensitive metric (`returns_sp500_roll_5`) that captures
         the worst observed SP500 return over a 5-day window:
         - Denominator SP500 price from yesterday (X-1)
         - Numerator: Lowest SP500 price observed from yesterday (X-1) to X+3
         - The final value on day X represents the worst return observed
           in this window, ensuring potential drawdowns are captured.
       - Any positive values in `returns_sp500_roll_5` are set to zero
         as only negative movements are relevant for stress testing.

    4. Data Alignment and Lookahead Bias Prevention:
       - The dataset is reindexed to match the `desired_index`.
       - One-day shift applied to avoid lookahead bias (ensuring that data
         used for predictions is only based on past information).

    5. Missing Value Handling:
       - Forward-fills missing values (`ffill()`) after the shift to maintain consistency.

    ### Returns:
    - `pd.DataFrame`: The processed macroeconomic dataset, including:
      - SP500 log returns (`returns_sp500`)
      - 5-day rolling worst-case returns (`returns_sp500_roll_5
      - Shifted and aligned macroeconomic features ready for modeling
    """

    # Concatenate vix_data (and fx_data if available)
    if fx_data is None:
        macro_df = pd.concat([macro_df, vix_data], axis=1)
    else:
        macro_df = pd.concat([macro_df, vix_data, fx_data], axis=1)

    # Calculate SP500 returns
    returns_sp500 = np.log(sp500_data / sp500_data.shift(1))

    # Calculate rolling minimum returns over 5 days and keep only negative values
    sp500_roll_min_5 = sp500_data.rolling(window=5, min_periods=1).min().shift(-4)
    returns_sp500_5 = np.log(sp500_roll_min_5 / sp500_data).to_frame(
        name="returns_sp500_roll_5"
    )
    returns_sp500_5["returns_sp500_roll_5"] = returns_sp500_5[
        "returns_sp500_roll_5"
    ].apply(lambda x: x if x <= 0 else 0)

    # Merge SP500 returns into macro_df
    macro_df = macro_df.merge(
        returns_sp500, left_index=True, right_index=True, how="left"
    )
    macro_df = macro_df.merge(
        returns_sp500_5, left_index=True, right_index=True, how="left"
    )

    # Reindex macro_df to the desired index
    macro_df = macro_df.reindex(desired_index)

    # plot_missing_values(macro_df, "Missing Values in Macro Data (Pre-Processing)")

    # Shift macro data by one day (to avoid lookahead bias)
    macro_df_shifted = macro_df.shift(1)
    macro_df_shifted = macro_df_shifted.fillna(method="ffill")
    macro_df_shifted = macro_df_shifted.loc[start_date:end_date]

    plot_missing_values(
        macro_df_shifted, "Missing Values in Macro Data (Post-Processing)"
    )

    return macro_df_shifted


def process_input_features(
    macro_df,
    returns_all,
    market_data,
    start_date,
    end_date,
    portfolio_weights,
):
    input_features = macro_df.loc[start_date:end_date]

    rolling_cov_matrix = returns_all.rolling(window=30).cov()
    portfolio_volatility = rolling_cov_matrix.groupby(level=0).apply(
        lambda cov_matrix: np.sqrt(
            np.dot(portfolio_weights.T, np.dot(cov_matrix, portfolio_weights))
        )
    )
    portfolio_volatility_df = portfolio_volatility.to_frame(name="portfolio_volatility")
    portfolio_volume = (market_data["Volume"] * portfolio_weights).sum(axis=1)
    portfolio_volume_df = portfolio_volume.to_frame(name="portfolio_volume")
    input_features = input_features.join(portfolio_volatility_df, how="inner")
    input_features = input_features.join(portfolio_volume_df, how="inner")

    return input_features
