# custom_libraries/stress_backtesting.py

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

import numpy as np
import pandas as pd

import tensorflow as tf

from custom_libraries.metrics_validation import (
    calculate_var_es,
)


def create_stressed_input_condition(
    input_condition, indices, stress_values, scaler, i=0
):
    """
    Copies the condition vector and sets specific macroeconomic indicators to stressed values.
    """
    stressed_condition = input_condition.copy()
    for key, idx in indices.items():
        if key in stress_values:
            stressed_condition.iloc[i, idx] = stress_values[key]
    return scaler.transform(stressed_condition[: i + 1]), stressed_condition


def generate_scenarios_for_one_day(
    input_condition_vector, num_scenarios, noise_dim, generator, decoder, returns_scaler
):
    """
    Generate synthetic scenarios for one day given the input condition
    """
    # Convert input condition vector into a tensor and repeat it
    input_condition = tf.convert_to_tensor(
        np.atleast_2d(input_condition_vector), dtype=tf.float32
    )
    input_condition = tf.repeat(input_condition, repeats=num_scenarios, axis=0)

    # Generate noise and latent code with generator
    noise = tf.random.normal([num_scenarios, noise_dim])
    generated_latent = generator([noise, input_condition]).numpy()

    # Decode the latent code
    synthetic_returns_scaled = decoder.predict([generated_latent, input_condition])

    # Check and fix shape
    if synthetic_returns_scaled.shape[0] != num_scenarios:
        synthetic_returns_scaled = synthetic_returns_scaled.T

    synthetic_returns = returns_scaler.inverse_transform(synthetic_returns_scaled)
    if synthetic_returns.shape[0] != num_scenarios:
        synthetic_returns = synthetic_returns.T

    return synthetic_returns, generated_latent, input_condition


def rolling_backtest(
    returns_test,
    input_test,
    input_scaler,
    input_test_scaled,
    noise_dim,
    generator,
    decoder,
    returns_scaler,
    num_scenarios,
    portfolio_weights=None,
    stress_values=None,
    by_ticker=False,
):
    """
    Performs rolling backtesting

    If by_ticker=False: Computes portfolio-level VaR and ES
    If by_ticker=True: Computes VaR and ES for each individual stock
    """

    input_test_indices = {col: idx for idx, col in enumerate(input_test.columns)}
    backtest_dates = returns_test.index
    backtest_results = []

    for i in range(len(backtest_dates)):
        forecast_date = backtest_dates[i]

        # Apply stress conditions if needed
        if stress_values is not None:
            input_test_scaled, input_test = create_stressed_input_condition(
                input_test, input_test_indices, stress_values, input_scaler, i
            )
        condition = input_test_scaled[i]

        # Generate synthetic scenarios
        synthetic_returns, _, _ = generate_scenarios_for_one_day(
            condition, num_scenarios, noise_dim, generator, decoder, returns_scaler
        )

        # Convert to DataFrame
        synthetic_returns_df = pd.DataFrame(
            synthetic_returns, columns=returns_test.columns
        )

        # Compute historical returns
        hist_stock_returns = returns_test.loc[forecast_date]

        if by_ticker:
            # Compute VaR and ES for each stock
            var_synth_dict = {}
            es_synth_dict = {}

            for stock in synthetic_returns_df.columns:
                stock_returns = synthetic_returns_df[stock]
                stock_returns = np.where(stock_returns > 0, 0, stock_returns)
                var_synth, es_synth = calculate_var_es(stock_returns, alpha=5)
                var_synth_dict[f"synthetic_VaR_{stock}"] = var_synth
                es_synth_dict[f"synthetic_ES_{stock}"] = es_synth

            # Save results for this date (by ticker)
            backtest_results.append(
                {
                    "forecast_date": forecast_date,
                    **{
                        f"hist_return_{stock}": hist_stock_returns[stock]
                        for stock in returns_test.columns
                    },
                    **var_synth_dict,
                    **es_synth_dict,
                }
            )
        else:
            # Compute portfolio returns
            synthetic_portfolio_returns = synthetic_returns_df.dot(portfolio_weights)
            synthetic_portfolio_returns = np.where(
                synthetic_portfolio_returns > 0, 0, synthetic_portfolio_returns
            )
            var_synth, es_synth = calculate_var_es(synthetic_portfolio_returns, alpha=5)
            hist_portfolio_return = returns_test.loc[forecast_date].dot(
                portfolio_weights
            )

            # Save results for this date (portfolio-level)
            backtest_results.append(
                {
                    "forecast_date": forecast_date,
                    "hist_portfolio_return": hist_portfolio_return,
                    "synthetic_VaR": var_synth,
                    "synthetic_ES": es_synth,
                    "synthetic_returns": synthetic_portfolio_returns.tolist(),
                }
            )

    return pd.DataFrame(backtest_results)


def detailed_evaluation_forecast(
    returns_test,
    input_test_scaled,
    generator,
    decoder,
    returns_scaler,
    num_scenarios,
    noise_dim,
    portfolio_weights,
    index_date=0,
):
    forecast_date = returns_test.index[index_date]
    print("Forecast date:", forecast_date)

    # Extract the condition vector for that date
    condition_vector = input_test_scaled[index_date]

    # Generate synthetic scenarios for that day
    synthetic_returns_d, _, _ = generate_scenarios_for_one_day(
        condition_vector, num_scenarios, noise_dim, generator, decoder, returns_scaler
    )

    # Convert to DataFrame for easier analysis
    synthetic_returns_df_d = pd.DataFrame(
        synthetic_returns_d, columns=returns_test.columns
    )

    # Compute synthetic portfolio returns
    synthetic_portfolio_returns_d = synthetic_returns_df_d.dot(portfolio_weights)

    # Extract the actual portfolio return for the selected forecast date
    actual_portfolio_return_d = returns_test.loc[forecast_date].dot(portfolio_weights)

    # Compute VaR and ES
    var_synth_d, es_synth_d = calculate_var_es(synthetic_portfolio_returns_d, alpha=5)

    print("Detailed evaluation for forecast date:", forecast_date)
    print("Synthetic Portfolio Returns - Summary:")
    print(pd.Series(synthetic_portfolio_returns_d).describe())
    print(f"Synthetic VaR (5th percentile): {var_synth_d:.4f}")
    print(f"Synthetic ES: {es_synth_d:.4f}")

    return synthetic_portfolio_returns_d, var_synth_d, actual_portfolio_return_d


def backtest_tickers_ret_syn(tickers_list, backtest_tickers_df):
    hist_returns = [
        backtest_tickers_df[f"hist_return_{ticker}"].mean() for ticker in tickers_list
    ]
    synthetic_vars = [
        backtest_tickers_df[f"synthetic_VaR_{ticker}"].mean() for ticker in tickers_list
    ]

    return hist_returns, synthetic_vars
