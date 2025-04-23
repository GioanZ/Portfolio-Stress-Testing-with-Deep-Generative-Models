# custom_libraries/utils_plot.py

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

import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf

from custom_libraries.utils_stress_testing import (
    cluster_extreme_latent,
)

SEED_RANDOM = 29
np.random.seed(SEED_RANDOM)
tf.random.set_seed(SEED_RANDOM)


def plot_missing_values(df, title="Missing Values Heatmap"):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cmap="viridis", cbar=False)
    plt.title(title)
    plt.show()


def plot_correlation_matrix(df, title="Correlation Matrix", figsize=(7, 4), annot=True):
    corr_matrix = df.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=annot,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
    )
    plt.title(title)
    plt.show()


def plot_stock_returns(returns):
    fig, ax = plt.subplots(figsize=(12, 6))
    for ticker in returns.columns:
        ax.plot(returns.index, returns[ticker], label=ticker, alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Returns")
    plt.title("Stock Returns Over Time")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.15), fontsize="small", ncol=10)
    fig.tight_layout()
    plt.show()


import matplotlib.pyplot as plt


def plot_macro_trend(macro_df):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)

    # Top subplot: Inflation, Fed Funds, Unemployment, VIX
    ax1 = axes[0]
    ax1.set_ylabel("Rates & VIX", color="red")
    ax1.plot(
        macro_df.index,
        macro_df["inflation"],
        color="orange",
        linestyle="dashed",
        label="Inflation",
    )
    ax1.plot(
        macro_df.index,
        macro_df["fed_funds"],
        color="red",
        linestyle="dotted",
        label="Fed Funds",
    )
    ax1.plot(
        macro_df.index,
        macro_df["unemployment"],
        color="green",
        linestyle="dashdot",
        label="Unemployment",
    )
    ax1.plot(
        macro_df.index, macro_df["vix"], color="purple", linestyle="solid", label="VIX"
    )
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.legend(loc="upper left")
    ax1.set_title("Macro Trends Over Time")

    # Middle subplot: SP500
    ax2 = axes[1]
    ax2.set_ylabel("SP500", color="blue")
    ax2.plot(macro_df.index, macro_df["sp500"], color="blue", label="SP500")
    ax2.tick_params(axis="y", labelcolor="blue")
    ax2.legend(loc="upper left")
    ax2.set_title("SP500 Market Trend")

    # Bottom subplot: Short-Term & Long-Term Rates
    ax3 = axes[2]
    ax3.set_ylabel("Interest Rates", color="black")
    ax3.plot(
        macro_df.index,
        macro_df["eurusd"],
        linestyle="dashed",
        label="EUR/USD",
        color="green",
    )
    ax3.plot(
        macro_df.index,
        macro_df["short_term"],
        linestyle="dotted",
        label="Short Term",
        color="orange",
    )
    ax3.plot(
        macro_df.index,
        macro_df["long_term"],
        linestyle="dashdot",
        label="Long Term",
        color="purple",
    )
    ax3.tick_params(axis="y")
    ax3.legend(loc="upper right")
    ax3.set_title("Short-Term vs Long-Term Interest Rates")

    # Formatting
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_gan_losses(gen_losses, critic_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(gen_losses, label="Generator Loss")
    plt.plot(critic_losses, label="Critic Loss")
    plt.plot(val_losses, label="Validation Loss", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def plot_historical_vs_synthetic_var_period(backtest_df):
    # Plot the historical portfolio returns and synthetic VaR over the backtest period
    plt.figure(figsize=(12, 6))
    plt.plot(
        backtest_df["forecast_date"],
        backtest_df["hist_portfolio_return"],
        label="Historical Portfolio Return",
        linestyle="-",
    )
    plt.plot(
        backtest_df["forecast_date"],
        backtest_df["synthetic_VaR"],
        label="Worst Case Scenario",
        linestyle="--",
        color="red",
    )
    plt.xlabel("Forecast Date")
    plt.ylabel("Return")
    plt.title("Rolling Backtest: Historical Returns vs. Worst Case Scenario")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_mean_grad_features(input_test, generator):
    """
    Computes and plots the mean gradients for each feature on a forecast date
    A fixed noise (zeros) is used to isolate the effect of the condition vector
    """
    num_scenarios = len(input_test)
    gradients_list = []
    noise_dim = generator.input_shape[0][1]

    for i in range(num_scenarios):
        input_condition_tensor = tf.convert_to_tensor(
            np.atleast_2d(input_test.iloc[i]), dtype=tf.float32
        )
        noise = tf.zeros([1, noise_dim])
        with tf.GradientTape() as tape:
            tape.watch(input_condition_tensor)
            output = generator([noise, input_condition_tensor])
        gradients = tape.gradient(output, input_condition_tensor)
        gradients_list.append(gradients.numpy())

    mean_gradients = np.mean(gradients_list, axis=0)

    plt.figure(figsize=(10, 6))
    plt.bar(input_test.columns, mean_gradients[0])
    plt.title("Mean Gradients for Each Feature on Forecast Date")
    plt.xlabel("Feature")
    plt.ylabel("Mean Gradient")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


def plot_historical_vs_synthetic_var_period_per_tickers(backtest_df, returns_test):
    for stock in returns_test.columns:
        plt.figure(figsize=(12, 6))

        # Plot historical returns
        plt.plot(
            backtest_df["forecast_date"],
            backtest_df[f"hist_return_{stock}"],
            label=f"Historical Return {stock}",
            linestyle="-",
            color="blue",
        )

        # Plot synthetic VaR (5%)
        plt.plot(
            backtest_df["forecast_date"],
            backtest_df[f"synthetic_VaR_{stock}"],
            label=f"Worst Case Scenario {stock} (VaR 5%)",
            linestyle="--",
            color="red",
        )

        # Formatting
        plt.xlabel("Date")
        plt.ylabel("Return")
        plt.title(
            f"Rolling Backtest: Historical Return vs. Worst Case Scenario for {stock}"
        )
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()

        # Show the plot for each stock
        plt.show()


def plot_historical_vs_synthetic_var_period_per_tickers(
    backtest_df, returns_test, cols=3
):
    stocks = returns_test.columns
    num_stocks = len(stocks)
    rows = math.ceil(num_stocks / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(30, 6 * rows))
    axes = axes.flatten()

    for i, stock in enumerate(stocks):
        ax = axes[i]

        # Plot historical returns
        ax.plot(
            backtest_df["forecast_date"],
            backtest_df[f"hist_return_{stock}"],
            label=f"Historical Return {stock}",
            linestyle="-",
            color="blue",
        )

        # Plot synthetic VaR (5%)
        ax.plot(
            backtest_df["forecast_date"],
            backtest_df[f"synthetic_VaR_{stock}"],
            label=f"Worst Case Scenario {stock} (Var 5%)",
            linestyle="--",
            color="red",
        )

        # Formatting
        ax.set_xlabel("Date")
        ax.set_ylabel("Return")
        ax.set_title(f"{stock}")
        ax.legend()
        ax.grid(True)

    # Remove empty subplots if number of stocks is not a multiple of col
    # for i in range(num_stocks, len(axes)):
    #    fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def plot_bar_diff(hist_returns, synthetic_vars, tickers):
    """
    Difference (synthetic VaR - historical return) for each ticker
    """
    diff = np.array(synthetic_vars) - np.array(hist_returns)
    colors = ["red" if d > 0 else "green" for d in diff]

    plt.figure(figsize=(14, 6))
    plt.bar(tickers, diff, color=colors)
    plt.xlabel("Ticker")
    plt.ylabel("Synthetic Worst Case Scenario - Actual Return")
    plt.title(
        "Difference between Synthetic Worst Case Scenario and Actual Return per Ticker"
    )
    plt.xticks(rotation=45)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


def plot_distribution_synthetic_portfolio(
    synthetic_returns, var_threshold, actual_return=None
):
    """
    Plot the histogram of synthetic portfolio returns and mark the VaR threshold
    """
    plt.figure(figsize=(10, 6))
    plt.hist(synthetic_returns, bins=50, alpha=0.7, edgecolor="black", density=True)

    # Mark the VaR threshold
    plt.axvline(
        var_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"VaR (5%): {var_threshold:.4f}",
    )

    if actual_return is not None:
        # Mark the actual observed portfolio return
        plt.axvline(
            actual_return,
            color="blue",
            linestyle="-",
            linewidth=2,
            label=f"Actual Return: {actual_return:.4f}",
        )

    plt.xlabel("Portfolio Return")
    plt.ylabel("Density")
    if actual_return is not None:
        plt.title(
            "Distribution of Synthetic Portfolio Returns with Actual Return (One Day)"
        )
    else:
        plt.title("Distribution of Synthetic Portfolio Returns (One Day)")
    plt.legend(loc="upper center")
    plt.show()


def plot_scatter_actual_vs_synthetic_oblique(hist_returns, synthetic_vars, tickers):
    """
    Plots a scatter plot comparing the mean historical return with the mean synthetic VaR
    """
    plt.figure(figsize=(12, 6))
    plt.scatter(hist_returns, synthetic_vars, color="orange", s=60)

    min_val = min(min(hist_returns), min(synthetic_vars))
    max_val = max(max(hist_returns), max(synthetic_vars))
    plt.plot([min_val, max_val], [min_val, max_val], "k--")
    for i, ticker in enumerate(tickers):
        plt.annotate(
            ticker,
            (hist_returns[i], synthetic_vars[i]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
        )
    plt.xlabel("Mean Historical Return")
    plt.ylabel("Mean Synthetic Worst Case Scenario (Var 5%)")
    plt.title("Historical Return vs. Synthetic Worst Case Scenario Per Ticker (Var 5%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_latent_space_clustering(
    encoder,
    returns_train_scaled,
    seed_random,
    colors=["blue", "green", "orange"],
):
    """
    Plots a scatter plot of the latent space with clusters and cluster centers
    """

    _, _, generated_latent = encoder.predict(returns_train_scaled)
    _, clusters, cluster_centers = cluster_extreme_latent(
        generated_latent, seed_random, n_clusters=len(colors)
    )

    plt.figure(figsize=(8, 6))
    for i, color in enumerate(colors):
        plt.scatter(
            generated_latent[clusters == i, 0],
            generated_latent[clusters == i, 1],
            c=color,
            alpha=0.6,
            label=f"Cluster {i+1}",
        )
    plt.scatter(
        cluster_centers[:, 0],
        cluster_centers[:, 1],
        c="red",
        s=100,
        marker="x",
        label="Cluster Centers",
    )
    plt.xlabel("Latent 1")
    plt.ylabel("Latent 2")
    plt.title("Clustering of Synthetic Scenarios in Latent Space")
    plt.legend()
    plt.show()


def plot_tail_histogram(real_tail, synthetic_tail):
    plt.figure(figsize=(10, 5))
    plt.hist(real_tail, bins=50, alpha=0.5, density=True, label="Real Tail")
    plt.hist(synthetic_tail, bins=50, alpha=0.5, density=True, label="Synthetic Tail")
    plt.title("Tail Distribution Comparison (Density)")
    plt.xlabel("Returns")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


def plot_series(items, title, ylabel):
    plt.figure(figsize=(12, 6))
    for label, series, style, color in items:
        plt.plot(series.index, series, style, label=label, color=color)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
