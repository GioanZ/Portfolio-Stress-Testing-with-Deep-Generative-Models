# custom_libraries/metrics_validation.py

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
from scipy.stats import entropy

SEED_RANDOM = 29
np.random.seed(SEED_RANDOM)


def compute_portfolio_returns(returns_array, weights):
    return returns_array.dot(weights)


def calculate_var_es(portfolio_returns, alpha=5):
    var_threshold = np.percentile(portfolio_returns, alpha)
    es = portfolio_returns[portfolio_returns <= var_threshold].mean()
    return var_threshold, es


def select_extreme_scenarios(synthetic_returns, portfolio_weights, alpha=5):
    portfolio_returns = compute_portfolio_returns(synthetic_returns, portfolio_weights)
    var_threshold, es = calculate_var_es(portfolio_returns, alpha)
    extreme_indices = np.where(portfolio_returns <= var_threshold)[0]
    extreme_scenarios = synthetic_returns[extreme_indices]
    return extreme_scenarios, var_threshold, es, portfolio_returns


def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))
