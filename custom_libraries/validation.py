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

from scipy.stats import ks_2samp, wasserstein_distance as emd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from custom_libraries.metrics_validation import jensen_shannon_divergence

SEED_RANDOM = 29
np.random.seed(SEED_RANDOM)


def filter_tail(real_returns, synthetic_returns, mode="negative", percentile=5):
    """
    Filters the tail of the distribution
    mode: "negative" for all returns < 0, or "percentile" for left-tail cutoff
    """
    if mode == "negative":
        real_tail = real_returns[real_returns < 0]
        synthetic_tail = synthetic_returns[synthetic_returns < 0]
    elif mode == "percentile":
        cutoff = np.percentile(real_returns, percentile)
        real_tail = real_returns[real_returns <= cutoff]
        synthetic_tail = synthetic_returns[synthetic_returns <= cutoff]
    else:
        raise ValueError("mode must be 'negative' or 'percentile'")

    return real_tail, synthetic_tail


def compare_tail_distributions(real_tail, synthetic_tail):
    """
    Returns KS, EMD, and JSD between the real and synthetic tail distributions
    """
    results = {}
    if len(real_tail) > 1 and len(synthetic_tail) > 1:
        (
            results["Kolmogorov Smirnov Statistic (KS Stat)"],
            results["KS p-value"],
        ) = ks_2samp(synthetic_tail, real_tail)
        results["Earth Mover's Distance (EMD)"] = emd(real_tail, synthetic_tail)

        hist_real, bin_edges = np.histogram(real_tail, bins=50, density=True)
        hist_syn, _ = np.histogram(synthetic_tail, bins=bin_edges, density=True)
        results["Jensen-Shannon Divergence (JSD)"] = jensen_shannon_divergence(
            hist_real + 1e-8, hist_syn + 1e-8
        )

    return results


def adversarial_tail_test(
    real_tail, synthetic_tail, seed, n_estimators=100, max_eval_size=2000, test_size=0.3
):
    """
    Random Forest classifier to distinguish real vs. synthetic tails
    """

    if len(real_tail) < 2 or len(synthetic_tail) < 2:
        return {"Train Accuracy": None, "Test Accuracy": None}

    real_size = min(max_eval_size, len(real_tail))
    syn_size = min(real_size, len(synthetic_tail))

    real_sample = np.random.choice(real_tail, real_size, replace=False).reshape(-1, 1)
    syn_sample = np.random.choice(synthetic_tail, syn_size, replace=False).reshape(
        -1, 1
    )

    X = np.vstack([real_sample, syn_sample])
    y = np.array([0] * real_size + [1] * syn_size)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(X_train, y_train)

    return {
        "Train Accuracy": clf.score(X_train, y_train),
        "Test Accuracy": clf.score(X_test, y_test),
    }
