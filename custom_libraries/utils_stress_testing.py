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

import tensorflow as tf

from scipy.stats import ks_2samp, wasserstein_distance

from sklearn.cluster import KMeans

from .metrics_validation import (
    compute_portfolio_returns,
    jensen_shannon_divergence,
    select_extreme_scenarios,
)


def generate_synthetic_scenarios(
    generator, decoder, macro_train_scaled, returns_scaler, noise_dim, num_samples
):
    idx = np.random.choice(macro_train_scaled.shape[0], num_samples, replace=True)
    macro_conditions_sample = macro_train_scaled[idx]
    macro_conditions_sample = tf.convert_to_tensor(
        macro_conditions_sample, dtype=tf.float32
    )
    noise = tf.random.normal([num_samples, noise_dim])
    generated_latent = generator([noise, macro_conditions_sample]).numpy()

    # Conditional decoding
    synthetic_returns_scaled = decoder.predict(
        [generated_latent, macro_conditions_sample]
    )
    synthetic_returns = returns_scaler.inverse_transform(synthetic_returns_scaled)

    return synthetic_returns, generated_latent, macro_conditions_sample


def compare_distributions(
    returns_test,
    synthetic_returns,
    synthetic_portfolio_returns,
    historical_returns,
):
    emd_distance = wasserstein_distance(historical_returns, synthetic_portfolio_returns)

    hist_real, bin_edges = np.histogram(
        returns_test.values[:, 0], bins=50, density=True
    )
    hist_syn, _ = np.histogram(synthetic_returns[:, 0], bins=bin_edges, density=True)
    js_div = jensen_shannon_divergence(hist_real + 1e-8, hist_syn + 1e-8)
    ks_stat, ks_pvalue = ks_2samp(returns_test.values[:, 0], synthetic_returns[:, 0])

    return emd_distance, js_div, ks_stat, ks_pvalue


def autocorr(x, lag=1):
    return np.corrcoef(x[:-lag], x[lag:])[0, 1]


def cluster_extreme_latent(z_codes, random_seed, n_clusters=3):
    """Clustering Function for Latent Space Scenarios"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)
    clusters = kmeans.fit_predict(z_codes)
    global_center = np.mean(z_codes, axis=0)
    distances = np.linalg.norm(kmeans.cluster_centers_ - global_center, axis=1)
    extreme_cluster = np.argmax(distances)
    extreme_indices = np.where(clusters == extreme_cluster)[0]
    return extreme_indices, clusters, kmeans.cluster_centers_


# TODO
def run_stress_testing(
    portfolio_weights, returns_test, generated_latent, synthetic_returns, random_seed
):
    extreme_scenarios, var_threshold, es, synthetic_portfolio_returns = (
        select_extreme_scenarios(synthetic_returns, portfolio_weights, alpha=5)
    )

    historical_portfolio_returns = compute_portfolio_returns(
        returns_test.values, portfolio_weights
    )

    emd_distance, js_div, ks_stat, ks_pvalue = compare_distributions(
        returns_test,
        synthetic_returns,
        synthetic_portfolio_returns,
        historical_portfolio_returns,
    )

    autocorr_real = np.mean(
        [
            autocorr(returns_test.values[:, i])
            for i in range(returns_test.values.shape[1])
        ]
    )
    autocorr_syn = np.mean(
        [autocorr(synthetic_returns[:, i]) for i in range(synthetic_returns.shape[1])]
    )

    extreme_indices_cluster, clusters, cluster_centers = cluster_extreme_latent(
        generated_latent, random_seed, n_clusters=3
    )

    print("--- Stress Testing Results ---")
    print(f"VaR (5th percentile): {var_threshold:.4f}")
    print(f"Expected Shortfall: {es:.4f}")
    print(f"Number of extreme scenarios (VaR): {extreme_scenarios.shape[0]}")
    print(f"Earth Mover's Distance: {emd_distance:.4f}")
    print(f"Jensen-Shannon Divergence (asset 1): {js_div:.4f}")
    print(f"KS-test statistic (asset 1): {ks_stat:.4f}, p-value: {ks_pvalue:.4f}")
    print(
        f"Mean autocorrelation lag-1: real = {autocorr_real:.4f}, synthetic = {autocorr_syn:.4f}"
    )
    print(
        f"Number of scenarios in extreme cluster (K-Means): {len(extreme_indices_cluster)}"
    )

    return (
        synthetic_portfolio_returns,
        extreme_scenarios,
        var_threshold,
        clusters,
        cluster_centers,
    )
