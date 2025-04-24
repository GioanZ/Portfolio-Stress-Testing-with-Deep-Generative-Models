# custom_libraries/validation.py

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

import statsmodels.api as sm

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


def prepare_returns(train, test, weights):
    all_r = pd.concat([train, test])
    all_r = all_r[~all_r.index.duplicated(keep="last")]
    return all_r.dot(weights)


def historical_var(port, window, level):
    alpha = 1 - level / 100
    return port.rolling(window).quantile(alpha)


def monte_carlo_var(port, window, level, sims):
    dates = port.index
    var = []
    for i in range(len(dates)):
        if i < window:
            var.append(np.nan)
            continue
        w = port.iloc[i - window : i]
        mu, sigma = w.mean(), w.std()
        sims_draws = np.random.default_rng().normal(mu, sigma, size=sims)
        var.append(np.percentile(sims_draws, 100 - level))
    return pd.Series(var, index=dates)


def conditional_var(port_train, fac_train, fac_test, level):
    alpha = 1 - level / 100
    df = pd.concat([port_train.rename("p"), fac_train.rename("f")], axis=1).dropna()
    model = sm.QuantReg(df["p"], sm.add_constant(df["f"])).fit(q=alpha)
    X_test = sm.add_constant(fac_test.rename("f"))
    return model.predict(X_test)


def scenario_loss(port, fac_full, fac_test, window):
    dates = fac_test.index
    losses = []
    for date in dates:
        pos = port.index.get_loc(date)
        if pos < window:
            losses.append(np.nan)
            continue
        idx = port.index[pos - window : pos]
        f = fac_full.loc[idx]
        p = port.loc[idx]
        beta = f.cov(p) / f.var()
        losses.append(beta * fac_test.loc[date])
    return pd.Series(losses, index=dates)


def calculate_all_var(
    returns_train,
    returns_test,
    portfolio_weights,
    input_train,
    input_test,
    var,
    window_size,
    num_scen,
):
    port_rets = prepare_returns(returns_train, returns_test, portfolio_weights)
    prt_train = port_rets.loc[returns_train.index]
    prt_test = port_rets.loc[returns_test.index]

    fac_full = pd.concat(
        [input_train["returns_sp500_roll_5"], input_test["returns_sp500_roll_5"]]
    )
    fac_test = input_test["returns_sp500_roll_5"]

    hist = historical_var(port_rets, window_size, var).loc[prt_test.index]
    mc = monte_carlo_var(port_rets, window_size, var, num_scen).loc[prt_test.index]
    cond = conditional_var(
        prt_train,
        input_train["returns_sp500_roll_5"],
        input_test["returns_sp500_roll_5"],
        var,
    ).loc[prt_test.index]
    scen = scenario_loss(port_rets, fac_full, fac_test, window_size).loc[prt_test.index]
    real = prt_test

    return real, cond, hist, mc, scen


def first_date_for_rolling(dates, start, window):
    """Earliest date so that a full `window` of history exists at start"""
    pos = dates.get_loc(pd.to_datetime(start))
    return dates[max(0, pos - window)]


def assemble_data(returns, factor, first, last):
    """Join returns with factor and select [first:last], dropping NaNs"""
    df = returns.join(factor.rename("factor"), how="inner")
    return df.loc[first:last].dropna()


def calc_rolling_betas(returns, factor, window):
    """Estimate daily betas = cov(R_i, F) / var(F) on a trailing window"""
    betas = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for i in range(window, len(returns)):
        idx = returns.index[i - window : i]
        Y = returns.loc[idx]
        X = factor.loc[idx]
        xc = X - X.mean()
        var = (xc**2).mean()
        cov = (Y.subtract(Y.mean(axis=0), axis=1)).multiply(xc, axis=0).mean(axis=0)
        betas.iloc[i] = cov / var
    return betas


def predict_loss(betas, factor, weights):
    """Compute predicted P&L = (weights · betas) * factor"""
    port_beta = betas.multiply(weights, axis=1).sum(axis=1)
    return port_beta * factor


def load_backtest_series(bt, start, end):
    """Extract historical & synthetic series from backtest_df"""
    df = bt.set_index("forecast_date").loc[start:end]
    return df["hist_portfolio_return"], df["synthetic_VaR"]


def align_series(hist, synth, pred):
    """Align three Series on their common dates"""
    idx = hist.index.intersection(synth.index).intersection(pred.index)
    return idx, hist.reindex(idx), synth.reindex(idx), pred.reindex(idx)


def run_pipeline(
    returns_all,
    input_features,
    weights,
    backtest_df,
    assets,
    start_backtest,
    end_backtest,
    window_size,
):
    # build factor series (same-day)
    factor = input_features["returns_sp500_roll_5"]
    # extend start backwards
    ext_start = first_date_for_rolling(returns_all.index, start_backtest, window_size)
    master = assemble_data(returns_all, factor, ext_start, end_backtest)
    # rolling betas
    betas_all = calc_rolling_betas(master[assets], master["factor"], window_size)
    betas_bt = betas_all.loc[start_backtest:end_backtest]
    # predicted loss
    pred_loss = predict_loss(
        betas_bt, master["factor"].loc[start_backtest:end_backtest], weights
    )
    # load backtest series
    hist, synth = load_backtest_series(backtest_df, start_backtest, end_backtest)
    # align
    idx, hist_s, synth_s, pred_s = align_series(hist, synth, pred_loss)

    return idx, hist_s, synth_s, pred_s
