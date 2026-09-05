"""Semiparametric quantile-regression CoVaR benchmark."""

import numpy as np
import statsmodels.api as sm


class QuantileCoVaR:
    """Adrian-Brunnermeier-style contemporaneous quantile benchmark."""

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def fit(self, conditioning_returns, outcome_returns):
        x = np.asarray(conditioning_returns, dtype=float)
        y = np.asarray(outcome_returns, dtype=float)
        design = sm.add_constant(x)
        fit = sm.QuantReg(y, design).fit(q=self.alpha, max_iter=2000)
        var_stress, var_median = np.quantile(x, [self.alpha, 0.50])
        covar_stress = float(fit.predict([1.0, var_stress])[0])
        covar_median = float(fit.predict([1.0, var_median])[0])
        return {"alpha": self.alpha, "intercept": float(fit.params[0]),
                "slope": float(fit.params[1]), "pseudo_r2": float(fit.prsquared),
                "covar_stress": covar_stress, "covar_median": covar_median,
                "delta_covar": covar_stress - covar_median}

    def bootstrap_interval(self, conditioning_returns, outcome_returns, n_boot=200,
                           block_size=20, seed=42):
        """Moving-block bootstrap interval for quantile-regression Delta-CoVaR."""
        x = np.asarray(conditioning_returns, dtype=float)
        y = np.asarray(outcome_returns, dtype=float)
        if len(x) != len(y):
            raise ValueError("Input series must have equal lengths.")
        rng = np.random.default_rng(seed)
        starts = np.arange(len(x) - block_size + 1)
        estimates = []
        blocks_needed = int(np.ceil(len(x) / block_size))
        for _ in range(n_boot):
            selected = rng.choice(starts, size=blocks_needed, replace=True)
            indices = np.concatenate([np.arange(start, start + block_size) for start in selected])
            indices = indices[:len(x)]
            estimates.append(self.fit(x[indices], y[indices])["delta_covar"])
        low, high = np.quantile(estimates, [0.025, 0.975])
        return {"method": "moving-block bootstrap", "n_boot": n_boot,
                "block_size": block_size, "confidence": 0.95,
                "lower": float(low), "upper": float(high)}

    def rolling(self, conditioning_returns, outcome_returns, window=500, step=20):
        """Estimate Delta-CoVaR over successive fixed-length windows."""
        x = np.asarray(conditioning_returns, dtype=float)
        y = np.asarray(outcome_returns, dtype=float)
        if window > len(x):
            return []
        return [{"end_observation": end,
                 "delta_covar": self.fit(x[end-window:end], y[end-window:end])["delta_covar"]}
                for end in range(window, len(x) + 1, step)]
