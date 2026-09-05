"""Univariate volatility filtering."""

import numpy as np
from arch import arch_model
from scipy.stats import rankdata


class GARCHModel:
    """Constant-mean GARCH(1,1) with an empirical probability transform."""

    def __init__(self, returns, asset_name="asset"):
        self.returns = np.asarray(returns, dtype=float)
        self.asset_name = asset_name
        self.results = self.sigma = self.residuals = None
        self.std_residuals = self.mu = self.u = None

    def fit(self, dist="skewt"):
        if not np.isfinite(self.returns).all():
            raise ValueError(f"{self.asset_name} contains non-finite observations.")
        model = arch_model(self.returns, mean="Constant", vol="GARCH", p=1, q=1,
                           dist=dist, rescale=False)
        self.results = model.fit(disp="off", update_freq=0)
        if not self.results.optimization_result.success:
            raise RuntimeError(f"GARCH fit failed for {self.asset_name}.")
        self.sigma = np.asarray(self.results.conditional_volatility)
        self.residuals = np.asarray(self.results.resid)
        self.mu = self.returns - self.residuals
        self.std_residuals = self.residuals / self.sigma
        return self.results

    def pit_transform(self):
        """Return rank-based pseudo-observations for semiparametric copula fitting."""
        n_obs = len(self.std_residuals)
        self.u = rankdata(self.std_residuals, method="average") / (n_obs + 1.0)
        return self.u

    def get_sigma(self): return self.sigma
    def get_mu(self): return self.mu
    def get_z(self): return self.std_residuals
