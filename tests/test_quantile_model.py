import numpy as np

from src.quantile_model import QuantileCoVaR


def test_quantile_model_recovers_positive_spillover():
    rng = np.random.default_rng(7)
    x = rng.normal(size=1000)
    y = 0.7 * x + rng.normal(scale=0.25, size=1000)
    result = QuantileCoVaR(alpha=0.05).fit(x, y)
    assert result["slope"] > 0.5
    assert result["delta_covar"] < 0
