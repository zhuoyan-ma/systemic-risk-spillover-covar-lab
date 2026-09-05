import numpy as np

from src.copula_model import CopulaModel


def test_independent_gaussian_conditional_quantile_matches_alpha():
    model = CopulaModel(np.linspace(0.01, 0.99, 100), np.linspace(0.99, 0.01, 100))
    model.best_name = "gaussian"
    model.best_theta = 0.0
    assert np.isclose(model.conditional_quantile(0.05, 0.10), 0.10, atol=1e-6)
