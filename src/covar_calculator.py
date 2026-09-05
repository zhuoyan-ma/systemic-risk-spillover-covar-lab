"""Dynamic CoVaR calculations."""

import numpy as np


class CoVaRCalculator:
    def __init__(self, outcome_garch, copula, conditioning_axis="x"):
        self.outcome = outcome_garch
        self.copula = copula
        self.conditioning_axis = conditioning_axis

    def compute_covar(self, conditioning_quantile, outcome_quantile):
        probability = self.copula.conditional_quantile(
            conditioning_quantile, outcome_quantile, axis=self.conditioning_axis
        )
        innovation = np.quantile(self.outcome.get_z(), probability)
        return self.outcome.get_mu() + self.outcome.get_sigma() * innovation

    def compute_all(self, conditioning_levels=(0.50, 0.10, 0.05, 0.01), alpha=0.05):
        return {level: self.compute_covar(level, alpha) for level in conditioning_levels}

    @staticmethod
    def compute_delta(covar_results, baseline=0.50):
        reference = covar_results[baseline]
        return {level: values - reference for level, values in covar_results.items()
                if level != baseline}
