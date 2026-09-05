"""End-to-end systemic-risk analysis orchestration."""

import numpy as np
import pandas as pd

from src.copula_model import CopulaModel
from src.covar_calculator import CoVaRCalculator
from src.data_loader import DataLoader
from src.garch_model import GARCHModel
from src.quantile_model import QuantileCoVaR


class AnalysisRunner:
    """Estimate directional Copula-CoVaR and a quantile-regression benchmark."""

    def __init__(self, file_path, input_type="returns"):
        self.loader = DataLoader(file_path, input_type=input_type)
        self.data = self.loader.load_data()
        self.results = {}

    def analyze_pair(self, asset_x, asset_y, alpha_list=(0.05,),
                     conditioning_levels=(0.50, 0.05, 0.01)):
        ret_x, ret_y = self.loader.get_two_returns(asset_x, asset_y)
        garch_x, garch_y = GARCHModel(ret_x, asset_x), GARCHModel(ret_y, asset_y)
        garch_x.fit()
        garch_y.fit()
        copula = CopulaModel(garch_x.pit_transform(), garch_y.pit_transform())
        copula.fit_all()

        output = {
            "asset_x": asset_x,
            "asset_y": asset_y,
            "copula": {"name": copula.get_best_name(), "theta": copula.get_theta(),
                       "aic": copula.get_aic(), "candidates": copula.results},
            "results": {},
        }
        xy = CoVaRCalculator(garch_y, copula, conditioning_axis="x")
        yx = CoVaRCalculator(garch_x, copula, conditioning_axis="y")
        for alpha in alpha_list:
            covar_xy = xy.compute_all(conditioning_levels, alpha)
            covar_yx = yx.compute_all(conditioning_levels, alpha)
            output["results"][f"alpha_{alpha}"] = {
                "direction_XY": {"covar": covar_xy, "delta": xy.compute_delta(covar_xy)},
                "direction_YX": {"covar": covar_yx, "delta": yx.compute_delta(covar_yx)},
            }

        benchmark = QuantileCoVaR(alpha=min(alpha_list))
        output["quantile_regression"] = {}
        for label, source, target in ((f"{asset_x}->{asset_y}", ret_x, ret_y),
                                      (f"{asset_y}->{asset_x}", ret_y, ret_x)):
            estimate = benchmark.fit(source, target)
            estimate["bootstrap_interval"] = benchmark.bootstrap_interval(source, target)
            estimate["rolling_delta_covar"] = benchmark.rolling(source, target)
            output["quantile_regression"][label] = estimate
        self.results[f"{asset_x}<->{asset_y}"] = output
        return output

    def analyze_all_pairs(self, max_pairs=None, **kwargs):
        pairs = self.loader.get_all_asset_pairs()[:max_pairs]
        for asset_x, asset_y in pairs:
            self.analyze_pair(asset_x, asset_y, **kwargs)
        return self.results

    def get_summary(self):
        rows = []
        for pair, data in self.results.items():
            for alpha_key, alpha_data in data["results"].items():
                alpha = float(alpha_key.split("_")[1])
                for suffix, source, target in (("XY", data["asset_x"], data["asset_y"]),
                                                ("YX", data["asset_y"], data["asset_x"])):
                    for beta, delta in alpha_data[f"direction_{suffix}"]["delta"].items():
                        rows.append({"pair": pair, "source": source, "target": target,
                                     "alpha": alpha, "beta": beta,
                                     "mean_delta_covar": float(np.mean(delta)),
                                     "latest_delta_covar": float(delta[-1]),
                                     "copula": data["copula"]["name"],
                                     "copula_aic": data["copula"]["aic"]})
        return pd.DataFrame(rows)
