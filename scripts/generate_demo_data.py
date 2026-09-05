"""Generate reproducible correlated, heteroskedastic demo returns."""

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    rng = np.random.default_rng(42)
    n_obs = 1500
    shocks = rng.multivariate_normal([0, 0], [[1.0, 0.55], [0.55, 1.0]], size=n_obs)
    returns = np.zeros((n_obs, 2))
    variances = np.ones((n_obs, 2)) * 0.8
    for t in range(1, n_obs):
        variances[t] = 0.03 + 0.08 * returns[t - 1] ** 2 + 0.88 * variances[t - 1]
        returns[t] = np.sqrt(variances[t]) * shocks[t]
    frame = pd.DataFrame({"date": pd.date_range("2018-01-01", periods=n_obs, freq="B"),
                          "equity_index": returns[:, 0], "credit_index": returns[:, 1]})
    destination = Path("data/demo_returns.csv")
    destination.parent.mkdir(exist_ok=True)
    frame.to_csv(destination, index=False)
    print(destination)


if __name__ == "__main__":
    main()
