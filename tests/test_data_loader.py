import numpy as np
import pandas as pd

from src.data_loader import DataLoader


def test_price_input_becomes_percentage_log_returns(tmp_path):
    dates = pd.date_range("2024-01-01", periods=101)
    frame = pd.DataFrame({"date": dates, "a": 100 * 1.01 ** np.arange(101),
                          "b": 80 * 1.005 ** np.arange(101)})
    path = tmp_path / "prices.csv"
    frame.to_csv(path, index=False)
    loader = DataLoader(path, input_type="prices")
    data = loader.load_data()
    assert len(data) == 100
    assert np.allclose(data["a"], 100 * np.log(1.01))
    assert loader.get_all_asset_pairs() == [("a", "b")]
