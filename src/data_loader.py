"""Data ingestion and validation utilities."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


class DataLoader:
    """Load aligned prices or returns from a wide CSV/Excel file."""

    def __init__(self, file_path, input_type="returns", return_scale=100.0):
        self.file_path = Path(file_path)
        self.input_type = input_type
        self.return_scale = float(return_scale)
        self.data = None
        self.asset_names = []

    def load_data(self, date_col="date"):
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        suffix = self.file_path.suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(self.file_path)
        elif suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(self.file_path)
        else:
            raise ValueError("Supported formats are CSV, XLSX and XLS.")
        if date_col not in frame:
            raise ValueError(f"Required date column '{date_col}' is missing.")
        if self.input_type not in {"prices", "returns"}:
            raise ValueError("input_type must be 'prices' or 'returns'.")

        dates = pd.to_datetime(frame.pop(date_col), errors="raise")
        values = frame.apply(pd.to_numeric, errors="coerce")
        if values.shape[1] < 2:
            raise ValueError("At least two asset columns are required.")
        if self.input_type == "prices":
            if (values <= 0).any().any():
                raise ValueError("Prices must be positive to compute log returns.")
            values = np.log(values).diff() * self.return_scale
        clean = pd.concat([dates.rename("date"), values], axis=1)
        clean = clean.sort_values("date").drop_duplicates("date", keep="last").dropna()
        if len(clean) < 100:
            raise ValueError("At least 100 complete observations are required.")
        self.data = clean.reset_index(drop=True)
        self.asset_names = list(values.columns)
        return self.data

    def get_asset_returns(self, name):
        if self.data is None:
            raise RuntimeError("Call load_data() first.")
        if name not in self.asset_names:
            raise ValueError(f"Unknown asset '{name}'. Available: {self.asset_names}")
        return self.data[name].to_numpy(dtype=float)

    def get_two_returns(self, asset_x, asset_y):
        return self.get_asset_returns(asset_x), self.get_asset_returns(asset_y)

    def get_all_asset_pairs(self):
        return list(combinations(self.asset_names, 2))
