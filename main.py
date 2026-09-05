"""Command-line entry point for Systemic Risk CoVaR Lab."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.analysis_runner import AnalysisRunner
from src.visualization import plot_delta_covar


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate directional systemic risk with CoVaR.")
    parser.add_argument("--data", required=True, help="Wide CSV/XLSX file with a date column.")
    parser.add_argument("--assets", nargs=2, metavar=("SOURCE", "TARGET"), required=True)
    parser.add_argument("--input-type", choices=("returns", "prices"), default="returns")
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.05])
    parser.add_argument("--output", default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = AnalysisRunner(args.data, input_type=args.input_type)
    result = runner.analyze_pair(*args.assets, alpha_list=args.alpha)
    (output_dir / "pair_analysis.json").write_text(
        json.dumps(result, indent=2, default=_json_default), encoding="utf-8"
    )
    runner.get_summary().to_csv(output_dir / "risk_summary.csv", index=False)
    plot_delta_covar(result, output_dir / "delta_covar.png")
    print(runner.get_summary().to_string(index=False))
    print(f"\nOutputs written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
