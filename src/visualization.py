"""Publication-ready plots for risk-spillover results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_delta_covar(result, output_path):
    """Plot time-varying delta-CoVaR for both directions and tail levels."""
    fig, axes = plt.subplots(len(result["results"]), 1, figsize=(10, 4.5), squeeze=False)
    for axis, (alpha_key, block) in zip(axes[:, 0], result["results"].items()):
        directions = (
            ("direction_XY", f'{result["asset_x"]} -> {result["asset_y"]}'),
            ("direction_YX", f'{result["asset_y"]} -> {result["asset_x"]}'),
        )
        for direction, label in directions:
            for beta, values in block[direction]["delta"].items():
                axis.plot(values, label=f"{label} | stress={beta:g}", linewidth=1.1)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_title(f"Dynamic Delta-CoVaR ({alpha_key.replace('_', '=')})")
        axis.set_ylabel("return units")
        axis.legend(frameon=False, ncol=2, fontsize=8)
    axes[-1, 0].set_xlabel("observation")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
