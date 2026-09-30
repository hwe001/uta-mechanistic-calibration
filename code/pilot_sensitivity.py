"""Pilot global sensitivity and practical-identifiability screen."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # CSV sensitivity outputs are primary.
    plt = None

from solver_adapter import simulate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    grids = {
        "radial_radius_mm": np.linspace(.10, .30, 17),
        "av_radius_mm": np.linspace(.08, .60, 14),
        "terminal_resistance": np.geomspace(.4, 16.0, 15),
        "terminal_compliance": np.geomspace(2e-9, 5e-8, 13),
        "youngs_modulus_pa": np.geomspace(5e5, 4e6, 13),
        "heart_rate_bpm": np.linspace(55, 105, 11),
    }
    defaults = {
        "radial_radius_mm": .2, "av_radius_mm": .2,
        "terminal_resistance": 1.6, "terminal_compliance": 1e-8,
        "youngs_modulus_pa": 1.5e6, "heart_rate_bpm": 72.0,
    }
    adapter_names = {
        "radial_radius_mm": "radial_radius", "av_radius_mm": "av_radius",
        "terminal_resistance": "terminal_resistance",
        "terminal_compliance": "terminal_compliance",
        "youngs_modulus_pa": "youngs_modulus", "heart_rate_bpm": "heart_rate",
    }
    rows = []
    for parameter, values in grids.items():
        for value in values:
            kwargs = {adapter_names[k]: v for k, v in defaults.items()}
            kwargs[adapter_names[parameter]] = float(value)
            result = simulate(args.repo, **kwargs)
            rows.append({"parameter": parameter, "value": float(value), **result})
    output = pd.DataFrame(rows)
    output.to_csv(args.output / "one_at_a_time_sensitivity.csv", index=False)

    summary_rows = []
    for parameter, group in output.groupby("parameter"):
        for metric in ["PI", "RI", "SD"]:
            summary_rows.append({
                "parameter": parameter,
                "metric": metric,
                "minimum": group[metric].min(),
                "maximum": group[metric].max(),
                "range": group[metric].max() - group[metric].min(),
                "spearman": group[["value", metric]].corr(method="spearman").iloc[0, 1],
            })
    pd.DataFrame(summary_rows).to_csv(args.output / "sensitivity_summary.csv", index=False)

    if plt is not None:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), constrained_layout=True)
        for ax, (parameter, group) in zip(axes.flat, output.groupby("parameter", sort=False)):
            ax.plot(group["value"], group["PI"], marker="o", ms=3, lw=1.5)
            if parameter in {"terminal_resistance", "terminal_compliance", "youngs_modulus_pa"}:
                ax.set_xscale("log")
            ax.set_title(parameter.replace("_", " "))
            ax.set_ylabel("Predicted PI")
        fig.savefig(args.output / "pilot_sensitivity_pi.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
