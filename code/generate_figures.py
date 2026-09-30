"""Manuscript figures for the UtA mechanistic modelling project.

Generates four publication figures from the saved analysis outputs:

    fig1_observed_vs_predicted   UtA-PI against gestational age: individual
                                 examinations, bin medians, the radial
                                 mechanistic trajectory with bootstrap band,
                                 and the M0 statistical benchmark (emphasis
                                 form; the retained model is the accent hue).
    fig2_model_comparison        Held-out validation metrics: RMSE (log-PI)
                                 and calibration slope per model, with the
                                 fold-level boundary-pinned proportions.
    fig3_identifiability         Local Jacobian singular values and the
                                 bounded synthetic-recovery parameter spans
                                 that demonstrate practical non-identifiability.
    fig4_mechanistic_trajectories Predicted PI trajectories with bootstrap
                                 bands plus per-model latent-parameter
                                 small multiples; constraint ceilings are
                                 annotated where fits are boundary-pinned.

Model identity colours are fixed across all figures (radial = blue, shunt =
aqua, distal = yellow; benchmark and observed data in neutral ink). Aqua and
yellow sit below 3:1 contrast on the light surface, so every coloured series
carries a direct text label. Outputs are written as 300-dpi PNG and PDF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_fit import load_core_examinations

MODEL_COLORS = {
    "radial": "#2a78d6",
    "shunt": "#1baf7a",
    "distal": "#eda100",
}
MODEL_LABELS = {
    "radial": "Radial-dilation model",
    "shunt": "A-V-shunt model",
    "distal": "Distal-resistance model",
}
MODEL_ORDER = ["radial", "shunt", "distal"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BENCHMARK_COLOR = "#898781"

plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_inputs(results: Path, input_path: Path) -> dict:
    calibration = json.loads((results / "mechanistic_calibration" / "calibration_summary.json").read_text())
    validation = json.loads((results / "woman_level_validation" / "validation_summary.json").read_text())
    identifiability = json.loads((results / "identifiability" / "identifiability_summary.json").read_text())
    return {
        "bins": pd.read_csv(results / "mechanistic_calibration" / "ga_bin_targets.csv"),
        "trajectories": pd.read_csv(results / "mechanistic_calibration" / "latent_trajectories.csv"),
        "benchmark_curve": pd.read_csv(
            results / "longitudinal_benchmark" / "trajectory_predictions.csv"),
        "comparison": pd.read_csv(results / "woman_level_validation" / "model_comparison.csv"),
        "core": load_core_examinations(input_path),
        "calibration": calibration,
        "validation": validation,
        "identifiability": identifiability,
    }


# ---------------------------------------------------------------------------
# Figure 1: observed vs predicted UtA-PI
# ---------------------------------------------------------------------------

def figure_observed_vs_predicted(data: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    style_axis(ax)

    core = data["core"]
    ax.scatter(core["GA_weeks"], core["UtA_PI"], s=7, color=MUTED, alpha=0.18,
               edgecolors="none", rasterized=True, label="Individual examinations")

    bins = data["bins"]
    ax.scatter(bins["ga_median"], bins["target_PI"], s=20, color=INK, zorder=3,
               label="Two-week bin medians", linewidths=0)

    radial = data["trajectories"]
    radial = radial[radial["model"] == "radial"]
    ax.fill_between(radial["ga"], radial["predicted_PI_ci_low"],
                    radial["predicted_PI_ci_high"], color=MODEL_COLORS["radial"],
                    alpha=0.15, linewidth=0, label="Radial model bootstrap 95% band")
    ax.plot(radial["ga"], radial["predicted_PI"], color=MODEL_COLORS["radial"],
            linewidth=1.8, label="Radial model")

    curve = data["benchmark_curve"]
    ax.plot(curve["GA_weeks"], curve["pred_PI"], color=BENCHMARK_COLOR,
            linewidth=1.4, linestyle=(0, (4, 2)), label="Statistical benchmark (M0)")

    ax.set_xlim(13.5, 40.5)
    ax.set_xlabel("Gestational age (weeks)")
    ax.set_ylabel("Uterine artery PI")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save_figure(fig, output, "fig1_observed_vs_predicted")


# ---------------------------------------------------------------------------
# Figure 2: model comparison (held-out validation)
# ---------------------------------------------------------------------------

def figure_model_comparison(data: dict, output: Path) -> None:
    comparison = data["comparison"].set_index("model")
    boundary = {
        row["model"]: row
        for row in data["validation"]["boundary_fits"]["fold_level"]
    }
    n_folds = int(boundary[MODEL_ORDER[0]]["folds"]) if boundary else 0

    fig, (ax_rmse, ax_cov) = plt.subplots(
        1, 2, figsize=(6.4, 2.3), gridspec_kw={"width_ratios": [1.55, 1]})
    y = np.arange(len(MODEL_ORDER))

    for ax, column, xlabel, reference in [
        (ax_rmse, "rmse_logPI", "Held-out RMSE, log(UtA-PI)", None),
        (ax_cov, "coverage_90pct", "90% interval coverage (nominal 0.90)", 0.90),
    ]:
        style_axis(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        values = [comparison.loc[m, column] for m in MODEL_ORDER]
        if reference is not None:
            ax.axvline(reference, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
        ax.hlines(y, 0, values, color=[MODEL_COLORS[m] for m in MODEL_ORDER],
                  linewidth=2.0)
        ax.scatter(values, y, s=34, color=[MODEL_COLORS[m] for m in MODEL_ORDER],
                   zorder=3, linewidths=0)
        ax.set_yticks(y, [MODEL_LABELS[m] for m in MODEL_ORDER])
        ax.set_xlabel(xlabel)
        ax.margins(y=0.18)
        ax.invert_yaxis()

    rmse_values = [comparison.loc[m, "rmse_logPI"] for m in MODEL_ORDER]
    ax_rmse.set_xlim(0, max(rmse_values) * 2.1)
    for i, model in enumerate(MODEL_ORDER):
        if model in boundary:
            ceiling = int(boundary[model]["folds_trajectory_at_ceiling"])
            ax_rmse.annotate(
                f"ceiling {ceiling}/{n_folds} folds",
                xy=(comparison.loc[model, "rmse_logPI"], i),
                xytext=(6, 0), textcoords="offset points",
                va="center", ha="left", fontsize=6.6, color=INK_2,
            )
        else:
            ax_rmse.annotate(
                "statistical benchmark",
                xy=(comparison.loc[model, "rmse_logPI"], i),
                xytext=(6, 0), textcoords="offset points",
                va="center", ha="left", fontsize=6.6, color=INK_2,
            )
    ax_cov.set_xlim(0.55, 1.02)

    fig.tight_layout()
    save_figure(fig, output, "fig2_model_comparison")


# ---------------------------------------------------------------------------
# Figure 3: identifiability
# ---------------------------------------------------------------------------

def figure_identifiability(data: dict, output: Path) -> None:
    ident = data["identifiability"]
    fig, (ax_sv, ax_recovery) = plt.subplots(
        1, 2, figsize=(6.6, 3.0), gridspec_kw={"width_ratios": [1, 1.6]}
    )

    style_axis(ax_sv)
    singular = ident["singular_values"]
    ax_sv.bar(np.arange(len(singular)), singular, width=0.55,
              color=MODEL_COLORS["radial"], linewidth=0)
    ax_sv.set_yscale("log")
    ax_sv.set_ylim(1e-3, 6)
    ax_sv.set_xticks(np.arange(len(singular)),
                     [f"SV{i + 1}" for i in range(len(singular))])
    ax_sv.set_ylabel("Singular value (log scale)")
    ax_sv.set_title(f"Local Jacobian, rank {ident['jacobian_rank_numeric']} "
                    f"of {ident['jacobian_shape'][1]} parameters")
    for i, value in enumerate(singular):
        ax_sv.text(i, value * 1.25, f"{value:.3g}", ha="center", fontsize=7, color=INK_2)
    ax_sv.text(0.03, 0.92,
               f"condition number {ident['condition_number_nonzero_subspace']:.0f}",
               transform=ax_sv.transAxes, fontsize=7, color=INK_2)

    # Recovery spans as fold-range bars (high/low) with the recovered range
    # annotated — a shared raw log axis would render the tighter spans
    # sub-pixel and unreadable.
    style_axis(ax_recovery)
    ranges = ident["recovery_parameter_ranges"]
    labels = {
        "radial_radius_mm": "Radial radius (mm)",
        "av_radius_mm": "A–V shunt radius (mm)",
        "terminal_resistance": "Terminal resistance (Pa·s/mm³)",
        "terminal_compliance": "Terminal compliance (mm³/Pa)",
        "youngs_modulus_pa": "Young's modulus (Pa)",
        "heart_rate": "Heart rate (bpm)",
    }
    order = sorted(ranges, key=lambda name: ranges[name][1] / ranges[name][0])
    y = np.arange(len(order))
    spans = [ranges[name][1] / ranges[name][0] for name in order]
    ax_recovery.barh(y, spans, height=0.55, color=MODEL_COLORS["radial"], linewidth=0)
    for i, name in enumerate(order):
        low, high = ranges[name]
        ax_recovery.text(spans[i] * 1.06, i, f"{low:.3g}–{high:.3g}",
                         va="center", fontsize=6.6, color=INK_2)
    ax_recovery.set_yticks(y, [labels[name] for name in order])
    ax_recovery.set_xlim(0, max(spans) * 1.45)
    ax_recovery.set_xlabel("Recovered span (fold range, high/low)")
    ax_recovery.set_title("Synthetic inverse recovery, 3 starts")
    fig.tight_layout()
    save_figure(fig, output, "fig3_identifiability")


# ---------------------------------------------------------------------------
# Figure 4: mechanistic trajectories with constraint ceilings
# ---------------------------------------------------------------------------

def figure_mechanistic_trajectories(data: dict, output: Path) -> None:
    trajectories = data["trajectories"]
    bins = data["bins"]
    fits = data["calibration"]["smooth_fits"]
    grid_spec = data["calibration"]["grid"]["spec"]
    adapter_of = {"radial": "radial_radius", "shunt": "av_radius",
                  "distal": "terminal_resistance"}

    fig = plt.figure(figsize=(6.9, 3.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.7, 1.0])
    ax_main = fig.add_subplot(gs[:, 0])
    style_axis(ax_main)

    ax_main.scatter(bins["ga_median"], bins["target_PI"], s=18, color=INK,
                    zorder=3, linewidths=0, label="Observed bin medians")
    for model in MODEL_ORDER:
        part = trajectories[trajectories["model"] == model]
        color = MODEL_COLORS[model]
        ax_main.fill_between(part["ga"], part["predicted_PI_ci_low"],
                             part["predicted_PI_ci_high"], color=color,
                             alpha=0.15, linewidth=0)
        ax_main.plot(part["ga"], part["predicted_PI"], color=color,
                     linewidth=1.8, label=MODEL_LABELS[model])

    ax_main.set_xlim(13.5, 40.5)
    ax_main.set_ylim(0.42, 1.44)
    ax_main.set_xlabel("Gestational age (weeks)")
    ax_main.set_ylabel("Predicted uterine artery PI")
    ax_main.legend(frameon=False, loc="upper left", fontsize=6.8)

    # Constraint-ceiling annotations for boundary-pinned models.
    pinned = {m: fits[m]["boundary_bins"] for m in MODEL_ORDER}
    total_bins = len(bins)
    if pinned["distal"]:
        ceiling = trajectories[trajectories["model"] == "distal"]["predicted_PI"].max()
        ax_main.annotate(
            f"at ceiling for {pinned['distal']}/{total_bins} bins",
            xy=(24.5, ceiling), xytext=(24.5, 0.46),
            fontsize=6.8, color=INK_2, ha="center",
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.7},
        )
    if pinned["shunt"]:
        shunt = trajectories[trajectories["model"] == "shunt"]
        early = shunt[shunt["ga"] < 20]["predicted_PI"].max()
        ax_main.annotate(
            f"at ceiling for {pinned['shunt']}/{total_bins} bins (early gestation)",
            xy=(15.5, early), xytext=(20.5, 0.98),
            fontsize=6.8, color=INK_2,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.7},
        )

    for row, model in enumerate(MODEL_ORDER):
        ax = fig.add_subplot(gs[row, 1])
        style_axis(ax)
        part = trajectories[trajectories["model"] == model]
        spec = grid_spec[adapter_of[model]]
        color = MODEL_COLORS[model]
        ax.axhspan(spec["low"], spec["high"], color=GRID, alpha=0.35, linewidth=0)
        ax.plot(part["ga"], part["param_estimate"], color=color, linewidth=1.6)
        ax.fill_between(part["ga"], part["param_ci_low"], part["param_ci_high"],
                        color=color, alpha=0.25, linewidth=0)
        ax.set_ylabel(f"{model}\n[{spec['low']}–{spec['high']}]",
                      fontsize=6.5, color=INK_2)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.tick_params(labelbottom=False)
    ax.set_xlabel("Gestational age (weeks)", fontsize=7.5)

    save_figure(fig, output, "fig4_mechanistic_trajectories")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path, help="Source workbook (read-only)")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    data = load_inputs(args.results, args.input)
    figure_observed_vs_predicted(data, args.output)
    figure_model_comparison(data, args.output)
    figure_identifiability(data, args.output)
    figure_mechanistic_trajectories(data, args.output)
    print(f"Figures written to {args.output}")


if __name__ == "__main__":
    main()
