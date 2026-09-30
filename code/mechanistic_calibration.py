"""Reproducible mechanistic calibration pipeline.

Replaces the earlier inline (unsaved) calibration attempt. Stages:

1. Cohort binning: 2-week gestational-age bins over the core analysis set
   (same eligibility rules as uta_data_audit.py), giving per-bin median GA,
   median UtA-PI and examination count.
2. Solver lookup grid: one-at-a-time response axes for radial-artery radius,
   terminal resistance and A-V shunt radius, evaluated with the full solver.
3. Interpolation validation: off-grid values per axis, full-solver PI compared
   against linear and monotone-cubic (PCHIP) interpolation of the grid, against
   a pre-specified 0.005 PI tolerance. A small radial x shunt corner check
   documents the non-additivity that motivates one-at-a-time calibration.
4. Per-bin one-dimensional inversion for each mechanism, with honest reporting
   of boundary estimates (axes whose PI range cannot reach the target).
5. Smooth monotone calibration: a four-parameter logistic transition in
   log-parameter space per mechanism, fitted to bin medians with an
   examination-count-weighted log-PI loss via multi-start bounded optimization;
   bootstrap confidence intervals with a fixed seed.
6. Model-comparison table and a summary JSON, including corrected grid
   metadata (the previously saved metadata described the coarse 31-point grid).

No stage other than grid construction and interpolation validation calls the
solver; fitting and bootstrap operate on the lookup grid alone.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

from solver_adapter import simulate

PI_TOLERANCE = 0.005
SD_REL_TOLERANCE = 0.01  # pre-specified acceptance: relative S/D error < 1%
MECHANISMS = {
    "radial": "radial_radius",
    "distal": "terminal_resistance",
    "shunt": "av_radius",
}
GRID_SPEC = {
    "radial_radius": (0.1, 0.5, 201),
    "terminal_resistance": (0.2, 5.0, 101),
    "av_radius": (0.05, 0.8, 101),
}


# ---------------------------------------------------------------------------
# Stage 1: cohort binning
# ---------------------------------------------------------------------------

def load_core_bins(input_path: Path) -> pd.DataFrame:
    """2-week GA bins of the core analysis set, replicating the audit rules."""
    data = pd.read_excel(input_path, sheet_name=0)
    composite = data["Exam_Date"].astype("string").str.contains("；", regex=False, na=False)
    ri_sd_error = (data["UtA_RI"] - (1.0 - 1.0 / data["UtA_SD"])).abs()
    eligible = (
        ~composite
        & data["GA_weeks"].ge(14)
        & data["UtA_PI"].notna()
        & ri_sd_error.le(0.02)
    )
    core = data.loc[eligible, ["GA_weeks", "UtA_PI"]].copy()
    bins = np.arange(14.0, 42.0, 2.0)
    core["ga_bin"] = pd.cut(core["GA_weeks"], bins=bins, right=False)
    grouped = core.groupby("ga_bin", observed=True).agg(
        ga_median=("GA_weeks", "median"),
        target_PI=("UtA_PI", "median"),
        n=("UtA_PI", "size"),
    ).reset_index()
    grouped["ga_median"] = grouped["ga_median"].astype(float)
    grouped["target_PI"] = grouped["target_PI"].astype(float)
    grouped["n"] = grouped["n"].astype(int)
    return grouped[["ga_median", "target_PI", "n"]]


# ---------------------------------------------------------------------------
# Stage 2: lookup grid
# ---------------------------------------------------------------------------

def build_grid(repo: Path) -> tuple[pd.DataFrame, float]:
    """One-at-a-time solver response axes.

    The `mechanism` column stores the adapter parameter name (radial_radius,
    terminal_resistance, av_radius), matching the archived inline-attempt
    intermediate so the regenerated file stays directly comparable to it.
    axis_arrays() translates model names ("radial", "distal", "shunt") to
    these labels.
    """
    rows: list[dict] = []
    started = time.perf_counter()
    for mechanism, adapter_arg in MECHANISMS.items():
        low, high, count = GRID_SPEC[adapter_arg]
        for value in np.linspace(low, high, count):
            result = simulate(repo, **{adapter_arg: float(value)})
            rows.append({
                "mechanism": adapter_arg,
                "value": float(value),
                "PI": result["PI"],
                "SD": result["SD"],
                "RI": result["RI"],
            })
    elapsed = time.perf_counter() - started
    return pd.DataFrame(rows), elapsed


def load_grid(path: Path) -> pd.DataFrame:
    grid = pd.read_csv(path)
    expected = sum(spec[2] for spec in GRID_SPEC.values())
    if len(grid) != expected:
        raise ValueError(
            f"Grid file has {len(grid)} rows; the canonical pipeline expects {expected}. "
            "Rebuild it without --reuse-grid."
        )
    return grid


def axis_arrays(grid: pd.DataFrame, model: str) -> tuple[np.ndarray, np.ndarray]:
    """(values, PI) for one calibration axis; `model` is the model name."""
    part = grid.loc[grid["mechanism"] == MECHANISMS[model]].sort_values("value")
    return part["value"].to_numpy(float), part["PI"].to_numpy(float)


# ---------------------------------------------------------------------------
# Stage 3: interpolation validation
# ---------------------------------------------------------------------------

def validate_interpolation(repo: Path, grid: pd.DataFrame, seed: int,
                           n_tests: int = 30) -> tuple[pd.DataFrame, dict]:
    """Off-grid full-solver checks against linear and PCHIP interpolation.

    Pre-specified acceptance (manuscript 2.6): absolute PI error <= 0.005 and
    relative S/D error < 1%; both linear and monotone-cubic variants are
    validated and recorded.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for mechanism in MECHANISMS:
        adapter_arg = MECHANISMS[mechanism]
        values, pis = axis_arrays(grid, mechanism)
        sds = grid.loc[grid["mechanism"] == adapter_arg].sort_values("value")[
            "SD"].to_numpy(float)
        step = float(np.min(np.diff(values)))
        # Midpoints plus seeded jitter, kept strictly between nodes and inside the range.
        interior = values[1:-1]
        picks = rng.choice(interior, size=min(n_tests, interior.size), replace=False)
        test_values = np.concatenate([
            picks + 0.5 * step,
            picks + 0.25 * step,
        ])
        test_values = test_values[(test_values > values[0]) & (test_values < values[-1])]
        pchip = PchipInterpolator(values, pis)
        pchip_sd = PchipInterpolator(values, sds)
        for value in test_values:
            full = simulate(repo, **{adapter_arg: float(value)})
            rows.append({
                "mechanism": mechanism,
                "value": float(value),
                "PI_full_solver": full["PI"],
                "PI_linear_interp": float(np.interp(value, values, pis)),
                "PI_pchip_interp": float(pchip(value)),
                "SD_full_solver": full["SD"],
                "SD_linear_interp": float(np.interp(value, values, sds)),
                "SD_pchip_interp": float(pchip_sd(value)),
            })
    detail = pd.DataFrame(rows)
    for method in ("linear", "pchip"):
        detail[f"abs_error_{method}"] = (
            detail[f"PI_{method}_interp"] - detail["PI_full_solver"]
        ).abs()
        detail[f"sd_rel_error_{method}"] = (
            (detail[f"SD_{method}_interp"] - detail["SD_full_solver"]).abs()
            / detail["SD_full_solver"].abs()
        )

    summary: dict[str, dict] = {}
    for mechanism, group in detail.groupby("mechanism"):
        summary[mechanism] = {
            "n_tests": int(len(group)),
            "linear_mean_abs_error": float(group["abs_error_linear"].mean()),
            "linear_max_abs_error": float(group["abs_error_linear"].max()),
            "pchip_mean_abs_error": float(group["abs_error_pchip"].mean()),
            "pchip_max_abs_error": float(group["abs_error_pchip"].max()),
            "within_tolerance_linear": bool(group["abs_error_linear"].max() <= PI_TOLERANCE),
            "within_tolerance_pchip": bool(group["abs_error_pchip"].max() <= PI_TOLERANCE),
            "sd_rel_max_error_linear": float(group["sd_rel_error_linear"].max()),
            "sd_rel_max_error_pchip": float(group["sd_rel_error_pchip"].max()),
            "sd_within_tolerance_linear": bool(
                group["sd_rel_error_linear"].max() < SD_REL_TOLERANCE),
            "sd_within_tolerance_pchip": bool(
                group["sd_rel_error_pchip"].max() < SD_REL_TOLERANCE),
        }

    # Cross-term corner check: quantifies how far the full solver departs from an
    # additive (one-at-a-time) approximation when radial radius and A-V shunt
    # radius both move. This documents why joint inversion is not attempted.
    corner_rows = []
    radial_values, radial_pis = axis_arrays(grid, "radial")
    av_values, av_pis = axis_arrays(grid, "shunt")
    baseline_pi = float(np.interp(0.2, radial_values, radial_pis))
    radial_test = np.linspace(0.12, 0.4, 5)
    av_test = np.linspace(0.1, 0.7, 5)
    for radial_value in radial_test:
        for av_value in av_test:
            full = simulate(repo, radial_radius=float(radial_value), av_radius=float(av_value))
            additive = (
                float(np.interp(radial_value, radial_values, radial_pis))
                + float(np.interp(av_value, av_values, av_pis))
                - baseline_pi
            )
            corner_rows.append({
                "radial_radius": float(radial_value),
                "av_radius": float(av_value),
                "PI_full_solver": full["PI"],
                "PI_additive_approx": additive,
                "abs_nonadditivity": abs(full["PI"] - additive),
            })
    corner = pd.DataFrame(corner_rows)
    summary["cross_term"] = {
        "n_combinations": int(len(corner)),
        "max_abs_nonadditivity": float(corner["abs_nonadditivity"].max()),
        "note": (
            "One-at-a-time axes are additive only to within this error; all "
            "calibration models therefore vary a single remodelling axis."
        ),
    }
    return detail, corner, summary


# ---------------------------------------------------------------------------
# Stage 4: per-bin one-dimensional inversion
# ---------------------------------------------------------------------------

def invert_axis(value_axis: np.ndarray, pi_axis: np.ndarray,
                target: float) -> tuple[float, float, bool]:
    """Value whose grid PI matches `target`; boundary-clamped with a flag."""
    order = np.argsort(pi_axis)
    pi_sorted = pi_axis[order]
    value_sorted = value_axis[order]
    lo, hi = float(pi_sorted[0]), float(pi_sorted[-1])
    if target < min(lo, hi) or target > max(lo, hi):
        boundary_value = value_sorted[0] if abs(target - lo) < abs(target - hi) else value_sorted[-1]
        boundary_pi = lo if abs(target - lo) < abs(target - hi) else hi
        return float(boundary_value), float(boundary_pi), True
    estimate = float(np.interp(target, pi_sorted, value_sorted))
    predicted = float(np.interp(estimate, value_axis, pi_axis))
    return estimate, predicted, False


def per_bin_calibration(bins: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mechanism in MECHANISMS:
        values, pis = axis_arrays(grid, mechanism)
        for _, bin_row in bins.iterrows():
            estimate, predicted, at_boundary = invert_axis(
                values, pis, float(bin_row["target_PI"])
            )
            rows.append({
                "model": mechanism,
                "ga_median": float(bin_row["ga_median"]),
                "target_PI": float(bin_row["target_PI"]),
                "estimate": estimate,
                "pred_PI": predicted,
                "at_boundary": at_boundary,
                "n": int(bin_row["n"]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 5: smooth monotone calibration
# ---------------------------------------------------------------------------

def logistic_log_parameter(params: np.ndarray, ga: np.ndarray) -> np.ndarray:
    """log(param(ga)) = l0 + (l1 - l0) / (1 + exp(-k (ga - g0)))."""
    l0, l1, k, g0 = params
    return l0 + (l1 - l0) / (1.0 + np.exp(-k * (ga - g0)))


def predicted_pi(params: np.ndarray, ga: np.ndarray, value_axis: np.ndarray,
                 pi_axis: np.ndarray) -> np.ndarray:
    log_param = logistic_log_parameter(params, np.asarray(ga, dtype=float))
    log_param = np.clip(log_param, np.log(value_axis[0]), np.log(value_axis[-1]))
    return np.interp(np.exp(log_param), value_axis, pi_axis)


def fit_smooth(bins: pd.DataFrame, grid: pd.DataFrame, mechanism: str,
               start: np.ndarray | None = None) -> dict:
    values, pis = axis_arrays(grid, mechanism)
    ga = bins["ga_median"].to_numpy(float)
    observed = bins["target_PI"].to_numpy(float)
    weights = np.sqrt(bins["n"].to_numpy(float))
    log_lo, log_hi = np.log(values[0]), np.log(values[-1])
    bounds = (
        [log_lo, log_lo, -2.0, 10.0],
        [log_hi, log_hi, 2.0, 44.0],
    )

    def residuals(params: np.ndarray) -> np.ndarray:
        return weights * (np.log(observed) - np.log(
            np.clip(predicted_pi(params, ga, values, pis), 1e-9, None)
        ))

    if start is None:
        span = log_hi - log_lo
        starts = [
            np.array([lo, hi, k, g0])
            for lo, hi in [(log_lo + 0.1 * span, log_hi - 0.1 * span),
                           (log_hi - 0.1 * span, log_lo + 0.1 * span)]
            for k in (-0.5, 0.5)
            for g0 in (22.0, 30.0)
        ]
    else:
        starts = [np.asarray(start, dtype=float)]

    best = None
    for candidate in starts:
        candidate = np.clip(candidate, bounds[0], bounds[1])
        try:
            result = least_squares(residuals, candidate, bounds=bounds, xtol=1e-12,
                                   ftol=1e-12, gtol=1e-12)
        except Exception:  # a failed start must not abort the pipeline
            continue
        if best is None or result.cost < best.cost:
            best = result
    if best is None:
        raise RuntimeError(f"All optimization starts failed for mechanism '{mechanism}'.")

    at_bounds = [bool(abs(b - bound) < 1e-6) for b, bound in zip(best.x, bounds[0] + bounds[1])]
    return {
        "mechanism": mechanism,
        "params": best.x,
        "cost": float(best.cost),
        "n_weighted_residuals": residuals(best.x),
        "at_lower_bound": at_bounds[:4],
        "at_upper_bound": at_bounds[4:],
        "bounds": bounds,
        "value_axis": values,
        "pi_axis": pis,
    }


def bootstrap_smooth(bins: pd.DataFrame, grid: pd.DataFrame, mechanism: str,
                     point_estimate: np.ndarray, seed: int, n_boot: int,
                     ga_grid: np.ndarray) -> np.ndarray:
    """Percentile bootstrap (resampling bins with replacement) of the fitted
    latent trajectory on `ga_grid`. Returns (n_boot, len(ga_grid)) log-params."""
    rng = np.random.default_rng(seed)
    values, pis = axis_arrays(grid, mechanism)
    samples = np.empty((n_boot, ga_grid.size))
    n_bins = len(bins)
    completed = 0
    for _ in range(n_boot):
        index = rng.integers(0, n_bins, size=n_bins)
        resampled = bins.iloc[index]
        if resampled["ga_median"].nunique() < 4:
            continue  # a degenerate resample carries no trajectory information
        try:
            fit = fit_smooth(resampled, grid, mechanism, start=point_estimate)
        except RuntimeError:
            continue
        samples[completed] = logistic_log_parameter(fit["params"], ga_grid)
        completed += 1
    if completed < 0.5 * n_boot:
        raise RuntimeError(
            f"Bootstrap for '{mechanism}' completed only {completed}/{n_boot} resamples."
        )
    return samples[:completed]


# ---------------------------------------------------------------------------
# Stage 6: outputs
# ---------------------------------------------------------------------------

def model_metrics(fit: dict, bins: pd.DataFrame) -> dict:
    ga = bins["ga_median"].to_numpy(float)
    observed = bins["target_PI"].to_numpy(float)
    n = bins["n"].to_numpy(float)
    predicted = predicted_pi(fit["params"], ga, fit["value_axis"], fit["pi_axis"])
    log_residuals = np.log(observed) - np.log(predicted)
    rss = float(np.sum(n * log_residuals**2))
    n_eff = float(np.sum(n))
    sigma_sq = rss / n_eff
    loglik = -0.5 * n_eff * (np.log(2.0 * np.pi * sigma_sq) + 1.0)
    aic = 2.0 * 5.0 - 2.0 * loglik  # 4 trajectory parameters + residual variance
    axis_min, axis_max = float(fit["pi_axis"].min()), float(fit["pi_axis"].max())
    return {
        "rmse_PI": float(np.sqrt(np.mean((observed - predicted) ** 2))),
        "mae_PI": float(np.mean(np.abs(observed - predicted))),
        "rmse_logPI": float(np.sqrt(np.mean(log_residuals**2))),
        "weighted_rmse_logPI": float(np.sqrt(rss / n_eff)),
        "sigma_logPI": float(np.sqrt(sigma_sq)),
        "aic_logPI_bins": float(aic),
        "boundary_bins": int(np.sum(
            np.isclose(predicted, axis_min) | np.isclose(predicted, axis_max)
        )),
    }


def trajectory_table(fits: dict, bootstraps: dict, ga_grid: np.ndarray) -> pd.DataFrame:
    rows = []
    for mechanism, fit in fits.items():
        values, pis = fit["value_axis"], fit["pi_axis"]
        point = logistic_log_parameter(fit["params"], ga_grid)
        boot = bootstraps[mechanism]
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5], axis=0)
        pi_point = np.interp(np.exp(point), values, pis)
        # The parameter-to-PI mapping can be decreasing (radial axis), so map
        # both percentile endpoints and order them, keeping low <= high.
        pi_ends = np.sort(
            np.stack([
                np.interp(np.exp(ci_low), values, pis),
                np.interp(np.exp(ci_high), values, pis),
            ]), axis=0,
        )
        pi_low, pi_high = pi_ends[0], pi_ends[1]
        for i, ga in enumerate(ga_grid):
            rows.append({
                "model": mechanism,
                "ga": float(ga),
                "param_estimate": float(np.exp(point[i])),
                "param_ci_low": float(np.exp(ci_low[i])),
                "param_ci_high": float(np.exp(ci_high[i])),
                "predicted_PI": float(pi_point[i]),
                "predicted_PI_ci_low": float(pi_low[i]),
                "predicted_PI_ci_high": float(pi_high[i]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="Source workbook (read-only)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reuse-grid", action="store_true",
                        help="Load solver_lookup_grid.csv instead of rebuilding it")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--bootstrap", type=int, default=400)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    bins = load_core_bins(args.input)
    bins.to_csv(args.output / "ga_bin_targets.csv", index=False)

    if args.reuse_grid:
        grid = load_grid(args.output / "solver_lookup_grid.csv")
        grid_runtime = None
    else:
        grid, grid_runtime = build_grid(args.repo)
        grid.to_csv(args.output / "solver_lookup_grid.csv", index=False)

    interp_detail, interp_corner, interp_summary = validate_interpolation(
        args.repo, grid, seed=args.seed
    )
    interp_detail.to_csv(args.output / "interpolation_validation.csv", index=False)
    interp_corner.to_csv(args.output / "interpolation_cross_term_check.csv", index=False)

    calibration = per_bin_calibration(bins, grid)
    calibration.to_csv(args.output / "one_dimensional_calibration.csv", index=False)

    fits = {m: fit_smooth(bins, grid, m) for m in MECHANISMS}
    ga_grid = np.arange(14.0, 40.0 + 1e-9, 0.25)
    bootstraps = {
        m: bootstrap_smooth(bins, grid, m, fits[m]["params"], args.seed,
                            args.bootstrap, ga_grid)
        for m in MECHANISMS
    }

    metrics = {m: model_metrics(fits[m], bins) for m in MECHANISMS}
    comparison = pd.DataFrame([
        {
            "model": m,
            **metrics[m],
            "parameters": [float(p) for p in fits[m]["params"]],
        }
        for m in MECHANISMS
    ])
    comparison.to_csv(args.output / "smooth_model_comparison.csv", index=False)

    for mechanism, fit in fits.items():
        ga = bins["ga_median"].to_numpy(float)
        observed = bins["target_PI"].to_numpy(float)
        predicted = predicted_pi(fit["params"], ga, fit["value_axis"], fit["pi_axis"])
        pd.DataFrame({
            "ga": ga,
            "observed_PI": observed,
            "predicted_PI": predicted,
            "model": mechanism,
        }).to_csv(args.output / f"{mechanism}_smooth_trajectory.csv", index=False)

    trajectories = trajectory_table(fits, bootstraps, ga_grid)
    trajectories.to_csv(args.output / "latent_trajectories.csv", index=False)

    summary = {
        "generated_by": "code/mechanistic_calibration.py",
        "seed": args.seed,
        "bootstrap_resamples": args.bootstrap,
        "core_bins": bins.to_dict(orient="records"),
        "grid": {
            "spec": {name: {"low": s[0], "high": s[1], "n": s[2]}
                     for name, s in GRID_SPEC.items()},
            "rows": int(len(grid)),
            "runtime_seconds": grid_runtime,
            "reused_existing_file": bool(args.reuse_grid),
            "interpolation_validation": interp_summary,
            "pi_tolerance": PI_TOLERANCE,
            "sd_relative_tolerance": SD_REL_TOLERANCE,
        },
        "smooth_fits": {
            m: {
                "params_log_space": [float(p) for p in fits[m]["params"]],
                "parametrization": "log(param) = l0 + (l1-l0)/(1+exp(-k*(ga-g0)))",
                "asymptote_at_lower_bound": fits[m]["at_lower_bound"][:2],
                "asymptote_at_upper_bound": fits[m]["at_upper_bound"][:2],
                "asymptote_note": (
                    "A logistic asymptote at an admissible-range edge lies outside the "
                    "observed 14-40-week window; it is not constrained by the data and "
                    "does not affect fitted values over the observed range."
                ),
                **metrics[m],
            }
            for m in MECHANISMS
        },
        "aic_note": (
            "AIC is computed on the bin-median targets with examination-count-weighted "
            "log-PI residuals; it is comparable across the three mechanistic models "
            "(identical targets and loss) but not against individual-level models."
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (args.output / "solver_lookup_grid_metadata.json").write_text(
        json.dumps(summary["grid"], indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "grid_rows": len(grid),
        "interpolation_max_abs_error": {
            m: interp_summary[m]["linear_max_abs_error"] for m in MECHANISMS
        },
        "comparison": comparison[["model", "rmse_PI", "mae_PI", "aic_logPI_bins"]].to_dict(
            orient="records"
        ),
        "elapsed_seconds": summary["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
