"""Woman-level validation of the M0 benchmark and mechanistic calibration models.

All splitting is at the woman level; examinations from one woman never appear
in both development and held-out data. The benchmark (M0, from benchmark_fit)
and the three one-dimensional mechanistic models (radial, distal-resistance,
A-V-shunt) are re-estimated on development women only within every fold, and
held-out examinations are predicted for entirely unseen women using
population-level trajectories.

Scheme: 5-fold grouped cross-validation by woman, repeated three times with
fixed seeds. Mechanistic held-out predictions are recomputed with the full
solver at each examination's model-equivalent latent parameter value (the
lookup grid is used only for fitting, having been validated separately).

Reported per model: RMSE and MAE for log(UtA-PI) and PI, calibration slope
and intercept (observed on predicted log-PI), 90% prediction-interval
coverage, performance by gestational-age band, and the proportion of
boundary-pinned fits/examinations. Boundary-pinned estimates are a
consequence of the pre-specified parameter constraints and practical
non-identifiability, not a software failure.

A secondary repeated-examination analysis tests whether a woman's earlier
log-PI residual (persistence) improves prediction of a later examination
beyond the population curve.

Outputs (results/woman_level_validation/):
    validation_predictions.csv        out-of-fold predictions, all repeats
    model_comparison.csv              pooled per-model validation metrics
    performance_by_ga_band.csv        per-model metrics within GA bands
    repeated_interval_metrics.csv     secondary early-to-late prediction
    validation_summary.json           scheme, fold records, all metrics
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from benchmark_fit import fit_benchmark, fixed_effects, load_core_examinations
from mechanistic_calibration import (
    logistic_log_parameter,
    fit_smooth,
    predicted_pi,
    MECHANISMS,
)
from solver_adapter import simulate

GA_BANDS = [14, 18, 22, 26, 30, 34, 38, 42]
N_SPLITS = 5
N_REPEATS = 3
Z_90 = 1.6449


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def repeat_fold_assignment(women: np.ndarray, seed: int) -> pd.DataFrame:
    """Seeded woman-level fold assignment for one repeat."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(women)
    return pd.DataFrame({
        "Patient_ID": shuffled,
        "fold": np.arange(shuffled.size) % N_SPLITS,
    })


def mechanistic_fold_predictions(train: pd.DataFrame, grid: pd.DataFrame,
                                 repo: Path, test: pd.DataFrame,
                                 model: str) -> dict:
    """Fit one latent trajectory on development bins; predict held-out
    examinations with the full solver at each exam's latent parameter."""
    bins = train.groupby("ga_bin", observed=True).agg(
        ga_median=("GA_weeks", "median"),
        target_PI=("UtA_PI", "median"),
        n=("UtA_PI", "size"),
    ).reset_index()
    bins["ga_median"] = bins["ga_median"].astype(float)
    bins["target_PI"] = bins["target_PI"].astype(float)
    fit = fit_smooth(bins, grid, model)
    value_axis, pi_axis = fit["value_axis"], fit["pi_axis"]

    ga_test = test["GA_weeks"].to_numpy(float)
    log_param = np.clip(
        logistic_log_parameter(fit["params"], ga_test),
        np.log(value_axis[0]), np.log(value_axis[-1]),
    )
    # Full-solver recomputation of the reported held-out predictions.
    predictions = np.array([
        simulate(repo, **{MECHANISMS[model]: float(np.exp(lp))})["PI"]
        for lp in log_param
    ])

    # Individual-level residual SD on the log-PI scale: held-out predictions
    # are for unseen women, so the interval must include between-woman
    # scatter, not the (much smaller) bin-median deviation.
    dev_pred = predicted_pi(fit["params"], train["GA_weeks"].to_numpy(float),
                            value_axis, pi_axis)
    dev_resid = np.log(train["UtA_PI"].to_numpy(float)) - np.log(
        np.clip(dev_pred, 1e-9, None))
    sigma = float(np.std(dev_resid, ddof=1))

    # Boundary behaviour, reported as two distinct phenomena:
    # (a) an unconstrained logistic asymptote at the admissible-range edge —
    #     the outer asymptote lies outside 14-40 weeks, so the data cannot
    #     identify it; fitted values over the observed range are unaffected;
    # (b) the predicted trajectory itself sitting on the axis PI ceiling or
    #     floor over more than half the gestational range — the model cannot
    #     leave its constraint (distal-resistance and, early in gestation,
    #     A-V-shunt behaviour).
    endpoints = np.exp([fit["params"][0], fit["params"][1]])
    endpoint_at_edge = any(
        abs(value - bound) <= 1e-3
        for value in endpoints
        for bound in (value_axis[0], value_axis[-1])
    )
    scan = predicted_pi(fit["params"], np.linspace(14.0, 40.0, 53),
                        value_axis, pi_axis)
    edge_fraction = float(np.isclose(
        scan[:, None], [pi_axis.min(), pi_axis.max()]
    ).any(axis=1).mean())
    trajectory_at_ceiling = bool(edge_fraction > 0.5)
    exam_at_boundary = np.isclose(
        predictions[:, None], [pi_axis.min(), pi_axis.max()]
    ).any(axis=1)
    return {
        "prediction": predictions,
        "sigma_logPI": sigma,
        "endpoint_at_edge": bool(endpoint_at_edge),
        "trajectory_at_ceiling": trajectory_at_ceiling,
        "exam_at_boundary": exam_at_boundary,
    }


def run_validation(core: pd.DataFrame, grid: pd.DataFrame, repo: Path,
                   seeds: list[int]) -> tuple[pd.DataFrame, list[dict]]:
    core = core.assign(ga_bin=pd.cut(core["GA_weeks"], bins=GA_BANDS, right=False))
    women = core["Patient_ID"].unique()
    records: list[dict] = []
    fold_log: list[dict] = []

    for repeat, seed in enumerate(seeds):
        assignment = repeat_fold_assignment(women, seed)
        for fold in range(N_SPLITS):
            held_out_women = assignment.loc[assignment["fold"] == fold, "Patient_ID"]
            test_mask = core["Patient_ID"].isin(held_out_women)
            train, test = core[~test_mask], core[test_mask]

            benchmark = fit_benchmark(train, random_slope=False)
            coef = fixed_effects(benchmark)
            bench_pred = np.exp(
                coef["Intercept"] + coef["ga_c"] * test["ga_c"] + coef["ga2"] * test["ga2"]
            ).to_numpy(float)
            # Marginal SD for a new (unseen) woman: between-woman plus residual.
            bench_sigma = float(np.sqrt(
                float(benchmark.cov_re.iloc[0, 0]) + float(benchmark.scale)
            ))

            row_base = {
                "repeat": repeat,
                "fold": fold,
                "Patient_ID": test["Patient_ID"].to_numpy(),
                "GA_weeks": test["GA_weeks"].to_numpy(float),
                "observed_PI": test["UtA_PI"].to_numpy(float),
                "pred_benchmark": bench_pred,
                "sigma_benchmark": bench_sigma,
            }
            for model in MECHANISMS:
                result = mechanistic_fold_predictions(
                    train, grid, repo, test, model
                )
                row_base[f"pred_{model}"] = result["prediction"]
                row_base[f"sigma_{model}"] = result["sigma_logPI"]
                row_base[f"{model}_exam_boundary"] = result["exam_at_boundary"]
                fold_log.append({
                    "repeat": repeat,
                    "fold": fold,
                    "model": model,
                    "n_train_women": int(train["Patient_ID"].nunique()),
                    "n_test_examinations": int(test_mask.sum()),
                    "endpoint_at_range_edge": result["endpoint_at_edge"],
                    "trajectory_at_pi_ceiling": result["trajectory_at_ceiling"],
                })
            records.append(pd.DataFrame(row_base))

    return pd.concat(records, ignore_index=True), fold_log


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calibration(observed_log: np.ndarray, predicted_log: np.ndarray) -> tuple[float, float, bool]:
    """OLS calibration slope and intercept of observed on predicted log-PI.

    Returns (slope, intercept, degenerate). For a model whose held-out
    predictions are (near-)constant — a boundary-pinned constraint ceiling —
    the regression is numerically undefined and is flagged rather than
    reported as a spuriously extreme slope.
    """
    if float(np.var(predicted_log)) < 1e-8:
        return float("nan"), float("nan"), True
    slope, intercept = np.polyfit(predicted_log, observed_log, 1)
    return float(slope), float(intercept), False


def model_comparison_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model_columns = {
        "benchmark": ("pred_benchmark", "sigma_benchmark"),
        **{m: (f"pred_{m}", f"sigma_{m}") for m in MECHANISMS},
    }
    for model, (pred_col, sigma_col) in model_columns.items():
        valid = predictions[pred_col].notna() & predictions["observed_PI"].notna()
        subset = predictions[valid]
        observed = subset["observed_PI"].to_numpy(float)
        predicted = subset[pred_col].to_numpy(float)
        obs_log, pred_log = np.log(observed), np.log(predicted)
        slope, intercept, degenerate = calibration(obs_log, pred_log)
        sigma = subset[sigma_col].to_numpy(float)
        covered = np.abs(obs_log - pred_log) <= Z_90 * sigma
        boundary_col = f"{model}_exam_boundary"
        rows.append({
            "model": model,
            "n_examinations": int(valid.sum()),
            "rmse_logPI": float(np.sqrt(np.mean((obs_log - pred_log) ** 2))),
            "mae_logPI": float(np.mean(np.abs(obs_log - pred_log))),
            "rmse_PI": float(np.sqrt(np.mean((observed - predicted) ** 2))),
            "mae_PI": float(np.mean(np.abs(observed - predicted))),
            "calibration_slope_logPI": slope,
            "calibration_intercept_logPI": intercept,
            "calibration_degenerate": degenerate,
            "coverage_90pct": float(covered.mean()),
            "mean_sigma_logPI": float(np.mean(sigma)),
            "boundary_exam_proportion": (
                float(subset[boundary_col].mean()) if boundary_col in subset else np.nan
            ),
        })
    return pd.DataFrame(rows)


def ga_band_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    predictions = predictions.assign(
        GA_band=pd.cut(predictions["GA_weeks"], bins=GA_BANDS, right=False)
    )
    model_columns = {
        "benchmark": "pred_benchmark",
        **{m: f"pred_{m}" for m in MECHANISMS},
    }
    for model, pred_col in model_columns.items():
        for band, group in predictions.groupby("GA_band", observed=True):
            valid = group[pred_col].notna()
            observed_log = np.log(group.loc[valid, "observed_PI"].to_numpy(float))
            predicted_log = np.log(group.loc[valid, pred_col].to_numpy(float))
            rows.append({
                "model": model,
                "GA_band": str(band),
                "n": int(valid.sum()),
                "rmse_logPI": float(np.sqrt(np.mean((observed_log - predicted_log) ** 2))),
                "bias_logPI": float(np.mean(observed_log - predicted_log)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Uncertainty quantification
# ---------------------------------------------------------------------------

MODEL_COLUMNS = {
    "benchmark": "pred_benchmark",
    **{m: f"pred_{m}" for m in MECHANISMS},
}


def per_fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Metrics within each repeat x fold, for fold-level distributions."""
    rows = []
    grouped = predictions.groupby(["repeat", "fold"])
    for (repeat, fold), group in grouped:
        obs_log = np.log(group["observed_PI"].to_numpy(float))
        for model, col in MODEL_COLUMNS.items():
            pred_log = np.log(group[col].to_numpy(float))
            sigma = group[f"sigma_{model}"].to_numpy(float)
            rows.append({
                "repeat": repeat,
                "fold": fold,
                "model": model,
                "n": int(len(group)),
                "rmse_logPI": float(np.sqrt(np.mean((obs_log - pred_log) ** 2))),
                "coverage_90pct": float(
                    np.mean(np.abs(obs_log - pred_log) <= Z_90 * sigma)),
                "calibration_slope_logPI": (
                    float(np.polyfit(pred_log, obs_log, 1)[0])
                    if np.var(pred_log) > 1e-8 else float("nan")
                ),
            })
    return pd.DataFrame(rows)


def bootstrap_uncertainty(predictions: pd.DataFrame, seed: int,
                          n_boot: int = 2000) -> dict:
    """Woman-level bootstrap CIs for each model's metrics and for paired
    RMSE differences, plus fold-level paired Wilcoxon tests.

    Held-out rows are resampled by woman (the clustering unit), never by
    examination, so each replicate respects the repeated-measures structure.
    """
    rng = np.random.default_rng(seed)
    predictions = predictions.reset_index(drop=True)
    woman_index = {
        woman: np.asarray(indices, dtype=int)
        for woman, indices in predictions.groupby("Patient_ID").indices.items()
    }
    women = np.array(list(woman_index))
    obs_log = np.log(predictions["observed_PI"].to_numpy(float))

    def metrics_for(index: np.ndarray) -> dict:
        out = {}
        for model, col in MODEL_COLUMNS.items():
            pred_log = np.log(predictions[col].to_numpy(float))[index]
            sigma = predictions[f"sigma_{model}"].to_numpy(float)[index]
            resid = obs_log[index] - pred_log
            entry = {
                "rmse_logPI": float(np.sqrt(np.mean(resid ** 2))),
                "mae_logPI": float(np.mean(np.abs(resid))),
                "coverage_90pct": float(np.mean(np.abs(resid) <= Z_90 * sigma)),
            }
            if np.var(pred_log) > 1e-8:
                entry["calibration_slope_logPI"] = float(np.polyfit(pred_log, obs_log[index], 1)[0])
            else:
                entry["calibration_slope_logPI"] = float("nan")
            out[model] = entry
        return out

    names = list(MODEL_COLUMNS)
    samples = {model: {m: [] for m in
                       ("rmse_logPI", "mae_logPI", "coverage_90pct",
                        "calibration_slope_logPI")}
               for model in names}
    diffs = {"radial-benchmark": [], "radial-shunt": [], "radial-distal": []}
    for _ in range(n_boot):
        pick = rng.choice(women, size=women.size, replace=True)
        index = np.concatenate([woman_index[w] for w in pick])
        rep = metrics_for(index)
        for model in names:
            for metric, value in rep[model].items():
                samples[model][metric].append(value)
        diffs["radial-benchmark"].append(
            rep["radial"]["rmse_logPI"] - rep["benchmark"]["rmse_logPI"])
        diffs["radial-shunt"].append(
            rep["radial"]["rmse_logPI"] - rep["shunt"]["rmse_logPI"])
        diffs["radial-distal"].append(
            rep["radial"]["rmse_logPI"] - rep["distal"]["rmse_logPI"])

    ci = {
        model: {
            metric: {
                "point": metrics_for(np.arange(len(predictions)))[model][metric],
                "ci_low": float(np.nanpercentile(values, 2.5)),
                "ci_high": float(np.nanpercentile(values, 97.5)),
            }
            for metric, values in metrics.items()
        }
        for model, metrics in samples.items()
    }
    paired = {
        name: {
            "mean_diff": float(np.mean(values)),
            "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5)),
        }
        for name, values in diffs.items()
    }
    return {"bootstrap_ci": ci, "paired_rmse_differences": paired,
            "n_bootstrap": n_boot, "resampling_unit": "woman"}


def fold_paired_tests(fold_metrics: pd.DataFrame) -> list[dict]:
    """Fold-level paired Wilcoxon tests on RMSE (radial vs each model)."""
    tests = []
    wide = fold_metrics.pivot_table(
        index=["repeat", "fold"], columns="model", values="rmse_logPI")
    for other in ("benchmark", "shunt", "distal"):
        stat, p = wilcoxon(wide["radial"], wide[other])
        tests.append({
            "comparison": f"radial vs {other}",
            "n_pairs": int(wide["radial"].notna().sum()),
            "median_rmse_radial": float(wide["radial"].median()),
            "median_rmse_other": float(wide[other].median()),
            "wilcoxon_p": float(p),
        })
    return tests


# ---------------------------------------------------------------------------
# Secondary: early-to-late examination prediction
# ---------------------------------------------------------------------------

def repeated_interval_predictions(core: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, float]:
    intervals = []
    for woman, group in core.sort_values("GA_weeks", kind="stable").groupby("Patient_ID"):
        if len(group) < 2:
            continue
        rows = group.to_dict(orient="records")
        for earlier, later in zip(rows[:-1], rows[1:]):
            intervals.append({
                "Patient_ID": woman,
                "ga_earlier": float(earlier["GA_weeks"]),
                "ga_later": float(later["GA_weeks"]),
                "PI_earlier": float(earlier["UtA_PI"]),
                "PI_later": float(later["UtA_PI"]),
                "timeline_consistent": bool(earlier["timeline_abs_error_weeks"] <= 2.0),
            })
    intervals = pd.DataFrame(intervals)

    rng = np.random.default_rng(seed)
    women = np.sort(intervals["Patient_ID"].unique())
    dev_women = set(rng.choice(women, size=int(0.7 * women.size), replace=False))
    intervals["in_development"] = intervals["Patient_ID"].isin(dev_women)

    dev_core = core[core["Patient_ID"].isin(dev_women)]
    pop = np.polyfit(dev_core["GA_weeks"], dev_core["logPI"], 2)
    resid = dev_core["logPI"].to_numpy(float) - np.polyval(pop, dev_core["GA_weeks"])
    dev_core = dev_core.assign(pop_resid=resid)

    rho_num = rho_den = 0.0
    for _, group in dev_core.groupby("Patient_ID"):
        if len(group) < 2:
            continue
        r = group.sort_values("GA_weeks", kind="stable")["pop_resid"].to_numpy(float)
        rho_num += float(np.sum(r[1:] * r[:-1]))
        rho_den += float(np.sum(r[:-1] ** 2))
    rho = rho_num / rho_den if rho_den > 0 else 0.0

    log_earlier = np.log(intervals["PI_earlier"].to_numpy(float))
    pop_later = np.polyval(pop, intervals["ga_later"].to_numpy(float))
    pop_earlier = np.polyval(pop, intervals["ga_earlier"].to_numpy(float))
    intervals["pop_logPI_prediction"] = pop_later
    intervals["residual_adjusted_prediction"] = pop_later + rho * (log_earlier - pop_earlier)
    return intervals, float(rho)


def repeated_interval_metrics(intervals: pd.DataFrame) -> pd.DataFrame:
    evaluation = intervals[~intervals["in_development"] & intervals["timeline_consistent"]]
    observed = np.log(evaluation["PI_later"].to_numpy(float))
    rows = []
    for column, label in [
        ("pop_logPI_prediction", "population_curve"),
        ("residual_adjusted_prediction", "earlier_exam_adjusted"),
    ]:
        predicted = evaluation[column].to_numpy(float)
        rows.append({
            "predictor": label,
            "n_intervals": int(len(evaluation)),
            "rmse_logPI": float(np.sqrt(np.mean((observed - predicted) ** 2))),
            "mae_logPI": float(np.mean(np.abs(observed - predicted))),
            "correlation_pearson": float(np.corrcoef(observed, predicted)[0, 1]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path, help="Source workbook (read-only)")
    parser.add_argument("--grid", required=True, type=Path,
                        help="solver_lookup_grid.csv from mechanistic_calibration.py")
    parser.add_argument("--repo", required=True, type=Path,
                        help="Archived solver repository (full-solver prediction)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    core = load_core_examinations(args.input)
    grid = pd.read_csv(args.grid)
    seeds = [args.seed + repeat for repeat in range(N_REPEATS)]

    predictions, fold_log = run_validation(core, grid, args.repo, seeds)
    predictions.to_csv(args.output / "validation_predictions.csv", index=False)

    comparison = model_comparison_table(predictions)
    comparison.to_csv(args.output / "model_comparison.csv", index=False)

    folds = per_fold_metrics(predictions)
    folds.to_csv(args.output / "fold_metrics.csv", index=False)
    uncertainty = bootstrap_uncertainty(predictions, args.seed)
    paired_tests = fold_paired_tests(folds)
    uncertainty["fold_paired_wilcoxon"] = paired_tests

    bands = ga_band_table(predictions)
    bands.to_csv(args.output / "performance_by_ga_band.csv", index=False)

    intervals, rho = repeated_interval_predictions(core, args.seed)
    repeated = repeated_interval_metrics(intervals)
    intervals.to_csv(args.output / "repeated_interval_predictions.csv", index=False)
    repeated.to_csv(args.output / "repeated_interval_metrics.csv", index=False)

    fold_frame = pd.DataFrame(fold_log)
    boundary_by_model = fold_frame.groupby("model").agg(
        folds=("endpoint_at_range_edge", "size"),
        folds_endpoint_at_edge=("endpoint_at_range_edge", "sum"),
        folds_trajectory_at_ceiling=("trajectory_at_pi_ceiling", "sum"),
    ).reset_index()
    boundary_by_model["endpoint_at_edge_proportion"] = (
        boundary_by_model["folds_endpoint_at_edge"] / boundary_by_model["folds"]
    )
    boundary_by_model["trajectory_at_ceiling_proportion"] = (
        boundary_by_model["folds_trajectory_at_ceiling"] / boundary_by_model["folds"]
    )

    summary = {
        "generated_by": "code/woman_level_validation.py",
        "seeds": seeds,
        "scheme": f"5-fold grouped cross-validation by woman, {N_REPEATS} repeats",
        "split_unit": "woman",
        "n_examinations": int(len(core)),
        "n_women": int(core["Patient_ID"].nunique()),
        "prediction_mode": {
            "benchmark": "fixed-effects population trajectory (unseen women)",
            "mechanistic": "full-solver evaluation at each examination's latent parameter",
        },
        "model_comparison": comparison.to_dict(orient="records"),
        "uncertainty": uncertainty,
        "boundary_fits": {
            "note": (
                "Boundary-pinned estimates reflect the pre-specified parameter "
                "constraints and practical non-identifiability, not a software failure. "
                "endpoint_at_range_edge = a logistic asymptote outside the observed "
                "14-40-week window pins at the admissible-range edge (fitted values over "
                "the observed range are unaffected). trajectory_at_pi_ceiling = the "
                "predicted trajectory sits on the axis PI ceiling/floor over more than "
                "half the gestational range."
            ),
            "fold_level": boundary_by_model.to_dict(orient="records"),
        },
        "performance_by_ga_band": bands.to_dict(orient="records"),
        "repeated_intervals": {
            "persistence_rho": rho,
            "metrics": repeated.to_dict(orient="records"),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "n_examinations": summary["n_examinations"],
        "n_women": summary["n_women"],
        "model_comparison": comparison.to_dict(orient="records"),
        "uncertainty": {
            "paired_rmse_differences": uncertainty["paired_rmse_differences"],
            "fold_paired_wilcoxon": paired_tests,
            "rmse_ci": {m: uncertainty["bootstrap_ci"][m]["rmse_logPI"]
                        for m in MODEL_COLUMNS},
            "coverage_ci": {m: uncertainty["bootstrap_ci"][m]["coverage_90pct"]
                            for m in MODEL_COLUMNS},
        },
        "boundary_fits": boundary_by_model.to_dict(orient="records"),
        "persistence_rho": rho,
        "repeated_metrics": repeated.to_dict(orient="records"),
        "elapsed_seconds": summary["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
