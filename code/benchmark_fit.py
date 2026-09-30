"""Saved mixed-effects benchmark (M0) for log(UtA-PI) across gestation.

Replaces the earlier inline (unsaved) benchmark run so the longitudinal
Results are reproducible. The benchmark is the statistical comparator for
the mechanistic calibration models:

    log(UtA_PI) ~ GA_centered + GA_centered^2
    random intercept: Patient_ID
    optional random slope: GA_centered  (--random-slope)

The random GA slope was near the variance boundary in the archived run and
is therefore off by default; --random-slope reproduces the archived
specification exactly. A sensitivity variant that retains the six
RI-S/D-discordant records (excluded from the pre-specified core set) is
fitted in the same run and recorded in the metadata.

Outputs (results/longitudinal_benchmark/):
    benchmark_summary.txt       full statsmodels summary of the primary fit
    benchmark_coefficients.csv  fixed effects with confidence intervals
    variance_components.csv     random-effects and residual variances
    trajectory_predictions.csv  population trajectory with 90% intervals
    benchmark_diagnostics.csv   residual summary by gestational-age band
    benchmark_metadata.json     formulae, sample sizes, sensitivity fits,
                                software versions, source checksum
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.stats
import statsmodels
import statsmodels.formula.api as smf

BENCHMARK_FORMULA = "logPI ~ ga_c + ga2"
GA_GRID_STEP = 0.25
Z_90 = 1.6449  # two-sided 90% normal quantile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_core_examinations(input_path: Path,
                           include_ri_sd_discordant: bool = False) -> pd.DataFrame:
    """Core analysis set (audit eligibility rules) with modelling columns.

    The pre-specified core set excludes composite examination rows, GA below
    14 weeks, missing UtA-PI and RI-S/D disagreement above 0.02. The last
    exclusion can be relaxed for the sensitivity variant.
    """
    data = pd.read_excel(input_path, sheet_name=0)
    composite = data["Exam_Date"].astype("string").str.contains("；", regex=False, na=False)
    ri_sd_error = (data["UtA_RI"] - (1.0 - 1.0 / data["UtA_SD"])).abs()
    eligible = (
        ~composite
        & data["GA_weeks"].ge(14)
        & data["UtA_PI"].notna()
        & (ri_sd_error.notna() if include_ri_sd_discordant else ri_sd_error.le(0.02))
    )
    core = data.loc[eligible, [
        "Patient_ID", "Exam_Date", "GA_weeks", "UtA_PI", "UtA_RI", "UtA_SD",
    ]].copy()
    core["Exam_Date_parsed"] = pd.to_datetime(core["Exam_Date"], errors="coerce")
    core["logPI"] = np.log(core["UtA_PI"].astype(float))
    core["ga_c"] = core["GA_weeks"] - core["GA_weeks"].mean()
    core["ga2"] = core["ga_c"] ** 2
    core = core.sort_values(["Patient_ID", "Exam_Date_parsed", "GA_weeks"], kind="stable")

    date_gap = core.groupby("Patient_ID")["Exam_Date_parsed"].diff().dt.total_seconds() / (7 * 86400)
    ga_gap = core.groupby("Patient_ID")["GA_weeks"].diff()
    core["timeline_abs_error_weeks"] = (date_gap - ga_gap).abs()
    return core.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Benchmark model
# ---------------------------------------------------------------------------

def fit_benchmark(data: pd.DataFrame, random_slope: bool = False):
    """M0 mixed-effects benchmark; returns the statsmodels result."""
    re_formula = "ga_c" if random_slope else None
    model = smf.mixedlm(
        BENCHMARK_FORMULA, data, groups=data["Patient_ID"], re_formula=re_formula
    )
    return model.fit(reml=True)


def fixed_effects(result) -> dict[str, float]:
    """Fixed-effect coefficients only (random-effect parameters excluded)."""
    return {name: float(value) for name, value in result.fe_params.items()}


def trajectory_frame(result, ga_grid: np.ndarray, mean_ga: float) -> pd.DataFrame:
    """Population (fixed-effects) trajectory with 90% prediction intervals."""
    coef = fixed_effects(result)
    ga_c = ga_grid - mean_ga
    pred_log = coef["Intercept"] + coef["ga_c"] * ga_c + coef["ga2"] * ga_c**2
    sigma = float(np.sqrt(result.scale))
    return pd.DataFrame({
        "ga_c": ga_c,
        "ga2": ga_c**2,
        "GA_weeks": ga_grid,
        "pred_logPI": pred_log,
        "pred_PI": np.exp(pred_log),
        "pred_PI_low90": np.exp(pred_log - Z_90 * sigma),
        "pred_PI_high90": np.exp(pred_log + Z_90 * sigma),
    })


def diagnostics_frame(core: pd.DataFrame, result) -> tuple[pd.DataFrame, dict]:
    """Fixed-effect residual summary by gestational-age band plus overall fit."""
    coef = fixed_effects(result)
    fitted_log = coef["Intercept"] + coef["ga_c"] * core["ga_c"] + coef["ga2"] * core["ga2"]
    residual = core["logPI"].to_numpy(float) - fitted_log.to_numpy(float)
    bands = [14, 18, 22, 26, 30, 34, 38, 42]
    band = pd.cut(core["GA_weeks"], bins=bands, right=False)
    diagnostics = pd.DataFrame({
        "GA_band": band.astype(str),
        "residual_logPI": residual,
    }).groupby("GA_band", observed=False).agg(
        n=("residual_logPI", "size"),
        mean_residual=("residual_logPI", "mean"),
        median_residual=("residual_logPI", "median"),
        sd_residual=("residual_logPI", "std"),
    ).reset_index()
    overall = {
        "rmse_logPI_fixed_effects": float(np.sqrt(np.mean(residual**2))),
        "marginal_r2_logPI": float(1.0 - np.var(residual) / np.var(core["logPI"])),
        "residual_sd": float(np.std(residual)),
        "n_examinations": int(len(core)),
    }
    return diagnostics, overall


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path, help="Source workbook (read-only)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    core = load_core_examinations(args.input)

    # Primary (protocol-supported): random intercept only.
    primary = fit_benchmark(core, random_slope=False)
    # Archived-specification reproduction: random intercept + GA slope.
    archived = fit_benchmark(core, random_slope=True)
    # Sensitivity: retain the six RI-S/D-discordant records.
    sensitivity_core = load_core_examinations(args.input, include_ri_sd_discordant=True)
    sensitivity = fit_benchmark(sensitivity_core, random_slope=False)

    # Likelihood-ratio test for the random GA slope, with the boundary
    # correction: under H0 (zero slope variance) the LRT follows a 50:50
    # mixture of chi-square(0) and chi-square(1).
    ml_noslope = smf.mixedlm(
        BENCHMARK_FORMULA, core, groups=core["Patient_ID"]
    ).fit(reml=False)
    ml_slope = smf.mixedlm(
        BENCHMARK_FORMULA, core, groups=core["Patient_ID"], re_formula="ga_c"
    ).fit(reml=False)
    lrt = float(2.0 * (ml_slope.llf - ml_noslope.llf))
    lrt_p = float(0.5 * scipy.stats.chi2.sf(max(lrt, 0.0), 1.0))
    slope_variance = float(archived.cov_re.iloc[-1, -1])

    coefficients = pd.DataFrame({
        "term": list(fixed_effects(primary)),
        "estimate": [fixed_effects(primary)[t] for t in fixed_effects(primary)],
        "se": [float(primary.bse[t]) for t in fixed_effects(primary)],
        "ci_low": [float(primary.conf_int().loc[t, 0]) for t in fixed_effects(primary)],
        "ci_high": [float(primary.conf_int().loc[t, 1]) for t in fixed_effects(primary)],
        "p_value": [float(primary.pvalues[t]) for t in fixed_effects(primary)],
    })
    coefficients.to_csv(args.output / "benchmark_coefficients.csv", index=False)

    re_var = float(primary.cov_re.iloc[0, 0])
    variance = pd.DataFrame([
        {"component": "random_intercept_var", "value": re_var},
        {"component": "residual_var", "value": float(primary.scale)},
        {"component": "residual_sd", "value": float(np.sqrt(primary.scale))},
        {"component": "archived_spec_random_intercept_var",
         "value": float(archived.cov_re.iloc[0, 0])},
        {"component": "archived_spec_random_slope_var",
         "value": float(archived.cov_re.iloc[-1, -1])},
    ])
    variance.to_csv(args.output / "variance_components.csv", index=False)

    ga_grid = np.arange(14.0, 40.0 + GA_GRID_STEP / 2, GA_GRID_STEP)
    trajectory_frame(primary, ga_grid, float(core["GA_weeks"].mean())).to_csv(
        args.output / "trajectory_predictions.csv", index=False
    )

    diagnostics, overall = diagnostics_frame(core, primary)
    diagnostics.to_csv(args.output / "benchmark_diagnostics.csv", index=False)

    with (args.output / "benchmark_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("Primary benchmark (random intercept only)\n")
        handle.write(str(primary.summary()))
        handle.write("\n\n\nArchived-specification reproduction (random intercept + GA slope)\n")
        handle.write(str(archived.summary()))
        handle.write("\n\n\nSensitivity (RI-S/D-discordant records retained)\n")
        handle.write(str(sensitivity.summary()))
        handle.write("\n")

    metadata = {
        "generated_by": "code/benchmark_fit.py",
        "formula": BENCHMARK_FORMULA,
        "primary_random_effects": "intercept only",
        "archived_spec_random_effects": "intercept + GA slope (re_formula='ga_c')",
        "primary_converged": bool(primary.converged),
        "archived_spec_converged": bool(archived.converged),
        "primary_fixed_effects": fixed_effects(primary),
        "archived_spec_fixed_effects": fixed_effects(archived),
        "primary_random_intercept_var": re_var,
        "archived_spec_random_intercept_var": float(archived.cov_re.iloc[0, 0]),
        "archived_spec_random_slope_var": slope_variance,
        "random_slope_justification": {
            "slope_variance_reml": slope_variance,
            "lrt_statistic": lrt,
            "lrt_df_note": ("boundary-corrected: 50:50 mixture of chi-square(0) and "
                            "chi-square(1) under H0"),
            "lrt_p_value": lrt_p,
            "loglik_ml_no_slope": float(ml_noslope.llf),
            "loglik_ml_slope": float(ml_slope.llf),
        },
        "n_examinations_primary": int(len(core)),
        "n_women_primary": int(core["Patient_ID"].nunique()),
        "n_examinations_sensitivity": int(len(sensitivity_core)),
        "n_women_sensitivity": int(sensitivity_core["Patient_ID"].nunique()),
        "sensitivity_fixed_effects": fixed_effects(sensitivity),
        "diagnostics_overall": overall,
        "mean_ga_centering_weeks": float(core["GA_weeks"].mean()),
        "source": str(args.input),
        "source_sha256": sha256(args.input),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "n_examinations": metadata["n_examinations_primary"],
        "n_women": metadata["n_women_primary"],
        "primary_fixed_effects": metadata["primary_fixed_effects"],
        "random_intercept_var": re_var,
        "archived_spec_fixed_effects": metadata["archived_spec_fixed_effects"],
        "diagnostics_overall": overall,
    }, indent=2))


if __name__ == "__main__":
    main()
