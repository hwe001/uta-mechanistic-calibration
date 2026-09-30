"""Reproducible audit of the anonymised uterine-artery Doppler cohort.

The source workbook is read-only. All derived outputs exclude free-text impressions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # Numerical audit remains usable in minimal environments.
    plt = None


REQUIRED = {
    "Exam_Date", "Age", "GA_weeks", "UtA_SD", "UtA_PI", "UtA_RI",
    "BPD_cm", "HC_cm", "AC_cm", "FL_cm", "EFW_g", "FHR", "Patient_ID",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_age(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = pd.read_excel(args.input, sheet_name=0)
    missing_columns = sorted(REQUIRED.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    d = data.copy()
    d["Exam_Date_parsed"] = pd.to_datetime(d["Exam_Date"], errors="coerce")
    d["composite_exam_date"] = d["Exam_Date"].astype("string").str.contains("；", regex=False, na=False)
    d["composite_impression"] = d["Impression"].astype("string").str.contains("；", regex=False, na=False)
    d["Age_years"] = parse_age(d["Age"])
    d["RI_from_SD"] = 1.0 - 1.0 / d["UtA_SD"]
    d["RI_SD_abs_error"] = (d["UtA_RI"] - d["RI_from_SD"]).abs()
    d["analysis_eligible"] = (
        ~d["composite_exam_date"]
        & d["GA_weeks"].ge(14)
        & d["UtA_PI"].notna()
        & d["RI_SD_abs_error"].le(.02)
    )

    d = d.sort_values(["Patient_ID", "Exam_Date_parsed", "GA_weeks"], kind="stable")
    d["date_gap_weeks"] = d.groupby("Patient_ID")["Exam_Date_parsed"].diff().dt.total_seconds() / (7 * 86400)
    d["ga_gap_weeks"] = d.groupby("Patient_ID")["GA_weeks"].diff()
    d["timeline_abs_error_weeks"] = (d["date_gap_weeks"] - d["ga_gap_weeks"]).abs()

    exact_duplicates = int(data.duplicated().sum())
    key_duplicates = int(data.duplicated(["Patient_ID", "Exam_Date", "GA_weeks"], keep=False).sum())
    counts = data.groupby("Patient_ID").size()
    age_nunique = d.groupby("Patient_ID")["Age_years"].nunique(dropna=True)

    checks = pd.DataFrame([
        ("Rows", len(data)),
        ("Women", data["Patient_ID"].nunique()),
        ("Women with >=2 exams", int((counts >= 2).sum())),
        ("Women with >=3 exams", int((counts >= 3).sum())),
        ("Maximum exams per woman", int(counts.max())),
        ("Exact duplicate rows", exact_duplicates),
        ("Rows sharing patient/date/GA key", key_duplicates),
        ("Women with recorded age span >1 year", int((d.groupby("Patient_ID")["Age_years"].max() - d.groupby("Patient_ID")["Age_years"].min()).gt(1).sum())),
        ("Composite rows with multiple examination dates", int(d["composite_exam_date"].sum())),
        ("Composite rows below 14 weeks", int((d["composite_exam_date"] & d["GA_weeks"].lt(14)).sum())),
        ("Non-composite rows below 14 weeks", int((~d["composite_exam_date"] & d["GA_weeks"].lt(14)).sum())),
        ("Core analysis-eligible rows", int(d["analysis_eligible"].sum())),
        ("Women in core analysis set", int(d.loc[d["analysis_eligible"], "Patient_ID"].nunique())),
        ("Missing UtA PI", int(d["UtA_PI"].isna().sum())),
        ("RI-S/D identity error >0.02", int((d["RI_SD_abs_error"] > 0.02).sum())),
        ("RI-S/D identity error >0.05", int((d["RI_SD_abs_error"] > 0.05).sum())),
        ("Repeated intervals with timeline error >1 week", int((d["timeline_abs_error_weeks"] > 1).sum())),
        ("Repeated intervals with timeline error >2 weeks", int((d["timeline_abs_error_weeks"] > 2).sum())),
        ("GA <14 weeks", int((d["GA_weeks"] < 14).sum())),
        ("GA 14-<18 weeks", int(((d["GA_weeks"] >= 14) & (d["GA_weeks"] < 18)).sum())),
    ], columns=["check", "value"])
    checks.to_csv(args.output / "audit_checks.csv", index=False)

    missingness = pd.DataFrame({
        "column": data.columns,
        "nonmissing_n": [int(data[c].notna().sum()) for c in data.columns],
        "missing_n": [int(data[c].isna().sum()) for c in data.columns],
        "missing_percent": [100.0 * float(data[c].isna().mean()) for c in data.columns],
    })
    missingness.to_csv(args.output / "missingness.csv", index=False)

    bins = [10, 14, 18, 22, 26, 30, 34, 38, 42]
    d["GA_band"] = pd.cut(d["GA_weeks"], bins=bins, right=False)
    ga_summary = d.groupby("GA_band", observed=True).agg(
        examinations=("Patient_ID", "size"),
        women=("Patient_ID", "nunique"),
        uta_pi_median=("UtA_PI", "median"),
        uta_pi_q25=("UtA_PI", lambda x: x.quantile(.25)),
        uta_pi_q75=("UtA_PI", lambda x: x.quantile(.75)),
        uta_ri_median=("UtA_RI", "median"),
        uta_sd_median=("UtA_SD", "median"),
    ).reset_index()
    ga_summary["GA_band"] = ga_summary["GA_band"].astype("string")
    ga_summary.to_csv(args.output / "gestational_age_summary.csv", index=False)

    flags = pd.DataFrame({
        "Patient_ID": d["Patient_ID"],
        "Exam_Date": d["Exam_Date_parsed"].dt.strftime("%Y-%m-%d"),
        "GA_weeks": d["GA_weeks"],
        "UtA_PI": d["UtA_PI"],
        "UtA_RI": d["UtA_RI"],
        "UtA_SD": d["UtA_SD"],
        "RI_SD_abs_error": d["RI_SD_abs_error"],
        "timeline_abs_error_weeks": d["timeline_abs_error_weeks"],
        "flag_nonpositive_PI": d["UtA_PI"].le(0),
        "flag_RI_outside_0_1": ~d["UtA_RI"].between(0, 1, inclusive="both"),
        "flag_SD_below_1": d["UtA_SD"].lt(1),
        "flag_identity_error": d["RI_SD_abs_error"].gt(.02),
        "flag_timeline_error": d["timeline_abs_error_weeks"].gt(2),
        "flag_early_GA": d["GA_weeks"].lt(14),
        "flag_composite_exam": d["composite_exam_date"],
        "analysis_eligible": d["analysis_eligible"],
    })
    flag_columns = [c for c in flags if c.startswith("flag_")]
    flags = flags.loc[flags[flag_columns].any(axis=1)]
    flags.to_csv(args.output / "flagged_records.csv", index=False)

    corr = d[["GA_weeks", "UtA_PI", "UtA_RI", "UtA_SD", "EFW_g"]].corr()
    corr.to_csv(args.output / "correlations.csv")

    eligible = d.loc[d["analysis_eligible"]].copy()
    weekly = eligible.assign(week=np.floor(eligible["GA_weeks"]).astype(int)).groupby("week")["UtA_PI"].agg(
        median="median", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75), n="count"
    ).reset_index()
    weekly.to_csv(args.output / "weekly_uta_pi_summary.csv", index=False)
    if plt is not None:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.scatter(eligible["GA_weeks"], eligible["UtA_PI"], s=9, alpha=.22, color="#326891", edgecolors="none", label="Core analysis records")
        composite = d.loc[d["composite_exam_date"]]
        ax.scatter(composite["GA_weeks"], composite["UtA_PI"], marker="x", s=28, color="#d68910", label="Composite rows excluded")
        ax.plot(weekly["week"], weekly["median"], color="#b03a2e", lw=2, label="Weekly median")
        ax.fill_between(weekly["week"], weekly["q25"], weekly["q75"], color="#b03a2e", alpha=.15, label="Weekly IQR")
        ax.axvspan(11, 14, color="#f4d03f", alpha=.15, label="Sparse early interval")
        ax.set(xlabel="Gestational age (weeks)", ylabel="Uterine artery PI", title="UtA-PI coverage and gestational pattern")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.output / "uta_pi_coverage.png", dpi=200)
        plt.close(fig)

    summary = {
        "source": str(args.input),
        "source_sha256": sha256(args.input),
        "rows": int(len(data)),
        "women": int(data["Patient_ID"].nunique()),
        "ga_min": float(d["GA_weeks"].min()),
        "ga_max": float(d["GA_weeks"].max()),
        "uta_pi_nonmissing": int(d["UtA_PI"].notna().sum()),
        "repeat_ge_2": int((counts >= 2).sum()),
        "repeat_ge_3": int((counts >= 3).sum()),
        "ri_sd_identity_within_0_02_percent": 100.0 * float(d["RI_SD_abs_error"].le(.02).mean()),
        "early_lt14_n": int(d["GA_weeks"].lt(14).sum()),
        "early_lt14_pi_median": float(d.loc[d["GA_weeks"].lt(14), "UtA_PI"].median()),
        "composite_exam_rows": int(d["composite_exam_date"].sum()),
        "noncomposite_early_lt14_n": int((~d["composite_exam_date"] & d["GA_weeks"].lt(14)).sum()),
        "noncomposite_early_lt14_pi_median": float(d.loc[~d["composite_exam_date"] & d["GA_weeks"].lt(14), "UtA_PI"].median()),
        "core_analysis_rows": int(d["analysis_eligible"].sum()),
        "core_analysis_women": int(d.loc[d["analysis_eligible"], "Patient_ID"].nunique()),
    }
    (args.output / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
