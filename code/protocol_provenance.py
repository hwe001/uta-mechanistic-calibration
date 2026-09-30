"""Protocol-related provenance recoverable from the source workbook.

A reviewer noted that protocol information may be recoverable from the raw
examination reports. This script verifies, read-only, which protocol facts the
extract itself supports and which it does not, so that the manuscript claims
only what the data contain:

  - examination-type composition (Chinese labels retained verbatim) with
    gestational-age coverage per type;
  - singleton versus multiple pregnancy as stated in the free-text reports;
  - narrative diastolic-notch mentions (切迹) — there is no structured notch
    field;
  - laterality terms in the reports — there are no structured left/right
    uterine-artery columns;
  - umbilical-artery Doppler availability by gestational-age band;
  - the complete column list, documenting the absence of bilateral, notch,
    MCA, CPR and ductus-venosus fields.

Output: results/data_audit/exam_type_summary.csv and
results/data_audit/protocol_provenance.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

NOTCH_TERM = "切迹"          # diastolic notch in the Chinese reports
SINGLETON_TERM = "单活胎"     # "single live fetus"
TWINS_TERM = "双胎"
TRIPLETS_TERM = "三胎"
UTERINE_ARTERY_TERM = "子宫动脉"   # maternal uterine artery
LEFT_TERM, RIGHT_TERM = "左", "右"

UTA_COLUMNS = ["UtA_SD", "UtA_PI", "UtA_RI"]
UA_COLUMNS = ["UA_SD", "UA_PI", "UA_RI"]  # umbilical artery


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = pd.read_excel(args.input, sheet_name=0)
    impression = data["Impression"].astype("string")

    types = data["Exam_Type"].astype("string")
    type_summary = types.value_counts().rename_axis("exam_type_raw").reset_index(
        name="n")
    ga = data.groupby(types)["GA_weeks"].agg(["min", "median", "max"]).round(2)
    type_summary = type_summary.merge(
        ga.reset_index().rename(columns={"Exam_Type": "exam_type_raw"}),
        on="exam_type_raw", how="left",
    )
    type_summary["pi_nonmissing"] = [
        int(data.loc[types == t, "UtA_PI"].notna().sum()) for t in type_summary["exam_type_raw"]
    ]
    type_summary.to_csv(args.output / "exam_type_summary.csv", index=False)

    ua_bands = pd.cut(data["GA_weeks"], [10, 14, 18, 22, 26, 30, 34, 38, 42],
                      right=False)
    ua_coverage = data.assign(GA_band=ua_bands.astype(str)).groupby(
        "GA_band", observed=False)["UA_PI"].count()
    ua_coverage = ua_coverage[ua_coverage > 0]

    summary = {
        "generated_by": "code/protocol_provenance.py",
        "n_rows": int(len(data)),
        "n_women": int(data["Patient_ID"].nunique()),
        "columns": list(data.columns),
        "singleton": {
            "reports_stating_single_live_fetus": int(
                impression.str.contains(SINGLETON_TERM, na=False).sum()),
            "reports_mentioning_twins": int(
                impression.str.contains(TWINS_TERM, na=False).sum()),
            "reports_mentioning_triplets": int(
                impression.str.contains(TRIPLETS_TERM, na=False).sum()),
        },
        "examination_types": {
            "n_distinct_raw_labels": int(types.nunique()),
            "composite_labels_semicolon_separated": int(
                types.str.contains("；", regex=False, na=False).sum()),
            "haemodynamic_assessment_rows": int(
                types.str.contains("血流动力学", na=False).sum()),
            "explicit_maternal_uterine_artery_rows": int(
                types.str.contains(UTERINE_ARTERY_TERM, na=False).sum()),
            "largest_types": type_summary.head(4).to_dict(orient="records"),
        },
        "notch": {
            "structured_field_present": any("notch" in c.lower() for c in data.columns),
            "narrative_mentions": int(impression.str.contains(NOTCH_TERM, na=False).sum()),
            "note": ("no structured notch column exists in the extract; the archived "
                     "solver's predicted notch outputs are not calibratable against "
                     "a structured field"),
        },
        "laterality": {
            "left_right_uta_columns_present": any(
                c.lower().startswith("uta_l") or c.lower().startswith("uta_r")
                for c in data.columns),
            "narrative_left_mentions": int(
                impression.str.contains(LEFT_TERM, na=False).sum()),
            "narrative_right_mentions": int(
                impression.str.contains(RIGHT_TERM, na=False).sum()),
            "note": ("a single set of uterine-artery indices is recorded per "
                     "examination; whether it represents the left artery, the "
                     "right artery or an average is not documented in the extract"),
        },
        "other_doppler": {
            "umbilical_artery_columns": [c for c in UA_COLUMNS if c in data.columns],
            "umbilical_pi_nonmissing": int(data["UA_PI"].notna().sum())
            if "UA_PI" in data.columns else 0,
            "umbilical_availability_by_ga_band": ua_coverage.to_dict(),
            "mca_cpr_ductus_venosus_columns_present": False,
            "note": ("umbilical-artery indices were not used in the present "
                     "analysis, which was pre-specified to the uterine-artery "
                     "summaries"),
        },
        "ga_coverage_by_type_weeks": {
            "nt_examinations": [float(data.loc[types.str.contains("NT", na=False),
                                               "GA_weeks"].min()),
                                float(data.loc[types.str.contains("NT", na=False),
                                               "GA_weeks"].max())],
            # exact label: composite rows (NT + system) share the 系统 substring
            # and would wrongly extend the range into the first trimester
            "system_anomaly_scans": [float(data.loc[types == "{胎儿系统彩超检查;}",
                                                    "GA_weeks"].min()),
                                     float(data.loc[types == "{胎儿系统彩超检查;}",
                                                    "GA_weeks"].max())],
            "routine_obstetric_scans": [float(data.loc[types == "{产科常规彩超检查;}",
                                                       "GA_weeks"].min()),
                                        float(data.loc[types == "{产科常规彩超检查;}",
                                                       "GA_weeks"].max())],
        },
    }
    (args.output / "protocol_provenance.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
