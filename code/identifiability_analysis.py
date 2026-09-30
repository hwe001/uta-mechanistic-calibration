"""Formal local sensitivity, rank and synthetic inverse-recovery analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from solver_adapter import simulate


PARAMS = [
    "radial_radius_mm", "av_radius_mm", "terminal_resistance",
    "terminal_compliance", "youngs_modulus_pa", "heart_rate",
]
DEFAULTS = {
    "radial_radius_mm": .20, "av_radius_mm": .20, "terminal_resistance": 1.6,
    "terminal_compliance": 1e-8, "youngs_modulus_pa": 1.5e6, "heart_rate": 72.0,
}
BOUNDS = {
    "radial_radius_mm": (.10, .30), "av_radius_mm": (.08, .60),
    "terminal_resistance": (.4, 16.0), "terminal_compliance": (2e-9, 5e-8),
    "youngs_modulus_pa": (5e5, 4e6), "heart_rate": (55.0, 105.0),
}


def outputs(repo: Path, values: dict[str, float]) -> np.ndarray:
    adapter_values = dict(values)
    adapter_values["radial_radius"] = adapter_values.pop("radial_radius_mm")
    adapter_values["av_radius"] = adapter_values.pop("av_radius_mm")
    adapter_values["youngs_modulus"] = adapter_values.pop("youngs_modulus_pa")
    r = simulate(repo, **adapter_values)
    # RI and S/D are mathematically redundant; use PI and S/D as the two
    # independent summary targets available from this workbook.
    return np.array([float(r["PI"]), float(r["SD"])])


def transformed(values: dict[str, float]) -> np.ndarray:
    return np.array([np.log(values[p]) if p != "heart_rate" else values[p] for p in PARAMS])


def untransformed(x: np.ndarray) -> dict[str, float]:
    return {p: float(np.exp(x[i]) if p != "heart_rate" else x[i]) for i, p in enumerate(PARAMS)}


def bounds_transformed() -> tuple[np.ndarray, np.ndarray]:
    lo, hi = [], []
    for p in PARAMS:
        a, b = BOUNDS[p]
        if p != "heart_rate":
            a, b = np.log(a), np.log(b)
        lo.append(a); hi.append(b)
    return np.array(lo), np.array(hi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--starts", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    y0 = outputs(args.repo, DEFAULTS)
    x0 = transformed(DEFAULTS)
    # Central finite-difference Jacobian in transformed coordinates.
    jac = np.zeros((2, len(PARAMS)))
    rows = []
    for j, p in enumerate(PARAMS):
        h = 1e-4
        xp, xm = x0.copy(), x0.copy()
        xp[j] += h; xm[j] -= h
        yp, ym = outputs(args.repo, untransformed(xp)), outputs(args.repo, untransformed(xm))
        jac[:, j] = (yp - ym) / (2 * h)
        rows.append({"parameter": p, "d_PI_d_transformed": jac[0, j], "d_SD_d_transformed": jac[1, j]})
    pd.DataFrame(rows).to_csv(args.output / "local_jacobian.csv", index=False)

    u, singular, vt = np.linalg.svd(jac, full_matrices=False)
    pairwise = np.corrcoef(jac)
    sensitivity = pd.DataFrame({"parameter": PARAMS, "jacobian_column_norm": np.linalg.norm(jac, axis=0), "right_singular_loading_1": vt[0], "right_singular_loading_2": vt[1]})
    sensitivity.to_csv(args.output / "jacobian_loadings.csv", index=False)

    rng = np.random.default_rng(20260915)
    lo, hi = bounds_transformed()
    target_x = rng.uniform(lo, hi)
    target_values = untransformed(target_x)
    target_y = outputs(args.repo, target_values)

    def residual(x: np.ndarray) -> np.ndarray:
        return (outputs(args.repo, untransformed(x)) - target_y) / np.array([.01, .03])

    recoveries = []
    for i in range(args.starts):
        start = rng.uniform(lo, hi)
        fit = least_squares(residual, start, bounds=(lo, hi), max_nfev=50)
        estimate = untransformed(fit.x)
        recoveries.append({"start": i, "success": bool(fit.success), "cost": float(fit.cost), "PI_residual": float(outputs(args.repo, estimate)[0] - target_y[0]), "SD_residual": float(outputs(args.repo, estimate)[1] - target_y[1]), **estimate})
    recovery = pd.DataFrame(recoveries)
    recovery.to_csv(args.output / "synthetic_inverse_recovery.csv", index=False)

    result = {
        "independent_observables": ["PI", "SD"],
        "all_reported_observables": ["PI", "RI", "SD"],
        "baseline_PI": float(y0[0]), "baseline_SD": float(y0[1]),
        "jacobian_shape": list(jac.shape),
        "n_starts": int(args.starts),
        "jacobian_rank_numeric": int(np.linalg.matrix_rank(jac, tol=1e-10)),
        "singular_values": singular.tolist(),
        "condition_number_nonzero_subspace": float(singular[0] / singular[-1]),
        "target_parameters": target_values,
        "recovery_cost_median": float(recovery["cost"].median()),
        "recovery_cost_max": float(recovery["cost"].max()),
        "recovery_parameter_ranges": {p: [float(recovery[p].min()), float(recovery[p].max())] for p in PARAMS},
    }
    (args.output / "identifiability_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
