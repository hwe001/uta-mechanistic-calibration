"""Non-destructive adapter around the archived Clark et al. solver."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np


def load_solver(repo: Path):
    # The 2018 code uses np.complex, removed in NumPy 1.24.
    if not hasattr(np, "complex"):
        np.complex = complex  # type: ignore[attr-defined]
    sys.path.insert(0, str(repo.resolve()))
    import VesselDefinition as params
    import FunctionDefinitions as funcs
    return params, funcs


def waveform_metrics(velocity: np.ndarray) -> dict[str, float | bool | None]:
    peak = float(np.max(velocity))
    trough = float(np.min(velocity))
    mean = float(np.mean(velocity))
    metrics: dict[str, float | bool | None] = {
        "SD": peak / trough,
        "RI": (peak - trough) / peak,
        "PI": (peak - trough) / mean,
        "peak_velocity_cm_s": peak,
        "minimum_velocity_cm_s": trough,
        "mean_velocity_cm_s": mean,
        "notch_present": False,
        "notch_ratio": None,
    }
    derivative = np.diff(velocity)
    peak_index = int(np.argmax(velocity))
    minima = np.where((derivative[:-1] < 0) & (derivative[1:] > 0))[0] + 1
    maxima = np.where((derivative[:-1] > 0) & (derivative[1:] < 0))[0] + 1
    minima = minima[minima > peak_index]
    if minima.size:
        later_maxima = maxima[maxima > minima[0]]
        if later_maxima.size:
            height = float(velocity[later_maxima[0]] - velocity[minima[0]])
            if height > 0:
                metrics["notch_present"] = True
                metrics["notch_ratio"] = height / (peak - trough)
    return metrics


def simulate(repo: Path, radial_radius: float = .2, av_radius: float = .2,
             terminal_resistance: float = 1.6, terminal_compliance: float = 1e-8,
             uterine_radius: float = 2.0, arcuate_radius: float = 2.0,
             youngs_modulus: float = 1.5e6, heart_rate: float = 72.0) -> dict:
    params, funcs = load_solver(repo)
    params.vessels["radius"][:] = [uterine_radius, arcuate_radius, radial_radius, av_radius]
    params.SA_IVS[:] = [terminal_resistance, terminal_compliance, 1]
    params.E = youngs_modulus
    params.HeartRate = heart_rate
    params.IWavHar[0, :] = np.arange(1, params.NHar + 1) * heart_rate / 60.0

    total_resistance, venous_resistance = funcs.total_resistance(params.vessels, params.SA_IVS)
    baseline_resistance, _ = funcs.total_resistance(params.vessels_bl, params.SA_IVS_bl)
    steady_flow = params.SteadyFlow * baseline_resistance / total_resistance
    char_admit, wave_prop, ut_compliance = funcs.characteristic_admittance(params.vessels, params.SA_IVS)
    _, reflection = funcs.effective_admittance(
        params.vessels, params.SA_IVS, char_admit, wave_prop, venous_resistance
    )
    reflect_coeff = np.transpose(np.column_stack((np.abs(reflection[0]), np.angle(reflection[0]))))
    char = np.transpose(np.column_stack((np.abs(char_admit[0]), np.angle(char_admit[0]))))
    propagation = np.transpose(np.column_stack((wave_prop[0].real, wave_prop[0].imag)))
    with contextlib.redirect_stdout(io.StringIO()):
        velocity, time = funcs.timecourse(
            params.StartTime, 60.0 / heart_rate, params.dt, reflect_coeff, char,
            propagation, steady_flow, ut_compliance, "Uterine"
        )
    result = waveform_metrics(velocity)
    result.update({
        "total_resistance": float(total_resistance),
        "steady_flow_ml_min": float(steady_flow),
        "radial_radius_mm": radial_radius,
        "av_radius_mm": av_radius,
        "terminal_resistance": terminal_resistance,
        "terminal_compliance": terminal_compliance,
        "uterine_radius_mm": uterine_radius,
        "arcuate_radius_mm": arcuate_radius,
        "youngs_modulus_pa": youngs_modulus,
        "heart_rate_bpm": heart_rate,
        "n_timepoints": int(len(time)),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = simulate(args.repo)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
