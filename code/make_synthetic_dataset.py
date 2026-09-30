"""Generate a synthetic example dataset with the same columns as the (private) clinical workbook.

The values are simulated from the aggregate benchmark (quadratic log-PI trajectory with a woman-level
random intercept) and from the solver's PI-S/D relation. No real patient record is used. The output lets
other users run the audit, benchmark and calibration scripts end to end; the numbers it produces are NOT
the results of the paper.

Usage:  python code/make_synthetic_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
coef = pd.read_csv(root / 'results' / 'longitudinal_benchmark' / 'benchmark_coefficients.csv').set_index('term')['estimate']
axis = pd.read_csv(root / 'results' / 'mechanistic_calibration' / 'solver_lookup_grid.csv')
axis = axis[axis.mechanism == 'radial_radius'].sort_values('PI')

rng = np.random.default_rng(20260930)
n_women = 300
ga_center = 26.0                       # approximate centring used by the real analysis
rows = []
for pid in range(1, n_women + 1):
    u = rng.normal(0, 0.21)            # between-woman SD (random intercept)
    k = rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
    ga0 = rng.uniform(14, 36)
    date0 = pd.Timestamp('2025-01-01') + pd.Timedelta(days=int(rng.integers(0, 600)))
    for j in range(k):
        ga = ga0 + 3.0 * j + rng.uniform(-0.5, 0.5)
        if ga > 40: break
        x = ga - ga_center
        logpi = coef['Intercept'] + coef['ga_c'] * x + coef['ga2'] * x ** 2 + u + rng.normal(0, 0.28)
        pi = float(np.exp(logpi))
        pi = min(max(pi, axis.PI.min() + 0.01), axis.PI.max() - 0.01)
        sd = float(np.interp(pi, axis.PI, axis.SD) * np.exp(rng.normal(0, 0.03)))
        rows.append(dict(Exam_Date=(date0 + pd.Timedelta(weeks=3 * j)).strftime('%Y-%m-%d %H:%M:%S'), Age='%d岁' % rng.integers(22, 42),
                         GA_weeks=round(ga, 2), UtA_SD=round(sd, 2), UtA_PI=round(pi, 2), UtA_RI=round(1 - 1 / sd, 2),
                         UA_SD=np.nan, UA_PI=np.nan, UA_RI=np.nan, BPD_cm=np.nan, HC_cm=np.nan, AC_cm=np.nan, FL_cm=np.nan,
                         HL_cm=np.nan, EFW_g=np.nan, FHR=int(rng.integers(120, 160)), Exam_Type='synthetic', Impression='synthetic record',
                         Patient_ID=pid))
df = pd.DataFrame(rows)
out = root / 'data' / 'synthetic'
out.mkdir(parents=True, exist_ok=True)
df.to_excel(out / 'UtA_Doppler_Synthetic_Example.xlsx', index=False)
print('wrote', out / 'UtA_Doppler_Synthetic_Example.xlsx', len(df), 'examinations,', df.Patient_ID.nunique(), 'women')
