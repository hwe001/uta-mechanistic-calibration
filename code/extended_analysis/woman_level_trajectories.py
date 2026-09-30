"""Woman-level bootstrap bands for the three original calibrated trajectories (radial, A-V shunt, distal resistance).

Women are resampled with replacement, 2-week bins are rebuilt, each model is refitted from the point estimate, and percentile bands are taken on the
0.25-week gestational-age grid. Output has the same columns as results/mechanistic_calibration/latent_trajectories.csv.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd

base = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get('UTA_DATA', base / 'data' / 'UtA_Doppler_Cleaned_Dataset_Anonymous.xlsx'))
sys.path.insert(0, str(base / 'code'))
from benchmark_fit import load_core_examinations
import mechanistic_calibration as mc

N_BOOT = 300
grid = pd.read_csv(base / 'results' / 'mechanistic_calibration' / 'solver_lookup_grid.csv')
core = load_core_examinations(DATA).rename(columns={'GA_weeks': 'GA'})


def make_bins(d):
    d = d.copy(); d['b'] = pd.cut(d.GA, bins=np.arange(14.0, 42.0, 2.0), right=False)
    return d.groupby('b', observed=True).agg(ga_median=('GA', 'median'), target_PI=('UtA_PI', 'median'), n=('UtA_PI', 'size')).reset_index(drop=True)


bins = make_bins(core)
ga_grid = np.arange(14.0, 40.0 + 1e-9, 0.25)
women = core.Patient_ID.unique(); byw = {w: g for w, g in core.groupby('Patient_ID')}
fits, boots = {}, {}
for m in mc.MECHANISMS:
    fits[m] = mc.fit_smooth(bins, grid, m); rng = np.random.default_rng(20260930); s = []
    for _ in range(N_BOOT):
        pick = rng.choice(women, size=len(women), replace=True)
        b = make_bins(pd.concat([byw[w] for w in pick], ignore_index=True))
        if b.ga_median.nunique() < 4: continue
        try: s.append(mc.logistic_log_parameter(mc.fit_smooth(b, grid, m, start=fits[m]['params'])['params'], ga_grid))
        except Exception: continue
    boots[m] = np.array(s); print(m, len(s), 'resamples', flush=True)
out = base / 'results' / 'mechanistic_calibration' / 'latent_trajectories_woman_level.csv'
mc.trajectory_table(fits, boots, ga_grid).to_csv(out, index=False); print('saved', out)
