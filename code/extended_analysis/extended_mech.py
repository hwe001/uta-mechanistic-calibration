import sys, json, time
import numpy as np, pandas as pd
from pathlib import Path
import os as _os, pathlib as _pl
base = str(_pl.Path(__file__).resolve().parents[2]).replace(chr(92), '/') + '/'
DATA = _os.environ.get('UTA_DATA', base + 'data/UtA_Doppler_Cleaned_Dataset_Anonymous.xlsx')
code = base + 'code'
sys.path.insert(0, code)
import mechanistic_calibration as mc
from mechanistic_calibration import fit_smooth, predicted_pi, logistic_log_parameter
from benchmark_fit import load_core_examinations
from woman_level_validation import repeat_fold_assignment, N_SPLITS, GA_BANDS
from solver_adapter import simulate

repo = Path(base + 'external/network-wave-transmission-master')
mc.MECHANISMS.update({'uterine': 'uterine_radius', 'arcuate': 'arcuate_radius'})

grid0 = pd.read_csv(base + 'results/mechanistic_calibration/solver_lookup_grid.csv')
ax = pd.read_csv('axis_grids.csv')
new = ax[ax.axis.isin(['uterine_radius', 'arcuate_radius'])].rename(columns={'axis': 'mechanism'})[['mechanism', 'value', 'PI', 'SD', 'RI']]
grid = pd.concat([grid0, new], ignore_index=True)
models = ['radial', 'uterine', 'arcuate', 'shunt', 'distal']

# ---- interpolation validation for the new axes (60 off-grid full-solver runs each)
rng = np.random.default_rng(3)
val = {}
for name, arg in (('uterine', 'uterine_radius'), ('arcuate', 'arcuate_radius')):
    v, p = mc.axis_arrays(grid, name)
    s = grid.loc[grid.mechanism == arg].sort_values('value').SD.to_numpy(float)
    pts = rng.uniform(v[0] * 1.001, v[-1] * 0.999, 60)
    err_pi = []; err_sd = []
    for x in pts:
        r = simulate(repo, **{arg: float(x)})
        err_pi.append(abs(r['PI'] - np.interp(x, v, p))); err_sd.append(abs(r['SD'] - np.interp(x, v, s)) / r['SD'])
    val[name] = dict(max_abs_PI_error=float(max(err_pi)), max_rel_SD_error=float(max(err_sd)))
print('interpolation validation', val)

core = load_core_examinations(DATA)
# ---- population fits to 2-week bins (as in the paper)
bins = mc.load_core_bins(Path(DATA))
pop = {}
for m in models:
    f = fit_smooth(bins, grid, m)
    ga = bins.ga_median.to_numpy(float)
    pred = predicted_pi(f['params'], ga, f['value_axis'], f['pi_axis'])
    rmse = float(np.sqrt(np.mean((bins.target_PI.to_numpy(float) - pred) ** 2)))
    lat14 = float(np.exp(logistic_log_parameter(f['params'], np.array([14.0]))[0]))
    lat40 = float(np.exp(logistic_log_parameter(f['params'], np.array([40.0]))[0]))
    pop[m] = dict(rmse_PI=rmse, latent_14w=lat14, latent_40w=lat40, params=f['params'].tolist())
    print(m, round(rmse, 4), round(lat14, 3), round(lat40, 3))

# ---- woman-level CV (same folds/seeds; interpolation instead of full-solver recomputation)
seeds = [20260915 + r for r in range(3)]
core = core.assign(ga_bin=pd.cut(core['GA_weeks'], bins=GA_BANDS, right=False))
women = core.Patient_ID.unique()
recs = []
sd_axes = {m: (mc.axis_arrays(grid, m)[0], grid.loc[grid.mechanism == mc.MECHANISMS[m]].sort_values('value').SD.to_numpy(float)) for m in models}
for rep, seed in enumerate(seeds):
    a = repeat_fold_assignment(women, seed)
    for fold in range(N_SPLITS):
        held = a.loc[a.fold == fold, 'Patient_ID']
        te = core.Patient_ID.isin(held); tr, ts = core[~te], core[te]
        b = tr.groupby('ga_bin', observed=True).agg(ga_median=('GA_weeks', 'median'), target_PI=('UtA_PI', 'median'), n=('UtA_PI', 'size')).reset_index()
        b['ga_median'] = b.ga_median.astype(float); b['target_PI'] = b.target_PI.astype(float)
        row = pd.DataFrame(dict(repeat=rep, fold=fold, Patient_ID=ts.Patient_ID.to_numpy(), GA=ts.GA_weeks.to_numpy(float),
                                y=np.log(ts.UtA_PI.to_numpy(float)), sd_obs=ts.UtA_SD.to_numpy(float)))
        for m in models:
            f = fit_smooth(b, grid, m)
            lp = np.clip(logistic_log_parameter(f['params'], row.GA.to_numpy(float)), np.log(f['value_axis'][0]), np.log(f['value_axis'][-1]))
            row['pi_' + m] = np.interp(np.exp(lp), f['value_axis'], f['pi_axis'])
            v, s = sd_axes[m]
            row['sd_' + m] = np.interp(np.exp(lp), v, s)
        recs.append(row)
cv = pd.concat(recs, ignore_index=True)
cv.to_csv('extended_cv_predictions.csv', index=False)
res = {}
for m in models:
    res[m] = dict(rmse_logPI=float(np.sqrt(np.mean((cv.y - np.log(cv['pi_' + m])) ** 2))),
                  rmse_logSD=float(np.sqrt(np.mean((np.log(cv.sd_obs) - np.log(cv['sd_' + m])) ** 2))))
print(json.dumps(res, indent=1))
# paired woman-bootstrap vs radial
ids = cv.Patient_ID.unique(); grp = {i: g for i, g in cv.groupby('Patient_ID')}
rng = np.random.default_rng(5); diffs = {m: [] for m in models if m != 'radial'}
def rm(d, m): return np.sqrt(np.mean((d.y - np.log(d['pi_' + m])) ** 2))
for bb in range(400):
    s = rng.choice(ids, len(ids), replace=True); d = pd.concat([grp[i] for i in s]); r0 = rm(d, 'radial')
    for m in diffs: diffs[m].append(rm(d, m) - r0)
ci = {m: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in diffs.items()}
print('paired vs radial', {k: np.round(v, 4).tolist() for k, v in ci.items()})
json.dump(dict(interp=val, pop=pop, cv=res, paired=ci), open('extended_mech.json', 'w'), indent=1)
