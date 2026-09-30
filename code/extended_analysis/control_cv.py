import sys, json
import numpy as np, pandas as pd
from scipy.optimize import least_squares
import os as _os, pathlib as _pl
base = str(_pl.Path(__file__).resolve().parents[2]).replace(chr(92), '/') + '/'
DATA = _os.environ.get('UTA_DATA', base + 'data/UtA_Doppler_Cleaned_Dataset_Anonymous.xlsx')
sys.path.insert(0, base + 'code')
from benchmark_fit import load_core_examinations
from woman_level_validation import repeat_fold_assignment, N_SPLITS

core = load_core_examinations(DATA)
pred = pd.read_csv(base + 'results/woman_level_validation/validation_predictions.csv')
women = core['Patient_ID'].unique()
seeds = [20260915 + r for r in range(3)]

def logistic(p, ga):
    l0, l1, k, g0 = p
    return l0 + (l1 - l0) / (1 + np.exp(-k * (ga - g0)))

def fit_logistic(ga, y):
    best = None
    for k0 in (0.1, 0.3, 0.6):
        for g0 in (18, 24, 30):
            p0 = [y[ga < 20].mean() if (ga < 20).any() else y.max(), y[ga > 34].mean(), k0, g0]
            r = least_squares(lambda p: logistic(p, ga) - y, p0,
                              bounds=([-3, -3, 0.01, 10], [3, 3, 3, 45]))
            if best is None or r.cost < best.cost: best = r
    return best.x

rows = []
for rep, seed in enumerate(seeds):
    a = repeat_fold_assignment(women, seed)
    for fold in range(N_SPLITS):
        held = a.loc[a.fold == fold, 'Patient_ID']
        te = core['Patient_ID'].isin(held); tr, ts = core[~te], core[te]
        ga_tr, y_tr, ga_ts = tr.GA_weeks.to_numpy(float), tr.logPI.to_numpy(float), ts.GA_weeks.to_numpy(float)
        p_const = np.full(len(ts), y_tr.mean())
        c1 = np.polyfit(ga_tr, y_tr, 1); p_lin = np.polyval(c1, ga_ts)
        c2 = np.polyfit(ga_tr, y_tr, 2); p_quad = np.polyval(c2, ga_ts)
        pl = fit_logistic(ga_tr, y_tr); p_log = logistic(pl, ga_ts)
        rows.append(pd.DataFrame(dict(repeat=rep, fold=fold, Patient_ID=ts.Patient_ID.to_numpy(), GA=ga_ts,
                                      y=ts.logPI.to_numpy(float), const=p_const, linear=p_lin, quadratic=p_quad, logistic=p_log)))
df = pd.concat(rows, ignore_index=True)
# merge mechanistic and benchmark predictions (same folds/seeds)
key = ['repeat', 'fold', 'Patient_ID', 'GA_weeks']
m = pred.rename(columns={'GA_weeks': 'GA'})
df = df.merge(m[['repeat', 'fold', 'Patient_ID', 'GA', 'pred_radial', 'pred_benchmark', 'pred_shunt']], on=['repeat', 'fold', 'Patient_ID', 'GA'], how='left')
print('rows', len(df), 'missing radial', df.pred_radial.isna().sum())
df['radial'] = np.log(df.pred_radial); df['benchmark'] = np.log(df.pred_benchmark)
out = {}
for k in ['const', 'linear', 'quadratic', 'logistic', 'radial', 'benchmark']:
    d = df.dropna(subset=[k])
    out[k] = float(np.sqrt(np.mean((d.y - d[k]) ** 2)))
print(json.dumps(out, indent=1))
# woman-level bootstrap of paired differences vs radial
rng = np.random.default_rng(1)
ids = df.Patient_ID.unique()
grp = {i: g for i, g in df.groupby('Patient_ID')}
def rmse(d, k): return np.sqrt(np.mean((d.y - d[k]) ** 2))
res = {k: [] for k in ['const', 'linear', 'quadratic', 'logistic', 'benchmark']}
for b in range(600):
    s = rng.choice(ids, len(ids), replace=True)
    d = pd.concat([grp[i] for i in s])
    r = rmse(d, 'radial')
    for k in res: res[k].append(rmse(d, k) - r)
ci = {k: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in res.items()}
print('paired RMSE difference vs radial (model - radial): mean, 2.5, 97.5')
for k, v in ci.items(): print(k, np.round(v, 4))
json.dump(dict(rmse=out, paired_vs_radial=ci), open('control_cv.json', 'w'), indent=1)
