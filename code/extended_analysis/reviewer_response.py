"""Analyses added in response to an internal senior review (variance decomposition and ICC, mixed-model form, reference comparison,
Jacobian at several points with noise weighting, Sobol sensitivity to failed-run handling, woman-level bootstrap of the calibrated
trajectory, bin-width sensitivity). Needs the private workbook (UTA_DATA) for the data-based parts."""
import json, os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan

warnings.filterwarnings('ignore')
base = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get('UTA_DATA', base / 'data' / 'UtA_Doppler_Cleaned_Dataset_Anonymous.xlsx'))
CENT = os.environ.get('UTA_CENTILES', '')
sys.path.insert(0, str(base / 'code'))
from benchmark_fit import load_core_examinations
import mechanistic_calibration as mc
import identifiability_analysis as ia
OUT = base / 'results' / 'reviewer_response.json'
SOB = Path(os.environ.get('UTA_SOBOL_DIR', base / 'results' / 'extended_analysis'))
res = {}
core = load_core_examinations(DATA)
core = core.rename(columns={'GA_weeks': 'GA'})

# ---- A. variance decomposition, ICC, mixed-model form, residual diagnostics
def fit(formula):
    m = smf.mixedlm(formula, core, groups=core['Patient_ID']).fit(reml=False)
    sb2 = float(m.cov_re.iloc[0, 0]); se2 = float(m.scale)
    fixed = np.asarray(m.model.exog @ m.fe_params)
    vf = float(np.var(fixed))
    resid = np.asarray(m.resid)
    bp = het_breuschpagan(resid, np.column_stack([np.ones(len(core)), core.GA.values]))[1]
    return m, dict(aic=float(m.aic), bic=float(m.bic), loglik=float(m.llf), between_var=sb2, within_var=se2, icc=sb2 / (sb2 + se2),
                   fixed_var=vf, marginal_R2=vf / (vf + sb2 + se2), conditional_R2=(vf + sb2) / (vf + sb2 + se2),
                   shapiro_p_conditional_resid=float(stats.shapiro(resid)[1]), breusch_pagan_p=float(bp))
forms = {'null': 'logPI ~ 1', 'quadratic': 'logPI ~ GA + I(GA**2)', 'cubic': 'logPI ~ GA + I(GA**2) + I(GA**3)', 'spline_df4': 'logPI ~ bs(GA, df=4)'}
models = {}; A = {}
for k, f in forms.items():
    models[k], A[k] = fit(f)
res['mixed_models'] = A
q = models['quadratic']
# fraction of total variance by component (quadratic model)
a = A['quadratic']; tot = a['fixed_var'] + a['between_var'] + a['within_var']
res['variance_decomposition_quadratic'] = {'total_variance': tot, 'fraction_gestational_age': a['fixed_var'] / tot, 'fraction_between_woman': a['between_var'] / tot, 'fraction_within_woman': a['within_var'] / tot,
                                          'icc_after_GA': a['icc'], 'icc_null_model': A['null']['icc']}
# fitted PI and comparison with the cohort's own centile table
fe = q.fe_params
def pi_at(g): return float(np.exp(fe['Intercept'] + fe['GA'] * g + fe['I(GA ** 2)'] * g ** 2))
res['fitted_pi_population_curve'] = {str(g): pi_at(g) for g in (14, 20, 24, 28, 34, 40)}
if CENT and Path(CENT).exists():
    c = pd.read_excel(CENT); res['centile_table_P50_and_mean'] = {int(r.GA_weeks): [float(r.P50), float(r.Mean)] for r in c.itertuples() if int(r.GA_weeks) in (14, 20, 24, 28, 34, 40)}

# ---- B. Sobol sensitivity to failed-run handling
from SALib.analyze import sobol
names = ['radial_radius', 'av_radius', 'terminal_resistance', 'terminal_compliance', 'youngs_modulus', 'heart_rate', 'uterine_radius', 'arcuate_radius']
lo = np.array([0.10, 0.05, 0.2, 1e-9, 0.5e6, 60, 1.0, 0.5]); hi = np.array([0.50, 0.80, 5.0, 1e-7, 12e6, 120, 3.0, 3.0])
problem = dict(num_vars=8, names=names, bounds=[[np.log(a_), np.log(b_)] for a_, b_ in zip(lo, hi)])
X = np.load(SOB / 'sobol_X.npy'); Y = np.load(SOB / 'sobol_Y.npy'); S = np.load(SOB / 'sobol_S.npy')
ok = np.isfinite(Y) & (Y > 0) & (S > 1)
lpv = np.log(np.where(ok, Y, np.nan)); res_sob = {'failure_rate': float(1 - ok.mean()), 'n_runs': int(len(Y))}
for label, fill in [('median', np.nanmedian(lpv)), ('p05', np.nanpercentile(lpv, 5)), ('p95', np.nanpercentile(lpv, 95)), ('min', np.nanmin(lpv)), ('max', np.nanmax(lpv))]:
    v = np.where(np.isnan(lpv), fill, lpv)
    r = sobol.analyze(problem, v, calc_second_order=False, print_to_console=False)
    res_sob['ST_fill_' + label] = {n: float(x) for n, x in zip(names, r['ST'])}
# failure rate by parameter quartile
fr = {}
for j, n in enumerate(names):
    q4 = pd.qcut(X[:, j], 4, labels=False, duplicates='drop')
    fr[n] = [float(1 - ok[q4 == k].mean()) for k in range(4)]
res_sob['failure_rate_by_parameter_quartile'] = fr
res['sobol_failed_run_sensitivity'] = res_sob

# ---- C. Jacobian at several points, with raw, log and noise-weighted outputs
def jac(values, log_out=False):
    x0 = ia.transformed(values); J = np.zeros((2, len(ia.PARAMS))); y0 = ia.outputs(base / 'external' / 'network-wave-transmission-master', values)
    for j in range(len(ia.PARAMS)):
        h = 1e-4; xp, xm = x0.copy(), x0.copy(); xp[j] += h; xm[j] -= h
        yp = ia.outputs(base / 'external' / 'network-wave-transmission-master', ia.untransformed(xp)); ym = ia.outputs(base / 'external' / 'network-wave-transmission-master', ia.untransformed(xm))
        J[:, j] = (np.log(yp) - np.log(ym)) / (2 * h) if log_out else (yp - ym) / (2 * h)
    return J, y0
sdlog_pi = float(np.sqrt(A['quadratic']['within_var'] + A['quadratic']['between_var']))
sdlog_sd = float(np.log(core.UtA_SD.astype(float)).std())
pts = {'baseline': dict(ia.DEFAULTS)}
for g, r in [('14w', 0.117), ('27w', 0.145), ('40w', 0.179)]:
    p = dict(ia.DEFAULTS); p['radial_radius_mm'] = r; pts['radial_at_' + g] = p
rng = np.random.default_rng(3); lo_t, hi_t = ia.bounds_transformed(); rand = []
for _ in range(60):
    xr = rng.uniform(lo_t, hi_t); rand.append(ia.untransformed(xr))
def summarize(vals):
    J, y0 = jac(vals); Jl, _ = jac(vals, True)
    Jn = Jl / np.array([[sdlog_pi], [sdlog_sd]])   # log outputs weighted by their observed between-examination SD
    sv = lambda M: np.linalg.svd(M, compute_uv=False).tolist()
    u = np.linalg.svd(Jl)[0]
    return dict(PI=float(y0[0]), SD=float(y0[1]), sv_raw=sv(J), cond_raw=float(sv(J)[0] / sv(J)[1]), sv_log=sv(Jl), cond_log=float(sv(Jl)[0] / sv(Jl)[1]), sv_noise_weighted=sv(Jn),
                left_vector_2_log=u[:, 1].tolist())
jr = {k: summarize(v) for k, v in pts.items()}
rs = [summarize(v) for v in rand if np.isfinite(ia.outputs(base / 'external' / 'network-wave-transmission-master', v)).all()]
res['jacobian_points'] = jr
res['jacobian_random_points'] = {'n': len(rs), 'cond_raw_median': float(np.median([r['cond_raw'] for r in rs])), 'cond_raw_range': [float(min(r['cond_raw'] for r in rs)), float(max(r['cond_raw'] for r in rs))],
                                 'cond_log_median': float(np.median([r['cond_log'] for r in rs])), 'cond_log_range': [float(min(r['cond_log'] for r in rs)), float(max(r['cond_log'] for r in rs))],
                                 'second_sv_noise_weighted_median': float(np.median([r['sv_noise_weighted'][1] for r in rs])), 'second_sv_noise_weighted_max': float(max(r['sv_noise_weighted'][1] for r in rs))}
res['noise_scales'] = {'sd_log_PI_between_examinations': sdlog_pi, 'sd_log_SD_between_examinations': sdlog_sd}

# ---- D. woman-level bootstrap of the calibrated trajectory and E. bin-width sensitivity
mc.MECHANISMS.update({'uterine': 'uterine_radius', 'arcuate': 'arcuate_radius'})
grid0 = pd.read_csv(base / 'results' / 'mechanistic_calibration' / 'solver_lookup_grid.csv')
ax = pd.read_csv(SOB / 'axis_grids.csv'); new = ax[ax.axis.isin(['uterine_radius', 'arcuate_radius'])].rename(columns={'axis': 'mechanism'})[['mechanism', 'value', 'PI', 'SD', 'RI']]
grid = pd.concat([grid0, new], ignore_index=True)
def make_bins(d, width=2.0):
    d = d.copy(); edges = np.arange(14.0, 42.0 + 1e-9, width)
    d['b'] = pd.cut(d.GA, bins=edges, right=False)
    g = d.groupby('b', observed=True).agg(ga_median=('GA', 'median'), target_PI=('UtA_PI', 'median'), n=('UtA_PI', 'size')).reset_index(drop=True)
    return g
bins2 = make_bins(core)
women = core.Patient_ID.unique(); byw = {w: g for w, g in core.groupby('Patient_ID')}
D = {}
for m_ in ('radial', 'uterine', 'arcuate'):
    f = mc.fit_smooth(bins2, grid, m_); pt = f['params']; samples = []
    brng = np.random.default_rng(20260930)
    for _ in range(300):
        pick = brng.choice(women, size=len(women), replace=True)
        d = pd.concat([byw[w] for w in pick], ignore_index=True)
        b = make_bins(d)
        if b.ga_median.nunique() < 4: continue
        try: samples.append(mc.logistic_log_parameter(mc.fit_smooth(b, grid, m_, start=pt)['params'], np.array([14.0, 40.0])))
        except Exception: continue
    v = np.exp(np.array(samples))
    D[m_] = {'n_boot': len(samples), '14w': [float(np.percentile(v[:, 0], 2.5)), float(np.percentile(v[:, 0], 97.5))], '40w': [float(np.percentile(v[:, 1], 2.5)), float(np.percentile(v[:, 1], 97.5))],
             'point_14w': float(np.exp(mc.logistic_log_parameter(pt, np.array([14.0]))[0])), 'point_40w': float(np.exp(mc.logistic_log_parameter(pt, np.array([40.0]))[0]))}
res['woman_level_bootstrap_trajectory'] = D
E_ = {}
for w in (1.0, 2.0, 4.0):
    b = make_bins(core, w); f = mc.fit_smooth(b, grid, 'radial'); p = f['params']
    E_[str(w)] = {'n_bins': int(len(b)), 'radius_14w': float(np.exp(mc.logistic_log_parameter(p, np.array([14.0]))[0])), 'radius_40w': float(np.exp(mc.logistic_log_parameter(p, np.array([40.0]))[0])),
                  'cost': float(f['cost'])}
res['bin_width_sensitivity_radial'] = E_
OUT.write_text(json.dumps(res, indent=2)); print('saved', OUT)
