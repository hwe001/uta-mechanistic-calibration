import sys, json
import numpy as np, pandas as pd
import os as _os, pathlib as _pl
base = str(_pl.Path(__file__).resolve().parents[2]).replace(chr(92), '/') + '/'
DATA = _os.environ.get('UTA_DATA', base + 'data/UtA_Doppler_Cleaned_Dataset_Anonymous.xlsx')
sys.path.insert(0, base+'code')
from benchmark_fit import load_core_examinations
core=load_core_examinations(DATA)
ax=pd.read_csv('axis_grids.csv')
res={}
for name in ('radial_radius','uterine_radius','arcuate_radius'):
    g=ax[ax.axis==name].sort_values('PI').drop_duplicates('PI')
    # PI->SD along that axis (monotone region)
    core['sd_pred_'+name]=np.interp(core.UtA_PI,g.PI,g.SD)
    inrange=(core.UtA_PI>=g.PI.min())&(core.UtA_PI<=g.PI.max())
    d=core[inrange]
    res[name]=dict(n=int(inrange.sum()),rmse_logSD=float(np.sqrt(np.mean((np.log(d.UtA_SD)-np.log(d['sd_pred_'+name]))**2))),
                   bias_logSD=float(np.mean(np.log(d.UtA_SD)-np.log(d['sd_pred_'+name]))),
                   rho=float(np.corrcoef(np.log(d.UtA_SD),np.log(d['sd_pred_'+name]))[0,1]))
# empirical quadratic regression logSD ~ logPI (in-sample, best case for a purely empirical relation)
x=np.log(core.UtA_PI); y=np.log(core.UtA_SD); c=np.polyfit(x,y,2); r=y-np.polyval(c,x)
res['empirical_quadratic_in_sample']=dict(n=len(core),rmse_logSD=float(np.sqrt(np.mean(r**2))))
# GA-only SD prediction: CV comparison already in extended_cv (mechanistic sd_ from GA); empirical quadratic in GA
cv=pd.read_csv('extended_cv_predictions.csv')
# empirical GA-only logSD via 5-fold same folds: quadratic in GA fitted on other women
rows=[]
for (rep,fold),d in cv.groupby(['repeat','fold']):
    test_ids=set(d.Patient_ID); tr=core[~core.Patient_ID.isin(test_ids)]
    cc=np.polyfit(tr.GA_weeks,np.log(tr.UtA_SD),2)
    rows.append(np.log(d.sd_obs)-np.polyval(cc,d.GA))
allr=np.concatenate(rows); res['empirical_GA_quadratic_logSD_cv_rmse']=float(np.sqrt(np.mean(allr**2)))
print(json.dumps(res,indent=1)); json.dump(res,open('sd_individual.json','w'),indent=1)
