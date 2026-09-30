import pandas as pd, numpy as np
import os as _os, pathlib as _pl
base = str(_pl.Path(__file__).resolve().parents[2]).replace(chr(92), '/') + '/'
DATA = _os.environ.get('UTA_DATA', base + 'data/UtA_Doppler_Cleaned_Dataset_Anonymous.xlsx')
df=pd.read_excel(DATA)
cal=pd.read_csv(base+'results/mechanistic_calibration/one_dimensional_calibration.csv')
grid=pd.read_csv(base+'results/mechanistic_calibration/solver_lookup_grid.csv')
rad=grid[grid.mechanism=='radial_radius'].sort_values('value')
core=df[(df.GA_weeks>=14)&df.UtA_PI.notna()&df.UtA_SD.notna()]
print(len(core), core.UtA_SD.describe())
# bins from calibration
r=cal[cal.model=='radial']
edges=None
out=[]
gas=r.ga_median.values
# assign each exam to nearest bin median (approx two-week bins)
core=core.copy(); core['bin']=[gas[np.argmin(abs(gas-g))] for g in core.GA_weeks]
for g,est in zip(r.ga_median,r.estimate):
    sub=core[core.bin==g]
    pi_pred=np.interp(est,rad.value,rad.PI); sd_pred=np.interp(est,rad.value,rad.SD); ri_pred=np.interp(est,rad.value,rad.RI)
    out.append(dict(ga=g,n=len(sub),PI_obs=sub.UtA_PI.median(),PI_fit=pi_pred,SD_obs=sub.UtA_SD.median(),SD_pred=sd_pred,RI_obs=sub.UtA_RI.median(),RI_pred=ri_pred,radius=est))
o=pd.DataFrame(out); print(o.round(3).to_string())
o.to_csv('sd_check.csv',index=False)
print('RMSE SD',np.sqrt(np.mean((o.SD_obs-o.SD_pred)**2)),'RMSE RI',np.sqrt(np.mean((o.RI_obs-o.RI_pred)**2)))
