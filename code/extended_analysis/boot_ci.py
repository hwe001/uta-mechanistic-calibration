import sys, json
import numpy as np, pandas as pd
from pathlib import Path
import os as _os, pathlib as _pl
base = str(_pl.Path(__file__).resolve().parents[2]).replace(chr(92), '/') + '/'
DATA = _os.environ.get('UTA_DATA', base + 'data/UtA_Doppler_Cleaned_Dataset_Anonymous.xlsx')
sys.path.insert(0, base+'code')
import mechanistic_calibration as mc
mc.MECHANISMS.update({'uterine':'uterine_radius','arcuate':'arcuate_radius'})
grid0=pd.read_csv(base+'results/mechanistic_calibration/solver_lookup_grid.csv')
ax=pd.read_csv('axis_grids.csv'); new=ax[ax.axis.isin(['uterine_radius','arcuate_radius'])].rename(columns={'axis':'mechanism'})[['mechanism','value','PI','SD','RI']]
grid=pd.concat([grid0,new],ignore_index=True)
bins=mc.load_core_bins(Path(DATA))
out={}
for m in ('radial','uterine','arcuate'):
    f=mc.fit_smooth(bins,grid,m)
    s=mc.bootstrap_smooth(bins,grid,m,f['params'],20260915,400,np.array([14.0,40.0]))
    v=np.exp(s); out[m]={'14w':[float(np.nanpercentile(v[:,0],2.5)),float(np.nanpercentile(v[:,0],97.5))],'40w':[float(np.nanpercentile(v[:,1],2.5)),float(np.nanpercentile(v[:,1],97.5))]}
    print(m,out[m])
json.dump(out,open('boot_ci.json','w'),indent=1)
