import sys, json
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import brentq
import os as _os, pathlib as _pl
base = str(_pl.Path(__file__).resolve().parents[2]).replace(chr(92), '/') + '/'
DATA = _os.environ.get('UTA_DATA', base + 'data/UtA_Doppler_Cleaned_Dataset_Anonymous.xlsx')
sys.path.insert(0, base+'code')
from solver_adapter import simulate
repo=Path(base+'external/network-wave-transmission-master')
def run(**kw):
    r=simulate(repo,**kw); return r
# fine grids per axis (only the axis varied; rest baseline)
axes={'radial_radius':np.linspace(0.10,0.50,161),'uterine_radius':np.linspace(1.0,3.5,201),'arcuate_radius':np.linspace(0.32,1.5,201),'av_radius':np.linspace(0.05,0.8,41),'terminal_resistance':np.geomspace(0.2,5,41),'youngs_modulus':np.geomspace(0.5e6,12e6,41)}
rows=[]
for ax,vals in axes.items():
    for v in vals:
        r=run(**{ax:float(v)})
        rows.append(dict(axis=ax,value=v,PI=r['PI'],SD=r['SD'],RI=r['RI'],notch=r['notch_present'],notch_ratio=r['notch_ratio']))
G=pd.DataFrame(rows); G.to_csv('axis_grids.csv',index=False)
# matched-PI comparison
targets=[1.2,1.0,0.8,0.65]
out=[]
for ax in ('radial_radius','uterine_radius','arcuate_radius','youngs_modulus'):
    g=G[G.axis==ax].sort_values('value')
    for t in targets:
        pis=g.PI.values
        if t<pis.min() or t>pis.max(): out.append(dict(axis=ax,PI=t,reachable=False)); continue
        # PI monotone in axis? find crossing
        idx=np.where(np.diff(np.sign(pis-t))!=0)[0]
        if len(idx)==0: out.append(dict(axis=ax,PI=t,reachable=False)); continue
        i=idx[0]; w=(t-pis[i])/(pis[i+1]-pis[i])
        val=g.value.values[i]+w*(g.value.values[i+1]-g.value.values[i])
        r=run(**{ax:float(val)})
        out.append(dict(axis=ax,PI=t,reachable=True,value=val,SD=r['SD'],RI=r['RI'],notch=r['notch_present'],notch_ratio=r['notch_ratio'],PI_check=r['PI']))
O=pd.DataFrame(out); print(O.round(3).to_string()); O.to_csv('matched_PI.csv',index=False)
