import sys, time, json
import numpy as np, pandas as pd
from pathlib import Path
import os as _os, pathlib as _pl
base = str(_pl.Path(__file__).resolve().parents[2]).replace(chr(92), '/') + '/'
DATA = _os.environ.get('UTA_DATA', base + 'data/UtA_Doppler_Cleaned_Dataset_Anonymous.xlsx')
sys.path.insert(0, base+'code')
from solver_adapter import simulate
repo=Path(base+'external/network-wave-transmission-master')
t=time.time(); r=simulate(repo); print('baseline',{k:r[k] for k in ('PI','SD','RI')}, 'time',time.time()-t)
rows=[]
for name,vals in (('uterine_radius',[0.6,0.8,1.0,1.25,1.5,2.0,2.5,3.0,4.0]),('arcuate_radius',[0.3,0.5,0.75,1.0,1.5,2.0,2.5,3.0]),('heart_rate',[60,72,80,90,100,110,120]),('youngs_modulus',[0.5e6,1.0e6,1.5e6,3.0e6,6.0e6,12e6])):
    for v in vals:
        kw={name:v}
        r=simulate(repo,**kw); rows.append(dict(param=name,value=v,PI=r['PI'],SD=r['SD'],RI=r['RI']))
df=pd.DataFrame(rows); print(df.round(3).to_string()); df.to_csv('other_axes.csv',index=False)
