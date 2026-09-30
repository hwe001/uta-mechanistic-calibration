import sys, json, time
import numpy as np, pandas as pd
from pathlib import Path
from SALib.sample import saltelli
from SALib.analyze import sobol
import os as _os, pathlib as _pl
base = str(_pl.Path(__file__).resolve().parents[2]).replace(chr(92), '/') + '/'
DATA = _os.environ.get('UTA_DATA', base + 'data/UtA_Doppler_Cleaned_Dataset_Anonymous.xlsx')
sys.path.insert(0, base+'code')
from solver_adapter import simulate
repo=Path(base+'external/network-wave-transmission-master')
names=['radial_radius','av_radius','terminal_resistance','terminal_compliance','youngs_modulus','heart_rate','uterine_radius','arcuate_radius']
lo=np.array([0.10,0.05,0.2,1e-9,0.5e6,60,1.0,0.5]); hi=np.array([0.50,0.80,5.0,1e-7,12e6,120,3.0,3.0])
problem=dict(num_vars=8,names=names,bounds=[[np.log(a),np.log(b)] for a,b in zip(lo,hi)])
X=saltelli.sample(problem,512,calc_second_order=False)
print('samples',X.shape)
t=time.time(); Y=[];S=[]
for x in X:
    v=np.exp(x); r=simulate(repo,**dict(zip(names,map(float,v))))
    Y.append(r['PI']); S.append(r['SD'])
Y=np.array(Y); S=np.array(S); print('time',time.time()-t,'nan',np.isnan(Y).sum(),'neg SD',(S<=0).sum(),'PI range',np.nanmin(Y),np.nanmax(Y))
ok=np.isfinite(Y)&(Y>0)&(S>1)
print('valid',ok.sum(),len(Y))
np.save('sobol_X.npy',X); np.save('sobol_Y.npy',Y); np.save('sobol_S.npy',S)
# replace invalid by nearest valid quantile to keep design: use log PI clipped
lp=np.log(np.where(ok,Y,np.nan)); lp=np.where(np.isnan(lp),np.nanmedian(lp),lp)
res=sobol.analyze(problem,lp,calc_second_order=False,print_to_console=False)
df=pd.DataFrame(dict(param=names,S1=res['S1'],S1_conf=res['S1_conf'],ST=res['ST'],ST_conf=res['ST_conf']))
print(df.round(3).to_string()); df.to_csv('sobol_logPI.csv',index=False)
# manifold thinness: SD as function of PI across the whole cloud
o=pd.DataFrame(dict(PI=Y[ok],SD=S[ok]))
from scipy.interpolate import UnivariateSpline
o=o.sort_values('PI'); 
lp_=np.log(o.PI.values); ls_=np.log(o.SD.values)
# running median as curve
import numpy.polynomial.polynomial as P
c=np.polyfit(lp_,ls_,4); res_=ls_-np.polyval(c,lp_)
print('log SD explained by 4th-order poly of log PI over cloud: R2=',1-res_.var()/ls_.var(),'resid SD',res_.std())
