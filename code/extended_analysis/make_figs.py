import sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os as _os, pathlib as _pl
base = str(_pl.Path(__file__).resolve().parents[2]).replace(chr(92), '/') + '/'
DATA = _os.environ.get('UTA_DATA', base + 'data/UtA_Doppler_Cleaned_Dataset_Anonymous.xlsx')
sys.path.insert(0, base+'code')
from benchmark_fit import load_core_examinations
from mechanistic_calibration import logistic_log_parameter
core=load_core_examinations(DATA)
ax_=pd.read_csv('axis_grids.csv'); ext=json.load(open('extended_mech.json')); ctl=json.load(open('control_cv.json')); sob=pd.read_csv('sobol_logPI.csv')
col={'radial_radius':'#1f5fa8','uterine_radius':'#2e7d32','arcuate_radius':'#8e5ea2','youngs_modulus':'#b9770e','av_radius':'#c0392b','terminal_resistance':'#7f8c8d'}
lab={'radial_radius':'Radial radius','uterine_radius':'Uterine radius','arcuate_radius':'Arcuate radius','youngs_modulus':"Young's modulus",'av_radius':'A-V shunt radius','terminal_resistance':'Terminal resistance'}
# ---- Fig 8: manifold
fig,a=plt.subplots(figsize=(5.6,4.2),dpi=200)
a.scatter(core.UtA_PI,core.UtA_SD,s=4,color='#bbbbbb',alpha=0.6,label='Observed examinations',zorder=1)
for k in ('radial_radius','uterine_radius','arcuate_radius','youngs_modulus','av_radius','terminal_resistance'):
    g=ax_[ax_.axis==k].sort_values('PI'); g=g[(g.SD>0)&(g.PI<3)]
    a.plot(g.PI,g.SD,'-' if k in ('radial_radius','uterine_radius','arcuate_radius') else '--',lw=1.6 if k in ('radial_radius','uterine_radius','arcuate_radius') else 1.0,color=col[k],label=lab[k],zorder=3)
a.set_xlim(0.3,2.0); a.set_ylim(1.2,5.2); a.set_xlabel('UtA-PI'); a.set_ylabel('UtA S/D')
a.legend(fontsize=6,frameon=False,loc='upper left')
for s in ('top','right'): a.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig('fig8.png')
# ---- Fig 9
fig,(a1,a2)=plt.subplots(1,2,figsize=(9.2,3.6),dpi=200,gridspec_kw={'width_ratios':[1,1.15]})
ga=np.linspace(14,40,100)
for m,cc,unit in (('radial','#1f5fa8','radius (mm)'),('uterine','#2e7d32','radius (mm)'),('arcuate','#8e5ea2','radius (mm)')):
    p=np.array(ext['pop'][m]['params']); v=np.exp(logistic_log_parameter(p,ga))
    a1.plot(ga,v/v[0],color=cc,label={'radial':'Radial','uterine':'Uterine','arcuate':'Arcuate'}[m]+' (%.2f to %.2f mm)'%(v[0],v[-1]))
a1.set_xlabel('Gestational age (weeks)'); a1.set_ylabel('Fitted radius relative to 14 weeks'); a1.legend(fontsize=6.5,frameon=False)
names=['Constant mean','Linear in GA','Quadratic (benchmark)','Logistic curve\n(no solver)','Radial','Uterine','Arcuate','A-V shunt','Distal resistance']
vals=[ctl['rmse']['const'],ctl['rmse']['linear'],ctl['rmse']['benchmark'],ctl['rmse']['logistic'],ext['cv']['radial']['rmse_logPI'],ext['cv']['uterine']['rmse_logPI'],ext['cv']['arcuate']['rmse_logPI'],ext['cv']['shunt']['rmse_logPI'],ext['cv']['distal']['rmse_logPI']]
cols=['#bbbbbb','#bbbbbb','#888888','#888888','#1f5fa8','#2e7d32','#8e5ea2','#c0392b','#7f8c8d']
a2.barh(range(len(vals))[::-1],vals,color=cols)
for i,v in zip(range(len(vals))[::-1],vals): a2.text(v+0.005,i,'%.3f'%v,va='center',fontsize=6.5)
a2.set_yticks(range(len(vals))[::-1]); a2.set_yticklabels(names,fontsize=6.5); a2.set_xlabel('Held-out RMSE, log(UtA-PI)'); a2.set_xlim(0,0.62)
for a,l in ((a1,'a'),(a2,'b')):
    a.text(-0.02,1.03,l,transform=a.transAxes,fontweight='bold')
    for s in ('top','right'): a.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig('fig9.png')
# ---- Fig 10 Sobol
fig,a=plt.subplots(figsize=(5.6,3.2),dpi=200)
order=sob.sort_values('ST',ascending=True)
y=np.arange(len(order)); pretty={'radial_radius':'Radial radius','av_radius':'A-V shunt radius','terminal_resistance':'Terminal resistance','terminal_compliance':'Terminal compliance','youngs_modulus':"Young's modulus",'heart_rate':'Heart rate','uterine_radius':'Uterine radius','arcuate_radius':'Arcuate radius'}
a.barh(y+0.18,order.ST,height=0.34,color='#1f5fa8',label='Total-order index')
a.barh(y-0.18,order.S1,height=0.34,color='#9dbbe0',label='First-order index')
a.set_yticks(y); a.set_yticklabels([pretty[p] for p in order.param],fontsize=7); a.set_xlabel('Sobol index for log(UtA-PI)'); a.legend(fontsize=6.5,frameon=False,loc='lower right')
for s in ('top','right'): a.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig('fig10.png'); print('done')
