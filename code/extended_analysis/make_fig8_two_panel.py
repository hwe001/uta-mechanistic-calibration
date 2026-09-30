"""Figure 8, two panels: (a) observed S/D against PI with the S/D-PI relation of each axis; (b) S/D of each axis relative to the radial axis at matched PI."""
import os, pathlib, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = pathlib.Path(__file__).resolve().parents[2]
DATA = pathlib.Path(os.environ.get('UTA_DATA', base / 'data' / 'UtA_Doppler_Cleaned_Dataset_Anonymous.xlsx'))
sys.path.insert(0, str(base / 'code'))
from benchmark_fit import load_core_examinations
core = load_core_examinations(DATA)
ax_ = pd.read_csv(base / 'results' / 'extended_analysis' / 'axis_grids.csv')
col = {'radial_radius': '#1f5fa8', 'uterine_radius': '#2e7d32', 'arcuate_radius': '#8e5ea2', 'youngs_modulus': '#b9770e', 'av_radius': '#c0392b', 'terminal_resistance': '#7f8c8d'}
lab = {'radial_radius': 'Radial radius', 'uterine_radius': 'Uterine radius', 'arcuate_radius': 'Arcuate radius', 'youngs_modulus': "Young's modulus", 'av_radius': 'A-V shunt radius', 'terminal_resistance': 'Terminal resistance'}
solid = ('radial_radius', 'uterine_radius', 'arcuate_radius')
curves = {}
for k in col:
    g = ax_[ax_.axis == k].sort_values('PI'); g = g[(g.SD > 0) & (g.PI < 3)].drop_duplicates('PI'); curves[k] = (g.PI.values, g.SD.values)
fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.0), dpi=200, gridspec_kw={'width_ratios': [1.1, 1]})
a.scatter(core.UtA_PI, core.UtA_SD, s=4, color='#bbbbbb', alpha=0.6, label='Observed examinations', zorder=1)
for k, (pi, sd) in curves.items():
    a.plot(pi, sd, '-' if k in solid else '--', lw=1.6 if k in solid else 1.0, color=col[k], label=lab[k], zorder=2)
a.set_xlim(0.3, 2.0); a.set_ylim(1.2, 5.2); a.set_xlabel('UtA-PI'); a.set_ylabel('UtA S/D'); a.legend(fontsize=6, frameon=False, loc='upper left'); a.set_title('a  S/D against PI', loc='left', fontsize=10)
rpi, rsd = curves['radial_radius']
for k, (pi, sd) in curves.items():
    if k == 'radial_radius': continue
    lo, hi = max(pi.min(), rpi.min()), min(pi.max(), rpi.max()); grid = np.linspace(lo, hi, 200)
    rel = (np.interp(grid, pi, sd) / np.interp(grid, rpi, rsd) - 1) * 100
    b.plot(grid, rel, '-' if k in solid else '--', lw=1.6 if k in solid else 1.0, color=col[k], label=lab[k])
b.axhline(0, color='#1f5fa8', lw=1.0)
b.set_xlim(0.3, 1.6); b.set_xlabel('UtA-PI (matched)'); b.set_ylabel('S/D relative to the radial axis (%)'); b.legend(fontsize=6, frameon=False, loc='upper left'); b.set_title('b  Difference between axes at matched PI', loc='left', fontsize=10)
for s in ('top', 'right'):
    a.spines[s].set_visible(False); b.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(base / 'results' / 'extended_analysis' / 'fig8_two_panel.png')
print('saved')
