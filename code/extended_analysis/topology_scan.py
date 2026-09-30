"""Topology check: terminal-resistance and A-V shunt scans at different shunt and radial settings (writes results/topology_scan.json)."""
import json, sys
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(here / 'code'))
import identifiability_analysis as ia

repo = here / 'external' / 'network-wave-transmission-master'
out = {}


def pi(vals):
    try:
        return float(ia.outputs(repo, vals)[0])
    except Exception:
        return float('nan')


rt = np.exp(np.linspace(np.log(0.2), np.log(100), 25))
for radial in (0.20, 0.117):
    for av in (0.05, 0.20, 3.0):
        a = []
        for r in rt:
            v = dict(ia.DEFAULTS); v['radial_radius_mm'] = radial; v['av_radius_mm'] = av; v['terminal_resistance'] = float(r); a.append(pi(v))
        a = np.array(a); out[f'terminal_resistance_scan_radial{radial}_av{av}'] = {'PI_min': float(np.nanmin(a)), 'PI_max': float(np.nanmax(a)), 'n_valid': int(np.isfinite(a).sum())}
for radial in (0.20, 0.117):
    a = []
    for s in np.exp(np.linspace(np.log(0.05), np.log(3.0), 25)):
        v = dict(ia.DEFAULTS); v['radial_radius_mm'] = radial; v['av_radius_mm'] = float(s); a.append(pi(v))
    a = np.array(a); out[f'av_radius_scan_radial{radial}'] = {'PI_min': float(np.nanmin(a)), 'PI_max': float(np.nanmax(a))}
(here / 'results' / 'topology_scan.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
