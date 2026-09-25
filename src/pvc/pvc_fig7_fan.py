import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
# -*- coding: utf-8 -*-
"""Fig. 7(a)(b) data - the three PVC strategies as a function of PVC share.

Merges what used to be two panels. The old panel (a) swept PVC share for the
'all five upcycled' case only; the old panel (b) showed all three strategies but
only at each region's actual share. Sweeping all three exposes the fan: removing
PVC upstream is flat in PVC share, upcycling everything degrades, and diverting
PVC to incineration degrades fastest.

Run for both energy supply scenarios so the figure can show that the constraint
is specific to the fossil baseline.
"""
import sys, os
import numpy as np
import pandas as pd

BASE = config.INTERMEDIATE
import pvc_bottleneck as M

REGION_W, FEEDS, OTHERS = M.REGION_W, M.FEEDS, M.OTHERS


def max_margin(region, period, scenario='base', w=None, pvc_mode='upcycle'):
    """Best achievable wCR - CR_target over the admissible alpha range.

    Copied verbatim from pvc_fig7_data.py so this script does not re-run that
    module's side effects. The dCR_pvc branch is dropped, it is unused here.
    """
    w = dict(REGION_W[region]) if w is None else dict(w)
    if pvc_mode == 'absent':
        s = sum(w[f] for f in OTHERS)
        w = {f: w[f] / s for f in OTHERS}
        active = OTHERS
    elif pvc_mode == 'incinerate':
        active = OTHERS
    else:
        active = FEEDS

    pb = M.PERIOD_BASE[period]
    tau = M.tau_for(region, period)
    CIg = M.REGION_GRID_CI[region][period]
    CIp, Pp, H2cx = pb['CI_PPA'], pb['P_PPA'], pb['H2cx']
    a_max = M.REGION_ALPHA_MAX[region][period]
    incin_shift = M.INCIN_B_ARR * (CIg - M.CI_BASE_INCIN) / 1000.0
    b_w = sum(w.get(f, 0.0) * M.INCIN_B[f] for f in w)
    CRt = pb['CR_tgt'] - M.RED_PCT[period] * b_w * (CIg - M.CI_BASE_INCIN) / 1000.0

    best = -np.inf
    for a in M.alpha_grid:
        if a > a_max:
            break
        P_elec = M.P_GRID_BASE + a * (Pp - M.P_GRID_BASE)
        CI_elec_g = CIg + a * (CIp - CIg)
        if scenario == 'elec':
            UP_x, CR_x = M.transform_electrification(P_elec, CI_elec_g, H2cx)
        else:
            UP_x, CR_x = M.transform_baseline(P_elec, CI_elec_g)
        CR_x = CR_x - incin_shift
        UP_net = UP_x + tau * CR_x
        cr = 0.0
        for f in active:
            idxs = M.feed_idx[f]
            j = idxs[np.argmax(UP_net[idxs])]
            cr += w[f] * CR_x[j]
        best = max(best, cr - CRt)
    return best, CRt


def reweight(region, x):
    w0 = REGION_W[region]
    s = sum(w0[f] for f in OTHERS)
    return {**{f: w0[f] / s * (1 - x) for f in OTHERS}, 'PVC': x}


MODES = [('upcycle', 'A'), ('incinerate', 'B'), ('absent', 'C')]
SCEN = [('base', 'Fossil baseline'), ('elec', 'Renewable electrification')]
sweep = np.round(np.arange(0.0, 0.2601, 0.01), 4)

rows = []
for sc, sc_name in SCEN:
    for region in REGION_W:
        for mode, tag in MODES:
            for x in sweep:
                m, _ = max_margin(region, '2050', sc, w=reweight(region, x),
                                  pvc_mode=mode)
                rows.append(dict(Scenario=sc_name, Strategy=tag, Region=region,
                                 w_PVC=x, margin=m))
    print(f'  swept {sc_name}')
fan = pd.DataFrame(rows)

pts = []
for sc, sc_name in SCEN:
    for region in REGION_W:
        xa = REGION_W[region]['PVC']
        for mode, tag in MODES:
            m, _ = max_margin(region, '2050', sc, pvc_mode=mode)
            pts.append(dict(Scenario=sc_name, Strategy=tag, Region=region,
                            w_PVC_actual=xa, margin=m,
                            grid_CI=M.REGION_GRID_CI[region]['2050']))
pt = pd.DataFrame(pts)

fan.to_csv(os.path.join(BASE, 'fig7_fan_sweep.csv'), index=False,
           float_format='%.5f')
pt.to_csv(os.path.join(BASE, 'fig7_fan_points.csv'), index=False,
          float_format='%.5f')

# ── checks ───────────────────────────────────────────────────────────
old = pd.read_csv(os.path.join(BASE, 'fig7_panelB.csv'))
b = pt[pt.Scenario == 'Fossil baseline'].pivot_table(
    index='Region', columns='Strategy', values='margin')
chk = old.set_index('Region')[['margin_A', 'margin_B', 'margin_C']].join(b)
err = np.abs(chk[['A', 'B', 'C']].values
             - chk[['margin_A', 'margin_B', 'margin_C']].values).max()
print(f'\nreproduces fig7_panelB.csv: max |diff| = {err:.2e}   '
      f'{"OK" if err < 1e-6 else "*** REVIEW ***"}')

for sc_name in [s[1] for s in SCEN]:
    d = pt[pt.Scenario == sc_name].pivot_table(index='Region',
                                               columns='Strategy',
                                               values='margin')
    print(f'\n{sc_name}: strategy A infeasible in '
          f'{(d.A < 0).sum()}/11  ({", ".join(d.index[d.A < 0])})')
    print(f'   B worse than A in {(d.B < d.A).sum()}/11, '
          f'C better than A in {(d.C > d.A).sum()}/11')
    print(f'   C range {d.C.min():+.3f} to {d.C.max():+.3f}   '
          f'(flat in PVC share)' if d.C.max() - d.C.min() < 0.1 else '')

print('\nwrote fig7_fan_sweep.csv, fig7_fan_points.csv')
