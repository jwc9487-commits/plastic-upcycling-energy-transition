import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
"""Fig. 7 data preparation — PVC as the binding constraint.

Exports three tidy CSVs, one per panel.

Convention: every panel reports the ENVIRONMENTAL HEADROOM

    margin = max over alpha <= alpha_max of ( wCR - CR_target )

that is, the best the system can do against its target, ignoring profit. This is
exactly equivalent to the feasibility test used elsewhere (margin >= 0 <=> feasible)
but stays finite for infeasible cases, so infeasible regions can still be plotted
and their distance from the target read off.
"""
import sys, os
import numpy as np
import pandas as pd

BASE = config.INTERMEDIATE
import pvc_bottleneck as M

REGION_W, FEEDS, OTHERS = M.REGION_W, M.OTHERS and M.FEEDS, M.OTHERS
SCALE = 2e-8


def max_margin(region, period, scenario='base', w=None, pvc_mode='upcycle', dCR_pvc=0.0):
    """Best achievable wCR - CR_target over the admissible alpha range."""
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
        if dCR_pvc:
            CR_x = CR_x + dCR_pvc * M.IS_PVC
        UP_net = UP_x + tau * CR_x
        cr = 0.0
        for f in active:
            idxs = M.feed_idx[f]
            j = idxs[np.argmax(UP_net[idxs])]
            cr += w[f] * CR_x[j]
        best = max(best, cr - CRt)
    return best, CRt


# ── consistency check against the feasibility flags already published ──
cfg = pd.read_csv(os.path.join(BASE, 'pvc_bottleneck_configs.csv'))
bad = 0
for r in cfg.itertuples():
    for mode, key in [('upcycle', 'A_upcycle_feas'), ('incinerate', 'B_incin_feas'),
                      ('absent', 'C_absent_feas')]:
        sc = 'base' if r.Scenario == 'Baseline' else 'elec'
        m = max_margin(r.Region, str(r.Period), sc, pvc_mode=mode)[0]
        if (m >= -1e-9) != bool(getattr(r, key)):
            bad += 1
            print(f'  MISMATCH {r.Region} {r.Scenario} {r.Period} {mode}: margin {m:+.4f}')
print(f'margin-vs-feasibility consistency: {len(cfg)*3 - bad}/{len(cfg)*3} agree'
      f'   {"OK" if bad == 0 else "*** REVIEW ***"}')

# ══ panel A — headroom versus PVC share, 2050 fossil baseline ════════
def reweight(region, x):
    w0 = REGION_W[region]
    s = sum(w0[f] for f in OTHERS)
    return {**{f: w0[f] / s * (1 - x) for f in OTHERS}, 'PVC': x}

sweep = np.arange(0.0, 0.2601, 0.005)
rows = []
for region in REGION_W:
    for x in sweep:
        m, _ = max_margin(region, '2050', 'base', w=reweight(region, x))
        rows.append(dict(Region=region, w_PVC=x, margin=m,
                         grid_CI=M.REGION_GRID_CI[region]['2050'],
                         w_PVC_actual=REGION_W[region]['PVC']))
pd.DataFrame(rows).to_csv(os.path.join(BASE, 'fig7_panelA.csv'),
                          index=False, float_format='%.5f')
print(f'panel A: {len(rows)} rows')

# ══ panel B — the three strategies, 2050 fossil baseline ═════════════
rows = []
for region in REGION_W:
    rec = dict(Region=region, w_PVC=REGION_W[region]['PVC'],
               grid_CI=M.REGION_GRID_CI[region]['2050'])
    for mode, key in [('upcycle', 'A'), ('incinerate', 'B'), ('absent', 'C')]:
        m, t = max_margin(region, '2050', 'base', pvc_mode=mode)
        rec[f'margin_{key}'] = m
        rec[f'CRt_{key}'] = t
    rows.append(rec)
pb = pd.DataFrame(rows).sort_values('w_PVC')
pb.to_csv(os.path.join(BASE, 'fig7_panelB.csv'), index=False, float_format='%.5f')
print(f'panel B: {len(pb)} regions   '
      f'(B worse than A in {(pb.margin_B < pb.margin_A).sum()}/11, '
      f'C better than A in {(pb.margin_C > pb.margin_A).sum()}/11)')

# ══ panel C — carbon accounting terms per feed ═══════════════════════
def col(i): return pd.to_numeric(M.df.iloc[:, i], errors='coerce').fillna(0).values
PD_kg = np.abs(col(171)) * SCALE
FC_kg = col(170) * SCALE
EG_kg = col(43) * SCALE
assert np.abs(PD_kg + FC_kg - EG_kg - M.CR0).max() < 1e-6, 'CR identity broken'

# evaluated at the 2050 fossil-baseline optimum of a mid-composition region
HOST = 'Korea'
pbm = M.PERIOD_BASE['2050']
CIg = M.REGION_GRID_CI[HOST]['2050']
a_max = M.REGION_ALPHA_MAX[HOST]['2050']
tau = M.tau_for(HOST, '2050')
incin_shift = M.INCIN_B_ARR * (CIg - M.CI_BASE_INCIN) / 1000.0
w = REGION_W[HOST]
b_w = sum(w[f] * M.INCIN_B[f] for f in FEEDS)
CRt = pbm['CR_tgt'] - M.RED_PCT['2050'] * b_w * (CIg - M.CI_BASE_INCIN) / 1000.0

best = None
for a in M.alpha_grid:
    if a > a_max:
        break
    P_elec = M.P_GRID_BASE + a * (pbm['P_PPA'] - M.P_GRID_BASE)
    CI_elec_g = CIg + a * (pbm['CI_PPA'] - CIg)
    UP_x, CR_x = M.transform_baseline(P_elec, CI_elec_g)
    dE = M.CR0 - CR_x
    CR_r = CR_x - incin_shift
    UP_net = UP_x + tau * CR_r
    sel = {f: M.feed_idx[f][np.argmax(UP_net[M.feed_idx[f]])] for f in FEEDS}
    wcr = sum(w[f] * CR_r[sel[f]] for f in FEEDS)
    wup = sum(w[f] * UP_net[sel[f]] for f in FEEDS)
    if wcr >= CRt and (best is None or wup > best[0]):
        best = (wup, sel, dE, a)
_, sel, dE, a_opt = best

rows = []
for f in FEEDS:
    j = sel[f]
    rows.append(dict(Feed=f, PD=PD_kg[j], FC=FC_kg[j] - incin_shift[j],
                     E=EG_kg[j] + dE[j],
                     CR=PD_kg[j] + FC_kg[j] - incin_shift[j] - EG_kg[j] - dE[j]))
pc = pd.DataFrame(rows)
oth = pc[pc.Feed != 'PVC'][['PD', 'FC', 'E', 'CR']].mean()
pc = pd.concat([pc, pd.DataFrame([dict(Feed='Others', **oth.to_dict())])],
               ignore_index=True)
pc.to_csv(os.path.join(BASE, 'fig7_panelC.csv'), index=False, float_format='%.4f')
print(f'panel C (host {HOST}, alpha* {a_opt:.2f}):')
print(pc.to_string(index=False))

pvc = pc[pc.Feed == 'PVC'].iloc[0]
gap = oth['CR'] - pvc['CR']
struct = oth['FC'] - pvc['FC']
print(f'\n  gap {gap:+.2f} = structural {struct:+.2f} (avoided incineration) '
      f'+ addressable {gap-struct:+.2f}')

dcr = pd.read_csv(os.path.join(BASE, 'pvc_bottleneck_dCR.csv'))
dcr = dcr[((dcr.Period == 2050) | (dcr.Period == '2050')) & (dcr.dCR_star > 0)]
print('  dCR* as a share of the addressable gap:')
for r in dcr.itertuples():
    print(f'    {r.Region:>14} {r.dCR_star:+.2f}  ->  {r.dCR_star/(gap-struct)*100:.0f}%')
dcr.assign(addressable=gap - struct,
           frac=dcr.dCR_star / (gap - struct)).to_csv(
    os.path.join(BASE, 'fig7_panelC_levers.csv'), index=False, float_format='%.4f')

print('\nDone. fig7_panelA.csv, fig7_panelB.csv, fig7_panelC.csv, fig7_panelC_levers.csv')
