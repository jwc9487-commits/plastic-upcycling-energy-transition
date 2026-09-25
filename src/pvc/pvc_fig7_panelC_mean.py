import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
# -*- coding: utf-8 -*-
"""Panel C recomputed as the mean over all eleven regions, per feed.

Replaces the hardcoded HOST = 'Korea' of pvc_fig7_data.py. Same per-region
procedure as before: at the 2050 fossil baseline, pick the alpha that maximizes
composition-weighted net profit subject to the carbon target; if no alpha
satisfies the target (China, Brazil, South Africa) fall back to alpha_max, which
is where wCR peaks. Per-feed selection is argmax UP_net within the feed, as in
every other panel.
"""
import sys, io, os
import numpy as np
import pandas as pd

BASE = config.INTERMEDIATE
import pvc_bottleneck as M

REGION_W, FEEDS, OTHERS = M.REGION_W, M.FEEDS, M.OTHERS
SCALE = 2e-8


def col(i):
    return pd.to_numeric(M.df.iloc[:, i], errors='coerce').fillna(0).values


PD_kg = np.abs(col(171)) * SCALE
FC_kg = col(170) * SCALE
EG_kg = col(43) * SCALE
assert np.abs(PD_kg + FC_kg - EG_kg - M.CR0).max() < 1e-6, 'CR identity broken'

pbm = M.PERIOD_BASE['2050']
regions = list(REGION_W.keys())
recs, meta = [], []

for region in regions:
    CIg = M.REGION_GRID_CI[region]['2050']
    a_max = M.REGION_ALPHA_MAX[region]['2050']
    tau = M.tau_for(region, '2050')
    incin_shift = M.INCIN_B_ARR * (CIg - M.CI_BASE_INCIN) / 1000.0
    w = REGION_W[region]
    b_w = sum(w[f] * M.INCIN_B[f] for f in FEEDS)
    CRt = pbm['CR_tgt'] - M.RED_PCT['2050'] * b_w * (CIg - M.CI_BASE_INCIN) / 1000.0

    best, fallback = None, None
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
        fallback = (a, sel, dE, wcr)          # last admissible alpha = alpha_max
        if wcr >= CRt and (best is None or wup > best[0]):
            best = (wup, sel, dE, a)

    if best is not None:
        _, sel, dE, a_opt = best
        feas = True
    else:
        a_opt, sel, dE, _ = fallback
        feas = False
    meta.append(dict(Region=region, alpha=a_opt, feasible=feas))

    for f in FEEDS:
        j = sel[f]
        pd_, fc_ = PD_kg[j], FC_kg[j] - incin_shift[j]
        e_ = EG_kg[j] + dE[j]
        recs.append(dict(Region=region, Feed=f, PD=pd_, FC=fc_, E=e_,
                         CR=pd_ + fc_ - e_))

per = pd.DataFrame(recs)
mt = pd.DataFrame(meta)
print(mt.to_string(index=False))
print(f"\nfeasible at 2050 baseline: {int(mt.feasible.sum())}/11 "
      f"(infeasible = {', '.join(mt[~mt.feasible].Region)})\n")

pc = per.groupby('Feed', sort=False)[['PD', 'FC', 'E', 'CR']].mean().reset_index()
pc['Feed'] = pd.Categorical(pc.Feed, FEEDS, ordered=True)
pc = pc.sort_values('Feed')
oth = per[per.Feed != 'PVC'].groupby('Region')[['PD', 'FC', 'E', 'CR']].mean().mean()
pc = pd.concat([pc, pd.DataFrame([dict(Feed='Others', **oth.to_dict())])],
               ignore_index=True)

print('panel C, mean over eleven regions:')
print(pc.to_string(index=False, float_format=lambda v: f'{v:6.4f}'))

pvc = pc[pc.Feed == 'PVC'].iloc[0]
gap = oth['CR'] - pvc['CR']
struct = oth['FC'] - pvc['FC']
addr = gap - struct
print(f'\ngap {gap:.4f} = structural {struct:.4f} + addressable {addr:.4f}')

# spread check, so the caption can say how little the choice of region matters
sp = per.groupby('Region').apply(
    lambda g: (g[g.Feed != 'PVC'].CR.mean() - g[g.Feed == 'PVC'].CR.iloc[0]),
    include_groups=False)
print(f'gap across regions: {sp.min():.4f} to {sp.max():.4f}')

# required improvement as a share of the addressable gap
dcr = pd.read_csv(os.path.join(BASE, 'pvc_bottleneck_dCR.csv'))
dcr = dcr[((dcr.Period == 2050) | (dcr.Period == '2050')) & (dcr.dCR_star > 0)]
lev = dcr.assign(addressable=addr, frac=dcr.dCR_star / addr)
print('\nrequired improvement as a share of the addressable gap:')
for r in lev.itertuples():
    print(f'  {r.Region:>14} {r.dCR_star:+.4f} -> {r.frac*100:.0f}%')

pc.to_csv(os.path.join(BASE, 'fig7_panelC_mean.csv'), index=False,
          float_format='%.4f')
lev.to_csv(os.path.join(BASE, 'fig7_panelC_levers_mean.csv'), index=False,
           float_format='%.4f')
per.to_csv(os.path.join(BASE, 'fig7_panelC_byregion.csv'), index=False,
           float_format='%.4f')
print('\nwrote fig7_panelC_mean.csv, fig7_panelC_levers_mean.csv, '
      'fig7_panelC_byregion.csv')
