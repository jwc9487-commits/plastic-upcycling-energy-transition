import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
"""PVC bottleneck — part 3: per-feed carbon reduction at the optimum.

Puts the required improvement dCR* on a relative footing: how far is PVC from the
other four feeds, and by what fraction must it improve?
"""
import sys, os
import numpy as np
import pandas as pd

BASE = config.INTERMEDIATE
import pvc_bottleneck as M

FEEDS, PERIODS, REGION_W = M.FEEDS, M.PERIODS, M.REGION_W


def per_feed_at_opt(region, period, scenario='base'):
    """CR and UP_net of each feed's profit-optimal pathway at the chosen alpha."""
    w = REGION_W[region]
    pb = M.PERIOD_BASE[period]
    tau = M.tau_for(region, period)
    CIg = M.REGION_GRID_CI[region][period]
    CIp, Pp, H2cx, CRt0 = pb['CI_PPA'], pb['P_PPA'], pb['H2cx'], pb['CR_tgt']
    a_max = M.REGION_ALPHA_MAX[region][period]
    incin_shift = M.INCIN_B_ARR * (CIg - M.CI_BASE_INCIN) / 1000.0
    b_w = sum(w[f] * M.INCIN_B[f] for f in FEEDS)
    CRt = CRt0 - M.RED_PCT[period] * b_w * (CIg - M.CI_BASE_INCIN) / 1000.0

    best = None
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
        cr_f, up_f = {}, {}
        for f in FEEDS:
            idxs = M.feed_idx[f]
            j = idxs[np.argmax(UP_net[idxs])]
            cr_f[f] = CR_x[j]; up_f[f] = UP_net[j]
        wCR = sum(w[f] * cr_f[f] for f in FEEDS)
        wUP = sum(w[f] * up_f[f] for f in FEEDS)
        if wCR >= CRt and (best is None or wUP > best[0]):
            best = (wUP, wCR, cr_f, up_f, a)
    if best is None:                      # infeasible: report the alpha that maximizes wCR
        bestc = None
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
            cr_f, up_f = {}, {}
            for f in FEEDS:
                idxs = M.feed_idx[f]
                j = idxs[np.argmax(UP_net[idxs])]
                cr_f[f] = CR_x[j]; up_f[f] = UP_net[j]
            wCR = sum(w[f] * cr_f[f] for f in FEEDS)
            if bestc is None or wCR > bestc[1]:
                bestc = (sum(w[f]*up_f[f] for f in FEEDS), wCR, cr_f, up_f, a)
        best = bestc
    return best, CRt


print('\n' + '=' * 84)
print('11. PER-FEED CARBON REDUCTION AT THE OPTIMUM — 2050 fossil baseline')
print('    (infeasible regions evaluated at the alpha that maximizes wCR)')
print('=' * 84)
hdr = f'{"Region":>14} ' + ' '.join(f'{f:>7}' for f in FEEDS) + f' | {"wCR":>7} {"CRt":>7} {"gap":>7}'
print(hdr)
rows = []
for region in REGION_W:
    (wUP, wCR, cr_f, up_f, a), CRt = per_feed_at_opt(region, '2050', 'base')
    print(f'{region:>14} ' + ' '.join(f'{cr_f[f]:>7.2f}' for f in FEEDS)
          + f' | {wCR:>7.2f} {CRt:>7.2f} {wCR-CRt:>+7.2f}')
    rows.append(dict(Region=region, Period='2050', alpha=a, wCR=wCR, CRt=CRt,
                     gap=wCR - CRt, **{f'CR_{f}': cr_f[f] for f in FEEDS},
                     **{f'UPnet_{f}': up_f[f] for f in FEEDS}))
pd.DataFrame(rows).to_csv(os.path.join(BASE, 'pvc_bottleneck_perfeed.csv'),
                          index=False, float_format='%.4f')

print('\n' + '=' * 84)
print('12. REQUIRED PVC IMPROVEMENT IN RELATIVE TERMS (2050 fossil baseline)')
print('=' * 84)
d = pd.read_csv(os.path.join(BASE, 'pvc_bottleneck_dCR.csv'))
d = d[(d.Period == 2050) | (d.Period == '2050')]
pf = pd.DataFrame(rows).set_index('Region')
print(f'{"Region":>14} {"w_PVC":>7} {"CR_PVC now":>11} {"others avg":>11} '
      f'{"dCR*":>7} {"as % of now":>12} {"still below others?":>20}')
for r in d.itertuples():
    if pd.isna(r.dCR_star) or r.dCR_star == 0:
        continue
    now = pf.loc[r.Region, 'CR_PVC']
    oth = np.mean([pf.loc[r.Region, f'CR_{f}'] for f in FEEDS if f != 'PVC'])
    print(f'{r.Region:>14} {REGION_W[r.Region]["PVC"]*100:>6.1f}% {now:>11.2f} {oth:>11.2f} '
          f'{r.dCR_star:>+7.2f} {r.dCR_star/now*100:>11.0f}% '
          f'{("yes, still " + f"{oth-(now+r.dCR_star):.2f} short"):>20}')

print('\nDone (part 3). CSV: pvc_bottleneck_perfeed.csv')
