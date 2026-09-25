import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
"""PVC bottleneck — two follow-up checks on the stringency sensitivity.

(i)  Is the IDENTITY of the first-failing regions stable, even though the count is not?
(ii) Is the 1.2x target physically attainable at all? If even a PVC-free stream cannot
     reach it, that stringency lies outside the meaningful range and the collapse to
     11/11 says nothing about PVC.
"""
import sys, os
import numpy as np
import pandas as pd

BASE = config.INTERMEDIATE
import pvc_bottleneck as M

REGION_W, FEEDS, OTHERS = M.REGION_W, M.FEEDS, M.OTHERS
evaluate = M.evaluate

print('\n\n' + '#' * 78)
print('# STRINGENCY FOLLOW-UP')
print('#' * 78)

print('\n' + '=' * 78)
print('(i) IDENTITY OF FAILING REGIONS where the target is binding')
print('=' * 78)
for mult, period, sc in [(1.0, '2050', 'base'), (1.2, '2040', 'base'),
                         (1.2, '2040', 'elec'), (1.2, '2050', 'elec')]:
    line = f'  target x{mult}, {period}, {"fossil baseline" if sc=="base" else "electrified"}'
    print(f'\n{line}')
    for mode, key in [('upcycle', 'A all five'), ('incinerate', 'B PVC->incin'),
                      ('absent', 'C PVC absent')]:
        bad = [r for r in REGION_W
               if np.isnan(evaluate(r, period, sc, pvc_mode=mode, crt_mult=mult)[0])]
        print(f'    {key:<14} {len(bad):>2}  {", ".join(bad) if bad else "-"}')

print('\n' + '=' * 78)
print('(ii) IS THE TARGET ATTAINABLE? ceiling = best achievable wCR with a PVC-free')
print('     stream at the region\'s own alpha_max, versus the target at each stringency')
print('=' * 78)


def ceiling_wCR(region, period, scenario='base'):
    """max achievable composition-weighted CR with PVC removed upstream."""
    w0 = REGION_W[region]
    s = sum(w0[f] for f in OTHERS)
    w = {f: w0[f] / s for f in OTHERS}
    pb = M.PERIOD_BASE[period]
    CIg = M.REGION_GRID_CI[region][period]
    CIp, Pp, H2cx = pb['CI_PPA'], pb['P_PPA'], pb['H2cx']
    a_max = M.REGION_ALPHA_MAX[region][period]
    incin_shift = M.INCIN_B_ARR * (CIg - M.CI_BASE_INCIN) / 1000.0
    b_w = sum(w[f] * M.INCIN_B[f] for f in OTHERS)
    CRt_nom = pb['CR_tgt'] - M.RED_PCT[period] * b_w * (CIg - M.CI_BASE_INCIN) / 1000.0
    best = -np.inf
    for a in M.alpha_grid:
        if a > a_max:
            break
        P_elec = M.P_GRID_BASE + a * (Pp - M.P_GRID_BASE)
        CI_elec_g = CIg + a * (CIp - CIg)
        if scenario == 'elec':
            _, CR_x = M.transform_electrification(P_elec, CI_elec_g, H2cx)
        else:
            _, CR_x = M.transform_baseline(P_elec, CI_elec_g)
        CR_r = CR_x - incin_shift
        # ceiling: pick each feed's max-CR pathway, ignoring profit
        val = sum(w[f] * CR_r[M.feed_idx[f]].max() for f in OTHERS)
        best = max(best, val)
    return best, CRt_nom


print(f'\n  2050, fossil baseline')
print(f'{"Region":>14} {"ceiling":>9} {"tgt x0.8":>9} {"tgt x1.0":>9} {"tgt x1.2":>9}  {"x1.2 reachable?":>16}')
rows = []
n_unreach = 0
for region in REGION_W:
    ceil, CRt = ceiling_wCR(region, '2050', 'base')
    reach = ceil >= 1.2 * CRt
    n_unreach += (not reach)
    rows.append(dict(Region=region, ceiling=ceil, CRt_nominal=CRt,
                     reach_08=ceil >= 0.8 * CRt, reach_10=ceil >= CRt,
                     reach_12=reach))
    print(f'{region:>14} {ceil:>9.2f} {0.8*CRt:>9.2f} {CRt:>9.2f} {1.2*CRt:>9.2f}  '
          f'{("yes" if reach else "NO"):>16}')
pd.DataFrame(rows).to_csv(os.path.join(BASE, 'pvc_bottleneck_ceiling.csv'),
                          index=False, float_format='%.4f')
print(f'\n  {n_unreach} of 11 regions cannot reach the 1.2x 2050 target even with a')
print('  PVC-free stream and every feed pushed to its maximum-CR pathway, ignoring cost.')
print('  Where that holds, the 1.2x collapse is a statement about target calibration,')
print('  not about PVC.')

print('\nDone. CSV: pvc_bottleneck_ceiling.csv')
