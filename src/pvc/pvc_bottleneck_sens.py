import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
"""PVC bottleneck — robustness to the environmental target.

The headline counts (3 / 7 / 0 infeasible regions in 2050) are all measured against
the nominal CR_target trajectory, which is an assumption rather than an observation.
S-N3 already showed the count is sensitive to target stringency, so every headline
number is recomputed at 0.8x / 1.0x / 1.2x of the regionalized target.

The question is which claims survive:
  - ORDERING   C better than A better than B   (expected robust: driven by CR, not target)
  - COUNTS     how many regions fail           (expected sensitive)
  - LEVERS     dCR* and w*                     (expected to move with the target)
"""
import sys, os
import numpy as np
import pandas as pd

BASE = config.INTERMEDIATE
import pvc_bottleneck as M

REGION_W, PERIODS, OTHERS = M.REGION_W, M.PERIODS, M.OTHERS
evaluate, reweight = M.evaluate, M.reweight
MULTS = [0.8, 1.0, 1.2]

print('\n\n' + '#' * 78)
print('# ROBUSTNESS TO ENVIRONMENTAL TARGET STRINGENCY')
print('#' * 78)

# ══ A. feasibility counts, all periods, both scenarios ═══════════════
print('\n' + '=' * 78)
print('A. INFEASIBLE REGIONS (out of 11) by configuration and target stringency')
print('=' * 78)
rows = []
for mult in MULTS:
    for sc, tag in [('base', 'Baseline'), ('elec', 'Electrified')]:
        for p in PERIODS:
            cnt = {}
            for mode, key in [('upcycle', 'A'), ('incinerate', 'B'), ('absent', 'C')]:
                n = sum(np.isnan(evaluate(r, p, sc, pvc_mode=mode, crt_mult=mult)[0])
                        for r in REGION_W)
                cnt[key] = n
            rows.append(dict(crt_mult=mult, Scenario=tag, Period=p, **cnt))
sens = pd.DataFrame(rows)
sens.to_csv(os.path.join(BASE, 'pvc_bottleneck_sens_counts.csv'), index=False)

for sc in ['Baseline', 'Electrified']:
    print(f'\n  {sc} scenario   (A = all five upcycled | B = PVC incinerated | C = PVC absent)')
    print(f'    {"target":>8} ' + ' '.join(f'{p:>14}' for p in PERIODS))
    for mult in MULTS:
        cells = []
        for p in PERIODS:
            r = sens[(sens.crt_mult == mult) & (sens.Scenario == sc) & (sens.Period == p)].iloc[0]
            cells.append(f'A{r.A} B{r.B} C{r.C}'.rjust(14))
        star = ' *' if mult == 1.0 else '  '
        print(f'    {mult:>6.1f}x{star}' + ' '.join(cells))

# ══ B. does the ordering survive? ════════════════════════════════════
print('\n' + '=' * 78)
print('B. ORDERING CHECK — is B always at least as bad as A, and C at least as good?')
print('=' * 78)
viol_BA = sens[sens.B < sens.A]
viol_CA = sens[sens.C > sens.A]
print(f'  cases where diverting PVC to incineration HELPED (B < A): {len(viol_BA)} of {len(sens)}')
print(f'  cases where removing PVC upstream HURT      (C > A): {len(viol_CA)} of {len(sens)}')
if len(viol_BA):
    print(viol_BA.to_string(index=False))
if len(viol_CA):
    print(viol_CA.to_string(index=False))
print('  -> ordering C <= A <= B holds everywhere' if not len(viol_BA) and not len(viol_CA)
      else '  -> ORDERING VIOLATED, review')

# ══ C. 2050 baseline detail — which regions, at each stringency ═══════
print('\n' + '=' * 78)
print('C. 2050 FOSSIL BASELINE — named failing regions at each stringency')
print('=' * 78)
for mult in MULTS:
    print(f'\n  target x{mult}')
    for mode, key in [('upcycle', 'A all five upcycled'),
                      ('incinerate', 'B PVC to incineration'),
                      ('absent', 'C PVC absent upstream')]:
        bad = [r for r in REGION_W
               if np.isnan(evaluate(r, '2050', 'base', pvc_mode=mode, crt_mult=mult)[0])]
        print(f'    {key:<24} {len(bad):>2}  {", ".join(bad) if bad else "-"}')

# ══ D. levers under each stringency ══════════════════════════════════
print('\n' + '=' * 78)
print('D. LEVERS at 2050 fossil baseline under each stringency')
print('=' * 78)

def min_dCR(region, period, mult, hi=8.0, tol=1e-3):
    if not np.isnan(evaluate(region, period, 'base', crt_mult=mult)[0]):
        return 0.0
    if np.isnan(evaluate(region, period, 'base', dCR_pvc=hi, crt_mult=mult)[0]):
        return np.nan
    lo = 0.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if np.isnan(evaluate(region, period, 'base', dCR_pvc=mid, crt_mult=mult)[0]):
            lo = mid
        else:
            hi = mid
    return hi

def crit_share(region, period, mult, tol=1e-4):
    f = lambda x: not np.isnan(evaluate(region, period, 'base',
                                        w=reweight(region, x), crt_mult=mult)[0])
    if f(REGION_W[region]['PVC']):
        return np.nan
    if not f(0.0):
        return -1.0
    lo, hi = 0.0, REGION_W[region]['PVC']
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if f(mid):
            lo = mid
        else:
            hi = mid
    return lo

lrows = []
print(f'{"Region":>14} {"w_PVC":>7} | ' +
      ' | '.join(f'{"x"+str(m)+" dCR*/w*":>18}' for m in MULTS))
for region in REGION_W:
    cells = []
    for mult in MULTS:
        d = min_dCR(region, '2050', mult)
        c = crit_share(region, '2050', mult)
        lrows.append(dict(Region=region, crt_mult=mult, dCR_star=d, w_crit=c))
        if d == 0.0:
            cells.append(f'{"feasible":>18}')
        elif np.isnan(d):
            cells.append(f'{">8.0 / -":>18}')
        else:
            cw = 'none' if (isinstance(c, float) and c < 0) else (
                 f'{c*100:.1f}%' if not np.isnan(c) else '-')
            cells.append(f'{d:+.2f} / {cw:>8}'.rjust(18))
    print(f'{region:>14} {REGION_W[region]["PVC"]*100:>6.1f}% | ' + ' | '.join(cells))
pd.DataFrame(lrows).to_csv(os.path.join(BASE, 'pvc_bottleneck_sens_levers.csv'),
                           index=False, float_format='%.4f')

print('\nDone. CSVs: pvc_bottleneck_sens_counts.csv, pvc_bottleneck_sens_levers.csv')
