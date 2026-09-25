import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
# -*- coding: utf-8 -*-
"""Table S7.1 robustness on the canonical (INCIN-regionalized) model.

Replaces robustness_SN3.py, which used the superseded model without incineration
credit regionalization (no INCIN_B). The model core is executed verbatim from the
canonical script m3_step5_RC_world_map_periods_INCIN.py (everything before the
time-series loop), so parameters and transforms cannot drift from Fig. 7.

Scaling rule: the regionalized 2050 target is multiplied by cr_mult (x0.8/1.0/1.2),
i.e. the policy reduction fraction is scaled while the regional incineration
baseline is kept.
"""
import sys, io, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CANON = os.path.join(HERE, 'm3_step5_RC_world_map_periods_INCIN.py')
src = io.open(CANON, encoding='utf-8').read()
core = src[:src.index("print('Computing time series per region")]
core = core.replace('sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=\'utf-8\')', '')
g = {'__name__': 'canon_core', '__file__': CANON}
exec(compile(core, CANON, 'exec'), g)
assert 'INCIN_B' in g and g['CI_BASE_INCIN'] == 369.0

FEEDS, alpha_grid = g['FEEDS'], g['alpha_grid']

def evaluate(region, period, scenario, cr_mult=1.0, incentive='tauCR'):
    w = g['REGION_W'][region]; pb = g['PERIOD_BASE'][period]
    tau = g['tau_for'](region, period); CIg = g['REGION_GRID_CI'][region][period]
    CIp, Pp, H2cx = pb['CI_PPA'], pb['P_PPA'], pb['H2cx']
    a_max = g['REGION_ALPHA_MAX'][region][period]
    incin_shift = g['INCIN_B_ARR'] * (CIg - g['CI_BASE_INCIN']) / 1000.0
    b_w = sum(w[f] * g['INCIN_B'][f] for f in FEEDS)
    CRt = (pb['CR_tgt'] - g['RED_PCT'][period] * b_w * (CIg - g['CI_BASE_INCIN']) / 1000.0) * cr_mult
    wUP = np.zeros_like(alpha_grid); wCR = np.zeros_like(alpha_grid)
    for i, a in enumerate(alpha_grid):
        P_elec = g['P_GRID_BASE'] + a * (Pp - g['P_GRID_BASE'])
        CI_elec_g = CIg + a * (CIp - CIg)
        if scenario == 'elec':
            UP_x, CR_x = g['transform_electrification'](P_elec, CI_elec_g, H2cx)
        else:
            UP_x, CR_x = g['transform_baseline'](P_elec, CI_elec_g)
        CR_x = CR_x - incin_shift
        if incentive == 'tauCR':   UP_net = UP_x + tau * CR_x
        elif incentive == 'neutral': UP_net = UP_x + tau * (CR_x - 1.0)
        else:                      UP_net = UP_x
        idx_best = {f: g['feed_idx'][f][np.argmax(UP_net[g['feed_idx'][f]])] for f in FEEDS}
        wUP[i] = sum(w[f] * UP_x[idx_best[f]] for f in FEEDS)
        wCR[i] = sum(w[f] * CR_x[idx_best[f]] for f in FEEDS)
    feas = (wCR >= CRt) & (alpha_grid <= a_max)
    if not feas.any(): return np.nan
    return wUP[np.argmax(np.where(feas, wUP, -np.inf))]

if __name__ == '__main__':
    # sanity: nominal case must reproduce the canonical Fig. 7 table (wUP_net there includes tau*CR)
    ref = pd.read_csv(os.path.join(config.INTERMEDIATE, 'm3_step5_RC_world_periods_INCIN.csv'))
    ref50 = ref[(ref.Period == 2050) | (ref.Period == '2050')]
    for sc, key in [('Baseline', 'base'), ('Electrified', 'elec')]:
        n_ref = ref50[ref50.Scenario == sc].wUP_net.isna().sum()
        n_new = sum(np.isnan(evaluate(r, '2050', key)) for r in g['REGION_W'])
        print(f'check 2050 {sc}: canonical infeasible={n_ref}  this script={n_new}')
        assert n_ref == n_new
    LABEL = {'tauCR': 'Credit-and-penalty (UP + tau*UCR)', 'neutral': 'Neutral-baseline (UP + tau*(UCR-1))',
             'none': 'No incentive'}
    rows = []
    for inc in ['tauCR', 'neutral', 'none']:
        for m in [0.8, 1.0, 1.2]:
            infB = [r for r in g['REGION_W'] if np.isnan(evaluate(r, '2050', 'base', m, inc))]
            infE = [r for r in g['REGION_W'] if np.isnan(evaluate(r, '2050', 'elec', m, inc))]
            rows.append({'Incentive formulation': LABEL[inc], 'Target multiplier': m,
                         'Baseline infeasible (of 11)': len(infB), 'Electrified infeasible (of 11)': len(infE),
                         'Baseline infeasible regions': '; '.join(infB), 'Electrified infeasible regions': '; '.join(infE)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(config.INTERMEDIATE, 'SN3_robustness_INCIN.csv'), index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80)
    print(out.to_string(index=False))
    for r in ['South Africa', 'Indonesia']:
        b = evaluate(r, '2030', 'base'); e = evaluate(r, '2030', 'elec')
        print(f'2030 {r}: base={b:+.3f} elec={"infeasible" if np.isnan(e) else f"{e:+.3f}"}')
