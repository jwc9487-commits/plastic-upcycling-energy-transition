import sys, io
if (getattr(sys.stdout, 'encoding', '') or '').lower().replace('-', '') != 'utf8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
"""PVC bottleneck analysis — is PVC merely a hard feedstock, or the system's binding constraint?

Model core is copied verbatim from `src/regional/m3_step5_RC_world_map_periods_INCIN.py`
(the script behind Fig. 12) so that the baseline reproduces exactly.

Four system configurations are compared for every region x period x scenario:

  A. 'upcycle'    - all five feeds upcycled                      (as published)
  B. 'incinerate' - PVC diverted to incineration, other four upcycled
                    -> PVC contributes CR = 0 and UP = UP_incin(PVC)
  C. 'absent'     - PVC removed from the waste stream upstream
                    (material substitution / design-out); weights renormalized
                    over the remaining four feeds
  D. 'improved'   - PVC upcycling improved by a uniform additive dCR

Outputs
  1. baseline reproduction check against m3_step5_RC_world_periods_INCIN.csv
  2. critical PVC share w* at which each region/period flips feasible
  3. minimum PVC carbon improvement dCR* that restores feasibility (lever: downstream)
  4. minimum PVC share reduction that restores feasibility (lever: upstream)
  5. effect of each configuration on wUP, wCR and on the year viability arrives

Counterfactual assumptions are declared explicitly at CF_NOTES below and carried
through a sensitivity band (see UP_INCIN_MODE).
"""
import sys, io, os
os.environ['PYTHONUTF8'] = '1'
import numpy as np
import pandas as pd

BASE = config.INTERMEDIATE
from ref_elec_coupling import ref_price, ref_ci

MASS = 50e6
FEEDS = ['PE', 'PET', 'PP', 'PS', 'PVC']
OTHERS = ['PE', 'PET', 'PP', 'PS']

# ── incineration benchmark ────────────────────────────────────────────
INCIN_B = {'PE': 2.221266, 'PET': 1.056216, 'PP': 2.105206, 'PS': 1.793923, 'PVC': 0.687119}
CI_BASE_INCIN = 369.0
RED_PCT = {'2026': 0.186, '2030': 0.502, '2040': 0.703, '2050': 0.900}

# Unit profit of incineration. -0.42 $/kg is the published benchmark, derived for a
# mixed-plastic-waste stream. Incineration revenue includes recovered electricity
# (INCIN_B kWh/kg), which differs strongly by resin, so the per-feed value is
# adjusted off the MPW reference composition. PVC recovers the least electricity
# (0.687 vs 1.82 kWh/kg for the reference mix), so its incineration profit is worse.
UP_INCIN_MPW = -0.42
UP_INCIN_MODE = 'feed_adjusted'   # 'feed_adjusted' (base) | 'uniform' (sensitivity)

CF_NOTES = """
Counterfactual assumptions (declared, per TEA/LCA convention 3.3):
  * The environmental target applies to the WHOLE regional plastic waste stream.
    Diverting a resin to incineration therefore does not remove it from the ledger;
    it enters at CR = 0 (incineration is the benchmark, so it abates nothing).
  * 'absent' represents upstream elimination of PVC (substitution / design-out).
    The stream itself changes, so both the composition weights AND the mixed-waste
    incineration baseline b_w are recomputed over the remaining four feeds.
  * 'improved' applies a uniform additive dCR to every PVC pathway. It is agnostic
    about which technology delivers the gain (dechlorination, chlorine recovery,
    higher-value product slate).
  * Carbon incentive is UP_net = UP + tau*CR throughout, so incinerated PVC receives
    neither credit nor penalty.
"""

REGION_W = {
    'Global/EU': {'PE': 0.406, 'PP': 0.264, 'PET': 0.105, 'PVC': 0.137, 'PS': 0.088},
    'U.S.':      {'PE': 0.562, 'PP': 0.200, 'PET': 0.146, 'PVC': 0.017, 'PS': 0.075},
    'China':     {'PE': 0.346, 'PP': 0.250, 'PET': 0.139, 'PVC': 0.204, 'PS': 0.061},
    'Japan':     {'PE': 0.418, 'PP': 0.278, 'PET': 0.175, 'PVC': 0.060, 'PS': 0.069},
    'Korea':     {'PE': 0.449, 'PP': 0.263, 'PET': 0.139, 'PVC': 0.090, 'PS': 0.059},
    'India':     {'PE': 0.709, 'PP': 0.105, 'PET': 0.092, 'PVC': 0.044, 'PS': 0.051},
    'Indonesia': {'PE': 0.358, 'PP': 0.326, 'PET': 0.126, 'PVC': 0.116, 'PS': 0.074},
    'Canada':    {'PE': 0.470, 'PP': 0.210, 'PET': 0.170, 'PVC': 0.080, 'PS': 0.070},
    'Brazil':    {'PE': 0.420, 'PP': 0.200, 'PET': 0.130, 'PVC': 0.180, 'PS': 0.070},
    'Australia': {'PE': 0.390, 'PP': 0.270, 'PET': 0.160, 'PVC': 0.080, 'PS': 0.100},
    'South Africa': {'PE': 0.420, 'PP': 0.220, 'PET': 0.160, 'PVC': 0.120, 'PS': 0.080},
}

PERIODS = ['2026', '2030', '2040', '2050']
PERIOD_YEAR = {'2026': 2026, '2030': 2030, '2040': 2040, '2050': 2050}

PERIOD_BASE = {
    '2026': dict(P_PPA=0.040, CI_PPA=30, H2cx=0.7, CR_tgt=0.50),
    '2030': dict(P_PPA=0.030, CI_PPA=20, H2cx=0.5, CR_tgt=1.35),
    '2040': dict(P_PPA=0.025, CI_PPA=15, H2cx=0.4, CR_tgt=1.89),
    '2050': dict(P_PPA=0.020, CI_PPA=10, H2cx=0.3, CR_tgt=2.42),
}

REGION_TAU_2026 = {
    'Global/EU': 0.080, 'U.S.': 0.000, 'China': 0.010, 'Japan': 0.040,
    'Korea': 0.030, 'India': 0.000, 'Brazil': 0.000, 'Australia': 0.000,
    'Indonesia': 0.000, 'Canada': 0.065, 'South Africa': 0.012,
}
TAU_BY_PERIOD = {'2030': 0.10, '2040': 0.15, '2050': 0.20}

def tau_for(region, period):
    return REGION_TAU_2026[region] if period == '2026' else TAU_BY_PERIOD[period]

REGION_GRID_CI = {
    'Global/EU':    {'2026': 200, '2030': 130, '2040':  80, '2050':  30},
    'U.S.':         {'2026': 230, '2030': 200, '2040': 125, '2050':  50},
    'China':        {'2026': 510, '2030': 460, '2040': 285, '2050': 110},
    'Japan':        {'2026': 290, '2030': 250, '2040': 150, '2050':  50},
    'Korea':        {'2026': 330, '2030': 280, '2040': 175, '2050':  70},
    'India':        {'2026': 510, '2030': 460, '2040': 295, '2050': 130},
    'Brazil':       {'2026':  90, '2030':  60, '2040':  40, '2050':  20},
    'Australia':    {'2026': 580, '2030': 420, '2040': 200, '2050':  40},
    'Indonesia':    {'2026': 720, '2030': 600, '2040': 350, '2050':  90},
    'Canada':       {'2026': 130, '2030':  90, '2040':  50, '2050':  20},
    'South Africa': {'2026': 870, '2030': 720, '2040': 380, '2050': 110},
}

REGION_ALPHA_MAX = {
    'Global/EU':    {'2026': 0.45, '2030': 0.65, '2040': 0.85, '2050': 0.95},
    'U.S.':         {'2026': 0.30, '2030': 0.50, '2040': 0.75, '2050': 0.92},
    'China':        {'2026': 0.35, '2030': 0.55, '2040': 0.80, '2050': 0.95},
    'Japan':        {'2026': 0.25, '2030': 0.45, '2040': 0.70, '2050': 0.90},
    'Korea':        {'2026': 0.15, '2030': 0.35, '2040': 0.65, '2050': 0.88},
    'India':        {'2026': 0.25, '2030': 0.45, '2040': 0.70, '2050': 0.90},
    'Brazil':       {'2026': 0.85, '2030': 0.92, '2040': 0.96, '2050': 0.99},
    'Australia':    {'2026': 0.35, '2030': 0.60, '2040': 0.85, '2050': 0.95},
    'Indonesia':    {'2026': 0.18, '2030': 0.30, '2040': 0.55, '2050': 0.85},
    'Canada':       {'2026': 0.65, '2030': 0.80, '2040': 0.90, '2050': 0.97},
    'South Africa': {'2026': 0.12, '2030': 0.25, '2040': 0.55, '2050': 0.85},
}

P_GRID_BASE = 0.080
P_ELEC_BASE, CI_ELEC_BASE_g = 0.08, 369.0
P_FH_BASE,  CI_FH_BASE_g = 0.03, 338.4
P_LP_BASE,  CI_LP_BASE_g = 0.00684, 190.08
P_HP_BASE,  CI_HP_BASE_g = 0.009, 190.08
P_H2_BASE,  CI_H2_BASE_kg = 2.0, 7.0
ETA_HEATER, ETA_BOILER = 0.95, 0.99

# reference MPW composition for the -0.42 benchmark = PlasticsEurope (Global/EU)
_b_ref = sum(REGION_W['Global/EU'][f] * INCIN_B[f] for f in FEEDS)
if UP_INCIN_MODE == 'feed_adjusted':
    UP_INCIN = {f: UP_INCIN_MPW + (INCIN_B[f] - _b_ref) * P_GRID_BASE for f in FEEDS}
else:
    UP_INCIN = {f: UP_INCIN_MPW for f in FEEDS}

print('Loading pathway data...')
df = pd.read_excel(config.MASTER_XLSX,
                   sheet_name='All pathways', header=3)
df['__feed'] = df.iloc[:, 0].astype(str).str.strip()
df = df[df['__feed'].isin(FEEDS)].reset_index(drop=True)

up_col = next(c for c in df.columns if isinstance(c, str) and c.lower().startswith('unit profit'))
cr_col = next(c for c in df.columns if isinstance(c, str) and c.lower().startswith('unit co2 reduc'))
UP0 = pd.to_numeric(df[up_col], errors='coerce').fillna(0).values
CR0 = pd.to_numeric(df[cr_col], errors='coerce').fillna(0).values
feed_arr = df['__feed'].values
INCIN_B_ARR = np.array([INCIN_B[f] for f in feed_arr])
IS_PVC = (feed_arr == 'PVC')

def coln(c): return pd.to_numeric(df.iloc[:, c], errors='coerce').fillna(0).values
ELEC_M = coln(7); FH_M = coln(8)
REF_M = np.array([coln(10 + k) for k in range(5)])
LP_M, HP_M = coln(15), coln(16); H2_M = coln(20)
ELEC_E = coln(31); FH_E = coln(32)
REF_E = np.array([coln(33 + k) for k in range(5)])
LP_E, HP_E = coln(38), coln(39); H2_E = coln(42)

P_REF_BASE = np.array([ref_price(k + 1, P_ELEC_BASE) for k in range(5)])
CI_REF_BASE_g = np.array([ref_ci(k + 1, CI_ELEC_BASE_g) for k in range(5)])

def transform_baseline(P_elec, CI_elec_g):
    r_e_c, r_e_ci = P_elec / P_ELEC_BASE, CI_elec_g / CI_ELEC_BASE_g
    r_ref_c = np.array([ref_price(k + 1, P_elec) / P_REF_BASE[k] for k in range(5)])
    r_ref_ci = np.array([ref_ci(k + 1, CI_elec_g) / CI_REF_BASE_g[k] for k in range(5)])
    new_ELEC_M = ELEC_M * r_e_c; new_ELEC_E = ELEC_E * r_e_ci
    new_REF_M_sum = REF_M.T @ r_ref_c; new_REF_E_sum = REF_E.T @ r_ref_ci
    dC_M = (new_ELEC_M - ELEC_M) + (new_REF_M_sum - REF_M.sum(axis=0))
    dE_kg = (new_ELEC_E - ELEC_E) + (new_REF_E_sum - REF_E.sum(axis=0))
    return UP0 - dC_M * 1e6 / MASS, CR0 - dE_kg / MASS

def transform_electrification(P_elec, CI_elec_g, h2_capex):
    r_e_c, r_e_ci = P_elec / P_ELEC_BASE, CI_elec_g / CI_ELEC_BASE_g
    r_ref_c = np.array([ref_price(k + 1, P_elec) / P_REF_BASE[k] for k in range(5)])
    r_ref_ci = np.array([ref_ci(k + 1, CI_elec_g) / CI_REF_BASE_g[k] for k in range(5)])
    new_ELEC_M = ELEC_M * r_e_c; new_ELEC_E = ELEC_E * r_e_ci
    new_REF_M_sum = REF_M.T @ r_ref_c; new_REF_E_sum = REF_E.T @ r_ref_ci
    new_FH_M = FH_M * ((P_elec / ETA_HEATER) / P_FH_BASE)
    new_FH_E = FH_E * ((CI_elec_g / ETA_HEATER) / CI_FH_BASE_g)
    new_LP_M = LP_M * ((P_elec / ETA_BOILER) / P_LP_BASE)
    new_LP_E = LP_E * ((CI_elec_g / ETA_BOILER) / CI_LP_BASE_g)
    new_HP_M = HP_M * ((P_elec / ETA_BOILER) / P_HP_BASE)
    new_HP_E = HP_E * ((CI_elec_g / ETA_BOILER) / CI_HP_BASE_g)
    new_H2_M = H2_M * ((53.0 * P_elec + h2_capex) / P_H2_BASE)
    new_H2_E = H2_E * ((53.0 * CI_elec_g / 1000.0) / CI_H2_BASE_kg)
    dC_M = ((new_ELEC_M - ELEC_M) + (new_REF_M_sum - REF_M.sum(axis=0))
            + (new_FH_M - FH_M) + (new_LP_M - LP_M)
            + (new_HP_M - HP_M) + (new_H2_M - H2_M))
    dE_kg = ((new_ELEC_E - ELEC_E) + (new_REF_E_sum - REF_E.sum(axis=0))
             + (new_FH_E - FH_E) + (new_LP_E - LP_E)
             + (new_HP_E - HP_E) + (new_H2_E - H2_E))
    return UP0 - dC_M * 1e6 / MASS, CR0 - dE_kg / MASS

feed_idx = {f: np.where(feed_arr == f)[0] for f in FEEDS}
alpha_grid = np.linspace(0, 1, 51)


def evaluate(region, period, scenario, w=None, pvc_mode='upcycle', dCR_pvc=0.0,
             crt_mult=1.0):
    """Return (wUP_net, wCR, CR_target) at the profit-maximizing feasible alpha.

    pvc_mode: 'upcycle' | 'incinerate' | 'absent'
    dCR_pvc : additive carbon-reduction improvement applied to every PVC pathway
    crt_mult: stringency multiplier on the regionalized environmental target
              (1.0 = nominal; the S-N3 robustness band is 0.8 / 1.2)
    """
    w = dict(REGION_W[region]) if w is None else dict(w)
    if pvc_mode == 'absent':
        s = sum(w[f] for f in OTHERS)
        w = {f: w[f] / s for f in OTHERS}
        active = OTHERS
    elif pvc_mode == 'incinerate':
        active = OTHERS            # PVC handled analytically below
    else:
        active = FEEDS

    pb = PERIOD_BASE[period]
    tau = tau_for(region, period)
    CIg = REGION_GRID_CI[region][period]
    CIp, Pp, H2cx, CRt = pb['CI_PPA'], pb['P_PPA'], pb['H2cx'], pb['CR_tgt']
    a_max = REGION_ALPHA_MAX[region][period]

    incin_shift = INCIN_B_ARR * (CIg - CI_BASE_INCIN) / 1000.0
    # mixed-waste incineration baseline of the stream that actually exists
    b_w = sum(w.get(f, 0.0) * INCIN_B[f] for f in w)
    CRt = crt_mult * (CRt - RED_PCT[period] * b_w * (CIg - CI_BASE_INCIN) / 1000.0)

    wUP = np.zeros_like(alpha_grid); wCR = np.zeros_like(alpha_grid)
    for i, a in enumerate(alpha_grid):
        P_elec = P_GRID_BASE + a * (Pp - P_GRID_BASE)
        CI_elec_g = CIg + a * (CIp - CIg)
        if scenario == 'elec':
            UP_x, CR_x = transform_electrification(P_elec, CI_elec_g, H2cx)
        else:
            UP_x, CR_x = transform_baseline(P_elec, CI_elec_g)
        CR_x = CR_x - incin_shift
        if dCR_pvc:
            CR_x = CR_x + dCR_pvc * IS_PVC
        UP_net = UP_x + tau * CR_x

        up_sum = 0.0; cr_sum = 0.0
        for f in active:
            idxs = feed_idx[f]
            j = idxs[np.argmax(UP_net[idxs])]
            up_sum += w[f] * UP_net[j]
            cr_sum += w[f] * CR_x[j]
        if pvc_mode == 'incinerate':
            up_sum += w['PVC'] * UP_INCIN['PVC']      # CR contribution is zero
        wUP[i] = up_sum; wCR[i] = cr_sum

    feas = (wCR >= CRt) & (alpha_grid <= a_max)
    if not feas.any():
        return np.nan, np.nan, CRt
    i_opt = int(np.argmax(np.where(feas, wUP, -np.inf)))
    return wUP[i_opt], wCR[i_opt], CRt


# ══ 1. baseline reproduction check ═══════════════════════════════════
print('\n' + '=' * 78)
print('1. BASELINE REPRODUCTION CHECK vs m3_step5_RC_world_periods_INCIN.csv')
print('=' * 78)
ref = pd.read_csv(os.path.join(BASE, 'm3_step5_RC_world_periods_INCIN.csv'))
ref_map = {(r.Region, r.Scenario, str(r.Period)): r.wUP_net for r in ref.itertuples()}
maxdiff = 0.0; nchk = 0
for region in REGION_W:
    for sc, tag in [('base', 'Baseline'), ('elec', 'Electrified')]:
        for p in PERIODS:
            got = evaluate(region, p, sc)[0]
            exp = ref_map.get((region, tag, p), np.nan)
            if np.isnan(got) and np.isnan(exp):
                nchk += 1; continue
            if np.isnan(got) != np.isnan(exp):
                print(f'  MISMATCH(feasibility) {region} {sc} {p}: got {got} exp {exp}')
                continue
            maxdiff = max(maxdiff, abs(got - exp)); nchk += 1
print(f'  checked {nchk} cells, max |diff| = {maxdiff:.2e}  '
      f'{"OK" if maxdiff < 1e-3 else "*** REVIEW ***"}')

# ══ 2. main configuration comparison ═════════════════════════════════
print('\n' + '=' * 78)
print('2. SYSTEM CONFIGURATIONS  (wCR / target, and wUP_net)')
print('=' * 78)
print(CF_NOTES)

rows = []
for region in REGION_W:
    for sc, tag in [('base', 'Baseline'), ('elec', 'Electrified')]:
        for p in PERIODS:
            rec = dict(Region=region, Scenario=tag, Period=p,
                       w_PVC=REGION_W[region]['PVC'])
            for mode, key in [('upcycle', 'A_upcycle'),
                              ('incinerate', 'B_incin'),
                              ('absent', 'C_absent')]:
                u, c, t = evaluate(region, p, sc, pvc_mode=mode)
                rec[f'{key}_wUP'] = u
                rec[f'{key}_wCR'] = c
                rec[f'{key}_CRt'] = t
                rec[f'{key}_feas'] = int(not np.isnan(u))
            rows.append(rec)
res = pd.DataFrame(rows)
res.to_csv(os.path.join(BASE, 'pvc_bottleneck_configs.csv'), index=False, float_format='%.4f')

piv = res[res.Scenario == 'Baseline'].pivot_table(
    index='Region', columns='Period',
    values=['A_upcycle_feas', 'B_incin_feas', 'C_absent_feas'])
print('\nBASELINE scenario feasibility (1 = feasible)')
print('  A = all five upcycled | B = PVC incinerated | C = PVC absent upstream')
hdr = f'{"Region":>14} ' + ' '.join(f'{p:>16}' for p in PERIODS)
print(hdr)
for region in REGION_W:
    cells = []
    for p in PERIODS:
        r = res[(res.Region == region) & (res.Scenario == 'Baseline') & (res.Period == p)].iloc[0]
        cells.append(f'A{int(r.A_upcycle_feas)} B{int(r.B_incin_feas)} C{int(r.C_absent_feas)}'.rjust(16))
    print(f'{region:>14} ' + ' '.join(cells))

# ══ 3. downstream lever: required PVC carbon improvement dCR* ═════════
print('\n' + '=' * 78)
print('3. DOWNSTREAM LEVER — minimum PVC improvement dCR* [kgCO2eq/kg]')
print('   (uniform additive gain on every PVC pathway that restores feasibility)')
print('=' * 78)

def min_dCR(region, period, scenario, hi=6.0, tol=1e-3):
    if not np.isnan(evaluate(region, period, scenario)[0]):
        return 0.0
    if np.isnan(evaluate(region, period, scenario, dCR_pvc=hi)[0]):
        return np.nan          # not reachable within the search range
    lo = 0.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if np.isnan(evaluate(region, period, scenario, dCR_pvc=mid)[0]):
            lo = mid
        else:
            hi = mid
    return hi

drows = []
print(f'{"Region":>14} {"w_PVC":>7} ' + ' '.join(f'{p:>9}' for p in PERIODS))
for region in REGION_W:
    vals = []
    for p in PERIODS:
        d = min_dCR(region, p, 'base')
        drows.append(dict(Region=region, Period=p, Scenario='Baseline', dCR_star=d))
        vals.append('  feasible' if d == 0.0 else (f'{d:>9.2f}' if not np.isnan(d) else '     >6.0'))
    print(f'{region:>14} {REGION_W[region]["PVC"]*100:>6.1f}% ' + ' '.join(vals))
pd.DataFrame(drows).to_csv(os.path.join(BASE, 'pvc_bottleneck_dCR.csv'),
                           index=False, float_format='%.4f')

# ══ 4. upstream lever: critical PVC share w* ══════════════════════════
print('\n' + '=' * 78)
print('4. UPSTREAM LEVER — critical PVC share w* [%]')
print('   (PVC share below which the region is feasible; others rescaled pro rata)')
print('=' * 78)

def reweight(region, w_pvc_new):
    w0 = REGION_W[region]
    s = sum(w0[f] for f in OTHERS)
    return {**{f: w0[f] / s * (1 - w_pvc_new) for f in OTHERS}, 'PVC': w_pvc_new}

def crit_pvc_share(region, period, scenario, tol=1e-4):
    feas_at = lambda x: not np.isnan(evaluate(region, period, scenario, w=reweight(region, x))[0])
    if feas_at(REGION_W[region]['PVC']):
        return np.nan          # already feasible, no constraint binding
    if not feas_at(0.0):
        return -1.0            # infeasible even with zero PVC -> PVC is not the cause
    lo, hi = 0.0, REGION_W[region]['PVC']
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if feas_at(mid):
            lo = mid
        else:
            hi = mid
    return lo

wrows = []
print(f'{"Region":>14} {"w_PVC":>7} ' + ' '.join(f'{p:>10}' for p in PERIODS))
for region in REGION_W:
    vals = []
    for p in PERIODS:
        c = crit_pvc_share(region, p, 'base')
        wrows.append(dict(Region=region, Period=p, Scenario='Baseline',
                          w_PVC=REGION_W[region]['PVC'], w_crit=c))
        if np.isnan(c):
            vals.append('  feasible')
        elif c < 0:
            vals.append('  not PVC')
        else:
            vals.append(f'{c*100:>9.1f}%')
    print(f'{region:>14} {REGION_W[region]["PVC"]*100:>6.1f}% ' + ' '.join(f'{v:>10}' for v in vals))
pd.DataFrame(wrows).to_csv(os.path.join(BASE, 'pvc_bottleneck_wcrit.csv'),
                           index=False, float_format='%.4f')

# ══ 5. PVC-share sweep for the bottleneck curve (figure panel a) ══════
print('\n' + '=' * 78)
print('5. PVC-SHARE SWEEP  (feasibility margin wCR - CRt, baseline scenario)')
print('=' * 78)
sweep = np.arange(0.0, 0.281, 0.01)
srows = []
for region in REGION_W:
    for p in PERIODS:
        for x in sweep:
            u, c, t = evaluate(region, p, 'base', w=reweight(region, x))
            srows.append(dict(Region=region, Period=p, w_PVC=x,
                              wUP=u, wCR=c, CRt=t,
                              margin=(c - t) if not np.isnan(c) else np.nan))
sw = pd.DataFrame(srows)
sw.to_csv(os.path.join(BASE, 'pvc_bottleneck_sweep.csv'), index=False, float_format='%.4f')
print(f'  wrote sweep: {len(sw)} rows  (11 regions x 4 periods x {len(sweep)} PVC shares)')

print('\nDone. CSVs written to data/intermediate:')
for f in ['pvc_bottleneck_configs.csv', 'pvc_bottleneck_dCR.csv',
          'pvc_bottleneck_wcrit.csv', 'pvc_bottleneck_sweep.csv']:
    print('  ', f)
