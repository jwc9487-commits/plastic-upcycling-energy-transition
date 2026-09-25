import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
"""M3 Step 5 (RC) — Decomposition of 2026→2050 electrification advantage.

Contribution of each lever:
  L1: rise in tau (environmental value priced into profit)
  L2: fall in P_PPA (renewable electricity price)
  L3: fall in CI_grid + CI_PPA (clean grid)
  L4: fall in green-H2 capex
  L5: rise in CR_target (NDC/NZE ambition)

Sequential counterfactual sweep:
  S0: 2026 base state
  S1: + τ → 2050 level
  S2: + P_PPA → 2050 level
  S3: + CI grid/PPA → 2050 level
  S4: + H2 capex → 2050 level
  S5: + CR_target → 2050 level (= 2050 full state)

For each region: ΔwUP_e from each step → stacked bar chart.
"""
import sys, io, os
os.environ['PYTHONUTF8'] = '1'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ref_elec_coupling import ref_price, ref_ci

BASE = config.INTERMEDIATE
MASS = 50e6
FEEDS = ['PE','PET','PP','PS','PVC']

# Incineration credit regionalization (2026-06-08): E_incin_net(feed)=a-b*CI_elec.
# Folded into the CI_grid lever (L3): dirtier grid -> bigger incineration credit ->
# lower CR; cleaner grid -> weaker incineration benchmark -> higher CR.
INCIN_B = {'PE':2.221266, 'PET':1.056216, 'PP':2.105206, 'PS':1.793923, 'PVC':0.687119}
CI_BASE_INCIN = 369.0

REGION_W = {
    'Global/EU': {'PE':0.406, 'PP':0.264, 'PET':0.105, 'PVC':0.137, 'PS':0.088},
    'U.S.':      {'PE':0.562, 'PP':0.200, 'PET':0.146, 'PVC':0.017, 'PS':0.075},
    'China':     {'PE':0.346, 'PP':0.250, 'PET':0.139, 'PVC':0.204, 'PS':0.061},
    'Japan':     {'PE':0.418, 'PP':0.278, 'PET':0.175, 'PVC':0.060, 'PS':0.069},
    'Korea':     {'PE':0.449, 'PP':0.263, 'PET':0.139, 'PVC':0.090, 'PS':0.059},
    'India':     {'PE':0.709, 'PP':0.105, 'PET':0.092, 'PVC':0.044, 'PS':0.051},
}

# 2026 state per region
S2026 = {
    'Global/EU': dict(tau=0.080, CI_grid=200, CI_PPA=30, P_PPA=0.040, H2cx=0.7, CR_tgt=0.00),
    'U.S.':      dict(tau=0.000, CI_grid=230, CI_PPA=30, P_PPA=0.040, H2cx=0.7, CR_tgt=0.00),
    'China':     dict(tau=0.010, CI_grid=510, CI_PPA=30, P_PPA=0.040, H2cx=0.7, CR_tgt=0.00),
    'Japan':     dict(tau=0.040, CI_grid=290, CI_PPA=30, P_PPA=0.040, H2cx=0.7, CR_tgt=0.00),
    'Korea':     dict(tau=0.030, CI_grid=330, CI_PPA=30, P_PPA=0.040, H2cx=0.7, CR_tgt=0.00),
    'India':     dict(tau=0.000, CI_grid=510, CI_PPA=30, P_PPA=0.040, H2cx=0.7, CR_tgt=0.00),
}

# 2050 endpoint per region
S2050 = {
    'Global/EU': dict(tau=0.200, CI_grid= 30, CI_PPA=10, P_PPA=0.020, H2cx=0.3, CR_tgt=2.42),
    'U.S.':      dict(tau=0.200, CI_grid= 50, CI_PPA=10, P_PPA=0.020, H2cx=0.3, CR_tgt=2.42),
    'China':     dict(tau=0.200, CI_grid=110, CI_PPA=10, P_PPA=0.020, H2cx=0.3, CR_tgt=2.42),
    'Japan':     dict(tau=0.200, CI_grid= 50, CI_PPA=10, P_PPA=0.020, H2cx=0.3, CR_tgt=2.42),
    'Korea':     dict(tau=0.200, CI_grid= 70, CI_PPA=10, P_PPA=0.020, H2cx=0.3, CR_tgt=2.42),
    'India':     dict(tau=0.200, CI_grid=130, CI_PPA=10, P_PPA=0.020, H2cx=0.3, CR_tgt=2.42),
}

P_GRID_BASE = 0.080
P_ELEC_BASE, CI_ELEC_BASE_g = 0.08, 369.0
P_FH_BASE,  CI_FH_BASE_g  = 0.03,     338.4
P_LP_BASE,  CI_LP_BASE_g  = 0.00684,  190.08
P_HP_BASE,  CI_HP_BASE_g  = 0.009,    190.08
P_H2_BASE,  CI_H2_BASE_kg = 2.0,      7.0
ETA_HEATER, ETA_BOILER = 0.95, 0.99

print('Loading data...')
df = pd.read_excel(config.MASTER_XLSX,
                   sheet_name='All pathways', header=3)
df['__feed'] = df.iloc[:, 0].astype(str).str.strip()
df = df[df['__feed'].isin(FEEDS)].reset_index(drop=True)
N = len(df)
print(f'N = {N}')

up_col = next(c for c in df.columns if isinstance(c, str) and c.lower().startswith('unit profit'))
cr_col = next(c for c in df.columns if isinstance(c, str) and c.lower().startswith('unit co2 reduc'))
UP0 = pd.to_numeric(df[up_col], errors='coerce').fillna(0).values
CR0 = pd.to_numeric(df[cr_col], errors='coerce').fillna(0).values
feed_arr = df['__feed'].values
INCIN_B_ARR = np.array([INCIN_B[f] for f in feed_arr])

def coln(c): return pd.to_numeric(df.iloc[:, c], errors='coerce').fillna(0).values

ELEC_M = coln(7);  FH_M = coln(8)
REF_M  = np.array([coln(10+k) for k in range(5)])
LP_M, HP_M = coln(15), coln(16)
H2_M   = coln(20)
ELEC_E = coln(31); FH_E = coln(32)
REF_E  = np.array([coln(33+k) for k in range(5)])
LP_E, HP_E = coln(38), coln(39)
H2_E   = coln(42)

P_REF_BASE    = np.array([ref_price(k+1, P_ELEC_BASE) for k in range(5)])
CI_REF_BASE_g = np.array([ref_ci(k+1, CI_ELEC_BASE_g) for k in range(5)])

def transform_electrification(P_elec, CI_elec_g, h2_capex):
    r_e_c, r_e_ci = P_elec/P_ELEC_BASE, CI_elec_g/CI_ELEC_BASE_g
    r_ref_c  = np.array([ref_price(k+1, P_elec) / P_REF_BASE[k] for k in range(5)])
    r_ref_ci = np.array([ref_ci(k+1, CI_elec_g) / CI_REF_BASE_g[k] for k in range(5)])
    new_ELEC_M = ELEC_M * r_e_c; new_ELEC_E = ELEC_E * r_e_ci
    new_REF_M_sum = REF_M.T @ r_ref_c; new_REF_E_sum = REF_E.T @ r_ref_ci
    new_FH_M = FH_M * ((P_elec/ETA_HEATER) / P_FH_BASE)
    new_FH_E = FH_E * ((CI_elec_g/ETA_HEATER) / CI_FH_BASE_g)
    new_LP_M = LP_M * ((P_elec/ETA_BOILER) / P_LP_BASE)
    new_LP_E = LP_E * ((CI_elec_g/ETA_BOILER) / CI_LP_BASE_g)
    new_HP_M = HP_M * ((P_elec/ETA_BOILER) / P_HP_BASE)
    new_HP_E = HP_E * ((CI_elec_g/ETA_BOILER) / CI_HP_BASE_g)
    new_H2_M = H2_M * ((53.0*P_elec + h2_capex) / P_H2_BASE)
    new_H2_E = H2_E * ((53.0*CI_elec_g/1000.0) / CI_H2_BASE_kg)
    dC_M = ((new_ELEC_M - ELEC_M) + (new_REF_M_sum - REF_M.sum(axis=0))
             + (new_FH_M - FH_M) + (new_LP_M - LP_M)
             + (new_HP_M - HP_M) + (new_H2_M - H2_M))
    dE_kg = ((new_ELEC_E - ELEC_E) + (new_REF_E_sum - REF_E.sum(axis=0))
             + (new_FH_E - FH_E) + (new_LP_E - LP_E)
             + (new_HP_E - HP_E) + (new_H2_E - H2_E))
    UP = UP0 - dC_M * 1e6/MASS
    CR = CR0 - dE_kg / MASS
    return UP, CR

feed_idx = {f: np.where(feed_arr == f)[0] for f in FEEDS}

def econ_opt_per_feed(UP_net, CR):
    UP_best = {}; CR_at_best = {}
    for f in FEEDS:
        idxs = feed_idx[f]
        UP_f = UP_net[idxs]
        best_local = np.argmax(UP_f)
        UP_best[f]  = UP_f[best_local]
        CR_at_best[f] = CR[idxs[best_local]]
    return UP_best, CR_at_best

alpha_grid = np.linspace(0, 1, 51)

def evaluate_state(region, state):
    """Return wUP_e_opt under given state (max over α s.t. wCR_e ≥ CR_tgt)."""
    w = REGION_W[region]
    tau  = state['tau']
    CIg  = state['CI_grid']
    CIp  = state['CI_PPA']
    Pp   = state['P_PPA']
    H2cx = state['H2cx']
    CRt  = state['CR_tgt']

    # ── Incineration credit regionalization (uses grid CIg, folded into L3) ──
    incin_shift = INCIN_B_ARR * (CIg - CI_BASE_INCIN) / 1000.0
    b_w = sum(w[f] * INCIN_B[f] for f in FEEDS)
    if CRt > 0:  # regionalize target consistently (red% = CRt/2.69)
        CRt = CRt - (CRt / 2.69) * b_w * (CIg - CI_BASE_INCIN) / 1000.0

    wUP = np.zeros_like(alpha_grid); wCR = np.zeros_like(alpha_grid)
    for i, a in enumerate(alpha_grid):
        P_elec    = P_GRID_BASE + a * (Pp  - P_GRID_BASE)
        CI_elec_g = CIg          + a * (CIp - CIg)
        UP_e, CR_e = transform_electrification(P_elec, CI_elec_g, H2cx)
        CR_e       = CR_e - incin_shift
        UP_e_net   = UP_e + tau * CR_e
        UPe_best, CRe_at = econ_opt_per_feed(UP_e_net, CR_e)
        wUP[i] = sum(w[f] * UPe_best[f] for f in FEEDS)
        wCR[i] = sum(w[f] * CRe_at[f]   for f in FEEDS)

    feas = wCR >= CRt
    wUP_feas = np.where(feas, wUP, -np.inf)
    if not np.isfinite(wUP_feas).any():
        return np.nan, np.nan, np.nan
    i_opt = np.argmax(wUP_feas)
    return wUP[i_opt], wCR[i_opt], alpha_grid[i_opt]

# ── Sequential counter-factual ──
LEVERS = ['L1: τ↑',
          'L2: P_PPA↓',
          'L3: CI_grid + CI_PPA↓',
          'L4: H2 capex↓',
          'L5: CR_target↑']

print('\nRunning sequential decomposition...\n')
print(f'{"Region":>10} {"S0(2026)":>10} {"+L1 τ":>10} {"+L2 P":>10} '
      f'{"+L3 CI":>10} {"+L4 H2":>10} {"+L5 CR":>10} {"Total Δ":>10}')

decomp_rows = []
for region in REGION_W:
    s0 = dict(S2026[region])
    s1 = dict(s0); s1['tau']     = S2050[region]['tau']
    s2 = dict(s1); s2['P_PPA']   = S2050[region]['P_PPA']
    s3 = dict(s2); s3['CI_grid'] = S2050[region]['CI_grid']; s3['CI_PPA'] = S2050[region]['CI_PPA']
    s4 = dict(s3); s4['H2cx']    = S2050[region]['H2cx']
    s5 = dict(s4); s5['CR_tgt']  = S2050[region]['CR_tgt']  # = S2050

    states = [s0, s1, s2, s3, s4, s5]
    wUPs = []
    for st in states:
        wUP, wCR, aopt = evaluate_state(region, st)
        wUPs.append(wUP)

    # Contributions
    contribs = [wUPs[i+1] - wUPs[i] if not (np.isnan(wUPs[i+1]) or np.isnan(wUPs[i])) else np.nan
                 for i in range(5)]
    total = wUPs[-1] - wUPs[0] if not (np.isnan(wUPs[-1]) or np.isnan(wUPs[0])) else np.nan
    print(f'{region:>10} {wUPs[0]:>+10.3f} {contribs[0]:>+10.3f} {contribs[1]:>+10.3f} '
          f'{contribs[2]:>+10.3f} {contribs[3]:>+10.3f} {contribs[4]:>+10.3f} {total:>+10.3f}')

    decomp_rows.append(dict(
        Region=region,
        S0_2026=wUPs[0],
        dL1_tau=contribs[0],
        dL2_P_PPA=contribs[1],
        dL3_CI=contribs[2],
        dL4_H2cx=contribs[3],
        dL5_CR_tgt=contribs[4],
        S5_2050=wUPs[-1],
        Total=total))

decomp_df = pd.DataFrame(decomp_rows)
decomp_df.to_csv(os.path.join(BASE, 'm3_step5_RC_decomposition_INCIN.csv'),
                 index=False, float_format='%.4f')

# ── Stacked bar figure ──
fig, ax = plt.subplots(figsize=(14, 7))
fig.subplots_adjust(top=0.88, bottom=0.20, left=0.08, right=0.78)

regions_list = list(REGION_W.keys())
x = np.arange(len(regions_list))
width = 0.55

contrib_matrix = np.array([
    [decomp_df.loc[decomp_df.Region == r, 'dL1_tau'].values[0] for r in regions_list],
    [decomp_df.loc[decomp_df.Region == r, 'dL2_P_PPA'].values[0] for r in regions_list],
    [decomp_df.loc[decomp_df.Region == r, 'dL3_CI'].values[0] for r in regions_list],
    [decomp_df.loc[decomp_df.Region == r, 'dL4_H2cx'].values[0] for r in regions_list],
    [decomp_df.loc[decomp_df.Region == r, 'dL5_CR_tgt'].values[0] for r in regions_list],
])
s0_vals = np.array([decomp_df.loc[decomp_df.Region == r, 'S0_2026'].values[0] for r in regions_list])
s5_vals = np.array([decomp_df.loc[decomp_df.Region == r, 'S5_2050'].values[0] for r in regions_list])

colors = ['#c0392b', '#2980b9', '#27ae60', '#8e44ad', '#d35400']

# Stack positive and negative separately
pos_bottom = np.zeros(len(regions_list))
neg_bottom = np.zeros(len(regions_list))

for j, lever in enumerate(LEVERS):
    vals = contrib_matrix[j]
    pos = np.where(vals > 0, vals, 0)
    neg = np.where(vals < 0, vals, 0)
    ax.bar(x, pos, width, bottom=pos_bottom, color=colors[j],
            edgecolor='white', linewidth=0.8, label=lever)
    ax.bar(x, neg, width, bottom=neg_bottom, color=colors[j],
            edgecolor='white', linewidth=0.8)
    pos_bottom += pos
    neg_bottom += neg

# Start (2026) and End (2050) markers
ax.plot(x, s0_vals, 'o', markersize=14, markerfacecolor='white',
         markeredgecolor='#1c2833', markeredgewidth=2.5,
         label='2026 (S0)', zorder=10)
ax.plot(x, s5_vals, 'D', markersize=12, markerfacecolor='gold',
         markeredgecolor='#1c2833', markeredgewidth=2.0,
         label='2050 (S5)', zorder=10)

# Connect via dashed
for i in range(len(regions_list)):
    ax.plot([x[i], x[i]], [s0_vals[i], s5_vals[i]],
             color='gray', linestyle=':', linewidth=0.8, zorder=1, alpha=0.5)

ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(regions_list, fontsize=11, fontweight='bold')
ax.set_ylabel('wUP_net ($/kg) — sequential contribution', fontsize=11)
ax.set_title('M3 Step 5 (RC) — Decomposition of 2026→2050 electrification economic gain\n'
              'Sequential counter-factual: 2026 baseline + L1→L2→L3→L4→L5 → 2050 final',
              fontsize=12, fontweight='bold')
ax.grid(linestyle=':', alpha=0.4, axis='y')

# Annotate totals
for i, r in enumerate(regions_list):
    total = s5_vals[i] - s0_vals[i]
    ax.annotate(f'Δ={total:+.3f}', xy=(x[i], s5_vals[i] + 0.025),
                ha='center', fontsize=9, fontweight='bold', color='#1c2833')

ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=9.5, framealpha=0.95)

out_png = os.path.join(config.FIG_OUT, 'fig_m3_step5_RC_decomposition_INCIN.png')
plt.savefig(out_png, dpi=180, bbox_inches='tight')
plt.close()
print(f'\nSaved: {out_png}')

# Summary share
print('\n── Lever contribution share (% of total Δ, region-mean) ──')
total_means = decomp_df['Total'].abs().mean()
for j, lever in enumerate(LEVERS):
    key = ['dL1_tau','dL2_P_PPA','dL3_CI','dL4_H2cx','dL5_CR_tgt'][j]
    share = decomp_df[key].abs().mean() / decomp_df.iloc[:, 2:7].abs().sum(axis=1).mean() * 100
    mean_val = decomp_df[key].mean()
    print(f'  {lever:<25}  mean Δ = {mean_val:+.3f}  share = {share:.1f}%')

print('\nDone.')
