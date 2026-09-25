import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
"""M2 Step 3 (RC) — Crossover analysis under two optimal frameworks.

Per feed, two independent optimal pathways:
  Econ-optimal pathway = argmax UP   (under UP framework)
  Env-optimal pathway  = argmax CR   (under UCR framework)

Panel A (Econ): x = P_elec, y = UP, lines = per-feed Econ-optimal UP under
                two scenarios (baseline / electrification).
Panel B (Env):  x = CI_elec, y = CR, lines = per-feed Env-optimal CR under
                two scenarios.

Baseline scenario:
  Only Electricity and Refrigerator are P_elec/CI_elec-dependent;
  Fired heat, LP/HP steam, Hydrogen are FOSSIL-fueled (independent prices/CIs).

Electrification scenario:
  All thermal/H2 utilities are produced from electricity:
    FH:   P = P_elec / η_heater (=0.95), CI same as P-eq.
    LP/HP: P = P_elec / η_boiler (=0.99)
    H2:   P = 53·P_elec + 0.7,  CI = 53·CI_elec (per kg H2)
  Refrigerator: ref_elec_coupling (unchanged from baseline).

Output: 2-panel figure with crossover markers + CSV.
"""
import sys, io, os
os.environ['PYTHONUTF8'] = '1'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch
# ── Journal style (2026-06-09): Arial, no titles, 9pt bold labels, 7pt ticks ──
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.labelsize': 9, 'axes.labelweight': 'bold',
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.titlesize': 9,
})
MM = 1/25.4
FIG_W = 159.2 * MM
FIG_W1 = 79.6 * MM

from ref_elec_coupling import ref_price, ref_ci

BASE = config.INTERMEDIATE
MASS = 50e6
FEEDS = ['PE','PET','PP','PS','PVC']

# Baseline reference unit prices and CIs (gCO2/kWh or kgCO2/kg)
P_ELEC_BASE  = 0.08;     CI_ELEC_BASE_g = 369.0      # g/kWh
P_FH_BASE    = 0.03;     CI_FH_BASE_g   = 338.4
P_LP_BASE    = 0.00684;  CI_LP_BASE_g   = 190.08
P_HP_BASE    = 0.009;    CI_HP_BASE_g   = 190.08
P_H2_BASE    = 2.0;      CI_H2_BASE_kg  = 7.0        # kgCO2/kg H2

ETA_HEATER = 0.95
ETA_BOILER = 0.99

print('Loading…')
df = pd.read_excel(config.MASTER_XLSX,
                   sheet_name='All pathways', header=3)
df['__feed'] = df.iloc[:, 0].astype(str).str.strip()
df['__tech'] = df.iloc[:, 1].astype(str).str.strip()
df = df[df['__feed'].isin(FEEDS)].reset_index(drop=True)
N = len(df)
print(f'N = {N}')

up_col = next(c for c in df.columns if isinstance(c, str) and c.lower().startswith('unit profit'))
cr_col = next(c for c in df.columns if isinstance(c, str) and c.lower().startswith('unit co2 reduc'))
UP0 = pd.to_numeric(df[up_col], errors='coerce').fillna(0).values
CR0 = pd.to_numeric(df[cr_col], errors='coerce').fillna(0).values
feed_arr = df['__feed'].values
tech_arr = df['__tech'].values
P1_arr   = df.iloc[:, 2].astype(str).str.strip().values   # Pathway1
P2_arr   = df.iloc[:, 3].astype(str).str.strip().values   # Pathway2
prod_arr = df.iloc[:, 4].astype(str).str.strip().values   # Product

# ── Pathway-naming → decoded process-block sequence (Source→units→Product) ──
PN = pd.read_excel(config.PATHWAY_NAMING,
                   sheet_name='Pathway naming', header=None)
PN = PN.iloc[4:].reset_index(drop=True); PN.columns = range(PN.shape[1])
PN[list(range(6, 12))] = PN.groupby([1, 2, 3])[list(range(6, 12))].ffill()  # oil shared in P1
for c in range(1, 6):
    PN[c] = PN[c].astype(str).str.strip()
_WH = {'PY','HT','HC','SC','AS','ARO','CO2','OS','HG','OG','SG','PRE'}
_MU = {'PY':'PY','HT':'HT','HC':'HC','SC':'SC','AS':'AS','ARO':'AS','CO2':'CA','OS':'OS',
       'HG':'HG','OG':'O-GS','SG':'S-GS','PRE':'PRE','P':'PRE','M':'MS','F':'FTS','R':'RWGS','W':'WGS'}
def _term(t):
    t = str(t).strip()
    for s in ('_HT','_LT','_OG','_SG'):
        if t.endswith(s): t = t[:-3]; break
    k = t.split('_')[-1]
    return _MU[k] if k in _WH else _MU.get(k[-1], '?')
def _seq(r, cols):
    return [_term(r[c]) for c in cols if pd.notna(r[c]) and str(r[c]).strip()]
_MAIN = {'HT':'H-PY','LT':'L-PY','OG':'O-GS','SG':'S-GS','HG':'HG'}
_PMAP = {'GASOLINE':'gasoline','DIESEL':'diesel','MeOH':'MeOH','FT':'FT fuel','Olefin':'olefin',
         'OLEFIN':'olefin','H2':'H2','Aromatics':'aromatics','AROMATICS':'aromatics','BA':'BA','WAX':'wax'}
_PUNIT = {'gasoline':['HC'],'diesel':['HC','SC'],'MeOH':['MS'],'FT fuel':['FTS'],
          'olefin':['OS'],'H2':['CA','PSA','PRE'],'aromatics':['AS'],'wax':['FTS'],'BA':['AS','HT']}
def decode_route(feed, tech, P1, P2, product):
    """Return ordered block list [feed, unit, unit, ..., product] for one pathway."""
    m = PN[(PN[1]==feed)&(PN[2]==tech)&(PN[3]==P1)&(PN[4]==P2)&(PN[5]==product)]
    if len(m) == 0:
        return [feed, _MAIN.get(tech, tech), str(product).split('_')[0].lower()]
    r = m.iloc[0]
    mt = _MAIN.get(tech, tech)
    oil = _seq(r, range(6, 12)); gas = _seq(r, range(12, 18)); gz = _seq(r, range(25, 29))
    pname = _PMAP.get(str(product).split('_')[0], str(product).split('_')[0].lower())
    finals = _PUNIT.get(pname, [])
    cands = []
    for line in (oil, gas, gz):
        if not line: continue
        units = [mt] + [u for u in line[1:] if u not in ('PY','HG','O-GS','S-GS')]
        cands.append(units)
    # choose the line whose terminal unit makes the labeled product
    for units in cands:
        if units and units[-1] in finals:
            return [feed] + units + [pname]
    # fallback: longest decoded line
    if cands:
        units = max(cands, key=len)
        return [feed] + units + [pname]
    return [feed, mt, pname]

_OUTU = {'HT':'UO','SC':'LG','CA':'CG','PRE':'SG','WGS':'HR','RWGS':'CR','HC':'fuel',
         'OS':'olefin','MS':'MeOH','FTS':'FT','PSA':'H2','HG':'PO','AS':'BTX'}
def decode_routes(feed, tech, P1, P2, product):
    """Return {'oil':chain, 'gas':chain}. Pyrolysis splits into an oil line AND a
    gas line (2 tiers); gasification has a single syngas line."""
    m = PN[(PN[1] == feed) & (PN[2] == tech) & (PN[3] == P1) & (PN[4] == P2) & (PN[5] == product)]
    mt = _MAIN.get(tech, tech)
    pname = _PMAP.get(str(product).split('_')[0], str(product).split('_')[0].lower())
    out = {}
    if len(m) == 0:
        return {'oil': [feed, mt, pname]}
    r = m.iloc[0]
    oil = _seq(r, range(6, 12)); gas = _seq(r, range(12, 18)); gz = _seq(r, range(25, 29))
    gz2 = [u for u in gz if u not in ('O-GS', 'S-GS')]
    if gz2:                                       # gasification → single syngas tier
        return {'gas': [feed, mt] + gz2 + [pname]}
    if oil:
        u = [mt] + [x for x in oil[1:] if x not in ('PY', 'HG')]
        out['oil'] = [feed] + u + [pname]
    if gas:
        u = [mt] + [x for x in gas if x not in ('PY', 'HG')]
        end = _OUTU.get(u[-1], 'gas') if len(u) > 1 else 'gas'
        out['gas'] = [feed] + u + [end]
    return out or {'oil': [feed, mt, pname]}

def coln(c): return pd.to_numeric(df.iloc[:, c], errors='coerce').fillna(0).values

# Cost columns ($M/yr)
ELEC_M = coln(7);  FH_M = coln(8)
REF_M  = np.array([coln(10+k) for k in range(5)])
LP_M, HP_M = coln(15), coln(16)
H2_M   = coln(20)
# Emission columns (kg/yr)
ELEC_E = coln(31); FH_E = coln(32)
REF_E  = np.array([coln(33+k) for k in range(5)])
LP_E, HP_E = coln(38), coln(39)
H2_E   = coln(42)

P_REF_BASE   = np.array([ref_price(k+1, P_ELEC_BASE) for k in range(5)])
CI_REF_BASE_g = np.array([ref_ci(k+1, CI_ELEC_BASE_g) for k in range(5)])

def transform_baseline(P_elec, CI_elec_g):
    """Baseline: only Electricity & Refrigerator are P_elec/CI_elec functions.
    FH, LP, HP, H2 fixed at fossil baselines."""
    r_e_c  = P_elec / P_ELEC_BASE
    r_e_ci = CI_elec_g / CI_ELEC_BASE_g
    r_ref_c  = np.array([ref_price(k+1, P_elec) / P_REF_BASE[k] for k in range(5)])
    r_ref_ci = np.array([ref_ci(k+1, CI_elec_g) / CI_REF_BASE_g[k] for k in range(5)])

    new_ELEC_M    = ELEC_M * r_e_c
    new_ELEC_E    = ELEC_E * r_e_ci
    new_REF_M_sum = REF_M.T @ r_ref_c
    new_REF_E_sum = REF_E.T @ r_ref_ci

    dC_M  = (new_ELEC_M - ELEC_M) + (new_REF_M_sum - REF_M.sum(axis=0))
    dE_kg = (new_ELEC_E - ELEC_E) + (new_REF_E_sum - REF_E.sum(axis=0))
    dC_pk = dC_M  * 1e6 / MASS
    dE_pk = dE_kg / MASS
    UP = UP0 - dC_pk
    CR = CR0 - dE_pk
    return UP, CR

def transform_electrification(P_elec, CI_elec_g):
    """Electrification: ALL thermal/H2 utilities converted to electric equivalents."""
    # Electricity + Refrigerator (same as baseline transform)
    r_e_c  = P_elec / P_ELEC_BASE
    r_e_ci = CI_elec_g / CI_ELEC_BASE_g
    r_ref_c  = np.array([ref_price(k+1, P_elec) / P_REF_BASE[k] for k in range(5)])
    r_ref_ci = np.array([ref_ci(k+1, CI_elec_g) / CI_REF_BASE_g[k] for k in range(5)])

    new_ELEC_M    = ELEC_M * r_e_c
    new_ELEC_E    = ELEC_E * r_e_ci
    new_REF_M_sum = REF_M.T @ r_ref_c
    new_REF_E_sum = REF_E.T @ r_ref_ci

    # FH → electric heater
    P_FH_new  = P_elec / ETA_HEATER
    CI_FH_new = CI_elec_g / ETA_HEATER
    r_FH_c  = P_FH_new / P_FH_BASE
    r_FH_ci = CI_FH_new / CI_FH_BASE_g
    new_FH_M = FH_M * r_FH_c
    new_FH_E = FH_E * r_FH_ci

    # LP steam → electric boiler
    P_LP_new  = P_elec / ETA_BOILER
    CI_LP_new = CI_elec_g / ETA_BOILER
    r_LP_c  = P_LP_new / P_LP_BASE
    r_LP_ci = CI_LP_new / CI_LP_BASE_g
    new_LP_M = LP_M * r_LP_c
    new_LP_E = LP_E * r_LP_ci

    # HP steam → electric boiler
    P_HP_new  = P_elec / ETA_BOILER
    CI_HP_new = CI_elec_g / ETA_BOILER
    r_HP_c  = P_HP_new / P_HP_BASE
    r_HP_ci = CI_HP_new / CI_HP_BASE_g
    new_HP_M = HP_M * r_HP_c
    new_HP_E = HP_E * r_HP_ci

    # H2 → green electrolysis
    P_H2_new   = 53.0 * P_elec + 0.7
    CI_H2_new  = 53.0 * CI_elec_g / 1000.0   # gCO2/kWh → kgCO2/kg via 53 kWh/kg
    r_H2_c  = P_H2_new / P_H2_BASE
    r_H2_ci = CI_H2_new / CI_H2_BASE_kg
    new_H2_M = H2_M * r_H2_c
    new_H2_E = H2_E * r_H2_ci

    dC_M  = ((new_ELEC_M - ELEC_M) + (new_REF_M_sum - REF_M.sum(axis=0))
              + (new_FH_M - FH_M) + (new_LP_M - LP_M)
              + (new_HP_M - HP_M) + (new_H2_M - H2_M))
    dE_kg = ((new_ELEC_E - ELEC_E) + (new_REF_E_sum - REF_E.sum(axis=0))
              + (new_FH_E - FH_E) + (new_LP_E - LP_E)
              + (new_HP_E - HP_E) + (new_H2_E - H2_E))
    dC_pk = dC_M  * 1e6 / MASS
    dE_pk = dE_kg / MASS
    UP = UP0 - dC_pk
    CR = CR0 - dE_pk
    return UP, CR

feed_idx = {f: np.where(feed_arr == f)[0] for f in FEEDS}

def best_UP(UP):
    return {f: UP[feed_idx[f]].max() for f in FEEDS}
def best_CR(CR):
    return {f: CR[feed_idx[f]].max() for f in FEEDS}
def best_UP_tech(UP):
    return {f: tech_arr[feed_idx[f][np.argmax(UP[feed_idx[f]])]] for f in FEEDS}
def best_CR_tech(CR):
    return {f: tech_arr[feed_idx[f][np.argmax(CR[feed_idx[f]])]] for f in FEEDS}
def best_path(vals):
    """Full optimal-pathway key (tech,P1,P2,product) per feed at argmax."""
    out = {}
    for f in FEEDS:
        idxs = feed_idx[f]; j = idxs[int(np.argmax(vals[idxs]))]
        out[f] = (tech_arr[j], P1_arr[j], P2_arr[j], prod_arr[j])
    return out

# ─────────────────────────────────────────────────────────────────────
# Panel A sweep: x = P_elec; CI fixed at baseline 369 (τ=0 → CI no effect on UP)
# ─────────────────────────────────────────────────────────────────────
P_grid = np.linspace(0.0, 0.06, 31)   # panel (a) x-range narrowed to 0–0.06 $/kWh
econ_UP_base = {f: np.zeros_like(P_grid) for f in FEEDS}
econ_UP_elec = {f: np.zeros_like(P_grid) for f in FEEDS}
econ_tech_base = {f: [] for f in FEEDS}
econ_tech_elec = {f: [] for f in FEEDS}
econ_path_base = {f: [] for f in FEEDS}
econ_path_elec = {f: [] for f in FEEDS}

for i, P in enumerate(P_grid):
    UP_b, CR_b = transform_baseline(P, CI_ELEC_BASE_g)
    UP_e, CR_e = transform_electrification(P, CI_ELEC_BASE_g)
    b_best = best_UP(UP_b); b_tech = best_UP_tech(UP_b); b_path = best_path(UP_b)
    e_best = best_UP(UP_e); e_tech = best_UP_tech(UP_e); e_path = best_path(UP_e)
    for f in FEEDS:
        econ_UP_base[f][i] = b_best[f]
        econ_UP_elec[f][i] = e_best[f]
        econ_tech_base[f].append(b_tech[f])
        econ_tech_elec[f].append(e_tech[f])
        econ_path_base[f].append(b_path[f])
        econ_path_elec[f].append(e_path[f])

# ─────────────────────────────────────────────────────────────────────
# Panel B sweep: x = CI_elec; P fixed at baseline 0.08
# ─────────────────────────────────────────────────────────────────────
CI_grid = np.linspace(0, 400, 41)   # panel (b) x-range narrowed to 0–400 (transition-focused)
env_CR_base = {f: np.zeros_like(CI_grid) for f in FEEDS}
env_CR_elec = {f: np.zeros_like(CI_grid) for f in FEEDS}
env_tech_base = {f: [] for f in FEEDS}
env_tech_elec = {f: [] for f in FEEDS}
env_path_base = {f: [] for f in FEEDS}
env_path_elec = {f: [] for f in FEEDS}

for i, CI in enumerate(CI_grid):
    UP_b, CR_b = transform_baseline(P_ELEC_BASE, CI)
    UP_e, CR_e = transform_electrification(P_ELEC_BASE, CI)
    b_best = best_CR(CR_b); b_tech = best_CR_tech(CR_b); b_path = best_path(CR_b)
    e_best = best_CR(CR_e); e_tech = best_CR_tech(CR_e); e_path = best_path(CR_e)
    for f in FEEDS:
        env_CR_base[f][i] = b_best[f]
        env_CR_elec[f][i] = e_best[f]
        env_tech_base[f].append(b_tech[f])
        env_tech_elec[f].append(e_tech[f])
        env_path_base[f].append(b_path[f])
        env_path_elec[f].append(e_path[f])

# ── Optimal-pathway regimes (segments where the full optimal pathway is constant) ──
def regimes(items, xgrid):
    segs = []
    cur = items[0]; start = xgrid[0]
    for i in range(1, len(items)):
        if items[i] != cur:
            segs.append((cur, start, xgrid[i]))
            cur = items[i]; start = xgrid[i]
    segs.append((cur, start, xgrid[-1]))
    return segs

# Collect distinct optimal pathways across all lanes (both panels, both scenarios)
distinct = {}  # key=(tech,P1,P2,prod) -> decoded route list
def register(pathlists):
    for f in FEEDS:
        for key in pathlists[f]:
            if key not in distinct:
                distinct[key] = decode_route(f, *key)
for pl in (econ_path_base, econ_path_elec, env_path_base, env_path_elec):
    register(pl)

print('\n=== Distinct optimal pathways (decoded process blocks) ===')
for key, route in distinct.items():
    print(f'  {key}: {" -> ".join(route)}')
print('\n=== Econ-optimal pathway regimes (Panel A, x=P_elec) ===')
for f in FEEDS:
    for scen, pl in (('elec', econ_path_elec), ('base', econ_path_base)):
        segs = [(distinct[k][1] if len(distinct[k])>1 else str(k), a, b)
                for k, a, b in regimes(pl[f], P_grid)]
        print(f'  {f} {scen}: {segs}')
import sys as _sys; _sys.stdout.flush()

# ─────────────────────────────────────────────────────────────────────
# Crossover finder
# ─────────────────────────────────────────────────────────────────────
def find_crossover(x, y1, y2):
    """Linear-interp x where y1 - y2 crosses zero. None if no crossing."""
    diff = y1 - y2
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        return None
    i = sign_change[0]
    if diff[i+1] == diff[i]:
        return x[i]
    xc = x[i] + (x[i+1]-x[i]) * (-diff[i]) / (diff[i+1]-diff[i])
    yc = y1[i] + (y1[i+1]-y1[i]) * (xc-x[i]) / (x[i+1]-x[i])
    return xc, yc

print('\n── Econ-optimal UP crossovers (Baseline vs Electrification) ──')
econ_crossovers = {}
for f in FEEDS:
    cx = find_crossover(P_grid, econ_UP_base[f], econ_UP_elec[f])
    econ_crossovers[f] = cx
    if cx is None:
        # Which dominates throughout?
        if econ_UP_elec[f].mean() > econ_UP_base[f].mean():
            dominant = 'Electrification dominates ALL P_elec range'
        else:
            dominant = 'Baseline dominates ALL P_elec range'
        print(f'  {f}: no crossover → {dominant}')
    else:
        print(f'  {f}: crossover at P_elec = {cx[0]:.4f} $/kWh, UP = {cx[1]:.3f}')

print('\n── Env-optimal CR crossovers (Baseline vs Electrification) ──')
env_crossovers = {}
for f in FEEDS:
    cx = find_crossover(CI_grid, env_CR_base[f], env_CR_elec[f])
    env_crossovers[f] = cx
    if cx is None:
        if env_CR_elec[f].mean() > env_CR_base[f].mean():
            dominant = 'Electrification dominates ALL CI_elec range'
        else:
            dominant = 'Baseline dominates ALL CI_elec range'
        print(f'  {f}: no crossover → {dominant}')
    else:
        print(f'  {f}: crossover at CI_elec = {cx[0]:.1f} gCO2/kWh, CR = {cx[1]:.3f}')

# Save CSV
rows = []
for i, P in enumerate(P_grid):
    rec = {'P_elec': P}
    for f in FEEDS:
        rec[f'econ_UP_base_{f}'] = econ_UP_base[f][i]
        rec[f'econ_UP_elec_{f}'] = econ_UP_elec[f][i]
    rows.append(rec)
pd.DataFrame(rows).to_csv(os.path.join(BASE, 'm2_step3_RC_econ.csv'),
                           index=False, float_format='%.4f')
rows = []
for i, CI in enumerate(CI_grid):
    rec = {'CI_elec': CI}
    for f in FEEDS:
        rec[f'env_CR_base_{f}'] = env_CR_base[f][i]
        rec[f'env_CR_elec_{f}'] = env_CR_elec[f][i]
    rows.append(rec)
pd.DataFrame(rows).to_csv(os.path.join(BASE, 'm2_step3_RC_env.csv'),
                           index=False, float_format='%.4f')

# ─────────────────────────────────────────────────────────────────────
# 2-panel figure with crossover markers + optimal-config tracks
# ─────────────────────────────────────────────────────────────────────
# Paper-consistent muted feed palette (matches Fig2 superstructure)
FEED_COLOR = {'PE':'#2980b9','PET':'#c0392b','PP':'#e67e22',
              'PS':'#8e44ad','PVC':'#16a085'}
LW = 0.75

# ── Process-block-flow chip rendering (paper/SI block style) ──
NODE_MAIN = {'H-PY', 'L-PY', 'S-GS', 'O-GS', 'HG'}
def draw_chain(axc, route, x0, y, cw=8.6, gap=2.0, h=4.4):
    """Draw Source→unit→…→Product as connected rounded chips; return end x."""
    x = x0; n = len(route)
    for i, node in enumerate(route):
        if i == 0:                       fc = FEED_COLOR.get(node, '#888'); tc = 'white'
        elif i == n - 1:                 fc = '#229954';                    tc = 'white'
        elif node in NODE_MAIN:          fc = '#34495e';                    tc = 'white'
        else:                            fc = '#aed6f1';                    tc = '#1c2833'
        axc.add_patch(FancyBboxPatch((x, y - h/2), cw, h,
                      boxstyle='round,pad=0.08,rounding_size=0.6',
                      fc=fc, ec='white', lw=0.4, zorder=4))
        axc.text(x + cw/2, y, node, ha='center', va='center',
                 fontsize=4.0, color=tc, fontweight='bold', zorder=5)
        if i < n - 1:
            axc.annotate('', xy=(x + cw + gap - 0.1, y), xytext=(x + cw + 0.1, y),
                         arrowprops=dict(arrowstyle='-|>', lw=0.6, color='#566573'),
                         zorder=4)
        x += cw + gap
    return x

from matplotlib.patches import Rectangle as _Rect
_ABBR = {'gasoline':'gaso','aromatics':'arom','olefin':'olef','FT fuel':'FT',
         'diesel':'dies','MeOH':'MeOH','wax':'wax','H2':'H2','BA':'BA'}
def draw_chain_seg(axc, route, x0, x1, ylo, lh, span, alpha=1.0):
    """Draw Source→…→Product as a horizontal chip chain scaled into the
    x-range [x0,x1] of a feed lane (data coords; x aligned to shared axis)."""
    n = len(route)
    pad = (x1 - x0) * 0.04
    aw = (x1 - x0) - 2 * pad
    gfrac = 0.28
    chip = aw / (n + (n - 1) * gfrac)
    gap = chip * gfrac
    yc = ylo + lh * 0.5; ch = lh * 0.56
    fs = min(4.2, max(2.3, 150 * chip / span))
    x = x0 + pad
    for i, node in enumerate(route):
        if i == 0:               fc = FEED_COLOR.get(node, '#888'); tc = 'white'
        elif i == n - 1:         fc = '#229954';                    tc = 'white'
        elif node in NODE_MAIN:  fc = '#34495e';                    tc = 'white'
        else:                    fc = '#aed6f1';                    tc = '#1c2833'
        axc.add_patch(_Rect((x, yc - ch/2), chip, ch, fc=fc, ec='white',
                            lw=0.3, zorder=4, alpha=alpha))
        axc.text(x + chip/2, yc, _ABBR.get(node, node), ha='center', va='center',
                 fontsize=fs, color=tc, fontweight='bold', zorder=5, clip_on=False,
                 alpha=min(1.0, alpha + 0.25))
        if i < n - 1:
            axc.annotate('', xy=(x + chip + gap*0.9, yc),
                         xytext=(x + chip + gap*0.1, yc),
                         arrowprops=dict(arrowstyle='-|>', lw=0.4, color='#566573',
                                         alpha=alpha), zorder=4)
        x += chip + gap
    return fs

SCEN_COL = {'elec': '#2471a3', 'base': '#7f8c8d'}   # winning-scenario band colors
def merge_comb(segs):
    """Merge consecutive regimes sharing the same (winning-scenario, main-conv);
    representative pathway = the widest sub-regime."""
    out = []
    for (item, x0, x1) in segs:
        scen, key = item
        mc = _MAIN.get(key[0], key[0])      # main conversion (from tech code)
        if out and out[-1][0] == scen and out[-1][1] == mc:
            _, _, pk, px0, px1 = out[-1]
            if (x1 - x0) > (px1 - px0): pk = key
            out[-1] = (scen, mc, pk, px0, x1)
        else:
            out.append((scen, mc, key, x0, x1))
    return out

def config_aligned(axc, comb, xgrid, xlabel, thr_fmt):
    """x-aligned configuration of the COMBINED optimum sharing the panel's x-axis.
    One lane per feed. At each x the better of {baseline, electrified} is taken;
    a top band marks the winning scenario and the chips below show that scenario's
    optimal process pathway. Dividers: red = scenario crossover, gray = path switch."""
    nb = len(FEEDS); span = xgrid[-1] - xgrid[0]
    axc.set_xlim(xgrid[0], xgrid[-1]); axc.set_ylim(0, nb)
    for bi, f in enumerate(FEEDS):
        ylo = nb - 1 - bi
        if bi > 0:
            axc.axhline(ylo + 1, color='#aeb6bf', lw=0.6, zorder=2)
        merged = merge_comb(regimes(comb[f], xgrid))
        for j, (scen, mc, key, x0, x1) in enumerate(merged):
            # winning-scenario band (top of lane)
            axc.add_patch(_Rect((x0, ylo + 0.84), x1 - x0, 0.13,
                                fc=SCEN_COL[scen], ec='white', lw=0.3, zorder=5))
            if (x1 - x0) > 0.085 * span:
                axc.text((x0 + x1) / 2, ylo + 0.905, scen, fontsize=3.3,
                         color='white', ha='center', va='center',
                         fontweight='bold', zorder=6)
            # optimal pathway block chain(s) below band — pyrolysis = 2 tiers
            ch = decode_routes(f, *key)
            oilc, gasc = ch.get('oil'), ch.get('gas')
            if oilc and gasc:
                draw_chain_seg(axc, oilc, x0, x1, ylo + 0.42, 0.40, span)  # upper: oil
                draw_chain_seg(axc, gasc, x0, x1, ylo + 0.02, 0.40, span)  # lower: gas
            else:
                draw_chain_seg(axc, oilc or gasc, x0, x1, ylo + 0.12, 0.60, span)
            if j < len(merged) - 1:
                nxt = merged[j + 1][0]
                col = '#c0392b' if nxt != scen else '#909497'
                axc.plot([x1, x1], [ylo + 0.02, ylo + 0.97], color=col,
                         lw=0.8 if nxt != scen else 0.6,
                         ls=(0, (2, 1.4)), zorder=6)
                axc.text(x1, ylo + 1.0, thr_fmt.format(x1), fontsize=3.5,
                         color=col, ha='center', va='top', zorder=7)
    axc.set_yticks([nb - 1 - i + 0.5 for i in range(nb)])
    axc.set_yticklabels(FEEDS, fontsize=6.5, fontweight='bold')
    for t, f in zip(axc.get_yticklabels(), FEEDS):
        t.set_color(FEED_COLOR[f])
    axc.set_xlabel(xlabel)
    axc.tick_params(labelsize=7)

fig = plt.figure(figsize=(FIG_W, FIG_W * 1.30))
outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.17,
                         top=0.985, bottom=0.075, left=0.068, right=0.995)
gA = outer[0].subgridspec(2, 1, height_ratios=[1.0, 2.15], hspace=0.06)
axA  = fig.add_subplot(gA[0])
cfgA = fig.add_subplot(gA[1], sharex=axA)
gB = outer[1].subgridspec(2, 1, height_ratios=[1.0, 2.15], hspace=0.06)
axB  = fig.add_subplot(gB[0])
cfgB = fig.add_subplot(gB[1], sharex=axB)

# ── Panel A: x=P_elec, y=UP ──
for f in FEEDS:
    axA.plot(P_grid, econ_UP_base[f], color=FEED_COLOR[f], linewidth=LW,
             linestyle='-', label=f'{f} (baseline)')
    axA.plot(P_grid, econ_UP_elec[f], color=FEED_COLOR[f], linewidth=LW,
             linestyle='--', label=f'{f} (electrified)')
    cx = econ_crossovers[f]
    if cx is not None:
        axA.plot(cx[0], cx[1], 'o', markersize=5,
                 markerfacecolor=FEED_COLOR[f], markeredgecolor='none',
                 zorder=10)
        axA.annotate(f'{f}\n{cx[0]:.3f}', xy=cx,
                     xytext=(cx[0]+0.003, cx[1]+0.04),
                     fontsize=5.5, fontweight='bold',
                     color=FEED_COLOR[f])

axA.axhline(-0.42, color='red', linestyle=':', linewidth=1.0, alpha=0.7)
axA.text(0.001, -0.41, 'UP threshold (−0.42)', color='red', fontsize=6)
axA.set_ylabel('UP at Econ-optimal pathway (\\$/kg)')
axA.text(0.012, 0.97, '(a)', transform=axA.transAxes, fontsize=9,
         fontweight='bold', va='top', ha='left')
axA.grid(linestyle=':', alpha=0.4)
axA.set_xlim(P_grid[0], P_grid[-1])
axA.tick_params(labelbottom=False)

# ── Panel B: x=CI_elec, y=CR ──
for f in FEEDS:
    axB.plot(CI_grid, env_CR_base[f], color=FEED_COLOR[f], linewidth=LW,
             linestyle='-', label=f'{f} (baseline)')
    axB.plot(CI_grid, env_CR_elec[f], color=FEED_COLOR[f], linewidth=LW,
             linestyle='--', label=f'{f} (electrified)')
    cx = env_crossovers[f]
    if cx is not None:
        axB.plot(cx[0], cx[1], 'o', markersize=5,
                 markerfacecolor=FEED_COLOR[f], markeredgecolor='none',
                 zorder=10)
        axB.annotate(f'{f}\n{cx[0]:.0f}', xy=cx,
                     xytext=(cx[0]+20, cx[1]+0.2),
                     fontsize=5.5, fontweight='bold',
                     color=FEED_COLOR[f])

axB.axhline(0, color='red', linestyle=':', linewidth=1.0, alpha=0.7)
axB.text(10, 0.02, 'CR sign-flip', color='red', fontsize=6, va='bottom')
axB.set_ylabel('UCR at Env-optimal pathway (kgCO$_2$/kg)')
axB.text(0.012, 0.97, '(b)', transform=axB.transAxes, fontsize=9,
         fontweight='bold', va='top', ha='left')
axB.grid(linestyle=':', alpha=0.4)
axB.set_xlim(CI_grid[0], CI_grid[-1])
axB.tick_params(labelbottom=False)

# ── Combined optimum per feed: at each x take the better of {base, elec} ──
econ_comb, env_comb = {}, {}
for f in FEEDS:
    econ_comb[f] = [('elec', econ_path_elec[f][i]) if econ_UP_elec[f][i] >= econ_UP_base[f][i]
                    else ('base', econ_path_base[f][i]) for i in range(len(P_grid))]
    env_comb[f]  = [('elec', env_path_elec[f][i]) if env_CR_elec[f][i] >= env_CR_base[f][i]
                    else ('base', env_path_base[f][i]) for i in range(len(CI_grid))]

# ── x-aligned configuration of the combined optimum (shares x with panel above) ──
config_aligned(cfgA, econ_comb, P_grid,
               'Electricity price P$_{elec}$ (\\$/kWh)', '{:.3f}')
config_aligned(cfgB, env_comb, CI_grid,
               'Electricity CI$_{elec}$ (gCO$_2$/kWh)', '{:.0f}')

# Feed legend (lines, from Panel A)
from matplotlib.patches import Patch
_h, _l = axA.get_legend_handles_labels()
fig.legend(_h, _l, loc='lower center', bbox_to_anchor=(0.24, 0.0), ncol=5,
           fontsize=4.6, framealpha=0.95, columnspacing=0.7,
           handlelength=1.4, title='Feed  (— baseline / -- electrified)',
           title_fontsize=5.3)
# Block-flow node legend + winning-scenario band
node_handles = [Patch(fc='#888888', ec='white', label='Feedstock (color = plastic)'),
                Patch(fc='#34495e', ec='white', label='Main conversion'),
                Patch(fc='#aed6f1', ec='white', label='Intermediate unit'),
                Patch(fc='#229954', ec='white', label='Product')]
fig.legend(node_handles, [h.get_label() for h in node_handles],
           loc='lower center', bbox_to_anchor=(0.58, 0.0), ncol=2,
           fontsize=4.8, framealpha=0.95, columnspacing=0.7,
           title='Process-block flow', title_fontsize=5.3)
scen_handles = [Patch(fc=SCEN_COL['elec'], ec='white', label='Electrified optimal'),
                Patch(fc=SCEN_COL['base'], ec='white', label='Baseline optimal')]
fig.legend(scen_handles, [h.get_label() for h in scen_handles],
           loc='lower center', bbox_to_anchor=(0.84, 0.0), ncol=1,
           fontsize=4.8, framealpha=0.95, columnspacing=0.7,
           title='Winning scenario (top band)', title_fontsize=5.3)

FIGDIR = config.FIG_OUT
out_png = os.path.join(FIGDIR, 'FigS7_9_crossover.png')
plt.savefig(out_png, dpi=400, bbox_inches='tight')
plt.close()
print(f'\nSaved: {out_png}')

# ── Fig.9 optimal-pathway names (convention: Feed_Main-tech_Product_#N) ──
import re as _re
from collections import defaultdict as _dd
FIGTECH = {'HT': 'H-PY', 'LT': 'L-PY', 'OG': 'O-GS', 'SG': 'S-GS', 'HG': 'HG'}
def _pcat(p):
    p = _re.sub(r'_\d+$', '', str(p).strip())
    return _PMAP.get(p.upper(), p)
_fam = _dd(list)                       # (feed, tech, prod-cat) -> ordered [(P1,P2)]
for _i in range(len(df)):              # build from master 'All pathways' (authoritative product label)
    fd = str(df.iloc[_i, 0]).strip(); tc = str(df.iloc[_i, 1]).strip()
    if tc not in _MAIN:
        continue
    _fam[(fd, tc, _pcat(df.iloc[_i, 4]))].append(
        (str(df.iloc[_i, 2]).strip(), str(df.iloc[_i, 3]).strip()))
def _name(feed, key):
    tech, p1, p2, prod = key
    pc = _pcat(prod); lst = _fam.get((feed, tech, pc), [])
    try:    n = lst.index((str(p1).strip(), str(p2).strip())) + 1
    except ValueError: n = '?'
    return f'{feed}_{FIGTECH.get(tech, tech)}_{pc}_#{n}'
print('\n=== Fig.9 optimal-pathway names (Feed_Main-tech_Product_#N) ===')
for panel, comb, grid, unit in [('(a) Econ-optimal', econ_comb, P_grid, '$/kWh'),
                                 ('(b) Env-optimal',  env_comb,  CI_grid, 'gCO2/kWh')]:
    print(panel)
    for f in FEEDS:
        for scen, mc, key, x0, x1 in merge_comb(regimes(comb[f], grid)):
            print(f'   {f:<4} [{scen:<4}] {x0:.3f}-{x1:.3f} {unit:<9}: {_name(f, key)}')
print('\nDone.')
