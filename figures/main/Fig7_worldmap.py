import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
"""Fig. 6 - regional and temporal roadmap.

Derived from m3_step5_RC_world_map_periods_INCIN.py, which produced Fig. 12 of
the EES manuscript. The model is untouched; only the figure width and a few
in-figure labels changed. The figure-only edits are listed in the README (Fig. 7 notes).
"""
import sys, io, os
os.environ['PYTHONUTF8'] = '1'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import geopandas as gpd

from ref_elec_coupling import ref_price, ref_ci

BASE = config.INTERMEDIATE
MASS = 50e6
FEEDS = ['PE','PET','PP','PS','PVC']

# ── Incineration credit regionalization (2026-06-08 fix) ──────────────
# Incineration benchmark (E_incin_net = "feedstock credit", Eqn 5-i) is a grid-
# dependent net electricity producer: E_incin_net(feed) = a - b*CI_elec[kgCO2/kWh].
# CR0 (provided) was computed at base grid CI = 369 gCO2/kWh. For a region with
# grid CI = CIg, the incineration credit grows (dirtier grid -> bigger displacement
# credit -> lower incineration net -> harder benchmark), lowering CR by:
#   dCR_incin(feed) = -b(feed) * (CIg - 369)/1000   [kgCO2/kg]
# Source: utility-electrification workbook, BAU(incineration) section.
# b(feed) = incineration recovered electricity [kWh/kg] (verified vs Carbon credit @CI=100).
INCIN_B = {'PE':2.221266, 'PET':1.056216, 'PP':2.105206, 'PS':1.793923, 'PVC':0.687119}
CI_BASE_INCIN = 369.0
# Policy reduction fractions implied by original CR_tgt schedule / 2.69 baseline
# (2030 NDC 50%, 2050 NZE 90%) — used to regionalize CR_target consistently.
RED_PCT = {'2026':0.186, '2030':0.502, '2040':0.703, '2050':0.900}

REGION_W = {
    # ── Primary-source verified (Phase 1 dataset) ──
    'Global/EU': {'PE':0.406, 'PP':0.264, 'PET':0.105, 'PVC':0.137, 'PS':0.088},  # PlasticsEurope 2018
    'U.S.':      {'PE':0.562, 'PP':0.200, 'PET':0.146, 'PVC':0.017, 'PS':0.075},  # Milbrandt 2022 (NREL)
    'China':     {'PE':0.346, 'PP':0.250, 'PET':0.139, 'PVC':0.204, 'PS':0.061},  # Jiang 2020 + Park 2023
    'Japan':     {'PE':0.418, 'PP':0.278, 'PET':0.175, 'PVC':0.060, 'PS':0.069},  # PWMI 2020
    'Korea':     {'PE':0.449, 'PP':0.263, 'PET':0.139, 'PVC':0.090, 'PS':0.059},  # Park 2023 KSEE
    'India':     {'PE':0.709, 'PP':0.105, 'PET':0.092, 'PVC':0.044, 'PS':0.051},  # CPCB 2015 via Goel 2019
    # ── Web-verified Phase 2 dataset (2026-05-31 search) ──
    'Indonesia': {'PE':0.358, 'PP':0.326, 'PET':0.126, 'PVC':0.116, 'PS':0.074},  # ✓ IPEN 2021 production share, normalized
    'Canada':    {'PE':0.470, 'PP':0.210, 'PET':0.170, 'PVC':0.080, 'PS':0.070},  # ◐ Statistics Canada 2019 (recycled flows) + CPP 2021
    'Brazil':    {'PE':0.420, 'PP':0.200, 'PET':0.130, 'PVC':0.180, 'PS':0.070},  # ◐ Mordor/Statista 2024 + construction-driven PVC
    'Australia': {'PE':0.390, 'PP':0.270, 'PET':0.160, 'PVC':0.080, 'PS':0.100},  # ◐ APFF 2020-21 + APCO 2023-24 (PVC halved)
    'South Africa': {'PE':0.420, 'PP':0.220, 'PET':0.160, 'PVC':0.120, 'PS':0.080},  # ◐ Plastics SA 2023 + SA Plastics Pact (LDPE-dominant)
}

# Verification tier (for visual cue on figure)
REGION_VERIFIED = {
    'Global/EU': 'primary',  'U.S.': 'primary',  'China': 'primary',
    'Japan': 'primary',  'Korea': 'primary',  'India': 'primary',
    'Indonesia': 'primary',  # IPEN 2021 verified
    'Canada': 'partial',  'Brazil': 'partial',  'Australia': 'partial',
    'South Africa': 'partial',
}

# 2040 = linear interpolation between 2030 and 2050
PERIODS = ['2026', '2030', '2040', '2050']

# Period-specific params (region-independent except CI_grid and τ-2026)
PERIOD_BASE = {
    '2026': dict(P_PPA=0.040, CI_PPA=30, H2cx=0.7, CR_tgt=0.50),  # Glasgow Pact pre-2030 ambition
    '2030': dict(P_PPA=0.030, CI_PPA=20, H2cx=0.5, CR_tgt=1.35),  # NDC 50% incin reduction
    '2040': dict(P_PPA=0.025, CI_PPA=15, H2cx=0.4, CR_tgt=1.89),  # mid-trajectory
    '2050': dict(P_PPA=0.020, CI_PPA=10, H2cx=0.3, CR_tgt=2.42),  # NZE 90% incin reduction
}

REGION_TAU_2026 = {
    'Global/EU': 0.080, 'U.S.': 0.000, 'China': 0.010,
    'Japan': 0.040, 'Korea': 0.030, 'India': 0.000,
    'Brazil':       0.000,    # no national carbon tax
    'Australia':    0.000,    # ETS pending, no current tax
    'Indonesia':    0.000,    # nascent carbon tax (token)
    'Canada':       0.065,    # federal carbon levy ~CAD 80/t ≈ $65/tCO2
    'South Africa': 0.012,    # SA Carbon Tax Act (2019), Tier 1 ~R190/t ≈ $10/t after rebates
}
TAU_BY_PERIOD = {'2030': 0.10, '2040': 0.15, '2050': 0.20}

def tau_for(region, period):
    if period == '2026':
        return REGION_TAU_2026[region]
    return TAU_BY_PERIOD[period]

# Grid CI roadmap (gCO2/kWh) — IEA STEPS/NZE midpoints, regional estimates
REGION_GRID_CI = {
    'Global/EU':    {'2026': 200, '2030': 130, '2040':  80, '2050':  30},
    'U.S.':         {'2026': 230, '2030': 200, '2040': 125, '2050':  50},
    'China':        {'2026': 510, '2030': 460, '2040': 285, '2050': 110},
    'Japan':        {'2026': 290, '2030': 250, '2040': 150, '2050':  50},
    'Korea':        {'2026': 330, '2030': 280, '2040': 175, '2050':  70},
    'India':        {'2026': 510, '2030': 460, '2040': 295, '2050': 130},
    'Brazil':       {'2026':  90, '2030':  60, '2040':  40, '2050':  20},   # hydro-heavy
    'Australia':    {'2026': 580, '2030': 420, '2040': 200, '2050':  40},   # rapid coal→renew
    'Indonesia':    {'2026': 720, '2030': 600, '2040': 350, '2050':  90},   # coal-heavy
    'Canada':       {'2026': 130, '2030':  90, '2040':  50, '2050':  20},   # hydro-clean
    'South Africa': {'2026': 870, '2030': 720, '2040': 380, '2050': 110},   # Eskom coal-dominant, slow transition
}

# α_max(region, period) — maximum feasible PPA adoption ratio for a new industrial facility.
# Constructed from: (i) grid renewable share [IRENA 2024 + IEA STEPS/NZE], (ii) corporate PPA
# market depth [BloombergNEF Corp. Energy Market Outlook 2024], (iii) RE100 Annual Disclosure 2024.
# Source citations to add in paper: IEA Renewables 2024/2025, IEA WEO 2024 NZE/STEPS,
# BNEF Corp. Energy Outlook 2024, RE100 2024 Disclosure, IRENA Renewable Capacity Statistics 2024.
REGION_ALPHA_MAX = {
    'Global/EU':    {'2026': 0.45, '2030': 0.65, '2040': 0.85, '2050': 0.95},  # RePowerEU + Fit-for-55
    'U.S.':         {'2026': 0.30, '2030': 0.50, '2040': 0.75, '2050': 0.92},  # IRA-driven, BNEF Americas lead
    'China':        {'2026': 0.35, '2030': 0.55, '2040': 0.80, '2050': 0.95},  # RE100 CN 59%, STEPS
    'Japan':        {'2026': 0.25, '2030': 0.45, '2040': 0.70, '2050': 0.90},  # RE100 JP 36%, 6th Energy Plan
    'Korea':        {'2026': 0.15, '2030': 0.35, '2040': 0.65, '2050': 0.88},  # 10th Basic Plan (2036)
    'India':        {'2026': 0.25, '2030': 0.45, '2040': 0.70, '2050': 0.90},  # RE100 IN 39%, 500 GW target
    'Brazil':       {'2026': 0.85, '2030': 0.92, '2040': 0.96, '2050': 0.99},  # hydro 89%, EPE 2050
    'Australia':    {'2026': 0.35, '2030': 0.60, '2040': 0.85, '2050': 0.95},  # IEA AUS 82% by 2030
    'Indonesia':    {'2026': 0.18, '2030': 0.30, '2040': 0.55, '2050': 0.85},  # NDC + JETP USD 21.5 B
    'Canada':       {'2026': 0.65, '2030': 0.80, '2040': 0.90, '2050': 0.97},  # hydro 60% + nuclear
    'South Africa': {'2026': 0.12, '2030': 0.25, '2040': 0.55, '2050': 0.85},  # JETP USD 8.5 B, Eskom slow
}

REGION_GEO = {
    # Northern hemisphere (top row of insets)
    'Canada':       dict(point=(-105, 58), inset=(-175,  62), label_pos='above'),
    'U.S.':         dict(point=(-100, 38), inset=(-130,  62), label_pos='below'),
    'Global/EU':    dict(point=(  10, 52), inset=( -30,  60), label_pos='above'),
    'China':        dict(point=( 100, 33), inset=(  35,  62), label_pos='above'),
    'Korea':        dict(point=( 126, 38), inset=(  85,  62), label_pos='above'),
    'Japan':        dict(point=( 142, 36), inset=( 140,  62), label_pos='above'),
    # Southern hemisphere / equatorial (bottom row of insets)
    'Brazil':       dict(point=( -55,-12), inset=(-100, -58), label_pos='above'),
    'South Africa': dict(point=(  25,-30), inset=( -45, -58), label_pos='above'),
    'India':        dict(point=(  78, 18), inset=(  10, -58), label_pos='below'),
    'Indonesia':    dict(point=( 118, -3), inset=(  70, -58), label_pos='below'),
    'Australia':    dict(point=( 134,-25), inset=( 130, -58), label_pos='above'),
}

P_GRID_BASE = 0.080
P_ELEC_BASE, CI_ELEC_BASE_g = 0.08, 369.0
P_FH_BASE,  CI_FH_BASE_g  = 0.03,     338.4
P_LP_BASE,  CI_LP_BASE_g  = 0.00684,  190.08
P_HP_BASE,  CI_HP_BASE_g  = 0.009,    190.08
P_H2_BASE,  CI_H2_BASE_kg = 2.0,      7.0
ETA_HEATER, ETA_BOILER = 0.95, 0.99

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
INCIN_B_ARR = np.array([INCIN_B[f] for f in feed_arr])  # per-pathway b [kWh/kg]

def coln(c): return pd.to_numeric(df.iloc[:, c], errors='coerce').fillna(0).values
ELEC_M = coln(7);  FH_M = coln(8)
REF_M  = np.array([coln(10+k) for k in range(5)])
LP_M, HP_M = coln(15), coln(16); H2_M = coln(20)
ELEC_E = coln(31); FH_E = coln(32)
REF_E  = np.array([coln(33+k) for k in range(5)])
LP_E, HP_E = coln(38), coln(39); H2_E = coln(42)

P_REF_BASE    = np.array([ref_price(k+1, P_ELEC_BASE) for k in range(5)])
CI_REF_BASE_g = np.array([ref_ci(k+1, CI_ELEC_BASE_g) for k in range(5)])

def transform_baseline(P_elec, CI_elec_g):
    r_e_c, r_e_ci = P_elec/P_ELEC_BASE, CI_elec_g/CI_ELEC_BASE_g
    r_ref_c  = np.array([ref_price(k+1, P_elec) / P_REF_BASE[k] for k in range(5)])
    r_ref_ci = np.array([ref_ci(k+1, CI_elec_g) / CI_REF_BASE_g[k] for k in range(5)])
    new_ELEC_M = ELEC_M * r_e_c; new_ELEC_E = ELEC_E * r_e_ci
    new_REF_M_sum = REF_M.T @ r_ref_c; new_REF_E_sum = REF_E.T @ r_ref_ci
    dC_M  = (new_ELEC_M - ELEC_M) + (new_REF_M_sum - REF_M.sum(axis=0))
    dE_kg = (new_ELEC_E - ELEC_E) + (new_REF_E_sum - REF_E.sum(axis=0))
    UP = UP0 - dC_M * 1e6/MASS
    CR = CR0 - dE_kg / MASS
    return UP, CR

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

def evaluate_period(region, period, scenario):
    """Evaluate region/period/scenario under α ≤ α_max(region, period) exogenous constraint."""
    w = REGION_W[region]
    pb = PERIOD_BASE[period]
    tau = tau_for(region, period)
    CIg = REGION_GRID_CI[region][period]
    CIp = pb['CI_PPA']; Pp = pb['P_PPA']; H2cx = pb['H2cx']; CRt = pb['CR_tgt']
    a_max = REGION_ALPHA_MAX[region][period]  # NEW: market-availability cap

    # ── Incineration credit regionalization (uses GRID CIg, independent of α) ──
    # Incineration is BAU and displaces the regional GRID (not the facility PPA),
    # so its credit scales with CIg, constant across α.
    incin_shift = INCIN_B_ARR * (CIg - CI_BASE_INCIN) / 1000.0   # per-pathway ΔCR (subtract)
    b_w = sum(w[f] * INCIN_B[f] for f in FEEDS)
    CRt = CRt - RED_PCT[period] * b_w * (CIg - CI_BASE_INCIN) / 1000.0  # regionalized target

    wUP = np.zeros_like(alpha_grid); wCR = np.zeros_like(alpha_grid)
    for i, a in enumerate(alpha_grid):
        P_elec    = P_GRID_BASE + a * (Pp  - P_GRID_BASE)
        CI_elec_g = CIg          + a * (CIp - CIg)
        if scenario == 'elec':
            UP_x, CR_x = transform_electrification(P_elec, CI_elec_g, H2cx)
        else:
            UP_x, CR_x = transform_baseline(P_elec, CI_elec_g)
        CR_x = CR_x - incin_shift                       # regionalize incineration benchmark
        UP_net = UP_x + tau * CR_x
        UPb, CRb = econ_opt_per_feed(UP_net, CR_x)
        wUP[i] = sum(w[f] * UPb[f] for f in FEEDS)
        wCR[i] = sum(w[f] * CRb[f] for f in FEEDS)

    # Apply α_max constraint: only allow α ∈ [0, α_max]
    alpha_allowed = alpha_grid <= a_max
    feas_env = wCR >= CRt
    feas = feas_env & alpha_allowed
    wUP_feas = np.where(feas, wUP, -np.inf)
    if not np.isfinite(wUP_feas).any():
        return np.nan, np.nan
    i_opt = np.argmax(wUP_feas)
    return wUP[i_opt], wCR[i_opt]

print('Computing time series per region...\n')
curves = {}
header_b = ' '.join(f'{p}_B'.rjust(8) for p in PERIODS)
header_e = ' '.join(f'{p}_E'.rjust(8) for p in PERIODS)
print(f'{"Region":>10}  {header_b}   {header_e}')

for region in REGION_W:
    b_arr = [evaluate_period(region, p, 'base')[0] for p in PERIODS]
    e_arr = [evaluate_period(region, p, 'elec')[0] for p in PERIODS]
    curves[region] = dict(b=b_arr, e=e_arr)
    def fmt(x): return f'{x:+.3f}' if not np.isnan(x) else ' infeas'
    print(f'{region:>10}  ' +
          ' '.join(f'{fmt(v):>8}' for v in b_arr) + '   ' +
          ' '.join(f'{fmt(v):>8}' for v in e_arr))

# CSV
rows = []
for region in REGION_W:
    for sc, key in [('Baseline','b'), ('Electrified','e')]:
        for k, p in enumerate(PERIODS):
            rows.append(dict(Region=region, Scenario=sc, Period=p,
                             wUP_net=curves[region][key][k]))
pd.DataFrame(rows).to_csv(os.path.join(BASE, 'm3_step5_RC_world_periods_INCIN.csv'),
                          index=False, float_format='%.4f')

# ─── World map figure ─────────────────────────────────────────────
print('\nBuilding world map...')
import matplotlib.colors as mcolors
_ADMIN = config.NE_ADMIN0   # vendored copy, no network access
world = gpd.read_file(_ADMIN)
GEN = {
 'China': 60.0, 'United States of America': 42.0, 'India': 26.0, 'Brazil': 11.3,
 'Indonesia': 9.1, 'Russia': 8.0, 'Japan': 7.9, 'Germany': 6.3, 'Mexico': 5.9,
 'Turkey': 5.2, 'United Kingdom': 4.9, 'France': 4.8, 'Thailand': 4.8, 'South Korea': 4.5,
 'Italy': 4.0, 'Saudi Arabia': 3.5, 'Spain': 3.5, 'Canada': 3.3, 'Pakistan': 3.3,
 'Vietnam': 3.1, 'Iran': 3.0, 'Egypt': 3.0, 'Philippines': 2.7, 'Argentina': 2.5,
 'Nigeria': 2.5, 'Poland': 2.5, 'Australia': 2.5, 'Malaysia': 2.4, 'South Africa': 2.0,
 'Colombia': 1.5, 'Ukraine': 1.5, 'Netherlands': 1.5, 'Venezuela': 1.2, 'Iraq': 1.2,
 'United Arab Emirates': 1.2, 'Belgium': 1.2, 'Algeria': 1.0, 'Chile': 1.0, 'Peru': 1.0,
 'Romania': 1.0, 'Morocco': 0.9, 'Israel': 0.9, 'Kazakhstan': 0.8, 'Bangladesh': 0.8,
 'Sweden': 0.8, 'Greece': 0.8, 'Czechia': 0.8, 'Austria': 0.8, 'Switzerland': 0.7,
 'Portugal': 0.7, 'Hungary': 0.6, 'Cuba': 0.5, 'Ecuador': 0.5, 'Tunisia': 0.5,
 'Norway': 0.6, 'Denmark': 0.6, 'Finland': 0.6, 'New Zealand': 0.6, 'Ireland': 0.5,
}
_ALIAS = {'United States of America': ['United States', 'United States of America'],
          'Czechia': ['Czech Rep.', 'Czech Republic'], 'South Korea': ['Korea']}
def _gen_for(name):
    if name in GEN: return GEN[name]
    for k, alist in _ALIAS.items():
        if name in alist and k in GEN: return GEN[k]
    return np.nan
world['gen'] = world['NAME'].map(_gen_for)
gen_cmap = mcolors.LinearSegmentedColormap.from_list(
    'teal_ref', ['#d6f2ea', '#b6e6d8', '#76dad4', '#37ccd9', '#2aabc8', '#2391a4'])
gen_norm = mcolors.PowerNorm(gamma=0.5, vmin=0.5, vmax=60)

import matplotlib as mpl
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
MM = 1/25.4
FIG_W = 174.0 * MM
fig = plt.figure(figsize=(FIG_W, FIG_W * 0.523))   # sized to the map, no letterbox
ax_map = fig.add_axes([0.02, 0.145, 0.96, 0.840])   # leaves a colour-bar strip below
ax_map.set_facecolor('white')                                          # ocean = white
world[world['gen'].isna()].plot(ax=ax_map, color='#e3e0d8', edgecolor='#b8b0a0', linewidth=0.4)
world[world['gen'].notna()].plot(ax=ax_map, column='gen', cmap=gen_cmap, norm=gen_norm,
                                 edgecolor='#9a9488', linewidth=0.4)
ax_map.set_xlim(-180, 180); ax_map.set_ylim(-70, 95)
ax_map.set_xticks([]); ax_map.set_yticks([])
for spine in ax_map.spines.values():
    spine.set_edgecolor('#7f8c8d'); spine.set_linewidth(0.5)

# colorbar — moved to the LEFT of the bottom strip
_sm = plt.cm.ScalarMappable(cmap=gen_cmap, norm=gen_norm); _sm.set_array([])
cax = fig.add_axes([0.055, 0.060, 0.24, 0.017])
cb = fig.colorbar(_sm, cax=cax, orientation='horizontal', ticks=[0.5, 2, 5, 10, 20, 40, 60])
cb.ax.set_xticklabels(['0.5', '2', '5', '10', '20', '40', '60'])
cb.set_label('Plastic waste generation (Mt year$^{-1}$)', fontsize=5, fontweight='bold')
cb.ax.tick_params(labelsize=4.2, length=2)
cb.outline.set_linewidth(0.5)

COLOR_B = '#34495e'; COLOR_E = '#e74c3c'

# Plastic composition colors — muted, harmonious palette that reads on the teal map
FEED_COLOR = {
    'PE':  '#4878d0',  # muted blue
    'PP':  '#6acc64',  # muted green
    'PET': '#e8a33d',  # muted amber
    'PVC': '#d65f5f',  # muted red (still flags PVC-heavy regions)
    'PS':  '#956cb4',  # muted purple
}
FEED_ORDER = ['PE', 'PP', 'PET', 'PVC', 'PS']

# One two-line label per region, on a single side of the pie.
# dlon shifts the two regions whose label would otherwise sit under a neighbouring
# inset box (Canada) or under the Australia pie (Indonesia).
LABEL_ADJ = {'Canada': (-22, 'below'), 'U.S.': (0, 'below'),
             'Global/EU': (0, 'below'), 'China': (0, 'above'),
             'Korea': (0, 'above'), 'Japan': (0, 'above'),
             'Brazil': (0, 'above'), 'South Africa': (0, 'above'),
             'India': (0, 'below'), 'Indonesia': (-16, 'below'),
             'Australia': (0, 'above')}
for region, geo in REGION_GEO.items():
    px, py = geo['point']
    pvc_pct = REGION_W[region]['PVC'] * 100
    dlon, side = LABEL_ADJ.get(region, (0, geo['label_pos']))
    above = side == 'above'
    ax_map.annotate(f'{region}\nPVC {pvc_pct:.1f}%', (px, py),
                     xytext=(px + dlon, py + (11 if above else -11)),
                     ha='center', va='bottom' if above else 'top',
                     fontsize=4.6, fontweight='bold', linespacing=1.25,
                     color='#c0392b' if pvc_pct >= 9 else '#1c2833', zorder=11)

INSET_W = 35; INSET_H = 25
PERIOD_X = np.arange(len(PERIODS))

all_vals = []
for r in curves:
    all_vals.extend([v for v in curves[r]['b'] if not np.isnan(v)])
    all_vals.extend([v for v in curves[r]['e'] if not np.isnan(v)])
y_min = min(all_vals) - 0.05
y_max = max(all_vals) + 0.10

def data_to_fig(ax, x, y):
    disp = ax.transData.transform((x, y))
    inv = fig.transFigure.inverted()
    return inv.transform(disp)

PIE_RADIUS_DEG = 5  # degrees of "longitude" for pie radius

for region, geo in REGION_GEO.items():
    px, py = geo['point']
    ix, iy = geo['inset']

    cx, cy = ix + INSET_W/2, iy + INSET_H/2
    ax_map.plot([px, cx], [py, cy],
                 color='#7f8c8d', linewidth=1.1, linestyle='-',
                 alpha=0.6, zorder=5)

    # ─── Pie chart marker at region location (physically square box → no empty margin) ───
    pcx, pcy = data_to_fig(ax_map, px, py)
    _W_in, _H_in = fig.get_size_inches()
    PIE_D = 0.050                      # pie diameter as fraction of figure width
    pie_w = PIE_D
    pie_h = PIE_D * (_W_in / _H_in)    # equal physical height
    pie_ax = fig.add_axes([pcx - pie_w/2, pcy - pie_h/2, pie_w, pie_h])
    pie_ax.set_facecolor('none')
    w_region = REGION_W[region]
    sizes = [w_region[f] for f in FEED_ORDER]
    colors = [FEED_COLOR[f] for f in FEED_ORDER]
    wedges, _ = pie_ax.pie(sizes, colors=colors, startangle=90,
                            wedgeprops=dict(edgecolor='white', linewidth=1.4))
    pie_ax.set_aspect('equal')
    pie_ax.set_zorder(10)
    # Outer ring: solid for primary-source, dashed for partial-verified
    from matplotlib.patches import Circle
    ring_style = '-' if REGION_VERIFIED.get(region) == 'primary' else '--'
    pie_ax.add_patch(Circle((0, 0), 1.04, fill=False, edgecolor='#1c2833',
                              linewidth=1.6, linestyle=ring_style, zorder=20))

    # PVC share is carried by the two-line region label on the map

    rect = FancyBboxPatch((ix, iy), INSET_W, INSET_H,
                          boxstyle='round,pad=0.5',
                          facecolor='white', edgecolor='#1c2833',
                          linewidth=0.7, alpha=0.97, zorder=6)
    ax_map.add_patch(rect)

    fig_x0, fig_y0 = data_to_fig(ax_map, ix, iy)
    fig_x1, fig_y1 = data_to_fig(ax_map, ix + INSET_W, iy + INSET_H)
    inset = fig.add_axes([fig_x0, fig_y0, fig_x1 - fig_x0, fig_y1 - fig_y0])
    for _sp in inset.spines.values():
        _sp.set_linewidth(0.5)

    b = np.array(curves[region]['b'], dtype=float)
    e = np.array(curves[region]['e'], dtype=float)

    # Find crossover year (first period where e >= b)
    valid = ~(np.isnan(b) | np.isnan(e))
    if valid.any():
        inset.fill_between(PERIOD_X[valid], b[valid], e[valid],
                            where=(e[valid] >= b[valid]),
                            color='#27ae60', alpha=0.22, interpolate=True)
        inset.fill_between(PERIOD_X[valid], b[valid], e[valid],
                            where=(e[valid] < b[valid]),
                            color='#34495e', alpha=0.12, interpolate=True)

    inset.plot(PERIOD_X, b, color=COLOR_B, linewidth=1.0, marker='o',
                markersize=2.5, markerfacecolor=COLOR_B,
                markeredgecolor='white', markeredgewidth=0.5,
                label='Baseline')
    inset.plot(PERIOD_X, e, color=COLOR_E, linewidth=1.0, marker='s',
                markersize=2.5, markerfacecolor=COLOR_E,
                markeredgecolor='white', markeredgewidth=0.5,
                linestyle='--', label='Electrified')

    # Mark each infeasible period individually (baseline) — not contiguous
    nan_b = np.where(np.isnan(b))[0]
    for j in nan_b:
        inset.axvspan(j - 0.45, j + 0.45,
                       color='#34495e', alpha=0.18)
        inset.plot(j, y_max - 0.04, marker='X', markersize=4,
                    color='#34495e', markeredgecolor='white',
                    markeredgewidth=0.5, zorder=12)
    # Mark each infeasible period individually (electrification)
    nan_e = np.where(np.isnan(e))[0]
    for j in nan_e:
        inset.axvspan(j - 0.45, j + 0.45,
                       color='#e74c3c', alpha=0.15)
        inset.plot(j, y_max - 0.10, marker='X', markersize=4,
                    color='#e74c3c', markeredgecolor='white',
                    markeredgewidth=1.0, zorder=12)
    # Legend annotation in one inset only (top-left = Canada)
    if region == 'Canada':
        inset.text(0.03, 0.96, 'X = infeasible',
                    transform=inset.transAxes, fontsize=4,
                    color='gray', ha='left', va='top', style='italic')

    # Annotate Δ2050 (or "Elec only")
    if not (np.isnan(e[-1]) or np.isnan(b[-1])):
        gap = e[-1] - b[-1]
        mid = (e[-1] + b[-1]) / 2
        inset.text(PERIOD_X[-1] - 0.02, mid,
                    f'Δ={gap:+.3f}', fontsize=4, fontweight='bold',
                    color='#27ae60', ha='right', va='center')

    inset.axhline(0, color='black', linewidth=0.6, linestyle=':')
    inset.set_xticks(PERIOD_X)
    inset.set_xticklabels(PERIOD_S := [p for p in PERIODS], fontsize=3.5)
    inset.set_ylim(y_min, y_max)
    inset.tick_params(labelsize=3.5, length=1.5)
    inset.grid(linestyle=':', alpha=0.3, axis='y')

# Scenario legend (bottom left)
legend_handles = [
    Line2D([0], [0], color=COLOR_B, marker='o', markersize=3,
            linewidth=1.0, label='Fossil baseline'),
    Line2D([0], [0], color=COLOR_E, marker='s', markersize=3,
            linewidth=1.0, linestyle='--', label='Electrification'),
    Patch(facecolor='#27ae60', alpha=0.22, edgecolor='none',
            label='Electrification favored'),
    Patch(facecolor='#34495e', alpha=0.18, edgecolor='none',
            label='Fossil baseline infeasible'),
    Patch(facecolor='#e74c3c', alpha=0.15, edgecolor='none',
            label='Electrification infeasible'),
    Patch(facecolor='#e3e0d8', edgecolor='#b8b0a0', linewidth=0.5, label='No data'),
]
plastic_handles = [Patch(facecolor=FEED_COLOR[f], edgecolor='white',
                          linewidth=1.0, label=f) for f in FEED_ORDER]
# Ring style indicators
verify_solid = Line2D([0], [0], color='#1c2833', linestyle='-', linewidth=1.8,
                       label='Primary-source verified')
verify_dashed = Line2D([0], [0], color='#1c2833', linestyle='--', linewidth=1.8,
                        label='Partial / best-estimate')

# ── Scenario + composition legends in the map's LOWER-LEFT (unified Fig.5/6 style) ──
leg_l = ax_map.legend(handles=legend_handles, loc='lower left', bbox_to_anchor=(0.008, 0.175),
                ncol=1, fontsize=4.5, framealpha=0.95, edgecolor='#888', fancybox=False,
                title='Scenario', title_fontsize=5.5, borderaxespad=0,
                handlelength=1.4, handletextpad=0.5, labelspacing=0.28)
leg_l.get_title().set_fontweight('bold'); leg_l.get_title().set_color('black')
ax_map.add_artist(leg_l)
leg_r = ax_map.legend(handles=plastic_handles + [verify_solid, verify_dashed],
                loc='lower left', bbox_to_anchor=(0.008, 0.015), ncol=2, fontsize=4.5,
                framealpha=0.95, edgecolor='#888', fancybox=False,
                title='Waste plastic composition', title_fontsize=5.5, borderaxespad=0,
                handlelength=1.4, handletextpad=0.5, labelspacing=0.32, columnspacing=1.0)
leg_r.get_title().set_fontweight('bold'); leg_r.get_title().set_color('black')
ax_map.add_artist(leg_r)

# (bottom period-roadmap caption removed — to be placed in the figure caption)

out = os.path.join(config.FIG_OUT, 'Fig7_worldmap.png')
plt.savefig(out, dpi=500, bbox_inches='tight', facecolor='white')
plt.close()
print(f'\nSaved: {out}')
print('\nDone.')
