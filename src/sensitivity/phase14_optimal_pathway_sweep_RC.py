import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
"""Phase 14 (Ref-Coupled): Per-feed optimal pathway sweep with Ref-Elec coupling.

Adds Ref1-5 cost AND CI scaling within the (P_elec, CI_elec) sweep loop,
so that refrigeration utilities also respond to grid electricity changes.

Compared to phase14_optimal_pathway_sweep.py:
  + 5 Ref columns scaled per ref_elec_coupling.py
  + Baseline pegged at P_elec=0.08 $/kWh, CI_elec=0.369 kgCO2/kWh
    (matches 'All pathways' computational baseline)
  + Other carriers baselines also aligned to 'All pathways'
"""
import sys, io, os
os.environ['PYTHONUTF8'] = '1'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'pdf.fonttype':42,'ps.fonttype':42,
    'axes.labelsize':9,'axes.labelweight':'bold',
    'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.titlesize':9,
    # subscripts upright Arial, as everywhere else in the paper
    'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial',
    'mathtext.bf':'Arial:bold','mathtext.default':'regular',})
MM=1/25.4; FIG_W=159.2*MM; FIG_W1=79.6*MM

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# Ref-Elec coupling helper
from ref_elec_coupling import ref_price, ref_ci, P_ELEC_BASE, CI_ELEC_BASE

BASE = config.INTERMEDIATE

# ════════════════════════════════════════════════════════════════════
# Load 959 pathways
# ════════════════════════════════════════════════════════════════════
print('Loading 959 pathways…')
df = pd.read_excel(config.MASTER_XLSX,
                   sheet_name='All pathways', header=3)
df['__feed'] = df.iloc[:, 0].astype(str).str.strip()
df['__tech'] = df.iloc[:, 1].astype(str).str.strip()
df['__p1']   = df.iloc[:, 2].astype(str).str.strip()
df['__p2']   = df.iloc[:, 3].astype(str).str.strip()
df = df[df['__feed'].isin(['PE','PET','PP','PS','PVC'])].copy().reset_index(drop=True)
print(f'Pathways: {len(df)}')

up_col = next(c for c in df.columns if isinstance(c, str) and c.lower().startswith('unit profit'))
cr_col = next(c for c in df.columns if isinstance(c, str) and c.lower().startswith('unit co2 reduc'))
UP_base = pd.to_numeric(df[up_col], errors='coerce').values
CR_base = pd.to_numeric(df[cr_col], errors='coerce').values
mask = ~(np.isnan(UP_base) | np.isnan(CR_base))
df = df[mask].reset_index(drop=True)
UP_base = UP_base[mask]; CR_base = CR_base[mask]
N = len(df)
print(f'After dropna: {N}')

def coln(c):
    return pd.to_numeric(df.iloc[:, c], errors='coerce').fillna(0).values

# Carrier columns
C_elec = coln(7);  G_elec = coln(31)
C_FH   = coln(8);  G_FH   = coln(32)
C_LP   = coln(15); G_LP   = coln(38)
C_HP   = coln(16); G_HP   = coln(39)
C_H2   = coln(20); G_H2   = coln(42)

# Ref columns (5 each)
C_REF = [coln(10), coln(11), coln(12), coln(13), coln(14)]
G_REF = [coln(33), coln(34), coln(35), coln(36), coln(37)]

# ════════════════════════════════════════════════════════════════════
# Baselines — aligned with 'All pathways' computational baseline
# ════════════════════════════════════════════════════════════════════
P_elec_kwh_base  = 0.08              # $/kWh
CI_elec_kwh_base = 0.369             # kgCO2/kWh
P_elec_mj_base   = P_elec_kwh_base / 3.6      # $/MJ
CI_elec_mj_base  = CI_elec_kwh_base / 3.6     # kgCO2/MJ

# Other carriers (from the utility-electrification workbook, rows 29-40)
P_FH_base   = 0.03  / 3.6;   CI_FH_base   = 0.3384 / 3.6   # $/MJ, kgCO2/MJ
P_LP_base   = 0.00684 / 3.6; CI_LP_base   = 0.19008 / 3.6
P_HP_base   = 0.009  / 3.6;  CI_HP_base   = 0.19008 / 3.6
P_H2_base   = 2.0;           CI_H2_base   = 7.0

E_H2 = 53.0;  H2_CAPOP = 0.7;  ETA_B = 0.99;  ETA_H = 0.95
MASS = 50e6

# Feeds & techs
feeds = df['__feed'].values
techs = df['__tech'].values
FEED_ORDER = ['PET', 'PP', 'PVC', 'PS', 'PE']   # 2026-08-26 USER order
unique_techs = sorted(set(techs))
print(f'Techs: {unique_techs}')

# 2026-08-26: house palette, taken from the main text (Fig. 3 driver colours)
TECH_COLORS = {'HT': '#b5651d', 'LT': '#e8b93d',
               'SG': '#3f8f5a', 'OG': '#4878d0', 'HG': '#8a8a8a'}
TECH_TO_ID = {t: i for i, t in enumerate(unique_techs)}

# Precompute baseline Ref values (for scaling)
P_REF_base = [ref_price(k, P_elec_kwh_base) for k in range(1, 6)]
CI_REF_base = [ref_ci(k, CI_elec_kwh_base*1000) for k in range(1, 6)]

# ════════════════════════════════════════════════════════════════════
# Sweep grid
# ════════════════════════════════════════════════════════════════════
P_grid  = np.linspace(0.04, 0.20, 33)         # $/kWh
CI_grid = np.linspace(0.03, 0.80, 33)         # kgCO2/kWh
print(f'Grid: {len(P_grid)} × {len(CI_grid)} = {len(P_grid)*len(CI_grid)}')

# Result containers
optimal_tech_id = {f: np.full((len(CI_grid), len(P_grid)), -1, dtype=int)
                   for f in FEED_ORDER}
optimal_UP      = {f: np.full((len(CI_grid), len(P_grid)), np.nan) for f in FEED_ORDER}
optimal_CR      = {f: np.full((len(CI_grid), len(P_grid)), np.nan) for f in FEED_ORDER}
optimal_idx     = {f: np.full((len(CI_grid), len(P_grid)), -1, dtype=int)
                   for f in FEED_ORDER}

feed_masks = {f: (feeds == f) for f in FEED_ORDER}
records = []

for i_ci, ci_kwh in enumerate(CI_grid):
    for i_p, p_kwh in enumerate(P_grid):
        p_mj  = p_kwh  / 3.6
        ci_mj = ci_kwh / 3.6

        # ─── Carrier electrification (H2, FH, LP, HP) ───
        P_H2_n  = E_H2 * p_kwh + H2_CAPOP;     CI_H2_n = E_H2 * ci_kwh
        P_FH_n  = p_mj / ETA_H;                CI_FH_n = ci_mj / ETA_H
        P_LP_n  = p_mj / ETA_B;                CI_LP_n = ci_mj / ETA_B
        P_HP_n  = p_mj / ETA_B;                CI_HP_n = ci_mj / ETA_B

        d_cost = (
            C_elec * (p_mj   / P_elec_mj_base - 1) +
            C_FH   * (P_FH_n / P_FH_base      - 1) +
            C_LP   * (P_LP_n / P_LP_base      - 1) +
            C_HP   * (P_HP_n / P_HP_base      - 1) +
            C_H2   * (P_H2_n / P_H2_base      - 1)
        )
        d_ghg = (
            G_elec * (ci_mj   / CI_elec_mj_base - 1) +
            G_FH   * (CI_FH_n / CI_FH_base      - 1) +
            G_LP   * (CI_LP_n / CI_LP_base      - 1) +
            G_HP   * (CI_HP_n / CI_HP_base      - 1) +
            G_H2   * (CI_H2_n / CI_H2_base      - 1)
        )

        # ─── Ref-Elec coupling (NEW) ───
        for k in range(5):
            P_REF_n  = ref_price(k+1, p_kwh)
            CI_REF_n = ref_ci(k+1, ci_kwh*1000)
            d_cost += C_REF[k] * (P_REF_n  / P_REF_base[k]  - 1)
            d_ghg  += G_REF[k] * (CI_REF_n / CI_REF_base[k] - 1)

        new_UP = UP_base - d_cost * 1e6 / MASS
        new_CR = CR_base - d_ghg / MASS

        for f in FEED_ORDER:
            m = feed_masks[f]
            sub_UP = new_UP[m]
            sub_CR = new_CR[m]
            sub_tech = techs[m]
            sub_idx_global = np.where(m)[0]
            qual = sub_CR > 0
            if qual.any():
                local_idx = np.argmax(np.where(qual, sub_UP, -np.inf))
            else:
                local_idx = int(np.argmax(sub_UP))
            opt_tech = sub_tech[local_idx]
            opt_g = int(sub_idx_global[local_idx])

            optimal_tech_id[f][i_ci, i_p] = TECH_TO_ID[opt_tech]
            optimal_UP[f][i_ci, i_p] = float(sub_UP[local_idx])
            optimal_CR[f][i_ci, i_p] = float(sub_CR[local_idx])
            optimal_idx[f][i_ci, i_p] = opt_g

            records.append({
                'P_elec_$/kWh': round(p_kwh, 4),
                'CI_elec_kgCO2/kWh': round(ci_kwh, 4),
                'feed': f,
                'optimal_idx': opt_g,
                'optimal_tech': opt_tech,
                'optimal_p1': df['__p1'].iloc[opt_g],
                'optimal_p2': df['__p2'].iloc[opt_g],
                'optimal_UP': round(float(sub_UP[local_idx]), 4),
                'optimal_CR': round(float(sub_CR[local_idx]), 4),
                'qualified_CR>0': bool(qual.any()),
            })

pd.DataFrame(records).to_csv(
    os.path.join(BASE, 'phase14_optimal_table_RC.csv'),
    index=False, encoding='utf-8-sig')
print(f'\nSaved: phase14_optimal_table_RC.csv ({len(records)} rows)')

# ════════════════════════════════════════════════════════════════════
# Sanity check: print baseline and renewable optimal per feed
# ════════════════════════════════════════════════════════════════════
print('\n--- Baseline P=0.08, CI=0.369 (NOT in grid, computed manually) ---')
# (skip — verified separately)

def find_grid_point(p_target, ci_target):
    i_p  = int(np.argmin(np.abs(P_grid  - p_target)))
    i_ci = int(np.argmin(np.abs(CI_grid - ci_target)))
    return i_ci, i_p

print('\n--- Approx baseline (P=0.08, CI=0.369) at nearest grid point ---')
i_ci0, i_p0 = find_grid_point(0.08, 0.369)
print(f'  Grid: P={P_grid[i_p0]:.3f}, CI={CI_grid[i_ci0]:.3f}')
for f in FEED_ORDER:
    tid = optimal_tech_id[f][i_ci0, i_p0]
    tech = unique_techs[tid] if tid >= 0 else 'n/a'
    print(f'  {f}: tech={tech}, UP={optimal_UP[f][i_ci0, i_p0]:.3f}, CR={optimal_CR[f][i_ci0, i_p0]:.3f}')

print('\n--- Renewable PPA (P=0.04, CI=0.03) at nearest grid point ---')
i_ci1, i_p1 = find_grid_point(0.04, 0.03)
print(f'  Grid: P={P_grid[i_p1]:.3f}, CI={CI_grid[i_ci1]:.3f}')
for f in FEED_ORDER:
    tid = optimal_tech_id[f][i_ci1, i_p1]
    tech = unique_techs[tid] if tid >= 0 else 'n/a'
    print(f'  {f}: tech={tech}, UP={optimal_UP[f][i_ci1, i_p1]:.3f}, CR={optimal_CR[f][i_ci1, i_p1]:.3f}')

# ════════════════════════════════════════════════════════════════════
# Visualization 1: regime map (5 panel)
# ════════════════════════════════════════════════════════════════════
print('\nGenerating regime map…')
fig, axes = plt.subplots(1, 5, figsize=(FIG_W, FIG_W*0.32), sharey=True)
n_tech = len(unique_techs)
cmap = ListedColormap([TECH_COLORS[t] for t in unique_techs])
norm = BoundaryNorm(np.arange(n_tech+1)-0.5, n_tech)

for ax, f in zip(axes, FEED_ORDER):
    M = optimal_tech_id[f]
    ax.imshow(M, origin='lower', aspect='auto', cmap=cmap, norm=norm,
              extent=[P_grid[0], P_grid[-1], CI_grid[0], CI_grid[-1]])
    ax.set_title(f, fontsize=8, fontweight='bold')
    ax.set_xlabel('P$_{elec}$ (\\$/kWh)', fontsize=9, fontweight='bold')
    ax.axhline(0.369, color='black', ls='--', lw=0.8, alpha=0.7)
    ax.axvline(0.08, color='black', ls='--', lw=0.8, alpha=0.7)
    ax.axhline(0.03, color='green', ls=':', lw=0.8, alpha=0.7)
    ax.axvline(0.04, color='green', ls=':', lw=0.8, alpha=0.7)

axes[0].set_ylabel('CI$_{elec}$ (kgCO$_{2eq}$/kWh)', fontsize=9, fontweight='bold')
# main-technology display names consistent with the main text (H-PY/L-PY/O-GS/S-GS/HG)
TECH_DISP = {'HT': 'H-PY', 'LT': 'L-PY', 'OG': 'O-GS', 'SG': 'S-GS', 'HG': 'HG'}
# 2026-08-26: legend runs H-PY, L-PY, S-GS, O-GS, HG (HG last); the raster keeps
# its own tech IDs from unique_techs, so only the legend order changes here.
LEGEND_ORDER = ['HT', 'LT', 'SG', 'OG', 'HG']
legend_h = [Patch(facecolor=TECH_COLORS[t], label=TECH_DISP.get(t, t))
            for t in LEGEND_ORDER if t in unique_techs]
# legend BELOW the panels (keeps total width at 159.2 mm; external right legend
# was widening the saved PNG to ~216 mm); journal style → no suptitle
fig.legend(handles=legend_h, loc='lower center', bbox_to_anchor=(0.5, -0.10),
           ncol=n_tech, fontsize=7, title='Optimal technology', title_fontsize=7,
           frameon=False, columnspacing=1.0, handlelength=1.4)
plt.tight_layout()
out = os.path.join(config.FIG_OUT, 'FigS7_8_regime_map.png')
plt.savefig(out, dpi=400, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# ════════════════════════════════════════════════════════════════════
# Visualization 2: UP heatmap (5 panel)
# ════════════════════════════════════════════════════════════════════
print('Generating UP heatmap…')
fig, axes = plt.subplots(1, 5, figsize=(FIG_W, FIG_W*0.30), sharey=True)
vmax = max(np.nanmax(optimal_UP[f]) for f in FEED_ORDER)
vmin = min(np.nanmin(optimal_UP[f]) for f in FEED_ORDER)
print(f'  UP range: {vmin:.3f} to {vmax:.3f}')

for ax, f in zip(axes, FEED_ORDER):
    im = ax.imshow(optimal_UP[f], origin='lower', aspect='auto',
                   cmap='RdYlGn', vmin=vmin, vmax=vmax,
                   extent=[P_grid[0], P_grid[-1], CI_grid[0], CI_grid[-1]])
    # Contour at UP = -0.42 (Z1 threshold)
    C = ax.contour(P_grid, CI_grid, optimal_UP[f], levels=[-0.42],
                   colors='red', linewidths=1.5)
    ax.clabel(C, inline=True, fmt='-0.42', fontsize=8)
    ax.set_title(f, fontsize=8, fontweight='bold')
    ax.set_xlabel('P_elec ($/kWh)', fontsize=10)
    ax.axhline(0.369, color='black', ls='--', lw=0.8, alpha=0.7)
    ax.axvline(0.08, color='black', ls='--', lw=0.8, alpha=0.7)

axes[0].set_ylabel('CI_elec (kgCO₂/kWh)', fontsize=10)
cbar = fig.colorbar(im, ax=axes, fraction=0.020, pad=0.02)
cbar.set_label('Optimal UP ($/kg)', fontsize=10)
fig.suptitle('Phase 14 (Ref-Coupled): Optimal UP per feed — red contour: −0.42 threshold',
             fontsize=12, fontweight='bold', y=1.02)
out = os.path.join(config.FIG_OUT, 'fig_phase14_optimal_UP_RC.png')
plt.savefig(out, dpi=400, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

print('\nDone.')
