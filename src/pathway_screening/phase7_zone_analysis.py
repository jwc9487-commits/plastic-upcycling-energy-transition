import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
"""Phase 7: ZONE-based classification & driver heatmap (Option B grouping).
ZONE definition (baseline):
  Profit threshold = -0.42 $/kg  (incineration economics)
  CO2 reduction threshold = 0 kg CO2eq/kg
  ZONE 1: profit > -0.42 AND CO2_red > 0   (Win-Win)
  ZONE 2: profit > -0.42 AND CO2_red ≤ 0   (economic OK, no CO2 benefit)
  ZONE 3: profit ≤ -0.42 AND CO2_red > 0   (CO2 benefit, uneconomic)
  ZONE 4: profit ≤ -0.42 AND CO2_red ≤ 0   (Lose-Lose)

Outputs:
  - zone_classification.csv  (959 paths × zone)
  - fig_phase7_1_zone_scatter.png   (Task 1: quadrant scatter)
  - fig_phase7_2_zone_distribution.png  (Task 1: feed/tech/product per zone)
  - fig_phase7_3_zone_driver_heatmap.png  (Task 2: zone × driver, cost+GHG)"""
import sys, io, os, json
from collections import Counter
os.environ['PYTHONUTF8'] = '1'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
# ── Journal style: Arial, no big titles, 9pt bold labels, 7pt ticks ──
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.labelsize': 9, 'axes.labelweight': 'bold',
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.titlesize': 9,
})
MM = 1 / 25.4
FIG_W = 159.2 * MM
FIGDIR = config.FIG_OUT

BASE = config.INTERMEDIATE

PROFIT_THRESHOLD = -0.42  # incineration economics
CO2_THRESHOLD    = 0.0    # baseline CO2 reduction

# ─── Load full pathway data (cost + GHG breakdown) from 'All pathways' ──
print('Loading Raw_result_20250223 / All pathways…')
df = pd.read_excel(config.MASTER_XLSX,
                   sheet_name='All pathways', header=3)

feed_col = df.columns[0]
tech_col = df.columns[1]
prod_col = df.columns[4]
up_col   = next(c for c in df.columns if isinstance(c, str) and c.startswith('Unit profit'))
cr_col   = next(c for c in df.columns if isinstance(c, str) and c.startswith('Unit CO2 reduc'))

df['__feed'] = df[feed_col].astype(str).str.strip()
df['__tech'] = df[tech_col].astype(str).str.strip()
df['__prod'] = df[prod_col].astype(str).str.strip()
df['__up']   = pd.to_numeric(df[up_col], errors='coerce')
df['__cr']   = pd.to_numeric(df[cr_col], errors='coerce')
df = df.dropna(subset=['__up', '__cr']).copy()
df = df[df['__feed'].isin(['PE', 'PET', 'PP', 'PS', 'PVC'])].copy()

# ZONE assignment
def assign_zone(p, c):
    if p > PROFIT_THRESHOLD and c > CO2_THRESHOLD: return 1
    if p > PROFIT_THRESHOLD and c <= CO2_THRESHOLD: return 2
    if p <= PROFIT_THRESHOLD and c > CO2_THRESHOLD: return 3
    return 4
df['__zone'] = df.apply(lambda r: assign_zone(r['__up'], r['__cr']), axis=1)

print(f'\nTotal paths: {len(df)}')
print('Zone counts:')
for z in range(1, 5):
    n = (df['__zone'] == z).sum()
    print(f'  ZONE {z}: {n}  ({n/len(df)*100:.1f}%)')

# Per-zone feed distribution
print('\nPer-zone feed distribution:')
zone_feed = df.groupby(['__zone', '__feed']).size().unstack(fill_value=0)
print(zone_feed)

# Per-zone tech distribution
print('\nPer-zone tech distribution:')
zone_tech = df.groupby(['__zone', '__tech']).size().unstack(fill_value=0)
print(zone_tech)

# Product category mapping
def product_category(p):
    p = str(p).upper()
    if 'GASOLINE' in p: return 'Gasoline'
    if 'DIESEL'   in p: return 'Diesel'
    if 'FT' in p:       return 'FT fuel'
    if 'MEOH' in p:     return 'MeOH'
    if 'H2' in p:       return 'H₂'
    if 'OLEFIN' in p:   return 'Olefin'
    if 'AROMATIC' in p: return 'Aromatics'
    if 'BTX' in p:      return 'BTX'
    if 'WAX' in p:      return 'Wax'
    if 'STYRENE' in p:  return 'Styrene'
    if 'BA_1' in p or 'AA_1' in p: return 'BA/AA'
    if 'NAP' in p:      return 'Naphtha'
    if 'C9' in p:       return 'C9+'
    if any(x in p for x in ['C2H4','C3H6','C2H6','C3H8','C4_']): return 'C2-C4 olefins/paraffins'
    return p[:8]
df['__prod_cat'] = df['__prod'].apply(product_category)

print('\nPer-zone product distribution:')
zone_prod = df.groupby(['__zone', '__prod_cat']).size().unstack(fill_value=0)
print(zone_prod)

df.to_csv(os.path.join(BASE, 'zone_classification.csv'),
          index=False, columns=['__feed','__tech','__prod','__prod_cat',
                                '__up','__cr','__zone'],
          encoding='utf-8-sig')
print(f'\nSaved: zone_classification.csv')

# ──────────────────────────────────────────────────────────────────────
# FIG 1: ZONE quadrant scatter
# ──────────────────────────────────────────────────────────────────────
ZONE_COLORS = {1: '#27ae60', 2: '#f39c12', 3: '#3498db', 4: '#c0392b'}
ZONE_LABELS = {
    1: 'ZONE 1 — Win-Win\n(profit>−0.42, CO₂ red>0)',
    2: 'ZONE 2 — Economic only\n(profit>−0.42, CO₂ red≤0)',
    3: 'ZONE 3 — Environmental only\n(profit≤−0.42, CO₂ red>0)',
    4: 'ZONE 4 — Lose-Lose\n(profit≤−0.42, CO₂ red≤0)',
}

fig, ax = plt.subplots(figsize=(11, 8))
for z in [4, 3, 2, 1]:  # plot worst first so ZONE 1 on top
    sub = df[df['__zone'] == z]
    ax.scatter(sub['__up'], sub['__cr'],
               c=ZONE_COLORS[z], s=28, alpha=0.55,
               edgecolors='white', linewidths=0.3,
               label=f'{ZONE_LABELS[z]}  (n={len(sub)})')

ax.axvline(PROFIT_THRESHOLD, color='black', lw=1.2, ls='--', alpha=0.7)
ax.axhline(CO2_THRESHOLD, color='black', lw=1.2, ls='--', alpha=0.7)
ax.text(PROFIT_THRESHOLD - 0.02, df['__cr'].max() * 0.97,
        f'  profit = −0.42\n  (incineration\n   economics)',
        ha='right', va='top', fontsize=8.5, color='#444',
        bbox=dict(facecolor='white', edgecolor='#ccc', alpha=0.85))
ax.text(df['__up'].max() * 0.95, CO2_THRESHOLD + 0.15,
        '  CO₂ red = 0 (baseline)',
        ha='right', va='bottom', fontsize=8.5, color='#444',
        bbox=dict(facecolor='white', edgecolor='#ccc', alpha=0.85))

ax.set_xlabel('Unit profit ($/kg)', fontsize=11)
ax.set_ylabel('Unit CO₂ reduction (kg CO₂eq/kg)', fontsize=11)
ax.set_title(f'ZONE-based classification of {len(df)} plastic upcycling pathways\n'
             f'(thresholds: incineration economics + incineration emission baseline)',
             fontsize=12, pad=10)
ax.legend(loc='lower right', fontsize=8.5, framealpha=0.92)
ax.grid(True, linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(config.FIG_OUT, 'fig_phase7_1_zone_scatter.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print('Saved: fig_phase7_1_zone_scatter.png')

# ──────────────────────────────────────────────────────────────────────
# FIG 2: ZONE × {Feed, Tech, Product} stacked bar
# ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 6))

# Feed
ax = axes[0]
feed_order = ['PE', 'PET', 'PP', 'PS', 'PVC']
feed_data = zone_feed.reindex(columns=feed_order, fill_value=0)
x = np.arange(len(feed_order))
bottom = np.zeros(len(feed_order))
for z in [1, 2, 3, 4]:
    if z not in feed_data.index: continue
    vals = feed_data.loc[z, feed_order].values
    ax.bar(x, vals, bottom=bottom, color=ZONE_COLORS[z],
           edgecolor='white', linewidth=0.6,
           label=f'ZONE {z}  (n={(df["__zone"]==z).sum()})')
    bottom = bottom + vals
ax.set_xticks(x)
ax.set_xticklabels(feed_order, fontsize=10)
ax.set_ylabel('Pathway count', fontsize=10)
ax.set_title('(a) Feed × ZONE', fontsize=11)
ax.legend(loc='upper right', fontsize=8)
ax.grid(axis='y', linestyle=':', alpha=0.4)

# Tech
ax = axes[1]
tech_order = ['HT', 'LT', 'SG', 'OG', 'HG']
tech_order = [t for t in tech_order if t in zone_tech.columns]
tech_data = zone_tech.reindex(columns=tech_order, fill_value=0)
x = np.arange(len(tech_order))
bottom = np.zeros(len(tech_order))
for z in [1, 2, 3, 4]:
    if z not in tech_data.index: continue
    vals = tech_data.loc[z, tech_order].values
    ax.bar(x, vals, bottom=bottom, color=ZONE_COLORS[z],
           edgecolor='white', linewidth=0.6)
    bottom = bottom + vals
ax.set_xticks(x)
ax.set_xticklabels(tech_order, fontsize=10)
ax.set_ylabel('Pathway count', fontsize=10)
ax.set_title('(b) Main technology × ZONE', fontsize=11)
ax.grid(axis='y', linestyle=':', alpha=0.4)

# Product
ax = axes[2]
prod_order = list(zone_prod.sum(axis=0).sort_values(ascending=False).index)[:10]
prod_data = zone_prod.reindex(columns=prod_order, fill_value=0)
x = np.arange(len(prod_order))
bottom = np.zeros(len(prod_order))
for z in [1, 2, 3, 4]:
    if z not in prod_data.index: continue
    vals = prod_data.loc[z, prod_order].values
    ax.bar(x, vals, bottom=bottom, color=ZONE_COLORS[z],
           edgecolor='white', linewidth=0.6)
    bottom = bottom + vals
ax.set_xticks(x)
ax.set_xticklabels(prod_order, fontsize=9, rotation=30, ha='right')
ax.set_ylabel('Pathway count', fontsize=10)
ax.set_title('(c) Product (top 10) × ZONE', fontsize=11)
ax.grid(axis='y', linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(config.FIG_OUT, 'fig_phase7_2_zone_distribution.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print('Saved: fig_phase7_2_zone_distribution.png')

# ──────────────────────────────────────────────────────────────────────
# TASK 2: ZONE × driver heatmap (Option B grouping)
# ──────────────────────────────────────────────────────────────────────
# Build cost & GHG breakdown per pathway (Option B columns)
cols = list(df.columns)

# Column index map (same as phase4_driver_pandas)
COST_GROUPS = {
    # 2026-08-26: cooling water (9) moved to 'Other materials'. The text names the
    # energy carriers as fired heat, electricity and steam, and panel (c) already
    # excluded cooling water, so panels (a) and (c) disagreed by 2.8 pp.
    'Energy carriers':           [7, 8, 10, 11, 12, 13, 14, 15, 16, 20],
    'Plastic feedstock':         [6],
    'Other materials':           [9, 17, 18, 19, 21, 22, 23, 24],
    'ACC (Capital)':             [5],
    'FOC (Fixed operating)':     [25],
}
GHG_GROUPS = {
    'Energy carriers':            [31, 32, 33, 34, 35, 36, 37, 38, 39, 42],
    'Direct emission':            [30],
    'Other materials':            [40, 41],
}

# Sub-component descriptions (used in legend/footnote only)
COST_GROUPS_DETAIL = {
    'Energy carriers': 'Elec + FH + Ref + Steam + H2',
    'Other materials': 'CW / Sulfolan / MEA / O2 / Water / sorbents',
}
GHG_GROUPS_DETAIL = {
    'Energy carriers': 'Elec + FH + Steam + Ref + H2',
    'Other materials': 'O2 + Steam_raw',
}

def numeric_sum(df, idx_list):
    return df.iloc[:, idx_list].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)

cost_df = pd.DataFrame()
for name, idxs in COST_GROUPS.items():
    cost_df[name] = numeric_sum(df, idxs)
cost_df['__zone'] = df['__zone'].values
cost_df['__total'] = cost_df[list(COST_GROUPS.keys())].sum(axis=1)
for k in COST_GROUPS:
    cost_df[k + '_share'] = cost_df[k] / cost_df['__total'].replace(0, np.nan)

ghg_df = pd.DataFrame()
for name, idxs in GHG_GROUPS.items():
    ghg_df[name] = numeric_sum(df, idxs)
ghg_df['__zone'] = df['__zone'].values
ghg_df['__total_abs'] = ghg_df[list(GHG_GROUPS.keys())].abs().sum(axis=1)
for k in GHG_GROUPS:
    ghg_df[k + '_share'] = ghg_df[k] / ghg_df['__total_abs'].replace(0, np.nan)

# Per-zone mean share
cost_share_M = np.zeros((4, len(COST_GROUPS)))
ghg_share_M  = np.zeros((4, len(GHG_GROUPS)))
for zi, z in enumerate([1, 2, 3, 4]):
    sub_c = cost_df[cost_df['__zone'] == z]
    sub_g = ghg_df[ghg_df['__zone'] == z]
    if not sub_c.empty:
        for j, k in enumerate(COST_GROUPS.keys()):
            cost_share_M[zi, j] = sub_c[k + '_share'].mean()
    if not sub_g.empty:
        for j, k in enumerate(GHG_GROUPS.keys()):
            ghg_share_M[zi, j] = sub_g[k + '_share'].mean()

# All-pathway column (2026-08-26): the shares the manuscript text quotes
cost_share_all = np.array([cost_df[k + '_share'].mean() for k in COST_GROUPS])
ghg_share_all  = np.array([ghg_df[k + '_share'].mean() for k in GHG_GROUPS])
print()
print(f'=== All-pathway share (n = {len(cost_df)}) ===')
for k, v in zip(COST_GROUPS, cost_share_all):
    print(f'  cost {k:<24} {v*100:5.2f}%')
for k, v in zip(GHG_GROUPS, ghg_share_all):
    print(f'  ghg  {k:<24} {v*100:5.2f}%')

# Print
print('\n=== Cost share by ZONE (Option B) ===')
for j, lbl in enumerate(COST_GROUPS):
    short = lbl.replace('\n', ' ')
    print(f'  {short:<60} ' + ' '.join(f'Z{z}={cost_share_M[z-1, j]*100:>4.1f}%'
                                        for z in range(1, 5)))
print('\n=== GHG share by ZONE (Option B) ===')
for j, lbl in enumerate(GHG_GROUPS):
    short = lbl.replace('\n', ' ')
    print(f'  {short:<60} ' + ' '.join(f'Z{z}={ghg_share_M[z-1, j]*100:>4.1f}%'
                                        for z in range(1, 5)))

# Plot heatmap — compact ZONE labels (single-column width)
# 2026-08-26: the descriptor line (Win-Win / Econ / ...) is dropped; at 72.1 mm
# it ran into the neighbouring column. The zones are defined in the caption.
ZONE_X_LABELS = [
    f'ZONE 1\n(n={(df["__zone"]==1).sum()})',
    f'ZONE 2\n(n={(df["__zone"]==2).sum()})',
    f'ZONE 3\n(n={(df["__zone"]==3).sum()})',
    f'ZONE 4\n(n={(df["__zone"]==4).sum()})',
    f'All\n(n={len(df)})',
]
COST_YLAB = ['Energy', 'Feedstock', 'Other mat.', 'Capital', 'Fixed op.']
GHG_YLAB  = ['Energy', 'Direct emis.', 'Other mat.']

# Palette tone (cost = blue #4E79A7, emission = terracotta #BC7645), darkened at top.
cmap_blue = LinearSegmentedColormap.from_list('blues', ['#f7fbff', '#4E79A7', '#2c4a6e'])
cmap_red  = LinearSegmentedColormap.from_list('terra', ['#f9f2ec', '#BC7645', '#6e3f22'])

# ── Unified layout (shared with Fig7c): figure 79.6×80 mm, graph 60×60 mm,
#    panel label at the figure top-left.  Heatmaps have NO colorbar (cells label %).
UW = 83.0                       # figure width mm (USER request)
UH_FIX = 45.0                   # figure height mm (USER request, same for both panels)
GX, GY, GW = 16.0,  9.0, 63.4   # graph box left, bottom, width mm
ROW_H = 9.0                     # uniform cell height mm (equal cell size across panels)
TOP = 2.0                       # top margin mm (panel letters are added at composition)
def make_panel(M, cmap, letter, outname, ylabels):
    Mp = M * 100.0
    UH = UH_FIX                          # both panels share one figure height (USER request)
    GH = UH - GY - TOP                   # the graph fills what is left, so cell height
                                         # now follows the row count rather than ROW_H
    fx = lambda mm: mm / UW
    fy = lambda mm: mm / UH
    fig = plt.figure(figsize=(UW * MM, UH * MM))
    ax = fig.add_axes([fx(GX), fy(GY), fx(GW), fy(GH)])   # 60×(nrows·ROW_H) mm graph
    im = ax.imshow(Mp, cmap=cmap, vmin=0, vmax=Mp.max() * 1.05, aspect='auto')
    col_max = np.argmax(Mp, axis=0)
    for i in range(Mp.shape[0]):
        for j in range(Mp.shape[1]):
            v = Mp[i, j]
            color = 'white' if v > Mp.max() * 0.55 else 'black'
            wt = 'bold' if i == col_max[j] else 'normal'
            ax.text(j, i, f'{v:.1f}%', ha='center', va='center',
                    color=color, fontsize=7, fontweight=wt)
    ax.set_xticks(range(Mp.shape[1]))
    ax.set_xticklabels(ZONE_X_LABELS[:Mp.shape[1]], fontsize=7)
    ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels, fontsize=7)
    out = os.path.join(FIGDIR, outname)
    plt.savefig(out, dpi=400)
    plt.close()
    print(f'Saved: {out}  (figure {UW}×{UH} mm, graph {GW}×{GH} mm)')

make_panel(np.column_stack([cost_share_M.T, cost_share_all]), cmap_blue, '(a)',
           'Fig3a_cost_driver_heatmap.png', COST_YLAB)
make_panel(np.column_stack([ghg_share_M.T, ghg_share_all]), cmap_red, '(b)',
           'Fig3b_ghg_driver_heatmap.png', GHG_YLAB)
# remove the old combined figure
_old = os.path.join(FIGDIR, 'Fig7ab_zone_driver_heatmap.png')
if os.path.exists(_old):
    os.remove(_old)

# Save JSON
result = {
    'zone_thresholds': {'profit': PROFIT_THRESHOLD, 'co2_red': CO2_THRESHOLD},
    'zone_counts': {f'ZONE {z}': int((df['__zone']==z).sum()) for z in range(1,5)},
    'cost_drivers': list(COST_GROUPS.keys()),
    'ghg_drivers':  list(GHG_GROUPS.keys()),
    'cost_share_M': cost_share_M.tolist(),
    'cost_share_all': cost_share_all.tolist(),
    'ghg_share_all':  ghg_share_all.tolist(),
    'n_all': int(len(cost_df)),
    'ghg_share_M':  ghg_share_M.tolist(),
    'zone_feed': zone_feed.to_dict(),
    'zone_tech': zone_tech.to_dict(),
    'zone_prod': zone_prod.to_dict(),
}
with open(os.path.join(BASE, 'phase7_zone_analysis.json'), 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2, default=int)
print('Saved: phase7_zone_analysis.json')
