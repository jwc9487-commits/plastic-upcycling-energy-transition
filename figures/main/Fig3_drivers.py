import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
# -*- coding: utf-8 -*-
"""Fig. 3 — energy carriers govern both cost and emissions.

Layout: (a) and (b) the same height on the left, (c) spanning exactly the two of
them on the right, so the block is aligned top and bottom.

House style follows the figures the first author drew: parenthesised panel
letters, per-cent signs inside the heat map cells, slash units, abbreviated row
labels, a heavy cell border, and a legend box under (c). Two items are NOT
reverted, because they were changed on evidence rather than preference:
  Win-Win / Lose-Lose   0 hits in the manuscript, 0 in the 16 reference papers
  cost-equivalent       0 hits in either
  Feed -> Feedstock     feedstock 33 hits in the manuscript, 289 in 15 of 16
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.labelsize': 7.5, 'axes.labelweight': 'bold',
    'xtick.labelsize': 6.0, 'ytick.labelsize': 6.0,
    'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.0, 'ytick.major.size': 2.0,
})
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial'
mpl.rcParams['mathtext.bf'] = 'Arial:bold'
# without this, anything inside $...$ renders italic serif and the 'eq' of
# CO_{2eq} would not match the surrounding Arial
mpl.rcParams['mathtext.default'] = 'regular'

DATA = config.INTERMEDIATE
OUT = config.FIG_OUT
MM = 1 / 25.4
FIG_W, FIG_H = 174.0 * MM, 80.0 * MM

J = json.load(open(os.path.join(DATA, 'phase7_zone_analysis.json'), encoding='utf-8'))
tau = pd.read_csv(os.path.join(DATA, 'fig3_driver_vs_tau.csv'))

COST_LAB = ['Energy', 'Feedstock', 'Other mat.', 'Capital', 'Fixed op.']
GHG_LAB = ['Energy', 'Direct emis.', 'Other mat.']
ZLAB = [('ZONE 1', 'better on both'), ('ZONE 2', 'profit only'),
        ('ZONE 3', 'carbon only'), ('ZONE 4', 'neither')]
ZN = [J['zone_counts'][f'ZONE {i}'] for i in (1, 2, 3, 4)]

cost = np.array(J['cost_share_M']) * 100      # 4 zones x 5
ghg = np.array(J['ghg_share_M']) * 100        # 4 zones x 3
# 2026-08-26: an all-pathway column, so the shares the text quotes can be read off
# the figure. Cooling water now sits in 'Other materials' (see the JSON note), so
# 'Energy' here is fired heat, electricity, steam, refrigeration and hydrogen.
cost = np.vstack([cost, np.array(J['cost_share_all']) * 100])
ghg = np.vstack([ghg, np.array(J['ghg_share_all']) * 100])
ZLAB = ZLAB + [('All', 'pathways')]
ZN = ZN + [J.get('n_all', sum(ZN))]

fig = plt.figure(figsize=(FIG_W, FIG_H))
axA = fig.add_axes([0.135, 0.6300, 0.340, 0.2925])
axB = fig.add_axes([0.135, 0.2625, 0.340, 0.2925])
axC = fig.add_axes([0.575, 0.2625, 0.340, 0.6600])


def heat(ax, M, rowlab, cmap, vmax, show_x):
    """M is zones x categories; drawn as categories (rows) by zones (columns)."""
    A = M.T
    ax.imshow(A, cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = A[i, j]
            ax.text(j, i, f'{v:.1f}%', ha='center', va='center', fontsize=6.2,
                    color='white' if v > vmax * 0.55 else '#1c2833',
                    fontweight='bold' if v == A[i].max() else 'normal')
    ax.set_yticks(range(A.shape[0]))
    ax.set_yticklabels(rowlab, fontsize=6.5)
    ax.set_xticks(range(A.shape[1]))
    if show_x:
        ax.set_xticklabels([f'{a}\n(n={n})' for (a, b), n in zip(ZLAB, ZN)],
                           fontsize=6.0, linespacing=1.30)
    else:
        ax.set_xticklabels([])
    ax.tick_params(length=0)
    for s in ax.spines.values():           # heavy border, as in the original
        s.set_linewidth(1.2)
        s.set_color('black')


heat(axA, cost, COST_LAB, 'Blues', 42, show_x=False)
heat(axB, ghg, GHG_LAB, 'Oranges', 100, show_x=True)

# ══════════════ (c) cost shares once carbon is priced ══════════════
SER = [('Energy_%', 'Energy', '#b5651d', 1.6),
       ('Feed_%', 'Feedstock', '#3f8f5a', 1.0),
       ('Capital_%', 'Capital', '#8a8a8a', 1.0),
       ('Operating_%', 'Fixed op.', '#4878d0', 1.0),
       ('Other_%', 'Other mat.', '#c9c2b6', 1.0),
       ('Process_%', 'Direct emis.', '#e8b93d', 1.0)]
for col, lab, c, lw in SER:
    axC.plot(tau['tau_$/kgCO2'], tau[col], color=c, lw=lw, label=lab,
             solid_capstyle='round', zorder=4)

d = tau['Energy_%'] - tau['Feed_%']
k = int(np.argmin(np.abs(d.values)))
tx = tau['tau_$/kgCO2'].iloc[k]
ty = tau['Energy_%'].iloc[k]
axC.axvline(tx, color='#b5651d', ls=':', lw=0.8, zorder=3)
axC.plot(tx, ty, marker='o', ms=3.8, mfc='#b5651d', mec='white', mew=0.6, zorder=8)
# both notes sit in bands no series passes through
axC.text(tx + 0.004, 46.6,
         f'Energy = Feedstock\n(~{tx*1000:.0f} \\$/tCO$_{{2eq}}$)',
         ha='left', va='top', fontsize=6.0, fontweight='bold', color='#b5651d',
         zorder=9)
ETS = 0.080
axC.axvline(ETS, color='#7f8c8d', ls=':', lw=0.8, zorder=3)
axC.text(ETS - 0.0015, 4.2, 'EU ETS (~80 \\$/tCO$_{2eq}$)', fontsize=6.0,
         ha='right', va='center', color='#5d6d7e', fontweight='bold', zorder=9)

axC.set_xlabel('Carbon pricing (\\$/kgCO$_{2eq}$)')
axC.set_ylabel('Share of total cost (%)')
axC.set_xlim(0, 0.10); axC.set_ylim(0, 48)
axC.grid(axis='y', ls=':', lw=0.35, alpha=0.35)
for s in ('top', 'right'):
    axC.spines[s].set_visible(False)

hC = [Line2D([0], [0], color=c, lw=max(lw, 1.4), label=lab) for _, lab, c, lw in SER]
lgC = fig.legend(handles=hC, loc='lower center', ncol=3,
                 bbox_to_anchor=(0.745, 0.012), fontsize=6.2, framealpha=1.0,
                 edgecolor='#bbb', fancybox=False, handletextpad=0.5,
                 borderpad=0.40, labelspacing=0.32, columnspacing=1.3,
                 handlelength=1.4)
lgC.get_frame().set_linewidth(0.4)


out = os.path.join(OUT, 'Fig3_drivers.png')
plt.savefig(out, dpi=500, facecolor='white')
plt.close()
print('saved', out)
print('cost rows:', COST_LAB)
print('ghg  rows:', GHG_LAB)
print(f'energy overtakes feedstock at tau = {tx:.4f} $/kgCO2 = {tx*1000:.0f} $/tCO2eq')
