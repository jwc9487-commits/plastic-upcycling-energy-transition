import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
# -*- coding: utf-8 -*-
"""Fig. 5 — the conditions under which the system becomes profitable, and under
which electrification beats the fossil baseline.

Replaces the old Fig. 10 + Fig. 11, which were two six-panel stacks of 210 and 212 mm.
Both are driven by the same file, m2_step4_RC_compare.csv, since wUP_elec is exactly
what the old Fig. 10 plotted. Three renewable shares are shown instead of six, with
the full six kept in the supplement.

In-figure wording follows the manuscript: 'baseline' and 'electrification' for the two
scenarios, 'break-even' as in Section 4.1, 'carbon price' as in Fig. 3.
"""
import os
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
    'xtick.major.size': 1.8, 'ytick.major.size': 1.8,
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
FIG_W, FIG_H = 174.0 * MM, 112.0 * MM

df = pd.read_csv(os.path.join(DATA, 'm2_step4_RC_compare.csv'))
ALPHAS = [0.2, 0.5, 0.8]          # interior levels; full six-level version in the supplement
CI = {a: df[df.alpha == a].CI_elec.iloc[0] for a in sorted(df.alpha.unique())}

P_GRID, PPA26, PPA50, ETS = 0.08, 0.04, 0.02, 0.08
C_CROSS, C_ELEC, C_BASE = '#1c2833', '#c0392b', '#34495e'
C_G50, C_G26, C_ETS = '#2a7f62', '#2d5f9a', '#8e44ad'


def grid(a, col):
    s = df[df.alpha == a]
    piv = s.pivot_table(index='tau', columns='P_elec', values=col)
    return piv.columns.values, piv.index.values, piv.values


fig = plt.figure(figsize=(FIG_W, FIG_H))
COLX = [0.088, 0.3487, 0.6094]
CW, CH = 0.2187, 0.320
# 6.7 mm between the rows, which is where the (d)(e)(f) letters go
ROWY = [0.575, 0.195]
axes = [[fig.add_axes([x, y, CW, CH]) for x in COLX] for y in ROWY]

# a point on each break-even contour that sits well inside the panel
CLAB = {0.2: (0.032, 0.145), 0.5: (0.040, 0.128), 0.8: (0.040, 0.112)}

# ══════════════ row 1: net profit of the electrified system ══════════════
vmax1 = max(abs(df.wUP_elec).max(), 0.25)
for j, a in enumerate(ALPHAS):
    ax = axes[0][j]
    X, Y, Z = grid(a, 'wUP_elec')
    im1 = ax.pcolormesh(X, Y, Z, cmap='RdYlGn', vmin=-vmax1, vmax=vmax1,
                        shading='auto', rasterized=True)
    cs = ax.contour(X, Y, Z, levels=[0], colors=[C_CROSS], linewidths=1.1)
    # automatic placement put the label on the left spine of (a) and past the
    # right spine of (c); anchor it near mid-panel on the contour instead
    ax.clabel(cs, fmt='break-even', fontsize=6.0, inline=True, inline_spacing=2,
              manual=[CLAB[a]])
    ax.set_title(f'$\\alpha$ = {a:.1f}\nCI = {CI[a]:.0f} gCO$_{{2eq}}$/kWh',
                 fontsize=6.5, fontweight='bold', pad=2.5, linespacing=1.3)

# ══════════════ row 2: electrification minus baseline ══════════════
vmax2 = max(abs(df.delta).max(), 0.25)
for j, a in enumerate(ALPHAS):
    ax = axes[1][j]
    X, Y, D = grid(a, 'delta')
    im2 = ax.pcolormesh(X, Y, D, cmap='RdBu', vmin=-vmax2, vmax=vmax2,
                        shading='auto', rasterized=True)
    _, _, Ze = grid(a, 'wUP_elec')
    _, _, Zb = grid(a, 'wUP_base')
    ax.contour(X, Y, D, levels=[0], colors=[C_CROSS], linewidths=1.1)
    ax.contour(X, Y, Ze, levels=[0], colors=[C_ELEC], linewidths=0.8,
               linestyles='--')
    ax.contour(X, Y, Zb, levels=[0], colors=[C_BASE], linewidths=0.8)

# reference lines and axis housekeeping on every panel
for i, row in enumerate(axes):
    for j, ax in enumerate(row):
        for xv, c in [(PPA50, C_G50), (PPA26, C_G26)]:
            ax.axvline(xv, color=c, ls=':', lw=0.8, zorder=6)
        ax.axhline(ETS, color=C_ETS, ls=':', lw=0.8, zorder=6)
        ax.set_xlim(0, 0.08); ax.set_ylim(0, 0.20)
        ax.set_xticks([0, 0.02, 0.04, 0.06, 0.08])
        ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
        if j > 0:
            ax.set_yticklabels([])
        if i == 0:
            ax.set_xticklabels([])
        for s in ax.spines.values():
            s.set_linewidth(0.5)
    row[0].set_ylabel('Carbon pricing\n(\\$/kgCO$_{2eq}$)', fontsize=7.2)
axes[1][1].set_xlabel('Electricity price (\\$/kWh)')

# ══════════════ colour bars, one per row ══════════════
cax1 = fig.add_axes([0.855, 0.575, 0.014, 0.320])
cb1 = fig.colorbar(im1, cax=cax1)
cb1.set_label('Unit profit (\\$/kg)',
              fontsize=6.2, fontweight='bold', labelpad=2)
cax2 = fig.add_axes([0.855, 0.195, 0.014, 0.320])
cb2 = fig.colorbar(im2, cax=cax2)
cb2.set_label(r'$\Delta$ UP$_{\mathbf{net}}$ = UP$_{\mathbf{e}}$ − UP$_{\mathbf{b}}$ (\$/kg)',
              fontsize=6.2, fontweight='bold', labelpad=2)
for cb in (cb1, cb2):
    cb.ax.tick_params(labelsize=6.0, length=1.6, width=0.4)
    cb.outline.set_linewidth(0.4)
# row 1 is the absolute profit of the electrified system, so it reads
# profitable / unprofitable; only row 2 is a comparison against the baseline
fig.text(0.955, 0.893, 'Profitable', fontsize=6.2, ha='center', va='top',
         color='#1a7a3c', fontweight='bold')
fig.text(0.955, 0.582, 'Unprofitable', fontsize=6.2, ha='center', va='bottom',
         color='#b2182b', fontweight='bold')
fig.text(0.955, 0.513, 'Electrification\nfavored', fontsize=6.2, ha='center',
         va='top', color='#2166ac', fontweight='bold')
fig.text(0.955, 0.202, 'Baseline\nfavored', fontsize=6.2, ha='center',
         va='bottom', color='#b2182b', fontweight='bold')

# ══════════════ shared legend ══════════════
H = [Line2D([0], [0], color=C_CROSS, lw=1.1, label='Electrification equals baseline'),
     Line2D([0], [0], color=C_ELEC, lw=0.8, ls='--', label='Electrification break-even'),
     Line2D([0], [0], color=C_BASE, lw=0.8, label='Baseline break-even'),
     Line2D([0], [0], color=C_G50, lw=0.8, ls=':', label='PPA 2050'),
     Line2D([0], [0], color=C_G26, lw=0.8, ls=':', label='PPA 2026'),
     Line2D([0], [0], color=C_ETS, lw=0.8, ls=':', label='EU ETS level')]
leg = fig.legend(handles=H, loc='lower center', bbox_to_anchor=(0.45, 0.014),
                 ncol=6, fontsize=6.2, framealpha=1.0, edgecolor='#bbb',
                 fancybox=False, handletextpad=0.4, borderpad=0.4,
                 columnspacing=1.2, handlelength=1.5)
leg.get_frame().set_linewidth(0.4)

for lab, (fx, fy) in {'a': (0.022, 0.930), 'b': (0.338, 0.930), 'c': (0.599, 0.930),
                      'd': (0.022, 0.524), 'e': (0.338, 0.524),
                      'f': (0.599, 0.524)}.items():
    fig.text(fx, fy, f'({lab})', fontsize=8.5, fontweight='bold', va='bottom', ha='left')

out = os.path.join(OUT, 'Fig5_conditions.png')
plt.savefig(out, dpi=500, facecolor='white')
plt.close()
print('saved', out)
print('alphas shown:', ALPHAS, ' CI:', [CI[a] for a in ALPHAS])
print(f'grid {df[df.alpha==0].P_elec.nunique()} x {df[df.alpha==0].tau.nunique()}'
      f'   vmax profit {vmax1:.3f}  vmax delta {vmax2:.3f}')
