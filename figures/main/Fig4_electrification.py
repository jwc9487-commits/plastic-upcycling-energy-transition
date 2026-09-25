import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
# -*- coding: utf-8 -*-
"""Fig. 4 — electrification is the lever, but only when the electricity is clean.

Replaces the old Fig. 8 + Fig. 9 (159 and 132 mm, both with large blank margins).
The old Fig. 8 plotted 959 pathways against a running index plus a separate count-bar
panel; the same message is carried by the distribution per feedstock. The old Fig. 9
carried a strip of optimal process configurations under each panel, which is method
detail and moves to the supplement.

Scenario names follow the manuscript: 'grid electrification' and 'renewable
electrification' both appear in the abstract.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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

DATA = config.INTERMEDIATE
OUT = config.FIG_OUT
MM = 1 / 25.4
FIG_W, FIG_H = 174.0 * MM, 66.0 * MM

UP_THR = -0.42
FEEDS = ['PE', 'PP', 'PET', 'PS', 'PVC']
SC = [('S0_Baseline', 'Fossil baseline', '#3b7dbd'),
      ('S1_GridElec', 'Grid electrification', '#d1802f'),
      ('S2_RenewElec', 'Renewable electrification', '#4a9b5c')]
FEED_C = {'PE': '#2f6f9f', 'PP': '#3f8f5a', 'PET': '#d99a2b',
          'PS': '#8a6bb1', 'PVC': '#c0504d'}

s3 = pd.read_csv(os.path.join(DATA, 'phase11_three_scenarios_RC.csv'))
key = {k: k for k in s3.scenario.unique()}
order = list(s3.scenario.unique())
print('scenario keys in file:', order)
SC = [(order[i], SC[i][1], SC[i][2]) for i in range(3)]

econ = pd.read_csv(os.path.join(DATA, 'm2_step3_RC_econ.csv'))
env = pd.read_csv(os.path.join(DATA, 'm2_step3_RC_env.csv'))

fig = plt.figure(figsize=(FIG_W, FIG_H))
axA = fig.add_axes([0.078, 0.245, 0.380, 0.660])
axB = fig.add_axes([0.588, 0.245, 0.380, 0.660])


def dist_panel(ax, metric, thr, thr_lab):
    """box per feedstock and scenario, with the all-pathway mean of each scenario"""
    W = 0.24
    for gi, f in enumerate(FEEDS):
        for si, (skey, slab, c) in enumerate(SC):
            v = s3[(s3.feed == f) & (s3.scenario == skey)][metric].values
            pos = gi + (si - 1) * W
            bp = ax.boxplot([v], positions=[pos], widths=W * 0.82, showfliers=False,
                            patch_artist=True, whis=(5, 95), zorder=4)
            bp['boxes'][0].set(facecolor=c, edgecolor='white', linewidth=0.4,
                               alpha=0.92)
            for k in ('whiskers', 'caps'):
                for e in bp[k]:
                    e.set(color=c, linewidth=0.6)
            bp['medians'][0].set(color='white', linewidth=0.7)
    # the mean line stops short of its label, otherwise the minus sign merges into
    # the line and the value reads as positive
    XEND = len(FEEDS) - 0.52
    for si, (skey, slab, c) in enumerate(SC):
        m = s3[s3.scenario == skey][metric].mean()
        ax.plot([-0.5, XEND], [m, m], color=c, lw=0.7, alpha=0.85, zorder=3)
        ax.annotate(f'{m:+.2f}', (XEND, m), textcoords='offset points',
                    xytext=(5, 0), fontsize=6.5, color=c, fontweight='bold',
                    va='center', ha='left', annotation_clip=False, zorder=9)
    ax.axhline(thr, color='#7b3f00', lw=0.8, ls='--', zorder=5)
    ax.annotate(thr_lab, (-0.46, thr), textcoords='offset points', xytext=(2, 3),
                fontsize=6.5, color='#7b3f00', fontweight='bold', ha='left', zorder=9)
    ax.set_xticks(range(len(FEEDS)))
    ax.set_xticklabels(FEEDS, fontsize=6.2, fontweight='bold')
    ax.set_xlim(-0.5, len(FEEDS) - 0.32)
    ax.grid(axis='y', ls=':', lw=0.35, alpha=0.35)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


dist_panel(axA, 'UP', UP_THR, 'Incineration benchmark')
axA.set_ylabel('Unit profit (\\$ kg$^{-1}$)')
axA.set_ylim(-4.0, 0.35)

dist_panel(axB, 'CR', 0.0, 'Incineration benchmark')
axB.set_ylabel('Unit CO$_2$ reduction\n(kgCO$_2$eq kg$^{-1}$)', fontsize=7.0)
axB.set_ylim(-7.5, 4.6)


# ══════════════ legends ══════════════
hs = [Patch(facecolor=c, edgecolor='white', linewidth=0.4, label=l)
      for _, l, c in SC]
leg1 = fig.legend(handles=hs, loc='lower center', bbox_to_anchor=(0.520, 0.012),
                  ncol=3, fontsize=6.5, framealpha=1.0, edgecolor='#bbb',
                  fancybox=False, handletextpad=0.4, borderpad=0.32,
                  columnspacing=1.0, handlelength=1.0)
leg1.get_frame().set_linewidth(0.4)

for lab, (fx, fy) in {'a': (0.012, 0.925), 'b': (0.520, 0.925)}.items():
    fig.text(fx, fy, lab, fontsize=8, fontweight='bold', va='bottom', ha='left')

out = os.path.join(OUT, 'Fig4_electrification.png')
plt.savefig(out, dpi=500, facecolor='white')
plt.close()
print('saved', out)
for skey, slab, _ in SC:
    g = s3[s3.scenario == skey]
    print(f'  {slab:<28} mean UP {g.UP.mean():+.3f}   mean CR {g.CR.mean():+.3f}')
