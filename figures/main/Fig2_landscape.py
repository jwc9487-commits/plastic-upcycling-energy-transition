import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
# -*- coding: utf-8 -*-
"""Fig. 2 — viability landscape of all 959 upcycling pathways.

Replaces the old Fig. 3 + Fig. 4 pair. Those drew the same 959-point scatter five
times over 273 mm of page height; here it is drawn once, and the feedstock,
technology and product breakdowns become compact zone-composition bars.

(a) scatter against the incineration benchmark, partitioned into ZONEs 1-4
(b) zone composition by feedstock
(c) zone composition by technology
(d) zone composition by product
(e) composition of ZONE 1 as feedstock, technology, product
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.labelsize': 8, 'axes.labelweight': 'bold',
    'xtick.labelsize': 6.2, 'ytick.labelsize': 6.2,
    'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.0, 'ytick.major.size': 2.0,
})
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial'
mpl.rcParams['mathtext.bf'] = 'Arial:bold'

DATA = config.INTERMEDIATE
OUT = config.FIG_OUT
MM = 1 / 25.4
FIG_W, FIG_H = 174.0 * MM, 165.0 * MM


def pretty(s):
    """Arial has no subscript glyphs, so subscripts must go through mathtext."""
    return (str(s).replace('₂', '$_2$')
                  .replace('JET FUEL', 'Jet fuel')
                  .replace('H2', 'H$_2$'))

UP_THR, CR_THR = -0.42, 0.0
ZC = {1: '#4a9b5c', 2: '#e8b93d', 3: '#5b8fc9', 4: '#b0b0b0'}
ZBG = {1: '#e8f3ea', 2: '#fbf3dc', 3: '#e6eef7', 4: '#f0f0f0'}
ZNAME = {1: 'ZONE 1', 2: 'ZONE 2', 3: 'ZONE 3', 4: 'ZONE 4'}
TECH_LAB = {'HT': 'H-PY', 'LT': 'L-PY', 'SG': 'S-GS', 'OG': 'O-GS', 'HG': 'HG'}

df = pd.read_csv(os.path.join(DATA, 'zone_classification.csv'))
df.columns = [c.lstrip('_') for c in df.columns]
df['tech'] = df['tech'].map(lambda t: TECH_LAB.get(t, t))
N = len(df)

fig = plt.figure(figsize=(FIG_W, FIG_H))
# Two rows. Three columns at 174 mm forced every label under 6 pt, which Cell
# does not accept; releasing the pressure vertically fixes it without cutting
# content. The sunburst needs the width most, so it gets half of the lower row.
axA = fig.add_axes([0.062, 0.640, 0.500, 0.320])
axB = fig.add_axes([0.700, 0.830, 0.250, 0.130])
axC = fig.add_axes([0.700, 0.655, 0.250, 0.130])
axE = fig.add_axes([0.045, 0.055, 0.470, 0.496])   # square, so the disc fills it
axD = fig.add_axes([0.645, 0.115, 0.305, 0.390])

# ══════════════ (a) landscape ══════════════
xlo, xhi, ylo, yhi = -1.35, 0.12, -8.8, 4.6
axA.add_patch(plt.Rectangle((UP_THR, CR_THR), xhi - UP_THR, yhi - CR_THR,
                            fc=ZBG[1], ec='none', zorder=0))
axA.add_patch(plt.Rectangle((UP_THR, ylo), xhi - UP_THR, CR_THR - ylo,
                            fc=ZBG[2], ec='none', zorder=0))
axA.add_patch(plt.Rectangle((xlo, CR_THR), UP_THR - xlo, yhi - CR_THR,
                            fc=ZBG[3], ec='none', zorder=0))
axA.add_patch(plt.Rectangle((xlo, ylo), UP_THR - xlo, CR_THR - ylo,
                            fc=ZBG[4], ec='none', zorder=0))

for z in (4, 3, 2, 1):
    s = df[df.zone == z]
    axA.scatter(s.up, s.cr, s=3.4, c=ZC[z], lw=0, alpha=0.80, zorder=3 + z)

axA.axvline(UP_THR, color='#333', lw=0.8, ls='--', zorder=9)
axA.axhline(CR_THR, color='#333', lw=0.8, ls='--', zorder=9)
axA.plot(UP_THR, CR_THR, marker='o', ms=4.2, mfc='#1c2833', mec='white',
         mew=0.7, zorder=11)
axA.annotate('Incineration benchmark', (UP_THR, CR_THR),
             textcoords='offset points', xytext=(11, -16), fontsize=6.2,
             fontweight='bold', color='#1c2833', ha='left', zorder=12,
             arrowprops=dict(arrowstyle='-', lw=0.5, color='#1c2833',
                             shrinkA=0, shrinkB=3))

cnt = df.zone.value_counts()
for z, (xf, yf, va, ha) in {1: (0.985, 0.975, 'top', 'right'),
                            2: (0.985, 0.022, 'bottom', 'right'),
                            3: (0.018, 0.975, 'top', 'left'),
                            4: (0.018, 0.022, 'bottom', 'left')}.items():
    axA.text(xf, yf, f'{ZNAME[z]}\n{cnt[z]/N*100:.0f}%  (n = {cnt[z]})',
             transform=axA.transAxes, ha=ha, va=va, fontsize=7.0,
             fontweight='bold', color=ZC[z] if z != 4 else '#6d6d6d', zorder=12)

axA.set_xlim(xlo, xhi); axA.set_ylim(ylo, yhi)
axA.set_xlabel('Unit profit (\\$ kg$^{-1}$)')
axA.set_ylabel('Unit CO$_2$ reduction (kgCO$_2$eq kg$^{-1}$)')

# ══════════════ (b)(c)(d) zone composition ══════════════
def comp_bar(ax, col, title=''):
    g = df.groupby(col).zone.value_counts().unstack(fill_value=0)
    for z in (1, 2, 3, 4):
        if z not in g.columns:
            g[z] = 0
    g = g[[1, 2, 3, 4]]
    g['n'] = g.sum(axis=1)
    # rank by ZONE 1 share so the best performers sit at the top of each panel
    g = g.assign(_s=g[1] / g['n']).sort_values('_s').drop(columns='_s')
    frac = g[[1, 2, 3, 4]].div(g['n'], axis=0) * 100
    y = np.arange(len(g))
    left = np.zeros(len(g))
    for z in (1, 2, 3, 4):
        ax.barh(y, frac[z], left=left, height=0.66, color=ZC[z],
                edgecolor='white', linewidth=0.4, zorder=3)
        left += frac[z].values
    for i, (idx, r) in enumerate(g.iterrows()):
        if frac.loc[idx, 1] >= 6:
            ax.text(frac.loc[idx, 1] / 2, i, f'{frac.loc[idx,1]:.0f}',
                    ha='center', va='center', fontsize=6.0, color='white',
                    fontweight='bold', zorder=6)
        ax.text(101.5, i, f'{int(r["n"])}', ha='left', va='center',
                fontsize=6.0, color='#555', zorder=6)
    ax.set_yticks(y); ax.set_yticklabels([pretty(i) for i in g.index], fontsize=6.5)
    ax.set_xlim(0, 100); ax.set_ylim(-0.62, len(g) - 0.38)
    ax.tick_params(axis='y', length=0)
    ax.set_xticks([0, 50, 100])
    ax.set_title(title, fontsize=7.2, fontweight='bold', pad=2.5, loc='left')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    return g

comp_bar(axB, 'feed', title='by feedstock')
comp_bar(axC, 'tech', title='by technology')
gD = comp_bar(axD, 'prod_cat', title='by product')
axB.set_xticklabels([])
axC.set_xlabel('Share of pathways (%)', fontsize=7.2)
axD.set_xlabel('Share of pathways (%)', fontsize=7.2)
axD.text(101.5, len(gD) - 0.15, 'n', fontsize=6.0, color='#555',
         ha='left', va='center', fontweight='bold')

# ══════════════ (e) ZONE 1 composition ══════════════
z1 = df[df.zone == 1]
FEED_C = {'PE': '#2f6f9f', 'PP': '#3f8f5a', 'PET': '#d99a2b',
          'PS': '#8a6bb1', 'PVC': '#c0504d'}

def lighten(hexc, f):
    c = np.array(mpl.colors.to_rgb(hexc))
    return tuple(c + (1 - c) * f)

feed_order = z1.feed.value_counts().index.tolist()
inner_v, inner_c, inner_l = [], [], []
mid_v, mid_c, mid_l = [], [], []
out_v, out_c, out_l = [], [], []
for f in feed_order:
    sub = z1[z1.feed == f]
    inner_v.append(len(sub)); inner_c.append(FEED_C[f]); inner_l.append(f)
    for t in sub.tech.value_counts().index:
        s2 = sub[sub.tech == t]
        mid_v.append(len(s2)); mid_c.append(lighten(FEED_C[f], 0.34)); mid_l.append(t)
        for p in s2.prod_cat.value_counts().index:
            n = len(s2[s2.prod_cat == p])
            out_v.append(n); out_c.append(lighten(FEED_C[f], 0.62)); out_l.append(p)

WEDGE = dict(edgecolor='white', linewidth=0.7)
R1, R2, R3 = 0.42, 0.71, 1.00
axE.pie(inner_v, radius=R1, colors=inner_c, startangle=90, counterclock=False,
        wedgeprops=dict(width=R1, **WEDGE))
axE.pie(mid_v, radius=R2, colors=mid_c, startangle=90, counterclock=False,
        wedgeprops=dict(width=R2 - R1, **WEDGE))
axE.pie(out_v, radius=R3, colors=out_c, startangle=90, counterclock=False,
        wedgeprops=dict(width=R3 - R2, **WEDGE))

def ring_labels(vals, labs, r, fs, minfrac, bold=False, suffix=False):
    tot = sum(vals); ang = 90.0
    for v, l in zip(vals, labs):
        d = 360.0 * v / tot
        if v / tot >= minfrac:
            a = np.deg2rad(ang - d / 2)
            txt = f'{l}\n{v}' if suffix else l
            axE.text(r * np.cos(a), r * np.sin(a), txt, ha='center', va='center',
                     fontsize=fs, fontweight='bold' if bold else 'normal',
                     color='#1c2833' if not bold else 'white', zorder=6)
        ang -= d

ring_labels(inner_v, inner_l, (R1) * 0.62, 7.5, 0.055, bold=True, suffix=True)
ring_labels(mid_v, mid_l, (R1 + R2) / 2, 6.5, 0.060)
# outer ring carries no labels: the product breakdown is panel (d), and at
# this radius the text could not reach 6 pt. Colour keeps the linkage visible.

# PVC is a single pathway and would otherwise vanish into the ring
i_pvc = inner_l.index('PVC')
ang = 90.0 - 360.0 * sum(inner_v[:i_pvc]) / sum(inner_v) - \
      360.0 * inner_v[i_pvc] / sum(inner_v) / 2
a = np.deg2rad(ang)
axE.annotate('PVC  n = 1', (R3 * np.cos(a), R3 * np.sin(a)),
             textcoords='offset points', xytext=(-16, 15), fontsize=6.2,
             fontweight='bold', color=FEED_C['PVC'], ha='right',
             arrowprops=dict(arrowstyle='-', lw=0.5, color=FEED_C['PVC'],
                             shrinkA=0, shrinkB=1), zorder=9)
axE.set_title(f'ZONE 1 composition  (n = {len(z1)})', fontsize=7.2,
              fontweight='bold', pad=4.0)
axE.set(aspect='equal')

# shared zone legend
ZTXT = {1: 'better on both', 2: 'profit only', 3: 'carbon only', 4: 'neither'}
handles = [Patch(facecolor=ZC[z], edgecolor='white', linewidth=0.5,
                 label=f'{ZNAME[z]}  {ZTXT[z]}') for z in (1, 2, 3, 4)]
leg = fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.500, 0.008),
                 ncol=4, fontsize=6.5, framealpha=1.0, edgecolor='#bbb',
                 fancybox=False, handletextpad=0.3, borderpad=0.32,
                 columnspacing=1.1, handlelength=0.9)
leg.get_frame().set_linewidth(0.4)

# panel letters placed in figure coordinates so they clear the tick labels
for lab, (fx, fy) in {'a': (0.010, 0.968), 'b': (0.640, 0.968),
                      'c': (0.640, 0.820), 'd': (0.588, 0.520),
                      'e': (0.010, 0.560)}.items():
    fig.text(fx, fy, lab, fontsize=8.5, fontweight='bold', va='bottom', ha='left')

out = os.path.join(OUT, 'Fig2_landscape.png')
plt.savefig(out, dpi=500, facecolor='white')
plt.close()
print('saved', out)
print('zones:', {int(k): int(v) for k, v in df.zone.value_counts().sort_index().items()})
print('ZONE 1 by feed:', z1.feed.value_counts().to_dict())
