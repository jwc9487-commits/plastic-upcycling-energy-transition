import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
# -*- coding: utf-8 -*-
"""Fig. 5 (UP-optimal): table-style layout. A 3-row header (Feedstock / Main technology /
Product) sits directly on top of the 4th row, the UP breakdown, with the two panels' borders
touching so each column flows into its bar. Dual axes on the breakdown row:
LEFT = absolute net profit ($/kg, navy, diamond); RIGHT = value relative to incineration
(%, orange, circle), with the right-axis 0 % pinned to the red incineration line."""
import sys, io, os, re
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.labelsize': 9, 'axes.labelweight': 'bold',
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7})
MM = 1/25.4; FIG_W = 159.2*MM
BASE = config.INTERMEDIATE
UP_INCIN = -0.42; SCALE = 0.02
LEFT_C = '#1b3a5b'; RIGHT_C = '#e67e22'

df = pd.read_excel(config.MASTER_XLSX,
                   sheet_name='All pathways', header=3)
def col(i): return pd.to_numeric(df.iloc[:, i], errors='coerce')
feed = df.iloc[:, 0].astype(str).str.strip(); tech = df.iloc[:, 1].astype(str).str.strip()
prod = df.iloc[:, 4].astype(str).str.strip(); up = col(216)

REV = {'Liquid fuel': 133, 'FT fuel': 137, 'Gas product': 134, 'Aromatics': 135,
       'MeOH': 136, 'Acid': 132, 'Hydrogen': 138, 'By-product': 139}
util = sum(col(i) for i in range(7, 25))
acc, plastic, foc, sumcost = col(5), col(6), col(25), col(26)
other_cost = sumcost - (acc + plastic + foc + util)

FEEDS = ['PET', 'PP', 'PVC', 'PS', 'PE']; TECHS = ['HT', 'LT', 'SG', 'OG', 'HG']
TMAP = {'HT': 'H-PY', 'LT': 'L-PY', 'SG': 'S-GS', 'OG': 'O-GS', 'HG': 'HG'}
TECH_FULL = {'H-PY': 'High-temp pyrolysis', 'L-PY': 'Low-temp pyrolysis',
             'S-GS': 'Steam gasification', 'O-GS': 'Oxygen gasification', 'HG': 'Hydrogenolysis'}
PROD_FULL = {'BA': 'Benzoic acid', 'GASOLINE': 'Gasoline', 'FT': 'FT fuels', 'OLEFIN': 'Olefins',
             'MEOH': 'Methanol', 'JET FUEL': 'Jet fuel', 'AROMATICS': 'Aromatics', 'DIESEL': 'Diesel',
             'H2': 'Hydrogen'}
def pname(p):
    p = re.sub(r'[_#]\s*[\d.]+$', '', p).strip().upper()
    return PROD_FULL.get(p, p.title())

RCOL = {'Liquid fuel': '#1a7a3a', 'FT fuel': '#2ca25f', 'Gas product': '#66c2a4',
        'Aromatics': '#99d8c9', 'MeOH': '#41ae76', 'Acid': '#7fcdbb', 'Hydrogen': '#addd8e',
        'By-product': '#c7e9b4'}
CCOL = {'Feedstock': '#7b6a58', 'Capital': '#9e9e9e', 'Fixed O&M': '#bdbdbd',
        'Utility': '#d9b38c', 'Other cost': '#e0e0e0'}
COST_KEYS = ['Feedstock', 'Capital', 'Fixed O&M', 'Utility', 'Other cost']

rows = []
for f in FEEDS:
    for t in TECHS:
        m = (feed == f) & (tech == t)
        if m.sum() == 0: continue
        i = up[m].idxmax()
        rec = dict(feed=f, tech=TMAP[t], prod=pname(prod[i]), up=float(up[i]),
                   rel=(float(up[i]) - UP_INCIN) / abs(UP_INCIN) * 100)
        rec['rev'] = {k: float(col(c)[i]) * SCALE for k, c in REV.items()}
        rec['cost'] = {'Feedstock': float(plastic[i]) * SCALE, 'Capital': float(acc[i]) * SCALE,
                       'Fixed O&M': float(foc[i]) * SCALE, 'Utility': float(util[i]) * SCALE,
                       'Other cost': float(other_cost[i]) * SCALE}
        rows.append(rec)

# ---- x layout ----
x = 0.0; gap_bar = 1.0; gap_feed = 1.0; W = 0.86; feed_cols = {}   # no inter-feed gap
for f in FEEDS:
    sub = sorted([r for r in rows if r['feed'] == f], key=lambda r: -r['up'])
    best = max(sub, key=lambda r: r['up']); xs = []
    for r in sub:
        r['_x'] = x; r['_best'] = (r is best); xs.append(x); x += gap_bar
    feed_cols[f] = (sub, xs); x += gap_feed - gap_bar
XLO0 = feed_cols[FEEDS[0]][1][0] - 0.5           # first bar-cell edge
XHI0 = feed_cols[FEEDS[-1]][1][-1] + 0.5         # last bar-cell edge
GUT = 1.7                                        # inside gutter (widens the end columns)
XLO, XHI = XLO0 - GUT, XHI0 + GUT
blocks = {f: (xs[0] - 0.5, xs[-1] + 0.5) for f, (sub, xs) in feed_cols.items()}
blocks[FEEDS[0]] = (XLO, blocks[FEEDS[0]][1])    # widen first (PET) column to the left
blocks[FEEDS[-1]] = (blocks[FEEDS[-1]][0], XHI)  # widen last (PE) column to the right
bar_center = {f: (xs[0] + xs[-1]) / 2 for f, (sub, xs) in feed_cols.items()}

# ---- figure: header table (top) sitting directly on the breakdown (bottom) ----
fig = plt.figure(figsize=(FIG_W, FIG_W * 1.18))
gs = fig.add_gridspec(2, 1, height_ratios=[0.62, 1.0], hspace=0.0,
                      left=0.060, right=0.988, top=0.985, bottom=0.205)
axh = fig.add_subplot(gs[0]); axb = fig.add_subplot(gs[1], sharex=axh)
axh.set_xlim(XLO, XHI); axh.set_ylim(0, 1); axh.axis('off')

Y_FS, Y_TE, Y_PR = (0.80, 1.0), (0.30, 0.80), (0.0, 0.30)
def hline(x0, x1, y, **kw): axh.plot([x0, x1], [y, y], **kw)
for fi, f in enumerate(FEEDS):
    sub, xs = feed_cols[f]; x0, x1 = blocks[f]
    axh.add_patch(Rectangle((x0, 0), x1 - x0, 1, fill=False, edgecolor='#888', lw=0.9, zorder=4))
    hline(x0, x1, Y_FS[0], color='#888', lw=0.9, zorder=4)
    hline(x0, x1, Y_TE[0], color='#888', lw=0.9, zorder=4)
    axh.text(bar_center[f], 0.90, f, ha='center', va='center', fontsize=9, fontweight='bold')
    for ci, (r, xc) in enumerate(zip(sub, xs)):
        if ci > 0:
            axh.plot([xc - 0.5, xc - 0.5], [0, Y_FS[0]], color='#cfcfcf', lw=0.5, zorder=3)
        axh.text(xc, sum(Y_TE) / 2, TECH_FULL[r['tech']], ha='center', va='center',
                 rotation=90, fontsize=7.0, color='#222')
        axh.text(xc, sum(Y_PR) / 2, r['prod'], ha='center', va='center',
                 rotation=90, fontsize=7.0, color='#222')

# ================= breakdown row =================
for f in FEEDS:
    sub, xs = feed_cols[f]; x0, x1 = blocks[f]
    for r in sub:
        xx = r['_x']; pos = 0.0
        for k in REV:
            v = r['rev'][k]
            if abs(v) < 1e-9: continue
            axb.bar(xx, v, bottom=pos, width=W, color=RCOL[k], edgecolor='none', zorder=3); pos += v
        neg = 0.0
        for k in COST_KEYS:
            v = r['cost'][k]
            if abs(v) < 1e-9: continue
            axb.bar(xx, -v, bottom=neg, width=W, color=CCOL[k], edgecolor='none', zorder=3); neg -= v
        r['_top'] = pos
ymin = min(-sum(r['cost'].values()) for r in rows); ymax = max(r['_top'] for r in rows)
yL0, yL1 = ymin * 1.10, ymax * 1.12
axb.set_ylim(yL0, yL1); axb.set_xlim(XLO, XHI)
# continue the per-feed column borders down into the bars
for f in FEEDS:
    x0, x1 = blocks[f]
    for xe in (x0, x1):
        axb.plot([xe, xe], [yL0, yL1], color='#bdbdbd', lw=0.8, zorder=1)
# gutter/bar separators: solid through the header table, dashed through the chart
for xsep in (XLO0, XHI0):
    axh.plot([xsep, xsep], [0, 1], color='#888', lw=0.9, zorder=4)
    axb.plot([xsep, xsep], [yL0, yL1], color='#888', lw=0.9, ls=(0, (4, 2)), zorder=2)

# right axis (orange); pin 0 % to the incineration line (UP = -0.42)
ax2 = axb.twinx(); ax2.patch.set_visible(False)
pL = (UP_INCIN - yL0) / (yL1 - yL0); Rspan = 112.0
ax2.set_ylim(-pL * Rspan, (1 - pL) * Rspan); ax2.set_xlim(XLO, XHI)
ax2.tick_params(axis='y', labelsize=7, colors=RIGHT_C, direction='in', length=3,
                labelright=False)
ax2.spines['right'].set_color(RIGHT_C); ax2.spines['left'].set_visible(False)

for r in rows:
    axb.plot(r['_x'], r['up'], marker='D', markersize=4.8, color=LEFT_C,
             markeredgecolor='none', zorder=8)
    ax2.plot(r['_x'], r['rel'], marker='o', markersize=4.8, color=RIGHT_C,
             markeredgecolor='none', zorder=8)
    if r['_best']:
        axb.plot(r['_x'], r['_top'] + 0.06, marker='*', markersize=8, color='#d4ac0d',
                 markeredgecolor='#7d6608', markeredgewidth=0.4, zorder=9, clip_on=False)

axb.axhline(UP_INCIN, color='#c0392b', lw=1.3, ls='--', zorder=10)   # = right-axis 0 %
axb.axhline(0, color='#999', lw=0.6, zorder=2)
axb.set_xticks([])
axb.tick_params(axis='y', labelsize=7, colors=LEFT_C, direction='in', length=3,
                labelleft=False)
axb.spines['left'].set_color(LEFT_C)
axb.grid(axis='y', linestyle=':', alpha=0.30, zorder=0)
# tick labels INSIDE the panel, in the widened end columns (no overlap with bars)
lticks = [yt for yt in axb.get_yticks() if yL0 + 1e-6 < yt < yL1 - 1e-6]
for yt in lticks:
    va = 'bottom' if yt == min(lticks) else ('top' if yt == max(lticks) else 'center')
    axb.text(XLO + 0.18, yt, f'{yt:.2f}', ha='left', va=va, fontsize=6.5,
             color=LEFT_C, zorder=15,
             bbox=dict(boxstyle='square,pad=0.06', fc='white', ec='none', alpha=0.65))
r0, r1 = ax2.get_ylim()
rticks = [yt for yt in ax2.get_yticks() if r0 + 1e-6 < yt < r1 - 1e-6]
for yt in rticks:
    va = 'bottom' if yt == min(rticks) else ('top' if yt == max(rticks) else 'center')
    ax2.text(XHI - 0.18, yt, f'{yt:.0f}', ha='right', va=va, fontsize=6.5,
             color=RIGHT_C, zorder=15,
             bbox=dict(boxstyle='square,pad=0.06', fc='white', ec='none', alpha=0.65))
# unit tags INSIDE the panel (top corners, in the headroom above the bars)
axb.text(XLO + 0.18, yL1 * 0.97, '\\$/kg', ha='left', va='top', fontsize=7.5,
         fontweight='bold', color=LEFT_C, zorder=12)
axb.text(XHI - 0.18, yL1 * 0.97, '% vs. incineration', ha='right', va='top', fontsize=7.5,
         fontweight='bold', color=RIGHT_C, zorder=12)

# ---- left header column: vertical labels in bordered cells (table row-headers) ----
fig.canvas.draw()
ph = axh.get_position(); pb = axb.get_position()
HX1 = ph.x0                 # right edge flush with the table's left edge (no gap)
HX0 = HX1 - 0.040
cells = [(ph.y0 + 0.80 * ph.height, ph.y1, 'Feed'),
         (ph.y0 + 0.30 * ph.height, ph.y0 + 0.80 * ph.height, 'Main technology'),
         (ph.y0, ph.y0 + 0.30 * ph.height, 'Product'),
         (pb.y0, pb.y1, 'UP breakdown')]
for y0, y1, lab in cells:
    fig.add_artist(Rectangle((HX0, y0), HX1 - HX0, y1 - y0, transform=fig.transFigure,
                   facecolor='#ededed', edgecolor='#888', lw=0.9, zorder=5))
    fig.text((HX0 + HX1) / 2, (y0 + y1) / 2, lab, rotation=90, ha='center', va='center',
             fontsize=9, fontweight='bold', zorder=6)

# ---- split legends: Revenue + Cost + marker key ----
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
rev_present = [k for k in REV if any(abs(r['rev'][k]) > 1e-6 for r in rows)]
h_rev = [Patch(facecolor=RCOL[k], edgecolor='white', label=k) for k in rev_present]
h_cost = [Patch(facecolor=CCOL[k], edgecolor='white', label=k) for k in COST_KEYS]
lr = axb.legend(handles=h_rev, loc='upper left', bbox_to_anchor=(0.0, -0.01), ncol=4,
                fontsize=7, title='Revenue', title_fontsize=8, framealpha=0.95,
                edgecolor='none', fancybox=False, columnspacing=1.0,
                handlelength=1.2, handletextpad=0.5)
lr.get_title().set_fontweight('bold'); lr.get_title().set_color('black'); axb.add_artist(lr)
lc = axb.legend(handles=h_cost, loc='upper right', bbox_to_anchor=(1.0, -0.01), ncol=3,
                fontsize=7, title='Cost', title_fontsize=8, framealpha=0.95,
                edgecolor='none', fancybox=False, columnspacing=1.0,
                handlelength=1.2, handletextpad=0.5)
lc.get_title().set_fontweight('bold'); lc.get_title().set_color('black'); axb.add_artist(lc)
h_mk = [Line2D([0], [0], marker='D', color='w', markerfacecolor=LEFT_C, markersize=6,
               label='Net unit profit (left)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=RIGHT_C, markersize=6,
               label='Relative to incin. (right)'),
        Line2D([0], [0], color='#c0392b', lw=1.3, ls='--', label='Incineration'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#d4ac0d',
               markeredgecolor='#7d6608', markersize=9, label='Best per feed')]
lm = axb.legend(handles=h_mk, loc='upper center', bbox_to_anchor=(0.5, -0.165), ncol=4,
                fontsize=7, framealpha=0.95, edgecolor='none', fancybox=False,
                columnspacing=1.2, handlelength=1.4, handletextpad=0.5)
axb.add_artist(lm)

out = os.path.join(config.FIG_OUT,
                   'FigS6_2_UP_optimal_bar.png')
plt.savefig(out, dpi=400); plt.close()   # no tight bbox -> output width == figure width (159.2 mm)
print('saved:', out, '| bars:', len(rows), '| pL=%.3f -> right 0%% at incineration' % pL)
