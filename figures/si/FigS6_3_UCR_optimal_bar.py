import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
# -*- coding: utf-8 -*-
"""Fig. 6 (UCR-optimal): table-style layout matching Fig. 5. A 3-row header
(Feedstock / Main technology / Product) sits directly on the 4th row, the UCR breakdown.
Each column is the best-UCR pathway of a (feed, technology). Bars: carbon credits up,
emissions down. Dual axes: LEFT = absolute unit CO2 reduction (kgCO2eq/kg, navy, diamond);
RIGHT = % of that feed's incineration emissions avoided (orange, circle), with the
right-axis 0 % pinned to the UCR = 0 incineration line.
Reconciliation: UCR = (credits - emissions) x 2e-8; credits = |CO2 credit[171]| (product
displacement) + SUM(CO2 reduction)[170]; emissions = SUM(EMISSION)[43] (cols 30-42).
Per-feed incineration reference E_incin_net at base grid CI = 369 g/kWh."""
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
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial'; mpl.rcParams['mathtext.bf'] = 'Arial:bold'
MM = 1/25.4; FIG_W = 159.2*MM
BASE = config.INTERMEDIATE
SCALE = 2e-8
LEFT_C = '#1b3a5b'; RIGHT_C = '#e67e22'
# per-feed incineration footprint (E_incin_net at base CI = 369 g/kWh), kgCO2eq/kg
E_INCIN = {'PE': 2.122, 'PET': 1.842, 'PP': 2.084, 'PS': 2.467, 'PVC': 0.998}

df = pd.read_excel(config.MASTER_XLSX,
                   sheet_name='All pathways', header=3)
def col(i): return pd.to_numeric(df.iloc[:, i], errors='coerce')
feed = df.iloc[:, 0].astype(str).str.strip(); tech = df.iloc[:, 1].astype(str).str.strip()
prod = df.iloc[:, 4].astype(str).str.strip(); ucr = col(220)

emis = {'Direct process': col(30), 'Electricity': col(31), 'Fired heat': col(32),
        'Refrigeration': sum(col(i) for i in range(33, 38)),
        'Steam': col(38) + col(39) + col(40), 'Raw material': col(41) + col(42)}
cred = {'Product displacement': -col(171), 'Feedstock & capture': col(170)}

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

EM_KEYS = list(emis.keys()); CR_KEYS = list(cred.keys())
# harmonious sequential palettes: warm brown/tan ramp for emissions, green ramp for credits
ECOL = {k: plt.get_cmap('YlOrBr')(v) for k, v in zip(EM_KEYS, np.linspace(0.86, 0.40, len(EM_KEYS)))}
CCOL = {k: plt.get_cmap('Greens')(v) for k, v in zip(CR_KEYS, [0.58, 0.34])}

rows = []
for f in FEEDS:
    for t in TECHS:
        m = (feed == f) & (tech == t)
        if m.sum() == 0: continue
        i = ucr[m].idxmax()
        rec = dict(feed=f, tech=TMAP[t], prod=pname(prod[i]), ucr=float(ucr[i]),
                   rel=float(ucr[i]) / E_INCIN[f] * 100)
        rec['emis'] = {k: float(v[i]) * SCALE for k, v in emis.items()}
        rec['cred'] = {k: float(v[i]) * SCALE for k, v in cred.items()}
        rows.append(rec)

# ---- x layout ----
x = 0.0; gap_bar = 1.0; gap_feed = 1.0; W = 0.86; feed_cols = {}   # no inter-feed gap
for f in FEEDS:
    sub = sorted([r for r in rows if r['feed'] == f], key=lambda r: -r['ucr'])
    best = max(sub, key=lambda r: r['ucr']); xs = []
    for r in sub:
        r['_x'] = x; r['_best'] = (r is best); xs.append(x); x += gap_bar
    feed_cols[f] = (sub, xs); x += gap_feed - gap_bar
XLO0 = feed_cols[FEEDS[0]][1][0] - 0.5
XHI0 = feed_cols[FEEDS[-1]][1][-1] + 0.5
GUT = 1.7
XLO, XHI = XLO0 - GUT, XHI0 + GUT
blocks = {f: (xs[0] - 0.5, xs[-1] + 0.5) for f, (sub, xs) in feed_cols.items()}
blocks[FEEDS[0]] = (XLO, blocks[FEEDS[0]][1])
blocks[FEEDS[-1]] = (blocks[FEEDS[-1]][0], XHI)
bar_center = {f: (xs[0] + xs[-1]) / 2 for f, (sub, xs) in feed_cols.items()}

# ---- figure ----
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
        for k in CR_KEYS:
            v = r['cred'][k]
            if abs(v) < 1e-9: continue
            axb.bar(xx, v, bottom=pos, width=W, color=CCOL[k], edgecolor='none', zorder=3); pos += v
        neg = 0.0
        for k in EM_KEYS:
            v = r['emis'][k]
            if abs(v) < 1e-9: continue
            axb.bar(xx, -v, bottom=neg, width=W, color=ECOL[k], edgecolor='none', zorder=3); neg -= v
        r['_top'] = pos
ymin = min(-sum(r['emis'].values()) for r in rows); ymax = max(sum(r['cred'].values()) for r in rows)
yL0, yL1 = ymin * 1.10, ymax * 1.12
axb.set_ylim(yL0, yL1); axb.set_xlim(XLO, XHI)
for f in FEEDS:
    x0, x1 = blocks[f]
    for xe in (x0, x1):
        axb.plot([xe, xe], [yL0, yL1], color='#bdbdbd', lw=0.8, zorder=1)
for xsep in (XLO0, XHI0):
    axh.plot([xsep, xsep], [0, 1], color='#888', lw=0.9, zorder=4)
    axb.plot([xsep, xsep], [yL0, yL1], color='#888', lw=0.9, ls=(0, (4, 2)), zorder=2)

# right axis (orange); pin 0 % to the incineration line (UCR = 0)
ax2 = axb.twinx(); ax2.patch.set_visible(False)
relmin = min(r['rel'] for r in rows); relmax = max(r['rel'] for r in rows)
p0 = (0.0 - yL0) / (yL1 - yL0)
Rspan = max(relmax / (0.94 - p0), relmin / (0.06 - p0)) * 1.04
ax2.set_ylim(-p0 * Rspan, (1 - p0) * Rspan); ax2.set_xlim(XLO, XHI)
ax2.tick_params(axis='y', labelsize=7, colors=RIGHT_C, direction='in', length=3, labelright=False)
ax2.spines['right'].set_color(RIGHT_C); ax2.spines['left'].set_visible(False)

for r in rows:
    axb.plot(r['_x'], r['ucr'], marker='D', markersize=4.8, color=LEFT_C,
             markeredgecolor='none', zorder=8)
    ax2.plot(r['_x'], r['rel'], marker='o', markersize=4.8, color=RIGHT_C,
             markeredgecolor='none', zorder=8)
    if r['_best']:
        axb.plot(r['_x'], r['_top'] + 0.12, marker='*', markersize=8, color='#d4ac0d',
                 markeredgecolor='#7d6608', markeredgewidth=0.4, zorder=9, clip_on=False)

axb.axhline(0, color='#c0392b', lw=1.3, ls='--', zorder=10)   # incineration = UCR 0 = right 0 %
axb.set_xticks([])
axb.tick_params(axis='y', labelsize=7, colors=LEFT_C, direction='in', length=3, labelleft=False)
axb.spines['left'].set_color(LEFT_C)
axb.grid(axis='y', linestyle=':', alpha=0.30, zorder=0)
# tick labels inside the widened end columns
lticks = [yt for yt in axb.get_yticks() if yL0 + 1e-6 < yt < yL1 - 1e-6]
for yt in lticks:
    va = 'bottom' if yt == min(lticks) else ('top' if yt == max(lticks) else 'center')
    axb.text(XLO + 0.18, yt, f'{yt:.0f}', ha='left', va=va, fontsize=6.5,
             color=LEFT_C, zorder=15,
             bbox=dict(boxstyle='square,pad=0.06', fc='white', ec='none', alpha=0.65))
r0, r1 = ax2.get_ylim()
rticks = [yt for yt in ax2.get_yticks() if r0 + 1e-6 < yt < r1 - 1e-6]
for yt in rticks:
    va = 'bottom' if yt == min(rticks) else ('top' if yt == max(rticks) else 'center')
    ax2.text(XHI - 0.18, yt, f'{yt:.0f}', ha='right', va=va, fontsize=6.5,
             color=RIGHT_C, zorder=15,
             bbox=dict(boxstyle='square,pad=0.06', fc='white', ec='none', alpha=0.65))
# unit tags inside the panel (top corners)
axb.text(XLO + 0.18, yL1 * 0.97, r'kgCO$_{\mathbf{2eq}}$/kg', ha='left', va='top', fontsize=7.5,
         fontweight='bold', color=LEFT_C, zorder=12)
axb.text(XHI - 0.18, yL1 * 0.97, '% of incin. avoided', ha='right', va='top', fontsize=7.5,
         fontweight='bold', color=RIGHT_C, zorder=12)

# ---- left header column ----
fig.canvas.draw()
ph = axh.get_position(); pb = axb.get_position()
HX1 = ph.x0; HX0 = HX1 - 0.040
cells = [(ph.y0 + 0.80 * ph.height, ph.y1, 'Feed'),
         (ph.y0 + 0.30 * ph.height, ph.y0 + 0.80 * ph.height, 'Main technology'),
         (ph.y0, ph.y0 + 0.30 * ph.height, 'Product'),
         (pb.y0, pb.y1, 'UCR breakdown')]
for y0, y1, lab in cells:
    fig.add_artist(Rectangle((HX0, y0), HX1 - HX0, y1 - y0, transform=fig.transFigure,
                   facecolor='#ededed', edgecolor='#888', lw=0.9, zorder=5))
    fig.text((HX0 + HX1) / 2, (y0 + y1) / 2, lab, rotation=90, ha='center', va='center',
             fontsize=9, fontweight='bold', zorder=6)

# ---- split legends: Carbon credit + Emission + marker key ----
from matplotlib.patches import Patch
h_cr = [Patch(facecolor=CCOL[k], edgecolor='white', label=k) for k in CR_KEYS]
h_em = [Patch(facecolor=ECOL[k], edgecolor='white', label=k) for k in EM_KEYS]
lc = axb.legend(handles=h_cr, loc='upper left', bbox_to_anchor=(0.0, -0.01), ncol=2,
                fontsize=7, title='Carbon credit', title_fontsize=8, framealpha=0.95,
                edgecolor='none', fancybox=False, columnspacing=1.0,
                handlelength=1.2, handletextpad=0.5)
lc.get_title().set_fontweight('bold'); lc.get_title().set_color('black'); axb.add_artist(lc)
le = axb.legend(handles=h_em, loc='upper right', bbox_to_anchor=(1.0, -0.01), ncol=3,
                fontsize=7, title='Emission', title_fontsize=8, framealpha=0.95,
                edgecolor='none', fancybox=False, columnspacing=1.0,
                handlelength=1.2, handletextpad=0.5)
le.get_title().set_fontweight('bold'); le.get_title().set_color('black'); axb.add_artist(le)
h_mk = [Line2D([0], [0], marker='D', color='w', markerfacecolor=LEFT_C, markersize=6,
               label='Net unit CO$_2$ reduction (left)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=RIGHT_C, markersize=6,
               label='% of incin. avoided (right)'),
        Line2D([0], [0], color='#c0392b', lw=1.3, ls='--', label='Incineration'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#d4ac0d',
               markeredgecolor='#7d6608', markersize=9, label='Best per feed')]
lm = axb.legend(handles=h_mk, loc='upper center', bbox_to_anchor=(0.5, -0.165), ncol=4,
                fontsize=7, framealpha=0.95, edgecolor='none', fancybox=False,
                columnspacing=1.2, handlelength=1.4, handletextpad=0.5)
axb.add_artist(lm)

out = os.path.join(config.FIG_OUT,
                   'FigS6_3_UCR_optimal_bar.png')
plt.savefig(out, dpi=400); plt.close()   # no tight bbox -> output width == figure width (159.2 mm)
print('saved:', out, '| bars:', len(rows))
print('UCR range %.2f..%.2f | rel %% range %.0f..%.0f | p0=%.3f' %
      (min(r['ucr'] for r in rows), max(r['ucr'] for r in rows), relmin, relmax, p0))
