import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
# -*- coding: utf-8 -*-
"""Fig. S7.5 (journal style) — sequential lever decomposition of the 2026->2050
change in composition-weighted net profit, by region. INCIN-corrected data.
Matches the main paper figure format (Arial, 159.2 mm, unified legend frame)."""
import sys, io, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
mpl.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'pdf.fonttype':42,'ps.fonttype':42,
    'mathtext.fontset':'custom','mathtext.rm':'Arial',
    'mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold',
    'mathtext.default':'regular',
    'axes.labelsize':9,'axes.labelweight':'bold',
    'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7.5,})
MM = 1/25.4; FIG_W = 159.2*MM

DATA = config.INTERMEDIATE
OUT  = os.path.join(config.FIG_OUT, 'FigS7_5_decomposition.png')
df = pd.read_csv(os.path.join(DATA, 'm3_step5_RC_decomposition_INCIN.csv'))

regions = list(df['Region'])
x = np.arange(len(regions))
# 2026-08-26: house palette, taken from the main text (Fig. 3 driver colours)
LEVERS = [('dL1_tau','Carbon price','#e8b93d'),
          ('dL2_P_PPA','PPA electricity price','#b5651d'),
          ('dL3_CI','Grid + PPA carbon intensity','#3f8f5a'),
          ('dL4_H2cx','Green H$_2$ capital cost','#4878d0'),
          ('dL5_CR_tgt','CO$_2$-reduction target','#8a8a8a')]
s0 = df['S0_2026'].values
s5 = df['S5_2050'].values

fig, ax = plt.subplots(figsize=(FIG_W, FIG_W*0.52))
fig.subplots_adjust(left=0.085, right=0.985, top=0.965, bottom=0.255)
width = 0.56
pos_b = np.zeros(len(regions)); neg_b = np.zeros(len(regions))
for key, lab, col in LEVERS:
    v = df[key].values
    pos = np.where(v > 0, v, 0); neg = np.where(v < 0, v, 0)
    ax.bar(x, pos, width, bottom=pos_b, color=col, edgecolor='white', linewidth=0.7, label=lab)
    ax.bar(x, neg, width, bottom=neg_b, color=col, edgecolor='white', linewidth=0.7)
    pos_b += pos; neg_b += neg

# 2026 / 2050 state markers + connector
for i in range(len(regions)):
    ax.plot([x[i], x[i]], [s0[i], s5[i]], color='#9aa0a6', ls=':', lw=0.8, zorder=1)
ax.plot(x, s0, 'o', ms=8, mfc='#1c2833', mec='none', mew=0, zorder=10, label='2026 baseline')
ax.plot(x, s5, 'D', ms=7.5, mfc='white', mec='#1c2833', mew=1.1, zorder=10, label='2050 endpoint')

ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(regions, fontsize=8, fontweight='bold')
ax.set_ylabel(r'Composition-weighted net profit (\$/kg)', fontsize=9)
ax.set_xlim(-0.6, len(regions)-0.4)
ax.grid(axis='y', ls=':', alpha=0.4)
for sp in ('top','right'):
    ax.spines[sp].set_visible(True); ax.spines[sp].set_color('#888'); ax.spines[sp].set_linewidth(0.6)
for i in range(len(regions)):
    ax.annotate(f'$\\mathbf{{\\Delta}}$={s5[i]-s0[i]:+.3f}', xy=(x[i], s5[i]),
                xytext=(0, 7), textcoords='offset points',
                ha='center', fontsize=7, fontweight='bold', color='#1c2833')

handles = [Patch(facecolor=c, label=l) for _,l,c in LEVERS] + [
    Line2D([],[], marker='o', ls='none', mfc='#1c2833', mec='none', mew=0, ms=8, label='2026 baseline'),
    Line2D([],[], marker='D', ls='none', mfc='white', mec='#1c2833', mew=1.1, ms=7.5, label='2050 endpoint')]
leg = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.13),
                ncol=4, frameon=False,
                fancybox=False, columnspacing=1.2, handletextpad=0.5)
leg.get_frame().set_linewidth(0.8)

plt.savefig(OUT, dpi=400, bbox_inches='tight', facecolor='white')
plt.close()
from PIL import Image
w,h = Image.open(OUT).size
print(f'saved {OUT}  {w}x{h}  ratio={w/h:.4f}')
