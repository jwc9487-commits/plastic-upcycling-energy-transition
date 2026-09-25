import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
# -*- coding: utf-8 -*-
"""Fig. S7.7 (journal style) — global (Sobol) variance decomposition for
(a) unit profit (UP) and (b) unit CO2 reduction (UCR). Two panels resolve the
legend / category-text overlap of the old single-panel figure, and the right
panel fulfils the 'and unit CO2 reduction' part of the caption."""
import sys, io, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
mpl.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'pdf.fonttype':42,'ps.fonttype':42,
    'mathtext.fontset':'custom','mathtext.rm':'Arial',
    'mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold',
    'mathtext.default':'regular',
    'axes.labelsize':9,'axes.labelweight':'bold',
    'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,})
MM = 1/25.4; FIG_W = 159.2*MM
DATA = config.INTERMEDIATE
OUT  = os.path.join(config.FIG_OUT, 'FigS7_7_sobol.png')

up = pd.read_csv(os.path.join(DATA, 'sobol_indices.csv'))
cr = pd.read_csv(os.path.join(DATA, 'sobol_indices_CR.csv'))

NAME = {'P_elec':'Electricity price','CI_elec':'Electricity CI','P_FH':'Fired-heat price',
        'CI_FH':'Fired-heat CI','P_LP':'LP-steam price','P_HP':'HP-steam price',
        'P_H2':'Hydrogen price','plastic_factor':'Plastic price','tau':'Carbon price',
        'i_rate':'Interest rate','alpha':'Renewable share'}
CATEGORY = {'P_elec':'Energy','CI_elec':'Energy','P_FH':'Energy','CI_FH':'Energy',
            'P_LP':'Energy','P_HP':'Energy','P_H2':'Energy','alpha':'Energy',
            'plastic_factor':'Market','tau':'Policy','i_rate':'Finance'}
# 2026-08-26: house palette, shared with Fig. S7.6
COL = {'Energy':'#b5651d','Policy':'#e8b93d','Market':'#3f8f5a','Finance':'#4878d0'}

# common driver order = by UP ST descending
order = list(up.sort_values('ST', ascending=False)['driver'])

def cat_shares(d):
    tot = d['ST'].sum()
    s = {}
    for _, r in d.iterrows():
        c = CATEGORY[r['driver']]; s[c] = s.get(c, 0) + r['ST']
    return {c: s.get(c, 0)/tot*100 for c in ['Energy','Policy','Market','Finance']}

def panel(ax, d, title, tag):
    dd = d.set_index('driver')
    y = np.arange(len(order))[::-1]
    ST = np.array([dd.loc[k,'ST'] for k in order])
    S1 = np.array([max(dd.loc[k,'S1'],0) for k in order])
    cols = [COL[CATEGORY[k]] for k in order]
    ax.barh(y, ST, color=cols, alpha=0.55, edgecolor='white', linewidth=0.6)
    ax.barh(y, S1, color=cols, alpha=1.0, edgecolor='white', linewidth=0.6, hatch='////')
    for yi, k in zip(y, order):
        v = dd.loc[k,'ST']
        if v > 0.004:
            ax.text(v+0.012, yi, f'{v:.2f}', va='center', fontsize=6.2, color='#333')
    ax.set_yticks(y); ax.set_yticklabels([NAME[k] for k in order], fontsize=7.2)
    ax.set_xlabel('Sobol index', fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', ls=':', alpha=0.4)
    for sp in ('top','right'): ax.spines[sp].set_color('#888'); ax.spines[sp].set_linewidth(0.6)
    sh = cat_shares(d)
    sub = '   '.join(f'{c} {sh[c]:.0f}%' for c in ['Energy','Policy','Market','Finance'] if sh[c] >= 0.5)
    ax.set_title(f'{title}\n{sub}', fontsize=8.5, fontweight='bold', pad=4)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(FIG_W, FIG_W*0.52))
fig.subplots_adjust(left=0.135, right=0.985, top=0.86, bottom=0.235, wspace=0.62)
panel(axL, up, 'Unit profit (UP)', '(a)')
panel(axR, cr, r'Unit CO$_\mathbf{2}$ reduction (UCR)', '(b)')

handles = [Patch(facecolor=COL[c], label=c) for c in ['Energy','Policy','Market','Finance']] + [
    Patch(facecolor='#bbb', alpha=0.55, label='Total-order (S$_T$)'),
    Patch(facecolor='#bbb', hatch='////', label='First-order (S$_1$)')]
leg = fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.01),
                 ncol=6, frameon=False,
                 fancybox=False, columnspacing=1.1, handletextpad=0.45)
leg.get_frame().set_linewidth(0.8)

plt.savefig(OUT, dpi=400, bbox_inches='tight', facecolor='white')
plt.close()
from PIL import Image
w,h = Image.open(OUT).size
print(f'saved {OUT}  {w}x{h}  ratio={w/h:.4f}')
