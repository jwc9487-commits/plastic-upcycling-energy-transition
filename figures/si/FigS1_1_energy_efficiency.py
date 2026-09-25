import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
# -*- coding: utf-8 -*-
"""Supplementary Fig. S1.1: distribution of energy efficiency across the 959 upcycling
pathways, with a normal fit and the incineration benchmark.

Energy efficiency = column 'Energy efficiency' of the master pathway table (959 rows).
The incineration benchmark efficiency (0.255) is the value of the incineration process
model described in Supplementary Section S1.
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.labelsize': 9, 'axes.labelweight': 'bold',
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7})
MM = 1 / 25.4
INCIN_EFF = 0.255
BIN_W = 0.02
BAR_C, EDGE_C, FIT_C, INC_C = '#5aa050', '#3d6b36', '#2f5a2a', '#d62728'

src = os.path.join(config.SOURCE_DATA, 'si', 'FigS1_1_energy_efficiency.csv')
eff = pd.read_csv(src)['Energy_efficiency'].to_numpy()
assert len(eff) == 959
mu, sigma = eff.mean(), eff.std(ddof=1)

bins = np.arange(0.20, 0.86 + 1e-9, BIN_W)
w = np.full(len(eff), 1.0 / len(eff))                       # relative frequency

fig, ax = plt.subplots(figsize=(159.2 * MM, 68 * MM))
ax.hist(eff, bins=bins, weights=w, color=BAR_C, edgecolor=EDGE_C, linewidth=0.5, zorder=2)
x = np.linspace(0.20, 0.86, 400)
pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
ax.plot(x, pdf * BIN_W, color=FIT_C, lw=1.2, zorder=3, label='Normal fit')
ax.axvline(INCIN_EFF, color=INC_C, ls='--', lw=1.3, zorder=4)
ax.text(INCIN_EFF + 0.006, 0.097, 'Incineration', color=INC_C, fontsize=7,
        rotation=90, va='top', ha='left', fontweight='bold')
ax.text(0.985, 0.95, f'Normal fit\nμ = {mu:.3f}\nσ = {sigma:.3f}', transform=ax.transAxes,
        ha='right', va='top', fontsize=7, linespacing=1.3,
        bbox=dict(boxstyle='square,pad=0.35', fc='white', ec='0.6', lw=0.5))
ax.set_xlim(0.20, 0.86); ax.set_ylim(0, 0.10)
ax.set_xticks(np.arange(0.2, 0.81, 0.1)); ax.set_yticks([0, 0.025, 0.05, 0.075, 0.10])
ax.set_xlabel('Energy efficiency'); ax.set_ylabel('Relative frequency')
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.tick_params(direction='out', length=3, width=0.6)
fig.tight_layout(pad=0.4)
out = os.path.join(config.FIG_OUT, 'FigS1_1_energy_efficiency.png')
fig.savefig(out, dpi=500)
print(f'mu={mu:.4f} sigma={sigma:.4f} n={len(eff)} -> {out}')
