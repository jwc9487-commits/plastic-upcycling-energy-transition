import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
"""Tornado plot: ±50% perturbation sensitivity (Phase 8 RC2 visualization).

Each driver:
  Δ_low  = mean ΔUP at -50% perturbation  (or 0→half range for zero-baseline)
  Δ_high = mean ΔUP at +50% perturbation
Plot bars from baseline UP center to Δ_low (left) and Δ_high (right).
Drivers sorted by max(|Δ_low|, |Δ_high|) descending.
"""
import sys, io, os
os.environ['PYTHONUTF8'] = '1'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.labelsize': 9, 'axes.labelweight': 'bold',
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.titlesize': 9,
    'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial',
    'mathtext.bf':'Arial:bold','mathtext.default':'regular',
})
MM = 1/25.4; FIG_W = 159.2*MM

from ref_elec_coupling import ref_price, ref_ci

BASE = config.INTERMEDIATE

# ── Baselines (verified) ───────────────────────────────────────────
P_ELEC_BASE  = 0.08;   CI_ELEC_BASE = 0.369
P_FH_BASE    = 0.03;   CI_FH_BASE   = 0.3384
P_LP_BASE    = 0.00684; CI_LP_BASE  = 0.19008
P_HP_BASE    = 0.009;  CI_HP_BASE   = 0.19008
P_H2_BASE    = 2.0;    CI_H2_BASE   = 7.0
MASS = 50e6

# ── Load All pathways ──────────────────────────────────────────────
print('Loading…')
df = pd.read_excel(config.MASTER_XLSX,
                   sheet_name='All pathways', header=3)
df['__feed'] = df.iloc[:, 0].astype(str).str.strip()
df = df[df['__feed'].isin(['PE','PET','PP','PS','PVC'])].reset_index(drop=True)
up_col = next(c for c in df.columns if isinstance(c, str) and c.lower().startswith('unit profit'))
cr_col = next(c for c in df.columns if isinstance(c, str) and c.lower().startswith('unit co2 reduc'))
df['__up_base'] = pd.to_numeric(df[up_col], errors='coerce')
df['__cr_base'] = pd.to_numeric(df[cr_col], errors='coerce')
df = df.dropna(subset=['__up_base', '__cr_base']).reset_index(drop=True)
N = len(df)
print(f'N = {N}')

def coln(c): return pd.to_numeric(df.iloc[:, c], errors='coerce').fillna(0).values

PLASTIC  = coln(6);  ACC = coln(5)
ELEC_C   = coln(7);  G_elec = coln(31)
FH_C     = coln(8);  G_FH   = coln(32)
REF_C    = [coln(10+k) for k in range(5)]
REF_G    = [coln(33+k) for k in range(5)]
LP_C     = coln(15); G_LP   = coln(38)
HP_C     = coln(16); G_HP   = coln(39)
H2_C     = coln(20); G_H2   = coln(42)
SUM_E    = coln(43)

# ── ΔUP functions ──────────────────────────────────────────────────
def d_elec(P_new):
    d_cost = ELEC_C * (P_new/P_ELEC_BASE - 1)
    for k in range(5):
        d_cost += REF_C[k] * (ref_price(k+1, P_new)/ref_price(k+1, P_ELEC_BASE) - 1)
    return float(np.mean(-d_cost * 1e6 / MASS))

def d_h2(P):     return float(np.mean(-H2_C * (P/P_H2_BASE - 1) * 1e6 / MASS))
def d_fh(P):     return float(np.mean(-FH_C * (P/P_FH_BASE - 1) * 1e6 / MASS))
def d_lp(P):     return float(np.mean(-LP_C * (P/P_LP_BASE - 1) * 1e6 / MASS))
def d_hp(P):     return float(np.mean(-HP_C * (P/P_HP_BASE - 1) * 1e6 / MASS))
def d_plastic(factor): return float(np.mean(-PLASTIC * (factor - 1) * 1e6 / MASS))
def d_carbontax(tau):  return float(np.mean(-SUM_E * tau / MASS))
def d_interest(i, i_base=0.08, n=20):
    CRF = lambda i, n: i*(1+i)**n / ((1+i)**n - 1)
    return float(np.mean(-ACC * (CRF(i,n)/CRF(i_base,n) - 1) * 1e6 / MASS))

# ── Tornado data ───────────────────────────────────────────────────
DRIVERS = [
    # label, dUP_low, dUP_high, range_label
    ('Plastic feed price',  d_plastic(0.5),   d_plastic(1.5),   '−50% / +50%'),
    ('Carbon tax',           0,                d_carbontax(0.05),'0 → +0.05'),
    ('Electricity price',    d_elec(0.04),     d_elec(0.12),     '−50% / +50%'),
    ('Fired heat price',     d_fh(0.015),      d_fh(0.045),      '−50% / +50%'),
    ('LP steam price',       d_lp(0.00342),    d_lp(0.01026),    '−50% / +50%'),
    ('HP steam price',       d_hp(0.0045),     d_hp(0.0135),     '−50% / +50%'),
    ('Hydrogen price',       d_h2(1.0),        d_h2(3.0),        '−50% / +50%'),
    ('Interest rate',        d_interest(0.04), d_interest(0.12), '−50% / +50%'),
]

# Sort by max(|low|, |high|) descending
DRIVERS.sort(key=lambda x: max(abs(x[1]), abs(x[2])), reverse=True)
labels   = [d[0] for d in DRIVERS]
lo_vals  = [d[1] for d in DRIVERS]
hi_vals  = [d[2] for d in DRIVERS]
ranges   = [d[3] for d in DRIVERS]

# ── Categorize for color ───────────────────────────────────────────
ENERGY_DRIVERS = {'Electricity price', 'Fired heat price', 'LP steam price',
                  'HP steam price', 'Hydrogen price'}
POLICY_DRIVERS = {'Carbon tax'}
FINANCE_DRIVERS = {'Interest rate'}

def color_for(lbl):
    if lbl in ENERGY_DRIVERS: return '#b5651d'   # house Energy
    if lbl in POLICY_DRIVERS: return '#e8b93d'   # house Policy
    if lbl in FINANCE_DRIVERS: return '#4878d0'   # house Finance
    return '#3f8f5a'                              # house Market

colors = [color_for(l) for l in labels]

# ── Plot ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_W*0.52))
y = np.arange(len(labels))[::-1]   # top = highest sensitivity

# x-extent and label offset (computed before plotting so labels sit just outside each bar tip)
xmax = max(max(abs(v) for v in lo_vals), max(abs(v) for v in hi_vals))
off = xmax * 0.022

# Bars + value labels placed OUTSIDE each bar tip (avoids in-bar / center overlap)
for i, (lo, hi, c) in enumerate(zip(lo_vals, hi_vals, colors)):
    yi = y[i]
    for v in (lo, hi):
        if v == 0:
            continue
        ax.barh(yi, v, color=c, alpha=0.85, edgecolor='white', linewidth=1.0, height=0.7)
        if v > 0:
            ax.text(v + off, yi, f'{v:+.3f}', ha='left', va='center',
                    fontsize=6, color='#222', fontweight='bold')
        else:
            ax.text(v - off, yi, f'{v:+.3f}', ha='right', va='center',
                    fontsize=6, color='#222', fontweight='bold')

ax.axvline(0, color='black', lw=1.2)
ax.set_yticks(y)
ax.set_yticklabels([f'{l}\n({r})' for l, r in zip(labels, ranges)],
                   fontsize=7)
ax.set_xlabel('ΔUP (mean over 959 pathways) [$/kg product]', fontsize=9)
ax.set_title('Tornado sensitivity (±50% perturbation)',
             fontsize=9, fontweight='bold', loc='left', pad=6)
ax.grid(axis='x', linestyle=':', alpha=0.4)

# Set xlim symmetric (extra margin so outside labels fit)
ax.set_xlim(-xmax*1.34, xmax*1.34)
for sp in ('top', 'right'):
    ax.spines[sp].set_color('#888'); ax.spines[sp].set_linewidth(0.6)

# Legend (color categories) — short labels so it does not overlap the bars
from matplotlib.patches import Patch
leg = ax.legend(handles=[
    Patch(facecolor='#b5651d', label='Energy carrier'),
    Patch(facecolor='#e8b93d', label='Carbon policy'),
    Patch(facecolor='#4878d0', label='Finance'),
    Patch(facecolor='#3f8f5a', label='Market'),
], loc='lower right', fontsize=6.5, frameon=False,
   title='Lever category', title_fontsize=7)
leg.get_title().set_fontweight('bold')

plt.tight_layout()
out = os.path.join(config.FIG_OUT, 'FigS7_6_tornado.png')
plt.savefig(out, dpi=400, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# ── Print summary ──────────────────────────────────────────────────
print('\n=== Tornado data (sorted by max sensitivity) ===')
print(f'{"Driver":<22} {"Δ_low":>10} {"Δ_high":>10} {"max(|·|)":>10}')
for lbl, lo, hi, rng in DRIVERS:
    print(f'{lbl:<22} {lo:>10.4f} {hi:>10.4f} {max(abs(lo),abs(hi)):>10.4f}')

# Energy aggregate
energy_max = sum(max(abs(d[1]), abs(d[2])) for d in DRIVERS if d[0] in ENERGY_DRIVERS)
total_max = sum(max(abs(d[1]), abs(d[2])) for d in DRIVERS)
print(f'\nEnergy carriers combined share = {energy_max/total_max*100:.1f}% of total |sensitivity|')

print('Done.')
