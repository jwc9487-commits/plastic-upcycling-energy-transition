import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
"""Sobol indices on mean CR (unit CO2 reduction) variance.

Same framework as sobol_indices.py but output = mean CR.
Hypothesis: CR variance is dominated by energy CI variables (CI_elec, CI_FH)
and renewable share α — supporting the "energy is key to emissions" message.
"""
import sys, io, os
os.environ['PYTHONUTF8'] = '1'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

from ref_elec_coupling import ref_price, ref_ci

BASE = config.INTERMEDIATE

P_ELEC_B = 0.08;  CI_ELEC_B = 0.369
P_FH_B   = 0.03;  CI_FH_B   = 0.3384
P_H2_B   = 2.0;   CI_H2_B   = 7.0
MASS = 50e6

# ── Load ───────────────────────────────────────────────────────────
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

def coln(c): return pd.to_numeric(df.iloc[:, c], errors='coerce').fillna(0).values

G_ELEC = coln(31); G_FH = coln(32)
G_LP   = coln(38); G_HP = coln(39)
G_H2   = coln(42); SUM_E = coln(43)
REF_G  = [coln(33+k) for k in range(5)]
CR_b   = df['__cr_base'].values

# ── Vectorized CR_mean function ────────────────────────────────────
def eval_mean_CR(P_elec, CI_elec, P_FH, CI_FH, P_LP, P_HP, P_H2,
                 plastic_factor, tau, i_rate, alpha):
    """Compute mean CR. Only CI variables affect CR (cost variables are no-op)."""
    CI_eff = (1-alpha)*CI_elec + alpha*0.030
    ci_mj = CI_eff / 3.6

    d_ghg = (
        G_ELEC * (ci_mj   / (CI_ELEC_B/3.6) - 1) +
        G_FH   * (CI_FH   / CI_FH_B    - 1)
    )
    # Ref-Elec coupling
    for k in range(5):
        d_ghg += REF_G[k] * (ref_ci(k+1, CI_eff*1000)/ref_ci(k+1, CI_ELEC_B*1000) - 1)

    CR_new = CR_b - d_ghg / MASS
    return float(np.mean(CR_new))


problem = {
    'num_vars': 11,
    'names': ['P_elec', 'CI_elec', 'P_FH', 'CI_FH',
              'P_LP', 'P_HP', 'P_H2',
              'plastic_factor', 'tau', 'i_rate', 'alpha'],
    'bounds': [
        [0.04, 0.12], [0.03, 0.55], [0.015, 0.045], [0.17, 0.51],
        [0.00342, 0.01026], [0.0045, 0.0135], [1.0, 3.0],
        [0.5, 1.5], [0.0, 0.10], [0.04, 0.12], [0.0, 1.0],
    ]
}

N_BASE = 1024
SEED = 20260925   # fixed so the Saltelli sample and bootstrap intervals are reproducible
print(f'Generating Saltelli: N_base={N_BASE} → {N_BASE*(2*11+2)} runs')
params = sobol_sample.sample(problem, N_BASE, calc_second_order=False, seed=SEED)
print(f'Sample shape: {params.shape}')

print('Evaluating mean CR…')
Y = np.zeros(len(params))
for i, p in enumerate(params):
    Y[i] = eval_mean_CR(*p)
print(f'Y range: [{Y.min():.3f}, {Y.max():.3f}], mean={Y.mean():.3f}, std={Y.std():.3f}')

print('Computing Sobol indices for CR…')
Si = sobol_analyze.analyze(problem, Y, calc_second_order=False, print_to_console=False, seed=SEED)
S1 = np.array(Si['S1']); ST = np.array(Si['ST'])
names = problem['names']

print(f'\n{"Driver":<18} {"S1":>10} {"S_T":>10} {"Inter.":>10}')
print('-'*55)
for j, n in enumerate(names):
    print(f'{n:<18} {S1[j]:>10.4f} {ST[j]:>10.4f} {ST[j]-S1[j]:>10.4f}')

# Save
res_df = pd.DataFrame({
    'driver': names, 'S1': S1, 'ST': ST, 'interaction': ST - S1,
}).sort_values('ST', ascending=False).reset_index(drop=True)
res_df.to_csv(os.path.join(BASE, 'sobol_indices_CR.csv'), index=False, encoding='utf-8-sig')

# Categorize
CATEGORY = {
    'P_elec': 'Energy', 'CI_elec': 'Energy', 'P_FH': 'Energy',
    'CI_FH': 'Energy', 'P_LP': 'Energy', 'P_HP': 'Energy',
    'P_H2':  'Energy', 'alpha': 'Energy',
    'plastic_factor': 'Market', 'tau': 'Policy', 'i_rate': 'Finance',
}
COL = {'Energy': '#e74c3c', 'Policy': '#34495e', 'Market': '#27ae60', 'Finance': '#9b59b6'}

cat_ST = {}
for j, n in enumerate(names):
    c = CATEGORY[n]
    cat_ST[c] = cat_ST.get(c, 0) + ST[j]
total_ST = sum(cat_ST.values()) or 1.0
print('\n=== CR variance share by category ===')
for c in ['Energy', 'Policy', 'Market', 'Finance']:
    print(f'  {c:<10}: ST_sum={cat_ST.get(c,0):.3f} ({cat_ST.get(c,0)/total_ST*100:.1f}%)')

# Combined plot: UP variance + CR variance pie charts
print('\nGenerating combined UP + CR pie figure…')
# Re-load UP results for combined display
df_up = pd.read_csv(os.path.join(BASE, 'sobol_indices.csv'))
cat_ST_UP = {}
for _, row in df_up.iterrows():
    c = CATEGORY[row['driver']]
    cat_ST_UP[c] = cat_ST_UP.get(c, 0) + row['ST']
total_ST_UP = sum(cat_ST_UP.values()) or 1.0

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
cats = ['Energy', 'Policy', 'Market', 'Finance']
colors_pie = [COL[c] for c in cats]

for ax, (data, title, total) in zip(axes,
        [(cat_ST_UP, 'UP variance', total_ST_UP),
         (cat_ST,    'CR variance', total_ST)]):
    sizes = [data.get(c, 0) for c in cats]
    wedges, _, autotexts = ax.pie(
        sizes, labels=cats, colors=colors_pie,
        autopct=lambda p: f'{p:.1f}%' if p > 1 else '',
        startangle=90, textprops={'fontsize': 11.5, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

fig.suptitle('Sobol variance decomposition: UP vs CR (by driver category)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
out = os.path.join(config.FIG_OUT, 'fig_sobol_UP_vs_CR_pies.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# Individual driver bar (CR)
order = np.argsort(-ST)
fig, ax = plt.subplots(figsize=(11, 6.5))
y = np.arange(len(names))[::-1]
bar_colors = [COL[CATEGORY[names[i]]] for i in order]
ax.barh(y, ST[order], color=bar_colors, alpha=0.85, edgecolor='white', linewidth=1.0)
ax.barh(y, S1[order], color=bar_colors, alpha=1.0, edgecolor='black',
        linewidth=0.5, hatch='//')

for i, j in enumerate(order):
    if ST[j] > 0.005:
        ax.text(ST[j] + 0.005, y[i], f'{ST[j]:.3f}', va='center', fontsize=9.5)

ax.set_yticks(y); ax.set_yticklabels([names[i] for i in order], fontsize=10.5)
ax.set_xlabel('Sobol index for mean CR', fontsize=11)
ax.set_title('CR variance — Sobol indices\nHatched = S_1 (1st order); solid = S_T (total order)',
             fontsize=12, fontweight='bold', pad=10)
ax.grid(axis='x', linestyle=':', alpha=0.4)

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor='#e74c3c', label='Energy (8 drivers)'),
    Patch(facecolor='#34495e', label='Policy'),
    Patch(facecolor='#27ae60', label='Market'),
    Patch(facecolor='#9b59b6', label='Finance'),
], loc='lower right', fontsize=9, framealpha=0.92)

cat_text = '  '.join(f'{c}: {cat_ST.get(c,0)/total_ST*100:.0f}%'
                    for c in cats)
ax.text(0.98, 0.02, f'Category share:  {cat_text}',
        transform=ax.transAxes, ha='right', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef9e7',
                  edgecolor='#d4ac0d', linewidth=1.2))

plt.tight_layout()
out2 = os.path.join(config.FIG_OUT, 'fig_sobol_CR.png')
plt.savefig(out2, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: {out2}')

print('Done.')
