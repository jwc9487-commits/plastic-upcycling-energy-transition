import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
"""Sobol indices for variance-based global sensitivity analysis.

Computes 1st-order (S_i) and total-order (S_T_i) Sobol indices for
each driver's contribution to the variance of mean UP (averaged over 959 paths).

Drivers (11 total):
  1. P_elec       [0.04, 0.12]    $/kWh
  2. CI_elec      [0.03, 0.55]    kgCO2/kWh
  3. P_FH         [0.015, 0.045]  $/kWh
  4. CI_FH        [0.17, 0.51]    kgCO2/kWh
  5. P_LP         [0.00342, 0.01026]
  6. P_HP         [0.0045, 0.0135]
  7. P_H2         [1.0, 3.0]      $/kg
  8. P_plastic_factor  [0.5, 1.5]  multiplier (× baseline)
  9. tau (carbon tax)  [0, 0.10]   $/kgCO2
  10. i (interest)     [0.04, 0.12]
  11. alpha (renew share) [0, 1.0]

Output: Sobol indices ranked, with energy/policy/market category breakdown.
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
    'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'pdf.fonttype':42,'ps.fonttype':42,
    'axes.labelsize':9,'axes.labelweight':'bold',
    'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.titlesize':9,})
MM=1/25.4; FIG_W=159.2*MM; FIG_W1=79.6*MM


from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

from ref_elec_coupling import ref_price, ref_ci

BASE = config.INTERMEDIATE

# ── Baselines ──────────────────────────────────────────────────────
P_ELEC_B = 0.08;  CI_ELEC_B = 0.369
P_FH_B   = 0.03;  CI_FH_B   = 0.3384
P_LP_B   = 0.00684; CI_LP_B = 0.19008
P_HP_B   = 0.009; CI_HP_B   = 0.19008
P_H2_B   = 2.0;   CI_H2_B   = 7.0
MASS = 50e6

# ── Load data once ─────────────────────────────────────────────────
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
N_path = len(df)
print(f'N pathways = {N_path}')

def coln(c): return pd.to_numeric(df.iloc[:, c], errors='coerce').fillna(0).values
PLASTIC = coln(6); ACC = coln(5)
ELEC_C  = coln(7); G_ELEC = coln(31)
FH_C    = coln(8); G_FH   = coln(32)
LP_C    = coln(15); G_LP  = coln(38)
HP_C    = coln(16); G_HP  = coln(39)
H2_C    = coln(20); G_H2  = coln(42)
REF_C   = [coln(10+k) for k in range(5)]
REF_G   = [coln(33+k) for k in range(5)]
SUM_E   = coln(43)

UP_b = df['__up_base'].values

# ── Vectorized UP_mean function (single point in 11D parameter space) ──
def CRF(i, n=20):
    return i * (1+i)**n / ((1+i)**n - 1)
CRF_BASE = CRF(0.08)

def eval_mean_UP(P_elec, CI_elec, P_FH, CI_FH, P_LP, P_HP, P_H2,
                 plastic_factor, tau, i_rate, alpha):
    """Compute mean UP across 959 paths for given driver values.

    alpha (renewable share) modifies P_elec and CI_elec:
       P_eff  = (1-α) × P_elec + α × 0.04
       CI_eff = (1-α) × CI_elec + α × 0.030
    (additive on top of P_elec, CI_elec to allow joint sweep)
    """
    P_eff  = (1-alpha)*P_elec  + alpha*0.04
    CI_eff = (1-alpha)*CI_elec + alpha*0.030

    # Carrier electrification (with renewable share already accounted)
    p_mj  = P_eff / 3.6
    ci_mj = CI_eff / 3.6
    P_H2_n  = 53.0*P_eff + 0.7   # green H2 from electrolysis
    CI_H2_n = 53.0*CI_eff
    # However user also provides direct P_H2 → we use the user value (override)
    # Convention: P_H2 in the sweep represents EXOGENOUS H2 price (grey or external)
    # We use user P_H2 directly, IGNORING electrolysis for simplicity in Sobol
    # (or could use min(P_H2, electrolysis price). Let's just use P_H2.)
    P_H2_use, CI_H2_use = P_H2, 7.0   # fixed CI for H2 (or could parameterize)

    d_cost = (
        ELEC_C * (p_mj   / (P_ELEC_B/3.6) - 1) +
        FH_C   * (P_FH   / P_FH_B    - 1) +
        LP_C   * (P_LP   / P_LP_B    - 1) +
        HP_C   * (P_HP   / P_HP_B    - 1) +
        H2_C   * (P_H2_use / P_H2_B  - 1) +
        PLASTIC * (plastic_factor - 1) +
        ACC * (CRF(i_rate)/CRF_BASE - 1)
    )
    d_ghg = (
        G_ELEC * (ci_mj   / (CI_ELEC_B/3.6) - 1) +
        G_FH   * (CI_FH   / CI_FH_B    - 1)
    )
    # Ref-Elec coupling (Refs scale with P_eff, CI_eff)
    for k in range(5):
        d_cost += REF_C[k] * (ref_price(k+1, P_eff)/ref_price(k+1, P_ELEC_B) - 1)
        d_ghg  += REF_G[k] * (ref_ci(k+1, CI_eff*1000)/ref_ci(k+1, CI_ELEC_B*1000) - 1)

    UP_new = UP_b - d_cost * 1e6 / MASS
    # Carbon tax penalty
    UP_eff_arr = UP_new - tau * SUM_E / MASS
    return float(np.mean(UP_eff_arr))


# ── Sobol problem definition ───────────────────────────────────────
problem = {
    'num_vars': 11,
    'names': ['P_elec', 'CI_elec', 'P_FH', 'CI_FH',
              'P_LP', 'P_HP', 'P_H2',
              'plastic_factor', 'tau', 'i_rate', 'alpha'],
    'bounds': [
        [0.04, 0.12],   # P_elec
        [0.03, 0.55],   # CI_elec
        [0.015, 0.045], # P_FH
        [0.17, 0.51],   # CI_FH
        [0.00342, 0.01026], # P_LP
        [0.0045, 0.0135],   # P_HP
        [1.0, 3.0],     # P_H2
        [0.5, 1.5],     # plastic_factor
        [0.0, 0.10],    # tau
        [0.04, 0.12],   # i_rate
        [0.0, 1.0],     # alpha
    ]
}

# Generate Saltelli samples. N (base) * (2k+2) total runs.
# For k=11 drivers, N=512 gives 12,288 runs. N=1024 gives 24,576 (good accuracy).
N_BASE = 1024
SEED = 20260925   # fixed so the Saltelli sample and bootstrap intervals are reproducible
print(f'\nGenerating Saltelli samples: N_base={N_BASE} × (2k+2)={2*11+2} = {N_BASE*(2*11+2)} runs')
params = sobol_sample.sample(problem, N_BASE, calc_second_order=False, seed=SEED)
print(f'Sample shape: {params.shape}')

# Evaluate
print('Evaluating mean UP across samples…')
Y = np.zeros(len(params))
for i, p in enumerate(params):
    Y[i] = eval_mean_UP(*p)
    if (i+1) % 5000 == 0:
        print(f'  {i+1}/{len(params)}')
print(f'Y range: [{Y.min():.3f}, {Y.max():.3f}], mean={Y.mean():.3f}, std={Y.std():.3f}')

# Sobol analyze
print('\nComputing Sobol indices…')
Si = sobol_analyze.analyze(problem, Y, calc_second_order=False, print_to_console=False, seed=SEED)
S1 = np.array(Si['S1'])
ST = np.array(Si['ST'])
S1_conf = np.array(Si['S1_conf'])
ST_conf = np.array(Si['ST_conf'])

names = problem['names']
print(f'\n{"Driver":<18} {"S1 (1st)":>12} {"S_T (total)":>14} {"S_T - S1 (inter.)":>20}')
print('-'*70)
for j, n in enumerate(names):
    print(f'{n:<18} {S1[j]:>12.4f} {ST[j]:>14.4f} {ST[j]-S1[j]:>20.4f}')

# Save CSV
res_df = pd.DataFrame({
    'driver': names,
    'S1': S1, 'S1_conf': S1_conf,
    'ST': ST, 'ST_conf': ST_conf,
    'interaction': ST - S1,
}).sort_values('ST', ascending=False).reset_index(drop=True)
res_df.to_csv(os.path.join(BASE, 'sobol_indices.csv'), index=False, encoding='utf-8-sig')

# ── Categorize ─────────────────────────────────────────────────────
CATEGORY = {
    'P_elec':    'Energy', 'CI_elec':   'Energy', 'P_FH':       'Energy',
    'CI_FH':     'Energy', 'P_LP':      'Energy', 'P_HP':       'Energy',
    'P_H2':      'Energy', 'alpha':     'Energy',  # renewable share is energy
    'plastic_factor': 'Market',
    'tau':       'Policy',
    'i_rate':    'Finance',
}
COL = {'Energy':'#e74c3c', 'Policy':'#34495e', 'Market':'#27ae60', 'Finance':'#9b59b6'}

# Sum by category
cat_S1 = {}; cat_ST = {}
for j, n in enumerate(names):
    c = CATEGORY[n]
    cat_S1[c] = cat_S1.get(c, 0) + S1[j]
    cat_ST[c] = cat_ST.get(c, 0) + ST[j]

print('\n=== Variance share by category ===')
total_ST = sum(cat_ST.values())
for c in ['Energy', 'Policy', 'Market', 'Finance']:
    print(f'  {c:<10}: S1_sum={cat_S1.get(c,0):.3f}, ST_sum={cat_ST.get(c,0):.3f} '
          f'({cat_ST.get(c,0)/total_ST*100:.1f}% of total ST)')

# ── Plot ────────────────────────────────────────────────────────────
# Sort by ST descending
order = np.argsort(-ST)
fig, ax = plt.subplots(figsize=(FIG_W, FIG_W*0.5))
y = np.arange(len(names))[::-1]
bar_colors = [COL[CATEGORY[names[i]]] for i in order]
ax.barh(y, ST[order], xerr=ST_conf[order],
        color=bar_colors, alpha=0.85, edgecolor='white', linewidth=1.0,
        label='S_T (total)', error_kw={'ecolor':'#333','capsize':3})
ax.barh(y, S1[order], color=bar_colors, alpha=1.0, edgecolor='black',
        linewidth=0.5, hatch='//', label='S_1 (first-order)')

for i, j in enumerate(order):
    ax.text(ST[j] + 0.005, y[i], f'{ST[j]:.3f}', va='center', fontsize=6)

ax.set_yticks(y)
ax.set_yticklabels([names[i] for i in order], fontsize=7)
ax.set_xlabel('Sobol index', fontsize=9)
ax.set_title('Sobol indices: variance contribution to mean UP\n'
             '(S_T = total-order incl. interactions; S_1 = first-order)',
             fontsize=9, fontweight='bold', pad=6)
ax.grid(axis='x', linestyle=':', alpha=0.4)

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor='#e74c3c', label='Energy (8 drivers)'),
    Patch(facecolor='#34495e', label='Policy (Carbon tax)'),
    Patch(facecolor='#27ae60', label='Market (Plastic)'),
    Patch(facecolor='#9b59b6', label='Finance (interest)'),
], loc='lower right', fontsize=6.5, framealpha=0.92)

# Annotate category totals
cat_text = '  '.join(f'{c}: {cat_ST.get(c,0)/total_ST*100:.0f}%'
                    for c in ['Energy', 'Policy', 'Market', 'Finance'])
ax.text(0.98, 0.02, f'Category ST sum:  {cat_text}',
        transform=ax.transAxes, ha='right', fontsize=6,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef9e7',
                  edgecolor='#d4ac0d', linewidth=1.2))

plt.tight_layout()
out = os.path.join(config.FIG_OUT, 'fig_sobol_indices.png')
plt.savefig(out, dpi=400, bbox_inches='tight')
plt.close()
print(f'\nSaved: {out}')

# Category-aggregated pie chart
fig, ax = plt.subplots(figsize=(FIG_W1, FIG_W1*0.85))
cats = ['Energy', 'Policy', 'Market', 'Finance']
sizes = [cat_ST[c] for c in cats]
colors_pie = [COL[c] for c in cats]
wedges, _, autotexts = ax.pie(sizes, labels=cats, colors=colors_pie,
                                 autopct='%1.1f%%', startangle=90,
                                 textprops={'fontsize': 11, 'fontweight': 'bold'})
for w in wedges:
    w.set_edgecolor('white')
    w.set_linewidth(2)
ax.set_title('Variance share of mean UP — by driver category\n'
             '(Sobol total-order indices, sum)',
             fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
out_pie = os.path.join(config.FIG_OUT, 'fig_sobol_category_pie.png')
plt.savefig(out_pie, dpi=400, bbox_inches='tight')
plt.close()
print(f'Saved: {out_pie}')

print('Done.')
