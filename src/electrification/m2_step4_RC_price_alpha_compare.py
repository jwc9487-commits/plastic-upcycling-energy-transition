import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
"""M2 Step 4 (RC) — Baseline vs Electrification optimality comparison.

For each (α, P_elec, τ) point compute BOTH scenarios and show:
  Δ wUP_net = wUP_e − wUP_b
  Blue (Δ > 0) = Electrification optimal
  Red  (Δ < 0) = Baseline optimal
  Black contour at Δ = 0: crossover boundary

Also overlays each scenario's break-even contour (Σ wUP = 0).
"""
import sys, io, os
os.environ['PYTHONUTF8'] = '1'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib as mpl
# ── Journal style: Arial, no suptitle, 9pt bold labels, 7pt ticks ──
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.labelsize': 9, 'axes.labelweight': 'bold',
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.titlesize': 9,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
})
MM = 1/25.4
FIG_W = 159.2 * MM

from ref_elec_coupling import ref_price, ref_ci

BASE = config.INTERMEDIATE
MASS = 50e6
FEEDS = ['PE','PET','PP','PS','PVC']
GLOBAL_W = {'PE':0.406, 'PP':0.264, 'PET':0.105, 'PVC':0.137, 'PS':0.088}

CI_GRID, CI_PPA = 369.0, 30.0
P_ELEC_BASE, CI_ELEC_BASE_g = 0.08, 369.0
P_FH_BASE,  CI_FH_BASE_g  = 0.03,     338.4
P_LP_BASE,  CI_LP_BASE_g  = 0.00684,  190.08
P_HP_BASE,  CI_HP_BASE_g  = 0.009,    190.08
P_H2_BASE,  CI_H2_BASE_kg = 2.0,      7.0
ETA_HEATER, ETA_BOILER = 0.95, 0.99
H2_CAPEX = 0.7

print('Loading...')
df = pd.read_excel(config.MASTER_XLSX,
                   sheet_name='All pathways', header=3)
df['__feed'] = df.iloc[:, 0].astype(str).str.strip()
df = df[df['__feed'].isin(FEEDS)].reset_index(drop=True)

up_col = next(c for c in df.columns if isinstance(c, str) and c.lower().startswith('unit profit'))
cr_col = next(c for c in df.columns if isinstance(c, str) and c.lower().startswith('unit co2 reduc'))
UP0 = pd.to_numeric(df[up_col], errors='coerce').fillna(0).values
CR0 = pd.to_numeric(df[cr_col], errors='coerce').fillna(0).values
feed_arr = df['__feed'].values

def coln(c): return pd.to_numeric(df.iloc[:, c], errors='coerce').fillna(0).values
ELEC_M = coln(7);  FH_M = coln(8)
REF_M  = np.array([coln(10+k) for k in range(5)])
LP_M, HP_M = coln(15), coln(16); H2_M = coln(20)
ELEC_E = coln(31); FH_E = coln(32)
REF_E  = np.array([coln(33+k) for k in range(5)])
LP_E, HP_E = coln(38), coln(39); H2_E = coln(42)

P_REF_BASE    = np.array([ref_price(k+1, P_ELEC_BASE) for k in range(5)])
CI_REF_BASE_g = np.array([ref_ci(k+1, CI_ELEC_BASE_g) for k in range(5)])

def transform_baseline(P_elec, CI_elec_g):
    r_e_c, r_e_ci = P_elec/P_ELEC_BASE, CI_elec_g/CI_ELEC_BASE_g
    r_ref_c  = np.array([ref_price(k+1, P_elec) / P_REF_BASE[k] for k in range(5)])
    r_ref_ci = np.array([ref_ci(k+1, CI_elec_g) / CI_REF_BASE_g[k] for k in range(5)])
    new_ELEC_M = ELEC_M * r_e_c; new_ELEC_E = ELEC_E * r_e_ci
    new_REF_M_sum = REF_M.T @ r_ref_c; new_REF_E_sum = REF_E.T @ r_ref_ci
    dC_M  = (new_ELEC_M - ELEC_M) + (new_REF_M_sum - REF_M.sum(axis=0))
    dE_kg = (new_ELEC_E - ELEC_E) + (new_REF_E_sum - REF_E.sum(axis=0))
    return UP0 - dC_M*1e6/MASS, CR0 - dE_kg/MASS

def transform_electrification(P_elec, CI_elec_g, h2_capex=H2_CAPEX):
    r_e_c, r_e_ci = P_elec/P_ELEC_BASE, CI_elec_g/CI_ELEC_BASE_g
    r_ref_c  = np.array([ref_price(k+1, P_elec) / P_REF_BASE[k] for k in range(5)])
    r_ref_ci = np.array([ref_ci(k+1, CI_elec_g) / CI_REF_BASE_g[k] for k in range(5)])
    new_ELEC_M = ELEC_M * r_e_c; new_ELEC_E = ELEC_E * r_e_ci
    new_REF_M_sum = REF_M.T @ r_ref_c; new_REF_E_sum = REF_E.T @ r_ref_ci
    new_FH_M = FH_M * ((P_elec/ETA_HEATER) / P_FH_BASE)
    new_FH_E = FH_E * ((CI_elec_g/ETA_HEATER) / CI_FH_BASE_g)
    new_LP_M = LP_M * ((P_elec/ETA_BOILER) / P_LP_BASE)
    new_LP_E = LP_E * ((CI_elec_g/ETA_BOILER) / CI_LP_BASE_g)
    new_HP_M = HP_M * ((P_elec/ETA_BOILER) / P_HP_BASE)
    new_HP_E = HP_E * ((CI_elec_g/ETA_BOILER) / CI_HP_BASE_g)
    new_H2_M = H2_M * ((53.0*P_elec + h2_capex) / P_H2_BASE)
    new_H2_E = H2_E * ((53.0*CI_elec_g/1000.0) / CI_H2_BASE_kg)
    dC_M = ((new_ELEC_M - ELEC_M) + (new_REF_M_sum - REF_M.sum(axis=0))
             + (new_FH_M - FH_M) + (new_LP_M - LP_M)
             + (new_HP_M - HP_M) + (new_H2_M - H2_M))
    dE_kg = ((new_ELEC_E - ELEC_E) + (new_REF_E_sum - REF_E.sum(axis=0))
             + (new_FH_E - FH_E) + (new_LP_E - LP_E)
             + (new_HP_E - HP_E) + (new_H2_E - H2_E))
    return UP0 - dC_M*1e6/MASS, CR0 - dE_kg/MASS

feed_idx = {f: np.where(feed_arr == f)[0] for f in FEEDS}

def econ_opt_weighted(UP_net):
    s = 0.0
    for f in FEEDS:
        idxs = feed_idx[f]
        s += GLOBAL_W[f] * np.max(UP_net[idxs])
    return s

# ── Sweep ──
# alpha = 0.5 is evaluated for Fig. 5 (formerly appended by add_alpha05.py);
# the six-panel supplementary layout below keeps the original six levels.
ALPHA_VALUES = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
ALPHA_FIG = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
P_elec_grid = np.linspace(0.0, 0.08, 41)
tau_grid    = np.linspace(0.0,  0.20, 41)

print('Computing baseline vs electrification matrices per α...')
results = {}
for alpha in ALPHA_VALUES:
    CI_e_g = CI_GRID + alpha * (CI_PPA - CI_GRID)
    Zb = np.zeros((len(tau_grid), len(P_elec_grid)))
    Ze = np.zeros((len(tau_grid), len(P_elec_grid)))
    for j, P_e in enumerate(P_elec_grid):
        UP_b, CR_b = transform_baseline(P_e, CI_e_g)
        UP_e, CR_e = transform_electrification(P_e, CI_e_g)
        for i, t in enumerate(tau_grid):
            Zb[i, j] = econ_opt_weighted(UP_b + t * CR_b)
            Ze[i, j] = econ_opt_weighted(UP_e + t * CR_e)
    results[alpha] = dict(Zb=Zb, Ze=Ze, dZ=Ze-Zb, CI=CI_e_g)
    print(f'  α={alpha:.1f}: CI={CI_e_g:.0f}  Δ range [{(Ze-Zb).min():+.3f}, {(Ze-Zb).max():+.3f}],  '
          f'crossover present: {((Ze-Zb).min()<0 and (Ze-Zb).max()>0)}')

# Common Δ range
all_d = np.concatenate([results[a]['dZ'].flatten() for a in ALPHA_FIG])
abs_max_d = float(np.nanmax(np.abs(all_d)))

# ── Figure 1: 6-panel Δ heatmap with crossover and break-even contours ──
fig, axes = plt.subplots(3, 2, figsize=(FIG_W, FIG_W * 1.28))
fig.subplots_adjust(top=0.965, bottom=0.115, left=0.11, right=0.865,
                     hspace=0.26, wspace=0.20)

P_grid_ref = 0.08; P_ppa_ref = 0.04; P_ppa_2050 = 0.02
tau_ets = 0.08

# ---- place a contour label at the point most isolated from the other contours ----
def _segs(cs):
    if cs is None: return []
    try: return [s for s in cs.allsegs[0] if len(s) > 2]
    except Exception:
        return [p.vertices for p in cs.get_paths() if len(p.vertices) > 2]
def _pts(cs):
    ss = _segs(cs)
    return np.vstack(ss) if ss else np.empty((0, 2))
def place_clabel(ax, cs, others, text, color, fs):
    """Original inline clabel style, but anchored at the contour point that is most
    isolated from the other contours (so the label does not cross other lines)."""
    ss = _segs(cs)
    if not ss: return
    me = max(ss, key=len)
    op = [o for o in (_pts(o) for o in others if o is not None) if len(o)]
    sx, sy = 1/0.08, 1/0.20            # normalise the two axis scales
    best, bestd = None, -1.0
    for pt in me[2:-2]:
        if not (0.012 < pt[0] < 0.066 and 0.035 < pt[1] < 0.165): continue
        d = (min(np.min(((o[:,0]-pt[0])*sx)**2 + ((o[:,1]-pt[1])*sy)**2) for o in op)
             if op else 1e9)
        dref = min(min(((rx-pt[0])*sx)**2 for rx in (0.02,0.04,0.08)), ((0.08-pt[1])*sy)**2)
        d = min(d, dref)
        if d > bestd: bestd, best = d, pt
    if best is None: best = me[len(me)//2]
    ax.clabel(cs, inline=True, manual=[(float(best[0]), float(best[1]))],
              fmt=text, fontsize=fs, colors=color)

panel_labels = ['a', 'b', 'c', 'd', 'e', 'f']
for k, alpha in enumerate(ALPHA_FIG):
    ax = axes[k // 2, k % 2]
    r = results[alpha]
    Zb = r['Zb']; Ze = r['Ze']; dZ = r['dZ']; CI_e_g = r['CI']

    # Δ heatmap (blue = elec wins, red = baseline wins)
    im = ax.imshow(dZ, origin='lower', aspect='auto',
                    extent=[P_elec_grid[0], P_elec_grid[-1],
                            tau_grid[0],    tau_grid[-1]],
                    cmap='RdBu', vmin=-abs_max_d, vmax=abs_max_d)

    cs_cross = cs_b = cs_e = None
    # Crossover contour (Δ = 0)
    if dZ.min() < 0 < dZ.max():
        cs_cross = ax.contour(P_elec_grid, tau_grid, dZ, levels=[0.0],
                              colors='black', linewidths=1.2, zorder=5)
    # Baseline break-even contour (Σ wUP_b = 0)
    if Zb.min() < 0 < Zb.max():
        cs_b = ax.contour(P_elec_grid, tau_grid, Zb, levels=[0.0],
                           colors='#34495e', linewidths=0.9, linestyles='-', zorder=5)
    # Electrification break-even contour (Σ wUP_e = 0)
    if Ze.min() < 0 < Ze.max():
        cs_e = ax.contour(P_elec_grid, tau_grid, Ze, levels=[0.0],
                           colors='#e74c3c', linewidths=0.9, linestyles='--', zorder=5)
    # inline labels at the most-isolated point of each curve (avoid other lines)
    place_clabel(ax, cs_cross, [cs_b, cs_e], 'Elec = Base', 'black', 6)
    place_clabel(ax, cs_b, [cs_cross, cs_e], 'Base BE', '#34495e', 5.5)
    place_clabel(ax, cs_e, [cs_cross, cs_b], 'Elec BE', '#e74c3c', 5.5)

    # Reference verticals/horizontal
    for x_ref, c in [(P_grid_ref,'gray'), (P_ppa_ref,'blue'), (P_ppa_2050,'green')]:
        ax.axvline(x_ref, color=c, linestyle=':', linewidth=1.3, alpha=0.95, zorder=1)
    ax.axhline(tau_ets, color='purple', linestyle=':', linewidth=1.3, alpha=0.95, zorder=1)

    ax.set_box_aspect(1)                                    # square panel
    title = 'α = %.1f   CI$_{\\mathbf{elec}}$ = %.0f gCO$_{\\mathbf{2eq}}$/kWh' % (alpha, CI_e_g)
    ax.set_title(title, fontsize=8, fontweight='bold', pad=9)
    if k // 2 == 2:
        ax.set_xlabel('Electricity price (\\$/kWh)')
    if k % 2 == 0:
        ax.set_ylabel(r'Carbon pricing (\$/kg$\mathbf{CO_{2eq}}$)', fontweight='bold')
    ax.tick_params(labelsize=7)

# Panel letters at the top-left CORNER of each panel (outside the plot)
fig.canvas.draw()
for k in range(len(ALPHA_FIG)):
    ax = axes[k // 2, k % 2]; pos = ax.get_position()
    xpos = 0.006 if (k % 2 == 0) else (pos.x0 - 0.095)   # left col at figure edge
    fig.text(xpos, pos.y1 + 0.018, f'({panel_labels[k]})',   # raised to clear tick labels
             ha='left', va='bottom', fontsize=11, fontweight='bold')

# Shared colorbar
cbar_ax = fig.add_axes([0.885, 0.115, 0.018, 0.85])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.tick_params(labelsize=7)
cbar.set_label(r'Δ UP$_{\mathbf{net}}$ = UP$_{\mathbf{e}}$ − UP$_{\mathbf{b}}$ (\$/kg)', size=9, weight='bold')
# annotate
cbar_ax.text(4.6, 0.97, 'Electrification favored',
              transform=cbar_ax.transAxes, fontsize=9, color='#1f77b4', rotation=90,
              fontweight='bold', va='top')
cbar_ax.text(4.6, 0.03, 'Conventional utility mix favored',
              transform=cbar_ax.transAxes, fontsize=9, color='#c0392b', rotation=90,
              fontweight='bold', va='bottom')

# Legend at bottom
legend_handles = [
    Line2D([0], [0], color='black', linewidth=2.4, label='Crossover (Elec = Base)'),
    Line2D([0], [0], color='#34495e', linewidth=1.6, label='Baseline break-even'),
    Line2D([0], [0], color='#e74c3c', linewidth=1.6, linestyle='--',
            label='Electrification break-even'),
    Line2D([0], [0], color='gray', linestyle=':', linewidth=1.4, label=r'Grid (0.08 \$/kWh)'),
    Line2D([0], [0], color='blue', linestyle=':', linewidth=1.4, label=r'PPA 2026 (0.04 \$/kWh)'),
    Line2D([0], [0], color='green', linestyle=':', linewidth=1.4, label=r'PPA 2050 (0.02 \$/kWh)'),
    Line2D([0], [0], color='purple', linestyle=':', linewidth=1.4, label=r'EU ETS (~\$80/tCO$_{2eq}$)'),
]
fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.0),
           ncol=4, fontsize=7, frameon=False,
           columnspacing=1.0, handlelength=1.6)

FIGDIR = config.FIG_OUT
out_png = os.path.join(FIGDIR, 'legacy_Fig11_compare_panels.png')
plt.savefig(out_png, dpi=400, bbox_inches='tight')
plt.close()
print(f'\nSaved: {out_png}')

# ── Figure 2: Crossover curves — τ_crossover(P_elec) per α ──
fig2, ax2 = plt.subplots(figsize=(11, 6.8))
fig2.subplots_adjust(top=0.91, bottom=0.13, left=0.08, right=0.78)

cmap_a = plt.cm.viridis(np.linspace(0.05, 0.92, len(ALPHA_FIG)))
for k, alpha in enumerate(ALPHA_FIG):
    dZ = results[alpha]['dZ']
    CI_e_g = results[alpha]['CI']
    cross_tau = []
    for j in range(len(P_elec_grid)):
        col = dZ[:, j]
        s = np.where(np.diff(np.sign(col)) != 0)[0]
        if len(s) == 0:
            # No crossover — check which scenario dominates entirely
            cross_tau.append(np.nan if col[0]*col[-1] > 0 else tau_grid[s[0]] if len(s)>0 else np.nan)
        else:
            i = s[0]
            denom = col[i+1] - col[i]
            if denom == 0:
                cross_tau.append(tau_grid[i])
            else:
                t_be = tau_grid[i] + (tau_grid[i+1]-tau_grid[i]) * (-col[i])/denom
                cross_tau.append(t_be)
    cross_tau = np.array(cross_tau)
    ax2.plot(P_elec_grid, cross_tau*1000, color=cmap_a[k],
              linewidth=2.4,
              label=f'α = {alpha:.1f}   (CI = {CI_e_g:.0f} gCO₂/kWh)')

# References
ax2.axvline(P_grid_ref, color='gray', linestyle=':', linewidth=1.0, alpha=0.7)
ax2.text(P_grid_ref+0.0005, 195, 'grid\n0.08', fontsize=8, color='gray', va='top')
ax2.axvline(P_ppa_ref, color='blue', linestyle=':', linewidth=1.0, alpha=0.7)
ax2.text(P_ppa_ref+0.0005, 195, 'PPA 2026\n0.04', fontsize=8, color='blue', va='top')
ax2.axvline(P_ppa_2050, color='green', linestyle=':', linewidth=1.0, alpha=0.7)
ax2.text(P_ppa_2050+0.0005, 195, 'PPA 2050\n0.02', fontsize=8, color='green', va='top')
ax2.axhline(80, color='purple', linestyle=':', linewidth=1.0, alpha=0.7)
ax2.text(0.079, 82, 'EU ETS ~$80/tCO₂', fontsize=8, color='purple',
          ha='right', va='bottom')
ax2.axhline(50, color='orange', linestyle=':', linewidth=1.0, alpha=0.6)
ax2.text(0.079, 52, 'US 45Z ~$50/tCO₂', fontsize=8, color='orange',
          ha='right', va='bottom')

ax2.set_xlabel('Electricity price P$_{elec}$ ($/kWh)', fontsize=11)
ax2.set_ylabel('Crossover τ ($/tCO$_2$)', fontsize=11)
ax2.set_title('M2 Step 4 (RC) — Carbon tax threshold at which Electrification overtakes Baseline\n'
              'Curve = lowest τ at which Δ wUP$_{net}$ = wUP$_e$ − wUP$_b$ ≥ 0 (Elec ≥ Base)',
              fontsize=12, fontweight='bold')
ax2.grid(linestyle=':', alpha=0.4)
ax2.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
            fontsize=9.5, framealpha=0.95, title='Renewable fraction',
            title_fontsize=10)
ax2.set_xlim(P_elec_grid[0], P_elec_grid[-1])
ax2.set_ylim(0, 200)

# Annotation: above curve = Electrification, below = Conventional utility mix
ax2.text(0.004, 175, '↑ Electrification (Δ ≥ 0)', fontsize=11, color='#1f77b4',
          fontweight='bold')
ax2.text(0.004, 15, '↓ Conventional utility mix (Δ < 0)', fontsize=11, color='#c0392b',
          fontweight='bold')

out_png2 = os.path.join(config.FIG_OUT, 'fig_m2_step4_RC_compare_crossover.png')
plt.savefig(out_png2, dpi=180, bbox_inches='tight')
plt.close()
print(f'Saved: {out_png2}')

# CSV
rows = []
for alpha in ALPHA_VALUES:
    r = results[alpha]
    for i, t in enumerate(tau_grid):
        for j, P in enumerate(P_elec_grid):
            rows.append(dict(alpha=alpha, CI_elec=r['CI'],
                             P_elec=P, tau=t,
                             wUP_base=r['Zb'][i,j],
                             wUP_elec=r['Ze'][i,j],
                             delta=r['dZ'][i,j]))
pd.DataFrame(rows).to_csv(os.path.join(BASE, 'm2_step4_RC_compare.csv'),
                          index=False, float_format='%.4f')
print('\nDone.')
