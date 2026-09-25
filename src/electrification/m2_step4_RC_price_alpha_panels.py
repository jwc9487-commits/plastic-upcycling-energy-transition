import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
"""M2 Step 4 (RC) — Price–τ heatmap panels at fixed α.

Decouples P_elec (electricity price) from α (PPA renewable fraction).
α determines CI_elec (g/kWh); P_elec is an independent market parameter.

For each α ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}, sweep over:
  x-axis: P_elec ∈ [0.02, 0.10] $/kWh
  y-axis: τ ∈ [0, 0.20] $/kgCO₂
Heatmap color: Σ wᶠ · UP_net (global composition).
Break-even contour: Σ wᶠ · UP_net = 0.
"""
import sys, io, os
os.environ['PYTHONUTF8'] = '1'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
# ── Journal style: Arial, no suptitle, 9pt bold labels, 7pt ticks ──
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.labelsize': 9, 'axes.labelweight': 'bold',
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.titlesize': 9,
})
# mathtext in Arial so subscripts (e.g. CO_2eq) stay Arial/bold
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial'
mpl.rcParams['mathtext.it'] = 'Arial:italic'
mpl.rcParams['mathtext.bf'] = 'Arial:bold'
MM = 1/25.4
FIG_W = 159.2 * MM

from ref_elec_coupling import ref_price, ref_ci

BASE = config.INTERMEDIATE
MASS = 50e6
FEEDS = ['PE','PET','PP','PS','PVC']
GLOBAL_W = {'PE':0.406, 'PP':0.264, 'PET':0.105, 'PVC':0.137, 'PS':0.088}

# Endpoints: α → CI_elec interpolation
CI_GRID, CI_PPA = 369.0, 30.0

# Baseline reference utilities
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
N = len(df)
print(f'N = {N}')

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

def transform_electrification(P_elec, CI_elec_g, h2_capex=H2_CAPEX):
    """Full electrification under arbitrary P_elec, CI_elec_g (decoupled)."""
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
    UP = UP0 - dC_M*1e6/MASS
    CR = CR0 - dE_kg/MASS
    return UP, CR

feed_idx = {f: np.where(feed_arr == f)[0] for f in FEEDS}

def econ_opt_weighted(UP_net):
    s = 0.0
    for f in FEEDS:
        idxs = feed_idx[f]
        s += GLOBAL_W[f] * np.max(UP_net[idxs])
    return s

# ── Sweep parameters ──────────────────────────────────────────────
ALPHA_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
P_elec_grid = np.linspace(0.0, 0.08, 41)
tau_grid    = np.linspace(0.0,  0.20, 41)

# Helper: per α, compute Z matrix
def compute_Z(alpha):
    CI_e_g = CI_GRID + alpha * (CI_PPA - CI_GRID)
    Z = np.zeros((len(tau_grid), len(P_elec_grid)))
    for j, P_e in enumerate(P_elec_grid):
        UP, CR = transform_electrification(P_e, CI_e_g)
        for i, t in enumerate(tau_grid):
            UP_net = UP + t * CR
            Z[i, j] = econ_opt_weighted(UP_net)
    return Z, CI_e_g

# Compute all Z matrices
print('Computing heatmaps...')
results = {}
for a in ALPHA_VALUES:
    Z, CI_e_g = compute_Z(a)
    results[a] = dict(Z=Z, CI=CI_e_g)
    print(f'  α={a:.1f}: CI={CI_e_g:.0f} g/kWh, '
          f'Z range [{Z.min():+.3f}, {Z.max():+.3f}], '
          f'break-even reachable: {(Z.min()<0 and Z.max()>0)}')

# Common color range
all_vals = np.concatenate([r['Z'].flatten() for r in results.values()])
vmax = float(np.nanmax(np.abs(all_vals)))
vmin = -vmax

# ── Figure: 3×2 grid of (P_elec × τ) heatmaps ─────────────────────
fig, axes = plt.subplots(3, 2, figsize=(FIG_W, FIG_W * 1.28))
fig.subplots_adjust(top=0.965, bottom=0.115, left=0.11, right=0.865,
                     hspace=0.30, wspace=0.20)

# Reference markers
P_grid_ref = 0.08
P_ppa_ref  = 0.04
P_ppa_2050 = 0.02
tau_ets    = 0.08   # EU ETS ~80 $/tCO2
tau_45z    = 0.05   # US 45Z baseline

def format_panel(ax, Z, CI_e_g, alpha, panel_label, show_y=True, show_x=True):
    im = ax.imshow(Z, origin='lower', aspect='auto',
                    extent=[P_elec_grid[0], P_elec_grid[-1],
                            tau_grid[0],    tau_grid[-1]],
                    cmap='RdYlGn', vmin=vmin, vmax=vmax)
    # Break-even contour
    cs0 = ax.contour(P_elec_grid, tau_grid, Z, levels=[0.0],
                      colors='black', linewidths=1.2, zorder=5)
    # place the inline label at the contour point farthest from any reference line
    # (within a safe vertical band, so it never clips at top/bottom nor overlaps the dotted refs)
    _segs = cs0.allsegs[0] if cs0.allsegs else []
    if len(_segs):
        _v = np.vstack(_segs)
        _tlo, _thi = tau_grid[0], tau_grid[-1]; _tb = (_thi - _tlo) * 0.14
        _xlo, _xhi = P_elec_grid[0], P_elec_grid[-1]; _xb = (_xhi - _xlo) * 0.14
        _m = ((_v[:, 1] > _tlo + _tb) & (_v[:, 1] < _thi - _tb) &
              (_v[:, 0] > _xlo + _xb) & (_v[:, 0] < _xhi - _xb))
        _c = _v[_m] if _m.any() else _v
        _refx = np.array([P_grid_ref, P_ppa_ref, P_ppa_2050]); _refy = np.array([tau_ets])
        _xr = _xhi - _xlo; _yr = _thi - _tlo
        _dxn = np.min(np.abs(_c[:, 0][:, None] - _refx[None, :]), axis=1) / _xr
        _dyn = np.min(np.abs(_c[:, 1][:, None] - _refy[None, :]), axis=1) / _yr
        _k = int(np.argmax(np.minimum(_dxn, _dyn)))
        _lbl0 = ax.clabel(cs0, inline=True, fmt='Σ wUP = 0', fontsize=6,
                          manual=[(_c[_k, 0], _c[_k, 1])])
    else:
        _lbl0 = ax.clabel(cs0, inline=True, fmt='Σ wUP = 0', fontsize=6)
    for _t in _lbl0:
        _t.set_clip_on(False); _t.set_zorder(6)
    # Faint reference contours
    levels_ref = np.round(np.linspace(vmin, vmax, 11), 2)
    levels_ref = [l for l in levels_ref if abs(l) > 1e-3]
    cs = ax.contour(P_elec_grid, tau_grid, Z, levels=levels_ref,
                     colors='black', linewidths=0.4, alpha=0.4)
    ax.clabel(cs, inline=True, fmt='%.2f', fontsize=5)

    # Reference markers
    ax.axvline(P_grid_ref, color='gray', linestyle=':', linewidth=1.3, alpha=0.95, zorder=1)
    ax.axvline(P_ppa_ref,  color='blue', linestyle=':', linewidth=1.3, alpha=0.95, zorder=1)
    ax.axvline(P_ppa_2050, color='green', linestyle=':', linewidth=1.3, alpha=0.95, zorder=1)
    ax.axhline(tau_ets, color='purple', linestyle=':', linewidth=1.3, alpha=0.95, zorder=1)

    ax.set_box_aspect(1)                                    # square panel
    title = 'α = %.1f   CI$_{\\mathbf{elec}}$ = %.0f gCO$_{\\mathbf{2eq}}$/kWh' % (alpha, CI_e_g)
    ax.set_title(title, fontsize=8, fontweight='bold', pad=9)
    if show_x:
        ax.set_xlabel('Electricity price (\\$/kWh)')
    if show_y:
        ax.set_ylabel(r'Carbon pricing (\$/kg$\mathbf{CO_{2eq}}$)', fontweight='bold')
    ax.tick_params(labelsize=7)
    return im

panel_labels = ['a', 'b', 'c', 'd', 'e', 'f']
for k, alpha in enumerate(ALPHA_VALUES):
    ax = axes[k // 2, k % 2]
    Z = results[alpha]['Z']; CI_e_g = results[alpha]['CI']
    show_y = (k % 2 == 0)
    show_x = (k // 2 == 2)
    im = format_panel(ax, Z, CI_e_g, alpha, panel_labels[k],
                       show_y=show_y, show_x=show_x)

# Panel letters at the top-left CORNER of each panel (outside the plot)
fig.canvas.draw()
for k in range(len(ALPHA_VALUES)):
    ax = axes[k // 2, k % 2]; pos = ax.get_position()
    xpos = 0.006 if (k % 2 == 0) else (pos.x0 - 0.095)   # left col at figure edge
    fig.text(xpos, pos.y1 + 0.013, f'({panel_labels[k]})',   # raised to clear tick labels
             ha='left', va='bottom', fontsize=11, fontweight='bold')

# Shared colorbar (right side)
cbar_ax = fig.add_axes([0.885, 0.115, 0.018, 0.85])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.tick_params(labelsize=7)
cbar.set_label('Unit profit (\\$/kg)', size=9, weight='bold')

# Legend (bottom centre) — unified handle legend (matches Fig. 11)
legend_handles = [
    Line2D([0], [0], color='black', linewidth=1.6, label='Break-even (Σ w·UP = 0)'),
    Line2D([0], [0], color='gray', linestyle=':', linewidth=1.4, label=r'Grid (0.08 \$/kWh)'),
    Line2D([0], [0], color='blue', linestyle=':', linewidth=1.4, label=r'PPA 2026 (0.04 \$/kWh)'),
    Line2D([0], [0], color='green', linestyle=':', linewidth=1.4, label=r'PPA 2050 (0.02 \$/kWh)'),
    Line2D([0], [0], color='purple', linestyle=':', linewidth=1.4, label=r'EU ETS (~\$80/tCO$_{2eq}$)'),
]
fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.0),
           ncol=5, fontsize=7, frameon=False,
           columnspacing=1.0, handlelength=1.6)

FIGDIR = config.FIG_OUT
out_png = os.path.join(FIGDIR, 'legacy_Fig10_price_alpha_panels.png')
plt.savefig(out_png, dpi=400, bbox_inches='tight')
plt.close()
print(f'\nSaved: {out_png}')

# ── Threshold line plot (compact summary) ─────────────────────────
# For each α, find τ-threshold at each P_elec where Z crosses 0.
print('\nComputing break-even τ vs P_elec curves per α...')

fig2, ax2 = plt.subplots(figsize=(11, 6.8))
fig2.subplots_adjust(top=0.91, bottom=0.13, left=0.08, right=0.78)

cmap_a = plt.cm.viridis(np.linspace(0.05, 0.92, len(ALPHA_VALUES)))
for k, alpha in enumerate(ALPHA_VALUES):
    Z = results[alpha]['Z']
    CI_e_g = results[alpha]['CI']
    # For each P_elec column, find τ where Z = 0
    breakeven_tau = []
    for j in range(len(P_elec_grid)):
        col = Z[:, j]
        s = np.where(np.diff(np.sign(col)) != 0)[0]
        if len(s) == 0:
            breakeven_tau.append(np.nan)
        else:
            i = s[0]
            denom = col[i+1] - col[i]
            if denom == 0:
                breakeven_tau.append(tau_grid[i])
            else:
                t_be = tau_grid[i] + (tau_grid[i+1]-tau_grid[i]) * (-col[i])/denom
                breakeven_tau.append(t_be)
    breakeven_tau = np.array(breakeven_tau)
    ax2.plot(P_elec_grid, breakeven_tau*1000, color=cmap_a[k],
              linewidth=2.4, label=f'α = {alpha:.1f}   (CI = {CI_e_g:.0f} gCO₂/kWh)')

ax2.axvline(P_grid_ref, color='gray', linestyle=':', linewidth=1.1, alpha=0.7)
ax2.text(P_grid_ref + 0.001, ax2.get_ylim()[1]*0.95 if ax2.get_ylim()[1] > 0 else 5,
          'grid\n0.08', fontsize=8, color='gray', va='top')
ax2.axvline(P_ppa_ref, color='blue', linestyle=':', linewidth=1.1, alpha=0.7)
ax2.text(P_ppa_ref + 0.001, 195, 'PPA 2026\n0.04', fontsize=8, color='blue', va='top')
ax2.axvline(P_ppa_2050, color='green', linestyle=':', linewidth=1.1, alpha=0.7)
ax2.text(P_ppa_2050 + 0.001, 195, 'PPA 2050\n0.02', fontsize=8, color='green', va='top')
ax2.axhline(80, color='purple', linestyle=':', linewidth=1.1, alpha=0.7)
ax2.text(0.079, 82, 'EU ETS ~$80/tCO₂', fontsize=8, color='purple', va='bottom', ha='right')
ax2.axhline(50, color='orange', linestyle=':', linewidth=1.0, alpha=0.6)
ax2.text(0.079, 52, 'US 45Z ~$50/tCO₂', fontsize=8, color='orange', va='bottom', ha='right')

ax2.set_xlabel('Electricity price P$_{elec}$ ($/kWh)', fontsize=11)
ax2.set_ylabel('Break-even τ ($/tCO$_2$)', fontsize=11)
ax2.set_title('M2 Step 4 (RC) — Carbon tax threshold vs electricity price under fixed α\n'
              'Curve = lowest τ at which Σ wᶠ·UP$_{net}$ = 0 (full implementation feasible)',
              fontsize=12, fontweight='bold')
ax2.grid(linestyle=':', alpha=0.4)
ax2.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
            fontsize=9.5, framealpha=0.95, title='Renewable fraction',
            title_fontsize=10)
ax2.set_xlim(P_elec_grid[0], P_elec_grid[-1])

out_png2 = os.path.join(config.FIG_OUT, 'fig_m2_step4_RC_threshold_curves.png')
plt.savefig(out_png2, dpi=180, bbox_inches='tight')
plt.close()
print(f'Saved: {out_png2}')

# Write CSV
csv_rows = []
for alpha in ALPHA_VALUES:
    Z = results[alpha]['Z']
    CI_e_g = results[alpha]['CI']
    for i, t in enumerate(tau_grid):
        for j, P in enumerate(P_elec_grid):
            csv_rows.append(dict(alpha=alpha, CI_elec=CI_e_g,
                                  P_elec=P, tau=t, wUP_net=Z[i, j]))
pd.DataFrame(csv_rows).to_csv(os.path.join(BASE, 'm2_step4_RC_price_alpha_panels.csv'),
                                index=False, float_format='%.4f')
print('\nDone.')
