import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
# -*- coding: utf-8 -*-
"""Fig. 7 - PVC as the binding constraint of the mixed waste stream.

  (a) why PVC is the outlier, resin by resin: carbon accounting above, net profit
      below, both against the incineration benchmark
  (b) what that does to each region and what relieving it achieves, region by
      region, with the renewable-electrification case on the same axis

Order follows the hierarchy resin then region, and (b) closes the figure because
the regional result is the conclusion. It also carries the regional identity
forward from Fig. 6, which the earlier draft lost by plotting regions as
unlabelled markers.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.labelsize': 8, 'axes.labelweight': 'bold',
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.0, 'ytick.major.size': 2.0,
})
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial'
mpl.rcParams['mathtext.bf'] = 'Arial:bold'
# without this, anything inside $...$ renders italic serif and the 'eq' of
# CO_{2eq} would not match the surrounding Arial
mpl.rcParams['mathtext.default'] = 'regular'

DATA = config.INTERMEDIATE
OUT = config.FIG_OUT
MM = 1 / 25.4
FIG_W, FIG_H = 174.0 * MM, 104.0 * MM

C = pd.read_csv(os.path.join(DATA, 'fig7_panelC_mean.csv'))
E = pd.read_csv(os.path.join(DATA, 'fig7_panelB_profit.csv'))
B = pd.read_csv(os.path.join(DATA, 'fig7_panelB.csv'))
PT = pd.read_csv(os.path.join(DATA, 'fig7_fan_points.csv'))

POS, NEG, GREY, INK = '#2a7f62', '#c0392b', '#7f8c8d', '#1c2833'
FEED_C = {'PE': '#2f6f9f', 'PP': '#3f8f5a', 'PET': '#d99a2b',
          'PS': '#8a6bb1', 'PVC': '#c0504d'}
# stack colours derive from the figure-set anchors POS / NEG / INK, so that no
# colour is invented here and green keeps one meaning across both panels
PD_C, FC_C, E_C, CR_C = '#2a7f62', '#9ecbb8', '#e0a39b', INK

fig = plt.figure(figsize=(FIG_W, FIG_H))
axA1 = fig.add_axes([0.090, 0.435, 0.355, 0.495])   # resin, carbon
axA2 = fig.add_axes([0.090, 0.225, 0.355, 0.160])   # resin, profit
axB = fig.add_axes([0.598, 0.225, 0.387, 0.705])    # region

# ══════════════ (a) resin by resin ══════════════
feeds = ['PE', 'PET', 'PP', 'PS', 'PVC']
sub = C.set_index('Feed')
order = sub.loc[feeds].sort_values('CR', ascending=False).index.tolist()
d = sub.loc[order]
xs, Wb = np.arange(len(order), dtype=float), 0.56
XLIM = (-0.75, len(order) - 0.25)   # bars span the full panel width

axA1.bar(xs, d.PD, width=Wb, color=PD_C, edgecolor='white', lw=0.4, zorder=3)
axA1.bar(xs, d.FC, width=Wb, bottom=d.PD, color=FC_C, edgecolor='white', lw=0.4,
         zorder=3)
axA1.bar(xs, -d.E, width=Wb, color=E_C, edgecolor='white', lw=0.4, zorder=3)
axA1.axhline(0, color='black', lw=0.5, zorder=4)

oth = sub.loc['Others']
i_break = order.index('PET')
# one continuous run that ends at PET: the four resins it averages are the four
# it spans, and PVC is deliberately outside it
for x0, x1 in [(-0.62, i_break + 0.30)]:
    axA1.plot([x0, x1], [oth.CR, oth.CR], color=GREY, lw=0.8, ls=(0, (4, 2)),
              zorder=5)
axA1.plot(xs, d.CR, marker='D', ms=4.0, ls='none', mfc=CR_C, mec='none',
          zorder=7)
# a label near the mean line is placed on the far side of the marker, so the
# reference line never runs through a number
for xv, v in zip(xs, d.CR):
    near = abs(v - oth.CR) < 0.5
    dy, va = (-8, 'top') if near else (7, 'bottom')
    axA1.annotate(f'{v:+.2f}', (xv, v), textcoords='offset points', xytext=(0, dy),
                  ha='center', va=va, fontsize=6.2, fontweight='bold', color=CR_C,
                  zorder=8)

# The gap decomposes exactly into the three accounting terms of the bars, so the
# split bar reuses their colours and needs no new vocabulary. An earlier version
# labelled it structural / addressable; 'addressable' is absent from both the
# manuscript and the sixteen reference papers, and it claimed a reachability the
# optimisation does not show.
pvc = sub.loc['PVC']
d_FC = oth.FC - pvc.FC              # avoided incineration, far smaller for PVC
d_PD = oth.PD - pvc.PD              # product displacement, smaller for PVC
d_E = -(oth.E - pvc.E)              # process emissions, PVC actually emits less
gap = d_FC + d_PD + d_E             # identical to oth.CR - pvc.CR

# the split bar no longer competes with the resin columns for horizontal room:
# it sits in its own inset over the short PVC column, top right
y0 = pvc.CR
SEG = [(y0, y0 + d_FC, FC_C, d_FC),
       (y0 + d_FC, y0 + d_FC + d_PD, PD_C, d_PD),
       (y0 + d_FC + d_PD, oth.CR, E_C, d_E)]
axA1.add_patch(FancyBboxPatch(
    (0.7565, 0.600), 0.195, 0.317, transform=axA1.transAxes,
    boxstyle='round,pad=0.0,rounding_size=0.020', facecolor='white',
    edgecolor='#bbb', lw=0.5, zorder=5, clip_on=False))
ins = axA1.inset_axes([0.7645, 0.614, 0.180, 0.290])
ins.set_facecolor('none')
for sp in ins.spines.values():
    sp.set_visible(False)
ins.set_xticks([]); ins.set_yticks([])
ins.set_xlim(0.0, 1.0)
ins.set_ylim(y0 - 0.08, y0 + d_FC + d_PD + 0.08)
ins.patch.set_alpha(0.0)
XB = 0.14
for lo, hi, col, val in SEG:
    ins.plot([XB, XB], [lo, hi], color=col, lw=3.4, solid_capstyle='butt',
             zorder=6, clip_on=False)
    ins.annotate('%+.2f' % val, (XB + 0.28, (lo + hi) / 2), fontsize=5.4,
                 va='center', ha='left', color=INK, fontweight='bold',
                 zorder=9, annotation_clip=False)
axA1.annotate('Gap to mean  %.2f' % gap, (0.8540, 0.928),
              xycoords='axes fraction', fontsize=5.8, ha='center', va='bottom',
              color=INK, fontweight='bold', zorder=9)

axA1.set_xticks(xs); axA1.set_xticklabels([])
axA1.set_xlim(*XLIM); axA1.set_ylim(-1.35, 4.85)
axA1.set_ylabel('Carbon accounting\n(kgCO$_{2eq}$/kg)', fontsize=7.2)
axA1.grid(axis='y', ls=':', lw=0.35, alpha=0.35)
axA1.tick_params(axis='x', length=0)

ec = E.set_index('Feed').loc[order]
axA2.axhline(0, color='black', lw=0.5, zorder=4)
for xv, (f, r) in zip(xs, ec.iterrows()):
    lo, hi = min(r.UP_incin, r.UPnet), max(r.UP_incin, r.UPnet)
    axA2.plot([xv, xv], [lo, hi], color=FEED_C[f], lw=3.2, alpha=0.40,
              solid_capstyle='butt', zorder=3)
    axA2.plot([xv - 0.20, xv + 0.20], [r.UP_incin, r.UP_incin], color=GREY,
              lw=1.0, zorder=5)
    axA2.plot(xv, r.UPnet, marker='D', ms=4.0, mfc=FEED_C[f], mec='none',
              mew=0.5, zorder=7)
# no call-out here: the zero line and the single marker below it already say
# that PVC is the only resin under break-even, and the caption states it

axA2.set_xticks(xs)
axA2.set_xticklabels(order, fontsize=7.5, fontweight='bold')
# the resin colour is already carried by the markers directly above each label,
# so colouring the words as well adds no information and costs legibility
for t in axA2.get_xticklabels():
    t.set_color(INK)
axA2.set_xlim(*XLIM); axA2.set_ylim(-0.72, 0.82)
axA2.set_yticks([-0.5, 0.0, 0.5])
axA2.set_ylabel('Net profit\n(\\$/kg)', fontsize=7.2)
axA2.grid(axis='y', ls=':', lw=0.35, alpha=0.35)
axA2.tick_params(axis='x', length=0)

# ══════════════ (b) region by region ══════════════
B = B.sort_values('w_PVC').reset_index(drop=True)
y = np.arange(len(B))
elec = PT[(PT.Scenario == 'Renewable electrification')
          & (PT.Strategy == 'A')].set_index('Region').margin

axB.axvline(0, color='black', lw=0.8, ls='--', zorder=4)
for i, r in B.iterrows():
    axB.plot([r.margin_B, r.margin_C], [i, i], color='#dde1e3', lw=1.8,
             solid_capstyle='round', zorder=3)
axB.scatter(B.margin_B, y, s=17, marker='s', facecolor=NEG, edgecolor='none',
            lw=0.4, zorder=6, label='PVC to incineration')
axB.scatter(B.margin_A, y, s=17, marker='o', facecolor=GREY, edgecolor='none',
            lw=0.4, zorder=7, label='All five upcycled')
axB.scatter(B.margin_C, y, s=19, marker='^', facecolor=POS, edgecolor='none',
            lw=0.4, zorder=6, label='PVC not in the stream')
# the electrified case on the same axis, so the shift is read directly
axB.scatter([elec[r.Region] for r in B.itertuples()], y, s=18, marker='o',
            facecolor='white', edgecolor=GREY, lw=0.8, zorder=5,
            label='All five upcycled, electrified')

axB.set_yticks(y)
axB.set_yticklabels([f'{r.Region}  {r.w_PVC*100:.0f}%' for _, r in B.iterrows()],
                    fontsize=6.5)
for t, r in zip(axB.get_yticklabels(), B.itertuples()):
    if r.margin_A < 0:
        t.set_color(NEG); t.set_fontweight('bold')
axB.set_ylim(-0.7, len(B) - 0.3)
axB.set_xlim(-0.44, 0.92)
# 'relative to' is the construction the manuscript uses to define its reduction
# metrics against a reference (P68, P73); 'minus' appears nowhere in the text
axB.set_xlabel('Carbon reduction relative to target (kgCO$_{2eq}$/kg)',
               fontsize=7.2)
axB.grid(axis='x', ls=':', lw=0.35, alpha=0.35)
axB.tick_params(axis='y', length=0)
axB.text(0.985, 0.022, 'Sorted by PVC share', transform=axB.transAxes,
         ha='right', va='bottom', fontsize=6.2, color=INK, style='italic')

# ══════════════ legends ══════════════
hA = [Patch(facecolor=PD_C, edgecolor='white', lw=0.4, label='Product displacement'),
      Patch(facecolor=FC_C, edgecolor='white', lw=0.4, label='Avoided incineration'),
      Patch(facecolor=E_C, edgecolor='white', lw=0.4, label='Process emissions'),
      Line2D([0], [0], color=CR_C, marker='D', ms=4.0, ls='none', mfc=CR_C,
             mec='none', label='Net carbon reduction'),
      Line2D([0], [0], color=GREY, lw=0.8, ls=(0, (4, 2)), label='Four-resin mean'),
      Line2D([0], [0], color=GREY, lw=1.0, label='Incineration benchmark')]
lgA = fig.legend(handles=hA, loc='lower center', ncol=3,
                 bbox_to_anchor=(0.260, 0.008), fontsize=6.2, framealpha=1.0,
                 edgecolor='#bbb', fancybox=False, handletextpad=0.4,
                 borderpad=0.36, labelspacing=0.28, columnspacing=1.1,
                 handlelength=1.0)
lgA.get_frame().set_linewidth(0.4)

hB, lB = axB.get_legend_handles_labels()
lgB = fig.legend(hB, lB, loc='lower center', ncol=2,
                 bbox_to_anchor=(0.790, 0.008), fontsize=6.2, framealpha=1.0,
                 edgecolor='#bbb', fancybox=False, handletextpad=0.3,
                 borderpad=0.36, labelspacing=0.28, columnspacing=1.1,
                 handlelength=0.9)
lgB.get_frame().set_linewidth(0.4)

for ax, lab in [(axA1, 'a'), (axB, 'b')]:
    ax.text(0.0, 1.020, f'({lab})', transform=ax.transAxes, fontsize=8.5,
            fontweight='bold', va='bottom', ha='left')

out = os.path.join(OUT, 'Fig6_pvc_bottleneck.png')
plt.savefig(out, dpi=500, facecolor='white')
plt.close()
print('saved', out)
print(f'resin order {order}')
print(f'gap {gap:.3f} = avoided incineration {d_FC:.3f} + product displacement {d_PD:.3f} + process emissions {d_E:.3f}')
print(f'regions failing at fossil baseline: {int((B.margin_A < 0).sum())}/11, '
      f'with PVC incinerated {int((B.margin_B < 0).sum())}/11, '
      f'with PVC not in the stream {int((B.margin_C < 0).sum())}/11')
print(f'electrified margin range {elec.min():+.3f} to {elec.max():+.3f}')
