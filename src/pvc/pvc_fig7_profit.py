import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
# -*- coding: utf-8 -*-
"""Per-resin economics for Fig. 7(b): net unit profit against the incineration
benchmark, 2050, averaged over the eleven regions."""
import sys, os
import pandas as pd
B = config.INTERMEDIATE
import pvc_bottleneck as M

FEEDS = ['PE', 'PET', 'PP', 'PS', 'PVC']
d = pd.read_csv(os.path.join(B, 'pvc_bottleneck_perfeed.csv'))
d = d[d.Period.astype(str) == '2050']
assert d[[f'UPnet_{f}' for f in FEEDS]].isna().sum().sum() == 0

b_ref = sum(M.REGION_W['Global/EU'][f] * M.INCIN_B[f] for f in FEEDS)
rows = []
for f in FEEDS:
    c = d[f'UPnet_{f}']
    rows.append(dict(Feed=f, UPnet=c.mean(), UPnet_min=c.min(), UPnet_max=c.max(),
                     UP_incin=-0.42 + (M.INCIN_B[f] - b_ref) * 0.08,
                     n_regions=len(c), n_negative=int((c < 0).sum())))
e = pd.DataFrame(rows)
e['advantage'] = e.UPnet - e.UP_incin
oth = e[e.Feed != 'PVC'].UPnet.mean()
e.loc[len(e)] = dict(Feed='Others', UPnet=oth, UPnet_min=float('nan'),
                     UPnet_max=float('nan'),
                     UP_incin=e[e.Feed != 'PVC'].UP_incin.mean(), n_regions=11,
                     n_negative=0, advantage=float('nan'))
e.to_csv(os.path.join(B, 'fig7_panelB_profit.csv'), index=False,
         float_format='%.4f')
print(e.to_string(index=False, float_format=lambda v: f'{v:8.3f}'))
print(f"\nfour-resin mean profit {oth:.3f}   PVC {e.loc[e.Feed=='PVC','UPnet'].iloc[0]:.3f}"
      f"   gap {oth - e.loc[e.Feed=='PVC','UPnet'].iloc[0]:.3f} $/kg")
print('wrote fig7_panelB_profit.csv')
