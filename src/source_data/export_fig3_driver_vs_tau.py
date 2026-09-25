import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config  # repository paths, see config.py
"""Fig. 3C data: mean cost-equivalent share of each cost origin versus carbon price.

For every pathway the carbon price tau ($/kgCO2) converts each emission origin into
a cost; the share of each origin in the pathway's total is then averaged over the
959 pathways. tau runs over 0-0.10 $/kgCO2 in 51 steps.

This is the tau loop of the authors' Source Data builder (make_fig_rawdata.py),
exported on its own because Fig3_drivers.py reads the result as a CSV.
Output: data/intermediate/fig3_driver_vs_tau.csv
"""
import os
import numpy as np
import pandas as pd

MASS = 50e6          # kg plastic per year, the plant basis of the master table
FEEDS = ['PE', 'PET', 'PP', 'PS', 'PVC']

df = pd.read_excel(config.MASTER_XLSX, sheet_name='All pathways', header=3)
df['__feed'] = df.iloc[:, 0].astype(str).str.strip()
df = df[df['__feed'].isin(FEEDS)].reset_index(drop=True)
print(f'{len(df)} pathways')


def coln(c):
    return pd.to_numeric(df.iloc[:, c], errors='coerce').fillna(0).values


def pk(x):   # M$/y -> $/kg
    return x * 1e6 / MASS


def epk(x):  # kgCO2/y -> kgCO2/kg
    return x / MASS


# cost columns (M$/y) and emission columns (kgCO2eq/y) of the master sheet
ACC = pk(coln(5)); FEED = pk(coln(6)); ELEC = pk(coln(7)); FH = pk(coln(8)); CW = pk(coln(9))
REF = pk(sum(coln(10 + k) for k in range(5)))
LP = pk(coln(15)); HP = pk(coln(16)); SULF = pk(coln(17)); STEAM = pk(coln(18))
WATER = pk(coln(19)); H2 = pk(coln(20)); MEA = pk(coln(21)); O2 = pk(coln(22))
COs = pk(coln(23)); H2s = pk(coln(24)); FOC = pk(coln(25))
DIR = epk(coln(30)); ELEC_g = epk(coln(31)); FH_g = epk(coln(32))
REF_g = epk(sum(coln(33 + k) for k in range(5)))
LP_g = epk(coln(38)); HP_g = epk(coln(39)); STEAM_g = epk(coln(40))
O2_g = epk(coln(41)); H2_g = epk(coln(42)); RAW_g = epk(coln(47))

energy_op = ELEC + FH + REF + LP + HP + STEAM + H2
energy_emis = ELEC_g + FH_g + REF_g + LP_g + HP_g + STEAM_g + H2_g
feed_op = FEED
feed_emis = RAW_g
process_emis = DIR + O2_g
capital = ACC
operating = FOC
other_op = CW + SULF + WATER + MEA + O2 + COs + H2s

rows = []
for t in np.linspace(0, 0.10, 51):
    e = energy_op + t * energy_emis
    fd = feed_op + t * feed_emis
    pr = t * process_emis
    tot = e + fd + capital + pr + operating + other_op
    rows.append({'tau_$/kgCO2': t,
                 'Energy_%': np.nanmean(e / tot) * 100,
                 'Feed_%': np.nanmean(fd / tot) * 100,
                 'Capital_%': np.nanmean(capital / tot) * 100,
                 'Process_%': np.nanmean(pr / tot) * 100,
                 'Operating_%': np.nanmean(operating / tot) * 100,
                 'Other_%': np.nanmean(other_op / tot) * 100})

out = os.path.join(config.INTERMEDIATE, 'fig3_driver_vs_tau.csv')
pd.DataFrame(rows).to_csv(out, index=False)
print('saved', out)
print('energy share at tau = 0: %.1f %%' % rows[0]['Energy_%'])
