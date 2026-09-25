"""Shared paths for every script in this repository.

All paths are relative to the repository root, so the code runs from any
location. Scripts add the repository root to ``sys.path`` and ``import config``.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

DATA = os.path.join(ROOT, 'data')
MASTER = os.path.join(DATA, 'master')              # inputs (values only)
INTERMEDIATE = os.path.join(DATA, 'intermediate')  # CSV/JSON written by src/ scripts
SOURCE_DATA = os.path.join(DATA, 'source_data')    # Source Data file for the paper
EXTERNAL = os.path.join(DATA, 'external')          # third-party public-domain data

FIGS = os.path.join(ROOT, 'figures')
FIG_OUT = os.path.join(FIGS, 'output')             # every PNG is written here

# Master pathway table: sheet 'All pathways', header on row 4 (pandas header=3),
# 959 pathways in rows 5-963. Same cell layout as the authors' working workbook.
MASTER_XLSX = os.path.join(MASTER, 'all_pathways_master.xlsx')
ALL_PATHWAYS_CSV = os.path.join(MASTER, 'all_pathways.csv')
PATHWAY_NAMING = os.path.join(MASTER, 'pathway_naming.xlsx')
COUNTRY_PRICES = os.path.join(MASTER, 'country_price_parameters.xlsx')

# Natural Earth 1:110m admin-0 countries (public domain), read by Fig. 7
NE_ADMIN0 = os.path.join(EXTERNAL, 'naturalearth', 'ne_110m_admin_0_countries.zip')

for _d in (INTERMEDIATE, FIG_OUT):
    os.makedirs(_d, exist_ok=True)

# helper modules shared between scripts
for _p in (os.path.join(ROOT, 'src', 'common'), os.path.join(ROOT, 'src', 'pvc')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
