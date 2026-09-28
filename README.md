# plastic-upcycling-energy-transition

Code and data for

> Woochang Jeong, Chanhee You, Jiyong Kim. **Plastic waste upcycling in a decarbonizing energy system: Regional and long-term viability.** Sungkyunkwan University. (Manuscript under review.)

The repository regenerates every quantitative figure and table of the paper from one input: a techno-economic and environmental table of **959 upcycling pathways** (5 plastic feedstocks x pre-treatment, conversion and upgrading routes). All pathway results in the paper derive from this enumerated 959-pathway table. Scenario, regional and sensitivity analyses rescale its cost and emission columns and select optima by exhaustive enumeration (argmax over the pathways of each feedstock), so no optimization solver is needed.

## Quick start

```bash
python -m pip install -r requirements.txt
python run_all.py
```

`run_all.py` runs 30 steps in order, writes intermediate tables to `data/intermediate/` and figures to `figures/output/`, and stops at the first error (a log of every step is kept in `logs/`). A full run takes about **3 minutes** (measured on Windows 11 with Python 3.14). Use `python run_all.py --list` to see the steps and `--from N` to resume.

The committed files in `data/intermediate/` are the outputs of this pipeline. After a run, `git diff data/intermediate` shows any difference from the published values.

## Repository layout

```
config.py            shared paths (repository root, data and figure folders)
run_all.py           the full pipeline
data/
  master/            inputs, values only
    all_pathways_master.xlsx   959-pathway table, sheet 'All pathways' (header on row 4, same cell layout the scripts index by position)
    all_pathways.csv           the same 959 rows as a tidy CSV with unique column names (not read by the scripts)
    pathway_naming.xlsx        feedstock / technology / product and process sequence of every pathway
    country_price_parameters.xlsx  global baseline utility prices and carbon intensities (only the cells the code reads)
  intermediate/      CSV and JSON written by the scripts (committed)
  source_data/
    Source_Data.xlsx           Source Data for Figs. 2-7 and Supplementary Figs./Tables, one sheet per panel, with an Index sheet
    si/                        data behind the Supplementary figures that were drawn outside Python
  external/naturalearth/       Natural Earth 1:110m admin-0 countries (public domain), used by Fig. 7
src/
  common/            ref_elec_coupling.py: electricity coupling of the refrigeration utilities
  pathway_screening/ ZONE classification and cost/emission drivers
  electrification/   three electrification scenarios, feedstock optima, price x carbon-price maps
  regional/          regional-temporal model (canonical core), decomposition, robustness table
  pvc/               PVC bottleneck analysis
  sensitivity/       tornado, Sobol, regime map
  source_data/       export of the Fig. 3C table
figures/
  main/              Fig2-Fig7 plotting scripts
  si/                Supplementary figure scripts
  output/            generated PNG files (not committed)
```

## Figures and tables

| Item | Plotting script | Data it reads | Produced by |
|---|---|---|---|
| Fig. 1 | schematic, no data | - | - |
| Fig. 2 | `figures/main/Fig2_landscape.py` | `zone_classification.csv` | `src/pathway_screening/phase7_zone_analysis.py` |
| Fig. 3 | `figures/main/Fig3_drivers.py` | `phase7_zone_analysis.json`, `fig3_driver_vs_tau.csv` | `phase7_zone_analysis.py`, `src/source_data/export_fig3_driver_vs_tau.py` |
| Fig. 4 | `figures/main/Fig4_electrification.py` | `phase11_three_scenarios_RC.csv` | `src/electrification/phase11_three_scenarios_RC.py` |
| Fig. 5 | `figures/main/Fig5_conditions.py` | `m2_step4_RC_compare.csv` (alpha = 0.2, 0.5, 0.8 shown) | `src/electrification/m2_step4_RC_price_alpha_compare.py` |
| Fig. 6 | `figures/main/Fig6_pvc_bottleneck.py` | `fig7_panelC_mean.csv`, `fig7_panelB_profit.csv`, `fig7_panelB.csv`, `fig7_fan_points.csv` | `src/pvc/pvc_bottleneck*.py`, `pvc_fig7_*.py` |
| Fig. 7 | `figures/main/Fig7_worldmap.py` | master table, Natural Earth; regional inputs are in the script (Tables S8.1-S8.4) | itself; writes `m3_step5_RC_world_periods_INCIN.csv` |
| Fig. S1.1 | `figures/si/FigS1_1_energy_efficiency.py` | `data/source_data/si/FigS1_1_energy_efficiency.csv` | master table |
| Section S5 cost breakdown | not scripted | `data/source_data/si/FigS5_cost_revenue_breakdown_average.csv` (average annual cost and revenue items, all pathways and optimal pathways) | authors' spreadsheet; the unit-process breakdowns of Figs. S5.3-S5.5 derive from the process simulation results and are available on request |
| Fig. S6.2 | `figures/si/FigS6_2_UP_optimal_bar.py` | master table | itself |
| Fig. S6.3 | `figures/si/FigS6_3_UCR_optimal_bar.py` | master table | itself |
| Fig. S7.1 | not scripted | `data/source_data/si/FigS7_1_*.csv` (interest rate, feedstock price) | authors' spreadsheet |
| Figs. S7.2-S7.4 | not scripted | `data/source_data/si/FigS7_2_*` (H2 price), `FigS7_3_*` (heat CI), `FigS7_4_*` (electricity CI) | authors' spreadsheet |
| Fig. S7.5 | `figures/si/FigS7_5_decomposition.py` | `m3_step5_RC_decomposition_INCIN.csv` | `src/regional/m3_step5_RC_decomposition_INCIN.py` |
| Fig. S7.6 | `src/sensitivity/tornado_plot.py` | master table | itself |
| Fig. S7.7 | `figures/si/FigS7_7_sobol.py` | `sobol_indices.csv`, `sobol_indices_CR.csv` | `src/sensitivity/sobol_indices.py`, `sobol_indices_CR.py` |
| Table S7.1 | - | `SN3_robustness_INCIN.csv` | `src/regional/robustness_SN3_INCIN.py` |
| Fig. S7.8 | `src/sensitivity/phase14_optimal_pathway_sweep_RC.py` | master table | itself; writes `phase14_optimal_table_RC.csv` |
| Fig. S7.9 | `src/electrification/m2_step3_RC_optimal.py` | master table, `pathway_naming.xlsx` | itself; writes `m2_step3_RC_econ.csv`, `m2_step3_RC_env.csv` |
| Tables S8.1-S8.4 | - | `Source_Data.xlsx`, sheets `Fig7_TableS8-*` | values hard-coded in `Fig7_worldmap.py` |

Notes

* The `si/FigS7_*` files are cell-by-cell exports of the spreadsheet sheets the figures were drawn from. The first column is the original Excel row and the header holds the Excel column letters, so the layout (one block of optimal pathways per parameter level) can be read as in the spreadsheet.
* The output names `fig7_*.csv` of the PVC scripts date from an earlier figure numbering; they feed Fig. 6.
* Figs. 5 and 6: the scripts produce the data panels. The final published layouts (panel lettering, panel arrangement and marker colours) were assembled by the authors in a graphics editor, so the script output differs from the published figures in layout only, not in data. For Fig. 6 the script labels the resin panels `a` (upper, lower) and the regional panel `b`, which appear as separate lettered panels in the paper.
* Fig. 7 is drawn from the regional model in `src/regional/m3_step5_RC_world_map_periods_INCIN.py`. The figure script carries the same model code; only the figure size and in-figure wording differ. `robustness_SN3_INCIN.py` executes the model core of the canonical file, so Table S7.1 and Fig. 7 share one model.
* The pathway-level analyses (Figs. 2-5, S6.2-S6.3, S7.6-S7.9) compare against a fixed incineration benchmark evaluated at the baseline grid intensity of 369 gCO2eq/kWh. The incineration credit is regionalized with the grid intensity only in the regional analyses (Figs. 6-7, S7.5, Table S7.1).
* `m2_step4_RC_price_alpha_compare.py` and `m2_step4_RC_price_alpha_panels.py` also draw six-panel maps (`legacy_Fig10/11_*.png`) from an earlier version of the manuscript. They are kept because they write `m2_step4_RC_compare.csv`, the input of Fig. 5.
* Sobol indices use a fixed seed (`SEED` in the two scripts). The published indices were computed without a seed; the difference is about 0.001 in total-order and at most 0.007 in first-order indices and changes no value printed in Fig. S7.7 (two bars with the same printed value, renewable share and electricity price, can swap places).
* Figures use Arial when it is installed and fall back to DejaVu Sans otherwise, which changes text widths slightly.

## Not included

* The superstructure optimization (MILP) models (GAMS) and the Aspen Plus process simulation models, including the stream results behind Supplementary Figs. S3.x and Tables S3.x, are available from the corresponding author upon reasonable request.
* The authors' working spreadsheet from which the master table is exported (it holds formulas linked to per-feedstock result workbooks). `data/master/all_pathways_master.xlsx` contains its values for every sheet the code reads.

## Requirements

Tested with Python 3.14 on Windows 11. Package versions are in `requirements.txt`. Fig. 7 needs `geopandas` and reads the Natural Earth file from `data/external/`, so no network access is required.

## Licence

* Code: MIT licence (`LICENSE`).
* Data in `data/master`, `data/intermediate` and `data/source_data`: Creative Commons Attribution 4.0 International (`data/LICENSE-DATA`).
* `data/external/naturalearth`: Natural Earth, public domain (https://www.naturalearthdata.com/about/terms-of-use/).

## Citation

Please cite the paper and this repository (`CITATION.cff`). Archived version: https://doi.org/10.5281/zenodo.23004984

## Contact

Jiyong Kim (corresponding author), Sungkyunkwan University.
