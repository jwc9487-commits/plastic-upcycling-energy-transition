import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
"""Regenerate every intermediate table and figure of the paper.

    python run_all.py            # full pipeline
    python run_all.py --list     # show the steps without running them
    python run_all.py --from 12  # resume from step 12

Intermediates are written to data/intermediate, figures to figures/output.
Each step runs in its own Python process; the pipeline stops at the first failure.
"""
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    # pathway screening (959 enumerated pathways -> ZONE 1-4, cost/emission drivers)
    'src/pathway_screening/phase7_zone_analysis.py',
    # electrification scenarios and feedstock-level optima
    'src/electrification/phase11_three_scenarios_RC.py',
    'src/electrification/m2_step3_RC_optimal.py',
    'src/electrification/m2_step4_RC_price_alpha_compare.py',
    'src/electrification/m2_step4_RC_price_alpha_panels.py',
    # regional and temporal model (also draws Fig. 7) and its decomposition
    'figures/main/Fig7_worldmap.py',
    'src/regional/m3_step5_RC_decomposition_INCIN.py',
    # PVC bottleneck analysis (Fig. 6)
    'src/pvc/pvc_bottleneck.py',
    'src/pvc/pvc_bottleneck3.py',
    'src/pvc/pvc_bottleneck_sens.py',
    'src/pvc/pvc_bottleneck_sens2.py',
    'src/pvc/pvc_fig7_data.py',
    'src/pvc/pvc_fig7_fan.py',
    'src/pvc/pvc_fig7_panelC_mean.py',
    'src/pvc/pvc_fig7_profit.py',
    # sensitivity and robustness (Supplementary Section S7)
    'src/sensitivity/tornado_plot.py',
    'src/sensitivity/sobol_indices.py',
    'src/sensitivity/sobol_indices_CR.py',
    'src/sensitivity/phase14_optimal_pathway_sweep_RC.py',
    'src/regional/robustness_SN3_INCIN.py',
    # Fig. 3C data
    'src/source_data/export_fig3_driver_vs_tau.py',
    # main figures
    'figures/main/Fig2_landscape.py',
    'figures/main/Fig3_drivers.py',
    'figures/main/Fig4_electrification.py',
    'figures/main/Fig5_conditions.py',
    'figures/main/Fig6_pvc_bottleneck.py',
    # supplementary figures
    'figures/si/FigS1_1_energy_efficiency.py',
    'figures/si/FigS6_2_UP_optimal_bar.py',
    'figures/si/FigS6_3_UCR_optimal_bar.py',
    'figures/si/FigS7_5_decomposition.py',
    'figures/si/FigS7_7_sobol.py',
]


def main(argv):
    if '--list' in argv:
        for i, s in enumerate(STEPS, 1):
            print(f'{i:2d}  {s}')
        return 0
    start = int(argv[argv.index('--from') + 1]) if '--from' in argv else 1
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    logdir = os.path.join(ROOT, 'logs')   # git-ignored
    os.makedirs(logdir, exist_ok=True)
    t_all = time.time()
    for i, step in enumerate(STEPS, 1):
        if i < start:
            continue
        t0 = time.time()
        print(f'[{i:2d}/{len(STEPS)}] {step} ...', end=' ', flush=True)
        r = subprocess.run([sys.executable, os.path.join(ROOT, step)], cwd=ROOT, env=env,
                           capture_output=True, text=True, encoding='utf-8', errors='replace')
        log = os.path.join(logdir, os.path.splitext(os.path.basename(step))[0] + '.log')
        with open(log, 'w', encoding='utf-8') as f:
            f.write(r.stdout)
            f.write(r.stderr)
        if r.returncode != 0:
            print(f'FAILED ({time.time() - t0:.0f} s)')
            print(r.stderr[-3000:])
            print(f'full log: {log}')
            return r.returncode
        print(f'ok ({time.time() - t0:.0f} s)')
    print(f'\nAll steps finished in {(time.time() - t_all) / 60:.1f} min.')
    print('Intermediates: data/intermediate   Figures: figures/output')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
