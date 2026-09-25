"""Ref-Elec coupling helper module.

5 Refrigeration utilities (Ref1-5) have prices and CO2 intensities that
are linear functions of electricity (P_elec, CI_elec).

Formulas (extracted from the authors' utility-electrification workbook, sheet 'electrification parameters'; not distributed):

  P_Ref_k  [$/kWh]        = (G_k + F_k * P_elec[$/kWh]) * 1e6 / 31,540,000
  CI_Ref_k [gCO2eq/kWh]   = F_k * CI_elec[gCO2eq/kWh] * 0.19180

Coefficients (rows 9-13):
  Ref1: F = 6.2545,  G = 0.2065
  Ref2: F = 16.744,  G = 0.35
  Ref3: F = 14.869,  G = 0.4184
  Ref4: F = 20.711,  G = 0.433
  Ref5: F = 54.793,  G = 0.935

Verification (baseline P_elec=0.08, CI_elec=369):
  Ref1: P = 0.02241 $/kWh, CI = 442.8 gCO2/kWh  ✓
  Ref2: P = 0.05357 $/kWh, CI = 1184.8 gCO2/kWh ✓
  Ref3: P = 0.05098 $/kWh, CI = 1052.6 gCO2/kWh ✓
  Ref4: P = 0.06626 $/kWh, CI = 1465.9 gCO2/kWh ✓
  Ref5: P = 0.16859 $/kWh, CI = 3878.3 gCO2/kWh ✓
"""

# ───────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────

# Linear coefficients for Ref1-5
F_REF = {1: 6.2545, 2: 16.744, 3: 14.869, 4: 20.711, 5: 54.793}   # kWh-elec equivalent
G_REF = {1: 0.2065, 2: 0.35,   3: 0.4184, 4: 0.433,  5: 0.935}    # capital component

# Multipliers
M_COST = 1.0e6 / 31_540_000  # = 0.031706 (s/year unit conversion for cost)
M_CI   = 0.19180             # empirically derived from 5 Ref data points

# Baseline reference values (used in 'All pathways' sheet)
P_ELEC_BASE  = 0.08    # $/kWh
CI_ELEC_BASE = 369.0   # gCO2eq/kWh (= 0.369 kgCO2/kWh)


# ───────────────────────────────────────────────────────────────
# Core functions
# ───────────────────────────────────────────────────────────────

def ref_price(k: int, P_elec: float) -> float:
    """Return Ref{k} price [$/kWh] at the given electricity price.

    Args:
        k: refrigerator index (1-5)
        P_elec: electricity price [$/kWh]
    """
    return (G_REF[k] + F_REF[k] * P_elec) * M_COST


def ref_ci(k: int, CI_elec: float) -> float:
    """Return Ref{k} CO2 intensity [gCO2eq/kWh] at the given electricity CI.

    Args:
        k: refrigerator index (1-5)
        CI_elec: electricity CO2 intensity [gCO2eq/kWh]
    """
    return F_REF[k] * CI_elec * M_CI


def ref_ci_kg(k: int, CI_elec_kg: float) -> float:
    """Same as ref_ci but with CI_elec in kgCO2eq/kWh (returns kg too)."""
    return F_REF[k] * CI_elec_kg * M_CI


# ───────────────────────────────────────────────────────────────
# Vectorized convenience (numpy / pandas)
# ───────────────────────────────────────────────────────────────

def all_ref_prices(P_elec):
    """Return dict {1..5: price} for given P_elec (scalar or array)."""
    return {k: ref_price(k, P_elec) for k in range(1, 6)}


def all_ref_cis(CI_elec):
    """Return dict {1..5: CI} for given CI_elec (scalar or array)."""
    return {k: ref_ci(k, CI_elec) for k in range(1, 6)}


# ───────────────────────────────────────────────────────────────
# Rescaling helpers (apply to 'All pathways' rows)
# ───────────────────────────────────────────────────────────────

def rescale_ref_cost(df, P_elec_new,
                    cost_cols=None,
                    P_elec_base=P_ELEC_BASE):
    """Multiply Ref cost columns by ratio P_Ref(P_new) / P_Ref(P_base).

    The All pathways sheet's Ref columns are baseline costs at P_elec_base.
    Linear scaling preserves per-pathway resource consumption.

    Args:
        df: pandas DataFrame with Ref1-5 cost columns
        P_elec_new: new electricity price [$/kWh]
        cost_cols: list of 5 column names (default: Refrigerator 1..5 ($M/y))
        P_elec_base: baseline P_elec [$/kWh]

    Returns:
        DataFrame copy with Ref cost columns scaled.
    """
    if cost_cols is None:
        cost_cols = [f'Refrigerator {k} ($M/y)' for k in range(1, 6)]
    out = df.copy()
    for k, col in enumerate(cost_cols, start=1):
        ratio = ref_price(k, P_elec_new) / ref_price(k, P_elec_base)
        out[col] = df[col] * ratio
    return out


def rescale_ref_ci(df, CI_elec_new,
                   ci_cols=None,
                   CI_elec_base=CI_ELEC_BASE):
    """Multiply Ref CO2 columns by ratio CI_Ref(CI_new) / CI_Ref(CI_base).

    Since b_k≈0, ratio simplifies to CI_elec_new / CI_elec_base (same for all k).
    But we keep per-k structure for clarity (and future b_k≠0 extensions).

    Args:
        df: pandas DataFrame with Ref1-5 CO2 columns
        CI_elec_new: new electricity CI [gCO2eq/kWh]
        ci_cols: list of 5 column names (default: CO2 Refrigerator 1..5)
        CI_elec_base: baseline CI_elec [gCO2eq/kWh]

    Returns:
        DataFrame copy with Ref CO2 columns scaled.
    """
    if ci_cols is None:
        ci_cols = [f'CO2 Refrigerator {k} (kg/year)' for k in range(1, 6)]
    out = df.copy()
    for k, col in enumerate(ci_cols, start=1):
        ratio = ref_ci(k, CI_elec_new) / ref_ci(k, CI_elec_base)
        out[col] = df[col] * ratio
    return out


def rescale_ref_both(df, P_elec_new, CI_elec_new, **kwargs):
    """Apply both cost and CI rescaling."""
    df1 = rescale_ref_cost(df, P_elec_new,
                          cost_cols=kwargs.get('cost_cols'),
                          P_elec_base=kwargs.get('P_elec_base', P_ELEC_BASE))
    df2 = rescale_ref_ci(df1, CI_elec_new,
                        ci_cols=kwargs.get('ci_cols'),
                        CI_elec_base=kwargs.get('CI_elec_base', CI_ELEC_BASE))
    return df2


# ───────────────────────────────────────────────────────────────
# Self-test
# ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print('=' * 60)
    print('Ref-Elec coupling self-test')
    print('=' * 60)
    print(f'\nBaseline: P_elec = {P_ELEC_BASE} $/kWh, CI_elec = {CI_ELEC_BASE} gCO2/kWh\n')

    # Expected values (from the utility-electrification workbook, rows 32-36)
    expected_P  = [0.022412, 0.053567, 0.050981, 0.066261, 0.16863]
    expected_CI = [442.8, 1184.8, 1052.6, 1465.9, 3878.3]

    print(f'{"k":<4} {"F":>9} {"G":>9} {"P_calc":>10} {"P_exp":>10} {"CI_calc":>10} {"CI_exp":>10}')
    print('-' * 70)
    for k in range(1, 6):
        P_calc  = ref_price(k, P_ELEC_BASE)
        CI_calc = ref_ci(k, CI_ELEC_BASE)
        print(f'{k:<4} {F_REF[k]:>9.4f} {G_REF[k]:>9.4f} '
              f'{P_calc:>10.5f} {expected_P[k-1]:>10.5f} '
              f'{CI_calc:>10.2f} {expected_CI[k-1]:>10.2f}')

    print('\n--- Test: renewable PPA scenario (P=0.04, CI=30) ---')
    P_ren, CI_ren = 0.04, 30.0
    for k in range(1, 6):
        print(f'  Ref{k}: P = {ref_price(k, P_ren):.5f} $/kWh  '
              f'(× baseline = {ref_price(k, P_ren)/ref_price(k, P_ELEC_BASE):.3f})  |  '
              f'CI = {ref_ci(k, CI_ren):.2f} gCO2/kWh  '
              f'(× baseline = {ref_ci(k, CI_ren)/ref_ci(k, CI_ELEC_BASE):.3f})')

    print('\n--- Test: 4x P_elec (matches ELEC_sensitivity sheet which has 0.30 $/kWh) ---')
    P_4x = 0.30   # ELEC_sensitivity uses 0.30 $/kWh (= 3.75x baseline 0.08)
    print(f'P_4x ratio: P_4x / P_base = {P_4x / P_ELEC_BASE:.3f}')
    for k in range(1, 6):
        ratio = ref_price(k, P_4x) / ref_price(k, P_ELEC_BASE)
        print(f'  Ref{k}: cost ratio = {ratio:.3f}')
    # Expected from inspect: Ref1 ratio = 2.947
    # So at P_elec = 0.30: cost ratio of Ref1 should be ~2.95
    # (G + F*0.30) / (G + F*0.08) = (0.2065 + 1.876) / (0.2065 + 0.5004) = 2.083/0.707 = 2.947 ✓
    print('\n  Expected Ref1 ratio from ELEC_sensitivity sheet: 2.947')
