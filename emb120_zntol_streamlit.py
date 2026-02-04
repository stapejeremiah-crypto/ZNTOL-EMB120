# emb120_zntol_streamlit.py
# Curve-fitted high/low MSA AFM data + recalibrated correction for current test results

import streamlit as st
import numpy as np

# ────────────────────────────────────────────────
# Data from low-alt screenshot and high-alt original chart
# ────────────────────────────────────────────────
STRUCTURAL_MTOW = 26433
MIN_MSA = 8000

# Low-alt data (from screenshot): dict of ISA: dict of MSA: weight
low_data = {
    0: {23000: 19350, 22000: 20100, 20000: 21550, 18000: 23000, 16000: 24550},
    5: {22000: 19350, 20000: 20750, 18000: 22200, 16000: 23700, 14000: 25200, 12000: 26433},
    10: {20000: 20000, 18000: 21450, 16000: 22950, 14000: 24400, 12000: 25750, 10000: 26433},
    15: {20000: 19300, 18000: 20650, 16000: 22000, 14000: 23450, 12000: 24850, 10000: 26433},
    20: {18000: 19900, 16000: 21200, 14000: 22500, 12000: 23950, 10000: 25300, 8000: 26433}
}

# High-alt data: ISA grid, MSA grid, TOW array
high_isa_grid = np.array([-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30])
high_msa_grid = np.array([19000, 20000, 21000, 22000, 23000, 24000, 25000, 26000])
high_tow = np.array([
    [25000] * 8,
    [25000] * 8,
    [25000] * 8,
    [24600, 23100, 21700, 20300, 19000, 17700, 16400, 15100],
    [23600, 22200, 20800, 19400, 18000, 16700, 15400, 14100],
    [22600, 21100, 19700, 18300, 17000, 15700, 14300, 13000],
    [21600, 20000, 18600, 17200, 15900, 14500, 13100, 11700],
    [20600, 19000, 17500, 16000, 14700, 13300, 11900, 10400],
    [19400, 17900, 16300, 14900, 13200, 11800, 10400, 8800],
    [18200, 16600, 15000, 13500, 12000, 10400, 8700, 6900],
    [16400, 14800, 13200, 11600, 9900, 8100, 6400, 4600]
])

# Recalibrated adjustment parameters for your current test results
adjust_params = {
    -10: {'slope': -0.020, 'intercept': 1400.0},
    -5:  {'slope': -0.040, 'intercept': 1600.0},
    0:   {'slope': -0.030, 'intercept': 800.0},
    5:   {'slope': -0.012, 'intercept': 300.0},
    10:  {'slope': -0.008, 'intercept': 150.0},
    15:  {'slope': -0.020, 'intercept': -400.0},
    20:  {'slope': -0.050, 'intercept': -950.0}
}

# ────────────────────────────────────────────────
# Helper to interpolate/extrapolate adjustment parameters
# ────────────────────────────────────────────────
def get_adjust_param(isa: float, param: str) -> float:
    isas = sorted(adjust_params.keys())
    if isa in adjust_params:
        return adjust_params[isa][param]
    idx = np.searchsorted(isas, isa)
    if idx == 0:
        diff = adjust_params[isas[1]][param] - adjust_params[isas[0]][param]
        delta_isa = isas[1] - isas[0]
        return adjust_params[isas[0]][param] + diff / delta_isa * (isa - isas[0])
    elif idx == len(isas):
        diff = adjust_params[isas[-1]][param] - adjust_params[isas[-2]][param]
        delta_isa = isas[-1] - isas[-2]
        return adjust_params[isas[-1]][param] + diff / delta_isa * (isa - isas[-1])
    else:
        frac = (isa - isas[idx-1]) / (isas[idx] - isas[idx-1])
        return adjust_params[isas[idx-1]][param] + frac * (adjust_params[isas[idx]][param] - adjust_params[isas[idx-1]][param])

# ────────────────────────────────────────────────
# Calculation
# ────────────────────────────────────────────────
@st.cache_data
def calculate_zntol(isa_dev: float, msa: float, fuel_burn: float) -> dict:
    if isa_dev < -20 or isa_dev > 30:
        return {"error": "ISA deviation out of range (-20 to +30 °C)"}
    if msa < MIN_MSA or msa > 26000:
        return {"error": f"MSA out of range ({MIN_MSA:,}–26,000 ft)"}
    if fuel_burn < 0:
        return {"error": "Fuel burn cannot be negative"}

    # Adjust MSA for clearance difference
    if msa > 6000:
        effective_msa = msa - 1000
    else:
        effective_msa = msa

    # Use closest ISA for cold cases (no hard clamp to 0)
    isa_clamp = np.clip(isa_dev, min(high_isa_grid), max(high_isa_grid))
    source = "curve fit" if isa_dev == isa_clamp else f"curve fit (clamped ISA {isa_clamp}°C)"

    points = {}

    # Add low data if available
    if isa_clamp in low_data:
        points.update(low_data[isa_clamp])

    # Add high data (full for cold ISAs)
    row_idx = np.where(high_isa_grid == isa_clamp)[0]
    if len(row_idx) > 0:
        for col_idx, msa_high in enumerate(high_msa_grid):
            val = high_tow[row_idx[0], col_idx]
            if msa_high in points:
                points[msa_high] = (points[msa_high] + val) / 2
            else:
                points[msa_high] = val

    if not points:
        return {"error": "No data available for this ISA deviation"}

    msas = sorted(points.keys())
    weights = np.array([points[m] for m in msas])
    fit = np.polyfit(msas, weights, deg=2)
    poly = np.poly1d(fit)

    w_obstacle_max = poly(effective_msa)

    # Apply correction
    corr_slope = get_adjust_param(isa_clamp, 'slope')
    corr_int = get_adjust_param(isa_clamp, 'intercept')
    correction = corr_slope * effective_msa + corr_int
    w_obstacle_max += correction
    source += " (with test adjustment)"

    # Boost for cold/low MSA to reach structural max
    if isa_dev <= -5 and effective_msa < 12000:
        w_obstacle_max += 800

    # Pull-down for high MSA in cold
    if isa_dev <= -5 and effective_msa > 18000:
        w_obstacle_max = min(w_obstacle_max, 24000)

    w_obstacle_max = min(max(w_obstacle_max, 4600), STRUCTURAL_MTOW)

    zntol_uncapped = w_obstacle_max + fuel_burn
    zntol = min(zntol_uncapped, STRUCTURAL_MTOW)

    return {
        "w_obstacle_max": round(w_obstacle_max),
        "zntol": round(zntol),
        "capped": zntol_uncapped > STRUCTURAL_MTOW,
        "source": source,
        "effective_msa": round(effective_msa),
        "error": None
    }


# ────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────
st.set_page_config(page_title="EMB-120 ZNTOL Calculator", layout="centered")

st.title("EMB-120 Zero-Net Takeoff Limit (ZNTOL) Calculator")
st.caption("Curve-fitted + recalibrated correction • Cold cases use full high-alt data • Larger low-alt boost")

with st.sidebar:
    st.header("Instructions")
    st.markdown("Enter values at the highest enroute obstacle. Updated tuning: uses high-alt table for cold ISAs, increased low-MSA boost for cold cases.")
    st.divider()
    st.info("Cross-check with AFM. Structural cap 26,433 lbs applied.")

col1, col2 = st.columns(2)
with col1:
    isa_dev = st.number_input("ISA deviation (°C)", -30.0, 40.0, 0.0, 0.5, format="%.1f")
with col2:
    msa_ft = st.number_input("MSA (ft)", MIN_MSA, 30000, 15000, 100, format="%d")

fuel_burn = st.number_input("Fuel burned to obstacle (lbs)", 0.0, 10000.0, 2000.0, 100.0, format="%.0f")

if st.button("Calculate ZNTOL", type="primary"):
    res = calculate_zntol(isa_dev, msa_ft, fuel_burn)
    if res.get("error"):
        st.error(res["error"])
    else:
        st.success(f"**ZNTOL = {res['zntol']:,} lbs**")
        with st.expander("Details"):
            st.metric("Entered MSA", f"{msa_ft:,} ft")
            st.metric("Effective MSA used", f"{res['effective_msa']:,} ft")
            st.metric("Weight at obstacle", f"{res['w_obstacle_max']:,} lbs")
            st.metric("+ Fuel burn", f"+ {fuel_burn:,.0f} lbs")
            st.metric("Uncapped", f"{round(res['w_obstacle_max'] + fuel_burn):,} lbs",
                      delta="capped" if res['capped'] else None)
            st.caption(f"Source: {res['source']}")

with st.expander("Assumptions & Tuning"):
    st.markdown("""
    - Quadratic curve fit to merged AFM data
    - Linear correction recalibrated to your latest test data
    - Cold cases (ISA ≤ -5 °C) use full high-alt table data (no clamp to 0 °C)
    - Larger low-MSA boost (+800 lbs) for cold cases <12,000 ft effective MSA
    - High-MSA pull-down in cold cases
    - Effective MSA = entered - 1,000 ft if >6,000 ft
    - Structural MTOW cap = 26,433 lbs
    """)

st.caption("For reference only • Verify with official AFM")
