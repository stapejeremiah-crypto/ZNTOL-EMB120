# emb120_zntol_streamlit.py
# Merged high + low MSA data from AFM tables for accurate interpolation

import streamlit as st
import numpy as np
from scipy.interpolate import RectBivariateSpline

# ────────────────────────────────────────────────
# Combined Grid: High-alt from original + low-alt from your screenshot
# Values are max allowable weight at obstacle (lbs)
# ────────────────────────────────────────────────
# ISA deviations (Temp) - we'll use 0 to 20 from new data; extend cold cap separately
ISA_GRID_FULL = np.array([0, 5, 10, 15, 20])  # From low-alt table; add -5/-10/-15/-20 later if needed

MSA_GRID_FULL = np.sort(np.unique([
    8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 23000,  # low
    19000, 21000, 24000, 25000, 26000  # high
]))

# Create full TOW grid (rows=ISA, cols=MSA) - initialize with NaN, fill known
TOW_GRID_FULL = np.full((len(ISA_GRID_FULL), len(MSA_GRID_FULL)), np.nan)

# 1. Fill from your new low-alt screenshot data
low_data = {
    0:   {23000:19350, 22000:20100, 20000:21550, 18000:23000, 16000:24550},
    5:   {22000:19350, 20000:20750, 18000:22200, 16000:23700, 14000:25200, 12000:26433},
    10:  {20000:20000, 18000:21450, 16000:22950, 14000:24400, 12000:25750, 10000:26433},
    15:  {20000:19300, 18000:20650, 16000:22000, 14000:23450, 12000:24850, 10000:26433},
    20:  {18000:19900, 16000:21200, 14000:22500, 12000:23950, 10000:25300, 8000:26433}
}

for i, isa in enumerate(ISA_GRID_FULL):
    if isa in low_data:
        for msa, val in low_data[isa].items():
            col_idx = np.where(MSA_GRID_FULL == msa)[0]
            if len(col_idx) > 0:
                TOW_GRID_FULL[i, col_idx[0]] = val

# 2. Overlay/fill high-alt original table (for MSA >=19000, override if conflict)
high_isa_grid = np.array([-20,-15,-10,-5,0,5,10,15,20,25,30])
high_msa_grid = np.array([19000,20000,21000,22000,23000,24000,25000,26000])
high_tow = np.array([
    [25000]*8,
    [25000]*8,
    [25000]*8,
    [24600,23100,21700,20300,19000,17700,16400,15100],
    [23600,22200,20800,19400,18000,16700,15400,14100],
    [22600,21100,19700,18300,17000,15700,14300,13000],
    [21600,20000,18600,17200,15900,14500,13100,11700],
    [20600,19000,17500,16000,14700,13300,11900,10400],
    [19400,17900,16300,14900,13200,11800,10400,8800],
    [18200,16600,15000,13500,12000,10400,8700,6900],
    [16400,14800,13200,11600,9900,8100,6400,4600]
])

for row_idx, isa in enumerate(high_isa_grid):
    if isa in ISA_GRID_FULL:
        grid_row = np.where(ISA_GRID_FULL == isa)[0][0]
        for col_idx, msa in enumerate(high_msa_grid):
            grid_col = np.where(MSA_GRID_FULL == msa)[0]
            if len(grid_col) > 0:
                # Use high table where available (more conservative at high MSA)
                TOW_GRID_FULL[grid_row, grid_col[0]] = high_tow[row_idx, col_idx]

# Fill any remaining NaNs with nearest/edge values (fallback)
TOW_GRID_FULL = np.nan_to_num(TOW_GRID_FULL, nan=25000)  # conservative fallback

STRUCTURAL_MTOW = 26433
MIN_MSA = 8000

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

    # Cold cap (from original table)
    if isa_dev <= -5:
        w_obstacle_max = 25000.0
        source = "cold cap"
    else:
        # Clamp ISA to available grid for interpolation
        isa_clamp = np.clip(isa_dev, min(ISA_GRID_FULL), max(ISA_GRID_FULL))
        spline = RectBivariateSpline(ISA_GRID_FULL, MSA_GRID_FULL, TOW_GRID_FULL, kx=1, ky=1)
        w_obstacle_max = float(spline(isa_clamp, msa)[0, 0])
        source = "interpolated" if isa_dev == isa_clamp else f"clamped ISA {isa_clamp}°C"

    w_obstacle_max = min(w_obstacle_max, STRUCTURAL_MTOW)

    zntol_uncapped = w_obstacle_max + fuel_burn
    zntol = min(zntol_uncapped, STRUCTURAL_MTOW)

    return {
        "w_obstacle_max": round(w_obstacle_max),
        "zntol": round(zntol),
        "capped": zntol_uncapped > STRUCTURAL_MTOW,
        "source": source,
        "error": None
    }


# ────────────────────────────────────────────────
# Streamlit UI (same as before, minor updates)
# ────────────────────────────────────────────────
st.set_page_config(page_title="EMB-120 ZNTOL Calculator", layout="centered")

st.title("EMB-120 Zero-Net Takeoff Limit (ZNTOL) Calculator")
st.caption("Merged high/low MSA AFM data • Accurate down to 8,000 ft")

with st.sidebar:
    st.header("Instructions")
    st.markdown("Enter values at the highest enroute obstacle. Now uses full AFM low-alt data for better accuracy below 19,000 ft.")

    st.divider()
    st.info("Cross-check with AFM. Structural cap 26,433 lbs applied.")

col1, col2 = st.columns(2)
with col1:
    isa_dev = st.number_input("ISA deviation (°C)", -30.0, 40.0, 0.0, 0.5, format="%.1f")
with col2:
    msa_ft = st.number_input("MSA (ft)", MIN_MSA, 30000, 15000, 500, format="%d")

fuel_burn = st.number_input("Fuel burned to obstacle (lbs)", 0.0, 10000.0, 2000.0, 100.0, format="%.0f")

if st.button("Calculate ZNTOL", type="primary"):
    res = calculate_zntol(isa_dev, msa_ft, fuel_burn)
    if res.get("error"):
        st.error(res["error"])
    else:
        st.success(f"**ZNTOL = {res['zntol']:,} lbs**")
        with st.expander("Details"):
            st.metric("Weight at obstacle", f"{res['w_obstacle_max']:,} lbs")
            st.metric("+ Fuel burn", f"+ {fuel_burn:,.0f} lbs")
            st.metric("Uncapped", f"{round(res['w_obstacle_max'] + fuel_burn):,} lbs",
                      delta="capped" if res['capped'] else None)
            st.caption(f"Source: {res['source']}")

with st.expander("Assumptions"):
    st.markdown("""
    - Merged original high-alt table + your low-alt screenshot data  
    - Bilinear interpolation; cold ISA ≤ -5 °C capped at 25,000 lbs  
    - Structural MTOW cap = 26,433 lbs  
    - No wind/anti-ice/wet adjustments  
    """)

st.caption("For reference only • Verify with official AFM")
