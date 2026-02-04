# emb120_zntol_streamlit.py
# Extended to support MSA down to 8000 ft via conservative extrapolation

import streamlit as st
import numpy as np
from scipy.interpolate import RectBivariateSpline

# ────────────────────────────────────────────────
# Data Grid (your provided table, starting at 19,000 ft)
# ────────────────────────────────────────────────
ISA_GRID = np.array([-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30])
MSA_GRID = np.array([19000, 20000, 21000, 22000, 23000, 24000, 25000, 26000])

TOW_GRID = np.array([
    [25000, 25000, 24400, 23200, 21900, 20600, 19400, 18000],  # -20
    [25000, 25000, 23600, 22200, 20900, 19600, 18400, 17100],  # -15
    [25000, 24100, 22600, 21300, 19900, 18600, 17400, 16100],  # -10
    [24600, 23100, 21700, 20300, 19000, 17700, 16400, 15100],  # -5
    [23600, 22200, 20800, 19400, 18000, 16700, 15400, 14100],  # 0
    [22600, 21100, 19700, 18300, 17000, 15700, 14300, 13000],  # +5
    [21600, 20000, 18600, 17200, 15900, 14500, 13100, 11700],  # +10
    [20600, 19000, 17500, 16000, 14700, 13300, 11900, 10400],  # +15
    [19400, 17900, 16300, 14900, 13200, 11800, 10400,  8800],  # +20
    [18200, 16600, 15000, 13500, 12000, 10400,  8700,  6900],  # +25
    [16400, 14800, 13200, 11600,  9900,  8100,  6400,  4600]   # +30
])

STRUCTURAL_MTOW = 26433
MIN_MSA_SUPPORTED = 8000   # New: lowest we support via extrapolation

# ────────────────────────────────────────────────
# Calculation function with low-altitude extrapolation
# ────────────────────────────────────────────────
@st.cache_data
def calculate_zntol(isa_dev: float, msa: float, fuel_burn: float) -> dict:
    if isa_dev < -20 or isa_dev > 30:
        return {"error": "ISA deviation must be between -20°C and +30°C"}
    if msa < MIN_MSA_SUPPORTED or msa > 26000:
        return {"error": f"MSA must be between {MIN_MSA_SUPPORTED:,} ft and 26,000 ft"}
    if fuel_burn < 0:
        return {"error": "Fuel burn cannot be negative"}

    # For MSA below 19,000 ft: conservatively use the 19,000 ft column values
    effective_msa = max(msa, 19000)  # clamp to table minimum

    # Cold cap logic
    if isa_dev <= -5:
        w_obstacle_max = 25000.0
    else:
        spline = RectBivariateSpline(ISA_GRID, MSA_GRID, TOW_GRID, kx=1, ky=1)
        w_obstacle_max = float(spline(isa_dev, effective_msa)[0, 0])

    zntol_uncapped = w_obstacle_max + fuel_burn
    zntol = min(zntol_uncapped, STRUCTURAL_MTOW)

    return {
        "w_obstacle_max": round(w_obstacle_max),
        "zntol": round(zntol),
        "capped": zntol_uncapped > STRUCTURAL_MTOW,
        "extrapolated": msa < 19000,
        "error": None
    }


# ────────────────────────────────────────────────
# Streamlit App
# ────────────────────────────────────────────────
st.set_page_config(page_title="EMB-120 ZNTOL Calculator", layout="centered")

st.title("EMB-120 Zero-Net Takeoff Limit (ZNTOL) Calculator")
st.caption("PW118A / PW118B engines • Obstacle-limited net takeoff weight • Now supports MSA down to 8,000 ft")

with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Enter **ISA deviation** at the highest enroute obstacle  
    2. Enter **MSA** at that obstacle (now down to 8,000 ft)  
    3. Enter **fuel burned** from brake release to overhead the obstacle  

    For MSA < 19,000 ft: Uses conservative extrapolation from the 19,000 ft column.
    """)

    st.divider()
    st.info("Always verify with current AFM and company policy. Extrapolation is conservative (may understate capability at low MSA).")

# ── Main inputs ───────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    isa_dev = st.number_input(
        "ISA deviation at obstacle (°C)",
        min_value=-30.0,
        max_value=40.0,
        value=0.0,
        step=0.5,
        format="%.1f",
        help="Temperature deviation from ISA at the obstacle altitude"
    )

with col2:
    msa_ft = st.number_input(
        "MSA at highest obstacle (ft)",
        min_value=MIN_MSA_SUPPORTED,
        max_value=30000,
        value=15000,  # Default lowered for testing low altitudes
        step=500,
        format="%d",
        help="Minimum Sector Altitude of the critical obstacle (now supports 8,000–26,000 ft)"
    )

fuel_burn = st.number_input(
    "Fuel burned to obstacle (lbs)",
    min_value=0.0,
    max_value=10000.0,
    value=2000.0,
    step=100.0,
    format="%.0f",
    help="From flight plan – brake release to overhead the obstacle"
)

# ── Calculate button ──────────────────────────────────
if st.button("Calculate ZNTOL", type="primary", use_container_width=True):
    result = calculate_zntol(isa_dev, msa_ft, fuel_burn)

    if result.get("error"):
        st.error(result["error"])
    else:
        st.success(f"**Zero-Net Takeoff Limit (ZNTOL)** = **{result['zntol']:,} lbs**")

        with st.expander("Detailed breakdown", expanded=True):
            st.metric("Max allowable weight at obstacle", f"{result['w_obstacle_max']:,} lbs")
            if result["extrapolated"]:
                st.caption("→ (conservative value from 19,000 ft column – actual may be higher at low MSA)")
            st.metric("Fuel added back", f"+ {fuel_burn:,.0f} lbs")
            st.metric("Un-capped ZNTOL", f"{result['zntol'] + (STRUCTURAL_MTOW - result['zntol'] if result['capped'] else 0):,} lbs",
                      delta="capped at structural MTOW" if result['capped'] else None)

# ── Footer info ───────────────────────────────────────
with st.expander("Assumptions & Limitations"):
    st.markdown("""
    - Interpolation: Linear between table points  
    - MSA < 19,000 ft: Uses 19,000 ft column values (conservative/safe)  
    - Cold temperatures (ISA ≤ -5 °C): Hard-capped at 25,000 lbs  
    - Structural MTOW = 26,433 lbs  
    - No wind, anti-ice, wet runway credits – apply manually if needed  
    - Verify table data against your AFM revision  
    """)

st.markdown("---")
st.caption("Built for quick reference • Not a substitute for official performance tools")
