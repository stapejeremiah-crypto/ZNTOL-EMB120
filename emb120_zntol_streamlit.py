# emb120_zntol_streamlit.py
# Run with: streamlit run emb120_zntol_streamlit.py

import streamlit as st
import numpy as np
from scipy.interpolate import RectBivariateSpline

# ────────────────────────────────────────────────
# Data (from your AFM table - corrected values)
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

# ────────────────────────────────────────────────
# Calculation function
# ────────────────────────────────────────────────
@st.cache_data
def calculate_zntol(isa_dev: float, msa: float, fuel_burn: float) -> dict:
    if isa_dev < -20 or isa_dev > 30:
        return {"error": "ISA deviation must be between -20°C and +30°C"}
    if msa < 19000 or msa > 26000:
        return {"error": "MSA must be between 19,000 ft and 26,000 ft"}
    if fuel_burn < 0:
        return {"error": "Fuel burn cannot be negative"}

    # Cold cap logic
    if isa_dev <= -5:
        w_obstacle_max = 25000.0
    else:
        spline = RectBivariateSpline(ISA_GRID, MSA_GRID, TOW_GRID, kx=1, ky=1)
        w_obstacle_max = float(spline(isa_dev, msa)[0, 0])

    zntol_uncapped = w_obstacle_max + fuel_burn
    zntol = min(zntol_uncapped, STRUCTURAL_MTOW)

    return {
        "w_obstacle_max": round(w_obstacle_max),
        "zntol": round(zntol),
        "capped": zntol_uncapped > STRUCTURAL_MTOW,
        "error": None
    }


# ────────────────────────────────────────────────
# Streamlit App
# ────────────────────────────────────────────────
st.set_page_config(page_title="EMB-120 ZNTOL Calculator", layout="centered")

st.title("EMB-120 Zero-Net Takeoff Limit (ZNTOL) Calculator")
st.caption("PW118A / PW118B engines • Obstacle-limited net takeoff weight")

with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Enter the **ISA deviation** at the highest enroute obstacle  
    2. Enter the **MSA** at that obstacle  
    3. Enter **fuel burned** from brake release to overhead the obstacle  
       (from your flight plan computer)  

    Result = maximum allowable takeoff weight so the **net** flight path  
    just meets zero extra margin over the obstacle.
    """)

    st.divider()

    st.info("Always cross-check with current AFM revision and company policy.")

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
        min_value=10000,
        max_value=30000,
        value=21000,
        step=500,
        format="%d",
        help="Minimum Sector Altitude of the critical obstacle"
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
            st.metric("Fuel added back", f"+ {fuel_burn:,.0f} lbs")
            st.metric("Un-capped ZNTOL", f"{result['zntol'] + (STRUCTURAL_MTOW - result['zntol'] if result['capped'] else 0):,} lbs",
                      delta="capped at structural MTOW" if result['capped'] else None)

# ── Footer info ───────────────────────────────────────
with st.expander("Assumptions & Limitations"):
    st.markdown("""
    - Interpolation uses linear method between table points  
    - Cold temperatures (ISA ≤ -5 °C) are hard-capped at 25,000 lbs per table  
    - Structural MTOW = 26,433 lbs  
    - No credit for wind, anti-ice penalty, wet runway, etc. – add manually if required  
    - Table data based on provided values – verify against your current AFM  
    """)

st.markdown("---")
st.caption("Built for quick reference • Not a substitute for official performance tools")