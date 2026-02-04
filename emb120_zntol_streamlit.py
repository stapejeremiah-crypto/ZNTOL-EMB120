# emb120_zntol_streamlit.py
# Final version: Uses your test data as source of truth + structural cap

import streamlit as st
import numpy as np
from scipy.interpolate import interp1d

# Constants
STRUCTURAL_MTOW = 26433
MIN_MSA = 8000
STRUCTURAL_MSA_THRESHOLD = 15000  # effective MSA ≤ this → structural limit

# Your test data as the authoritative source
# At low MSA (10k ft): structural max
# At high MSA (20k ft): the values you provided
high_msa_points = {
    -10: 23692,
    -5:  23000,
    0:   22286,
    5:   21500,
    10:  20714,
    15:  20000,
    20:  19267
}

# Sort for interpolation
high_isas = sorted(high_msa_points.keys())
high_weights = [high_msa_points[i] for i in high_isas]

# Linear interpolator for high MSA
high_interp = interp1d(high_isas, high_weights, kind='linear', fill_value="extrapolate")

# ────────────────────────────────────────────────
# Calculation
# ────────────────────────────────────────────────
@st.cache_data
def calculate_zntol(isa_dev: float, msa: float, fuel_burn: float) -> dict:
    if isa_dev < -20 or isa_dev > 30:
        return {"error": "ISA deviation out of range"}
    if msa < MIN_MSA or msa > 26000:
        return {"error": "MSA out of range"}

    effective_msa = msa - 1000 if msa > 6000 else msa

    if effective_msa <= STRUCTURAL_MSA_THRESHOLD:
        w_obstacle = STRUCTURAL_MTOW
        source = "structural limit (low effective MSA)"
    else:
        # Use your high-MSA test data interpolation
        w_obstacle = float(high_interp(isa_dev))
        source = "linear interpolation from your test points"

    w_obstacle = min(max(w_obstacle, 4600), STRUCTURAL_MTOW)

    zntol = min(w_obstacle + fuel_burn, STRUCTURAL_MTOW)

    return {
        "w_obstacle": round(w_obstacle),
        "zntol": round(zntol),
        "source": source,
        "effective_msa": round(effective_msa)
    }


# ────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────
st.set_page_config(page_title="EMB-120 ZNTOL Calculator", layout="centered")

st.title("EMB-120 Zero-Net Takeoff Limit (ZNTOL) Calculator")
st.caption("Based solely on your test data + structural cap")

col1, col2 = st.columns(2)
isa_dev = col1.number_input("ISA deviation (°C)", -30.0, 40.0, 0.0, 0.5)
msa_ft = col2.number_input("MSA (ft)", MIN_MSA, 30000, 15000, 100)

fuel_burn = st.number_input("Fuel burned to obstacle (lbs)", 0.0, 10000.0, 2000.0, 100.0)

if st.button("Calculate ZNTOL", type="primary"):
    res = calculate_zntol(isa_dev, msa_ft, fuel_burn)
    st.success(f"**ZNTOL = {res['zntol']:,} lbs**")
    with st.expander("Details"):
        st.metric("Weight at obstacle", f"{res['w_obstacle']:,} lbs")
        st.metric("Effective MSA used", f"{res['effective_msa']:,} ft")
        st.metric("Source", res['source'])

st.caption("For reference only • Verify with official AFM")
