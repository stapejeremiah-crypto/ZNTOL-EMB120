# emb120_zntol_streamlit.py
# Prioritizes your test points for high MSA + structural cap + linear interp

import streamlit as st
import numpy as np
from scipy.interpolate import interp1d

# Constants
STRUCTURAL_MTOW = 26433
MIN_MSA = 8000
STRUCTURAL_MSA_THRESHOLD = 15000  # below this effective MSA → structural limit
HIGH_MSA_THRESHOLD = 18000        # use test points for MSA >= this

# Your test points as a dict: ISA -> {MSA: weight}
test_points = {
    -10: {20000: 23692, 10000: 26433},
    -5:  {20000: 23000, 10000: 26433},
    0:   {20000: 22286, 10000: 26433},
    5:   {20000: 21500, 10000: 26433},
    10:  {20000: 20714, 10000: 26433},
    15:  {20000: 20000, 10000: 26433},
    20:  {20000: 19267, 10000: 25875}
}

# Fallback table for low MSA or missing data (your original high + low merged)
fallback_table = {
    -20: {m: 25000 for m in [19000,20000,21000,22000,23000,24000,25000,26000]},
    -15: {m: 25000 for m in [19000,20000,21000,22000,23000,24000,25000,26000]},
    -10: {m: 25000 for m in [19000,20000,21000,22000,23000,24000,25000,26000]},
    -5: {19000:24600, 20000:23100, 21000:21700, 22000:20300, 23000:19000, 24000:17700, 25000:16400, 26000:15100},
    0: {19000:23600, 20000:22200, 21000:20800, 22000:19400, 23000:18000, 24000:16700, 25000:15400, 26000:14100,
        23000:19350, 22000:20100, 20000:21550, 18000:23000, 16000:24550},
    5: {19000:22600, 20000:21100, 21000:19700, 22000:18300, 23000:17000, 24000:15700, 25000:14300, 26000:13000,
        22000:19350, 20000:20750, 18000:22200, 16000:23700, 14000:25200, 12000:26433},
    # ... (add more if needed, but we prioritize test_points for high MSA)
}

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

    # Structural cap for low effective MSA
    if effective_msa <= STRUCTURAL_MSA_THRESHOLD:
        w_obstacle = STRUCTURAL_MTOW
        source = "structural limit (low MSA)"
    else:
        # Prioritize test points for high MSA
        if effective_msa >= HIGH_MSA_THRESHOLD and isa_dev in test_points:
            test_data = test_points[isa_dev]
            if effective_msa in test_data:
                w_obstacle = test_data[effective_msa]
                source = "direct test point match"
            else:
                # Interpolate between test points if available
                test_msas = sorted(test_data.keys())
                test_weights = [test_data[m] for m in test_msas]
                interp = interp1d(test_msas, test_weights, kind='linear', fill_value="extrapolate")
                w_obstacle = float(interp(effective_msa))
                source = "test point interpolation"
        else:
            # Fallback to table for missing test data
            isas = sorted(fallback_table.keys())
            closest_isa = min(isas, key=lambda x: abs(x - isa_dev))
            points = fallback_table[closest_isa]
            msas = sorted(points.keys())
            weights = [points[m] for m in msas]
            interp = interp1d(msas, weights, kind='linear', fill_value="extrapolate")
            w_obstacle = float(interp(effective_msa))
            source = f"fallback table interp (ISA {closest_isa}°C)"

    w_obstacle = min(max(w_obstacle, 4600), STRUCTURAL_MTOW)

    zntol = min(w_obstacle + fuel_burn, STRUCTURAL_MTOW)

    return {
        "w_obstacle": round(w_obstacle),
        "zntol": round(zntol),
        "source": source,
        "eff_msa": round(effective_msa)
    }

# ────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────
st.title("EMB-120 ZNTOL Calculator")
st.caption("Prioritizes your manual AFM test points for high MSA + structural cap")

col1, col2 = st.columns(2)
isa_dev = col1.number_input("ISA deviation (°C)", -30.0, 40.0, 0.0, 0.5)
msa_ft = col2.number_input("MSA (ft)", MIN_MSA, 30000, 15000, 100)

fuel_burn = st.number_input("Fuel burned to obstacle (lbs)", 0.0, 10000.0, 2000.0, 100.0)

if st.button("Calculate"):
    res = calculate_zntol(isa_dev, msa_ft, fuel_burn)
    st.success(f"**ZNTOL = {res['zntol']:,} lbs**")
    with st.expander("Details"):
        st.metric("Weight at obstacle", f"{res['w_obstacle']:,} lbs")
        st.metric("Effective MSA used", f"{res['eff_msa']:,} ft")
        st.metric("Source", res['source'])
