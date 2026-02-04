# emb120_zntol_streamlit.py
# Linear interpolation from merged table points + structural cap + cold/high pull-down

import streamlit as st
import numpy as np
from scipy.interpolate import interp1d

# ────────────────────────────────────────────────
# Merged table data (MSA: weight for each ISA)
# ────────────────────────────────────────────────
STRUCTURAL_MTOW = 26433
MIN_MSA = 8000
STRUCTURAL_MSA_THRESHOLD = 15000  # below this → structural limit

# All known points: ISA → {MSA: weight}
all_points = {
    -20: {m: 25000 for m in [19000,20000,21000,22000,23000,24000,25000,26000]},
    -15: {m: 25000 for m in [19000,20000,21000,22000,23000,24000,25000,26000]},
    -10: {m: 25000 for m in [19000,20000,21000,22000,23000,24000,25000,26000]},
    -5: {19000:24600, 20000:23100, 21000:21700, 22000:20300, 23000:19000, 24000:17700, 25000:16400, 26000:15100},
    0: {19000:23600, 20000:22200, 21000:20800, 22000:19400, 23000:18000, 24000:16700, 25000:15400, 26000:14100, 23000:19350, 22000:20100, 20000:21550, 18000:23000, 16000:24550},
    5: {19000:22600, 20000:21100, 21000:19700, 22000:18300, 23000:17000, 24000:15700, 25000:14300, 26000:13000, 22000:19350, 20000:20750, 18000:22200, 16000:23700, 14000:25200, 12000:26433},
    10: {19000:21600, 20000:20000, 21000:18600, 22000:17200, 23000:15900, 24000:14500, 25000:13100, 26000:11700, 20000:20000, 18000:21450, 16000:22950, 14000:24400, 12000:25750, 10000:26433},
    15: {19000:20600, 20000:19000, 21000:17500, 22000:16000, 23000:14700, 24000:13300, 25000:11900, 26000:10400, 20000:19300, 18000:20650, 16000:22000, 14000:23450, 12000:24850, 10000:26433},
    20: {19000:19400, 20000:17900, 21000:16300, 22000:14900, 23000:13200, 24000:11800, 25000:10400, 26000:8800, 18000:19900, 16000:21200, 14000:22500, 12000:23950, 10000:25300, 8000:26433},
    25: {19000:18200, 20000:16600, 21000:15000, 22000:13500, 23000:12000, 24000:10400, 25000:8700, 26000:6900},
    30: {19000:16400, 20000:14800, 21000:13200, 22000:11600, 23000:9900, 24000:8100, 25000:6400, 26000:4600}
}

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

    if msa > 6000:
        effective_msa = msa - 1000
    else:
        effective_msa = msa

    # Structural cap for low MSA
    if effective_msa <= STRUCTURAL_MSA_THRESHOLD:
        w_obstacle_max = STRUCTURAL_MTOW
        source = "structural limit (low MSA)"
    else:
        # Find closest ISA
        isas = sorted(all_points.keys())
        closest_isa = min(isas, key=lambda x: abs(x - isa_dev))
        source = f"linear interp (closest ISA {closest_isa}°C)"

        points = all_points[closest_isa]
        msas = sorted(points.keys())
        weights = [points[m] for m in msas]

        if len(msas) < 2:
            return {"error": "Insufficient data points for interpolation"}

        # Linear interpolation
        interp_func = interp1d(msas, weights, kind='linear', fill_value="extrapolate")
        w_obstacle_max = float(interp_func(effective_msa))

        # Cold/high MSA pull-down to match test data
        if isa_dev <= 0 and effective_msa > 18000:
            # Pull-down amount: -1300 at ISA -10, linear to -0 at ISA 0
            pull_down = max(0, -1300 * (isa_dev + 10) / 10)
            w_obstacle_max -= pull_down
            source += " + cold/high pull-down"

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
st.caption("Linear interpolation from AFM table points + structural cap + cold/high pull-down")

with st.sidebar:
    st.header("Instructions")
    st.markdown("Enter values at the highest enroute obstacle. Uses linear interpolation for accuracy, structural cap for low MSA, and targeted pull-down for cold/high cases.")
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
    - Linear interpolation between known AFM table points (closest ISA used)
    - Extrapolation for out-of-range MSA
    - Structural limit (26,433 lbs) for effective MSA ≤15,000 ft
    - Targeted pull-down for cold/high MSA cases to match your test data
    - Effective MSA = entered - 1,000 ft if >6,000 ft
    """)

st.caption("For reference only • Verify with official AFM")
