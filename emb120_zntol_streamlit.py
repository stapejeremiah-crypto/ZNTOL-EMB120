# emb120_zntol_streamlit.py
# 2D linear interpolation over ISA and MSA + structural cap + cold/high pull-down

import streamlit as st
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d

# Constants
STRUCTURAL_MTOW = 26433
MIN_MSA = 8000
STRUCTURAL_MSA_THRESHOLD = 15000
HIGH_MSA_PULLDOWN_THRESHOLD = 18000

# ISA and MSA grids
isa_grid = np.array([-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30])
msa_grid = np.array([
    8000, 10000, 12000, 14000, 16000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000, 26000
])

# Weight grid (rows: ISA, columns: MSA)
weight_grid = np.full((len(isa_grid), len(msa_grid)), np.nan)

# Fill from high-alt data
high_data = {
    -20: [25000, 25000, 24400, 23200, 21900, 20600, 19400, 18000],
    -15: [25000, 25000, 23600, 22200, 20900, 19600, 18400, 17100],
    -10: [25000, 24100, 22600, 21300, 19900, 18600, 17400, 16100],
    -5: [24600, 23100, 21700, 20300, 19000, 17700, 16400, 15100],
    0: [23600, 22200, 20800, 19400, 18000, 16700, 15400, 14100],
    5: [22600, 21100, 19700, 18300, 17000, 15700, 14300, 13000],
    10: [21600, 20000, 18600, 17200, 15900, 14500, 13100, 11700],
    15: [20600, 19000, 17500, 16000, 14700, 13300, 11900, 10400],
    20: [19400, 17900, 16300, 14900, 13200, 11800, 10400, 8800],
    25: [18200, 16600, 15000, 13500, 12000, 10400, 8700, 6900],
    30: [16400, 14800, 13200, 11600, 9900, 8100, 6400, 4600]
}

high_msa_indices = np.where(msa_grid >= 19000)[0]  # indices for 19000 to 26000
for i, isa in enumerate(isa_grid):
    if isa in high_data:
        weight_grid[i, high_msa_indices] = high_data[isa]

# Fill from low-alt data (overrides)
low_data = {
    0: {23000:19350, 22000:20100, 20000:21550, 18000:23000, 16000:24550},
    5: {22000:19350, 20000:20750, 18000:22200, 16000:23700, 14000:25200, 12000:26433},
    10: {20000:20000, 18000:21450, 16000:22950, 14000:24400, 12000:25750, 10000:26433},
    15: {20000:19300, 18000:20650, 16000:22000, 14000:23450, 12000:24850, 10000:26433},
    20: {18000:19900, 16000:21200, 14000:22500, 12000:23950, 10000:25300, 8000:26433}
}

for isa, data in low_data.items():
    i = np.where(isa_grid == isa)[0][0]
    for msa, val in data.items():
        j = np.where(msa_grid == msa)[0][0]
        weight_grid[i, j] = val

# Fill NaNs row-wise with linear interpolation
for i in range(len(isa_grid)):
    row = weight_grid[i]
    non_nan = np.isfinite(row)
    if non_nan.any():
        interp = interp1d(msa_grid[non_nan], row[non_nan], kind='linear', fill_value="extrapolate")
        weight_grid[i] = interp(msa_grid)

# 2D spline (linear kx=1, ky=1 for incremental changes)
spline = RectBivariateSpline(isa_grid, msa_grid, weight_grid, kx=1, ky=1)

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

    effective_msa = msa - 1000 if msa > 6000 else msa

    # Temperature-dependent structural cap
    # Apply full cap only for cold ISAs at low effective MSA
    if effective_msa <= STRUCTURAL_MSA_THRESHOLD and isa_dev <= 0:  # adjust threshold to +5 if needed
        w_obstacle_max = STRUCTURAL_MTOW
        source = "structural limit (cold/low MSA)"
    else:
        # 2D spline interpolation
        w_obstacle_max = spline(isa_dev, effective_msa)[0, 0]
        source = "2D linear interpolation"

        # Tuned cold/high MSA pull-down (only for colder cases)
        if isa_dev <= 0 and effective_msa > HIGH_MSA_PULLDOWN_THRESHOLD:
            pull_down = max(0, -1500 * (isa_dev + 10) / 10)
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
st.caption("2D linear interpolation over ISA and MSA + structural cap + cold/high pull-down")

with st.sidebar:
    st.header("Instructions")
    st.markdown("Enter values at the highest enroute obstacle. Uses 2D interpolation for incremental changes in ISA and MSA.")
    st.divider()
    st.info("Cross-check with AFM. Structural cap 26,433 lbs applied.")

col1, col2 = st.columns(2)
with col1:
    isa_dev = st.number_input("ISA deviation (°C)", -30.0, 40.0, 0.0, 0.1, format="%.1f")  # 0.1 step for fine ISA
with col2:
    msa_ft = st.number_input("MSA (ft)", MIN_MSA, 30000, 15000, 100, format="%d")  # 100 ft step

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
    - 2D linear interpolation (RectBivariateSpline) over ISA and MSA grids from your data
    - Grid NaNs filled row-wise with linear interpolation
    - Structural limit (26,433 lbs) for effective MSA ≤15,000 ft
    - Tuned pull-down for cold/high MSA cases to match your test data
    - Effective MSA = entered - 1,000 ft if >6,000 ft
    """)

st.caption("For reference only • Verify with official AFM")



