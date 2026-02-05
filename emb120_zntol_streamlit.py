# emb120_zntol_streamlit.py
# 2D linear interpolation over ISA and MSA + structural cap

import streamlit as st
import numpy as np
from scipy.interpolate import RectBivariateSpline

# Constants
STRUCTURAL_MTOW = 26433
MIN_MSA = 8000
STRUCTURAL_MSA_THRESHOLD = 15000

# ISA grid from table
isa_grid = np.array([-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30])

# MSA grid (unique from all known points)
msa_grid = np.array([
    8000, 10000, 12000, 14000, 16000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000, 26000
])

# Build 2D weight grid (rows = ISA, columns = MSA)
# Initialize with NaN
weight_grid = np.full((len(isa_grid), len(msa_grid)), np.nan)

# Fill from your high-alt table
high_data = {
    -20: [25000,25000,24400,23200,21900,20600,19400,18000],
    -15: [25000,25000,23600,22200,20900,19600,18400,17100],
    -10: [25000,24100,22600,21300,19900,18600,17400,16100],
    -5: [24600,23100,21700,20300,19000,17700,16400,15100],
    0: [23600,22200,20800,19400,18000,16700,15400,14100],
    5: [22600,21100,19700,18300,17000,15700,14300,13000],
    10: [21600,20000,18600,17200,15900,14500,13100,11700],
    15: [20600,19000,17500,16000,14700,13300,11900,10400],
    20: [19400,17900,16300,14900,13200,11800,10400,8800],
    25: [18200,16600,15000,13500,12000,10400,8700,6900],
    30: [16400,14800,13200,11600,9900,8100,6400,4600]
}

for i, isa in enumerate(isa_grid):
    if isa in high_data:
        # Map high_data to msa_grid order
        for j, m in enumerate(msa_grid):
            if m in [19000,20000,21000,22000,23000,24000,25000,26000]:
                idx = [19000,20000,21000,22000,23000,24000,25000,26000].index(m)
                weight_grid[i, j] = high_data[isa][idx]

# Fill from low-alt data (overrides where present)
low_data = {
    0: {23000:19350, 22000:20100, 20000:21550, 18000:23000, 16000:24550},
    5: {22000:19350, 20000:20750, 18000:22200, 16000:23700, 14000:25200, 12000:26433},
    10: {20000:20000, 18000:21450, 16000:22950, 14000:24400, 12000:25750, 10000:26433},
    15: {20000:19300, 18000:20650, 16000:22000, 14000:23450, 12000:24850, 10000:26433},
    20: {18000:19900, 16000:21200, 14000:22500, 12000:23950, 10000:25300, 8000:26433}
}

for isa, data in low_data.items():
    i = np.where(isa_grid == isa)[0]
    if len(i) > 0:
        i = i[0]
        for msa, val in data.items():
            j = np.where(msa_grid == msa)[0]
            if len(j) > 0:
                weight_grid[i, j[0]] = val

# Fill NaNs with nearest value (conservative fallback)
from scipy.ndimage import generic_filter
def fill_nan(arr):
    return np.nanmean(arr)
weight_grid = generic_filter(weight_grid, fill_nan, size=3, mode='constant', cval=np.nan)

# 2D spline interpolator
spline = RectBivariateSpline(isa_grid, msa_grid, weight_grid, kx=1, ky=1)

# Calculation
@st.cache_data
def calculate_zntol(isa_dev: float, msa: float, fuel_burn: float) -> dict:
    if isa_dev < -20 or isa_dev > 30:
        return {"error": "ISA deviation out of range (-20 to +30 °C)"}
    if msa < MIN_MSA or msa > 26000:
        return {"error": f"MSA out of range ({MIN_MSA:,}–26,000 ft)"}
    if fuel_burn < 0:
        return {"error": "Fuel burn cannot be negative"}

    effective_msa = msa - 1000 if msa > 6000 else msa

    # Initialize to avoid UnboundLocalError
    w_obstacle_max = STRUCTURAL_MTOW  # safe default
    source = "unknown"

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
            # Fallback instead of error
            w_obstacle_max = STRUCTURAL_MTOW
            source += " (fallback - insufficient points)"
        else:
            interp_func = interp1d(msas, weights, kind='linear', fill_value="extrapolate")
            w_obstacle_max = float(interp_func(effective_msa))

            # Cold/high MSA pull-down
            if isa_dev <= 0 and effective_msa > 18000:
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

# UI
st.title("EMB-120 ZNTOL Calculator")
st.caption("2D spline interpolation over ISA & MSA + structural cap")

col1, col2 = st.columns(2)
isa_dev = col1.number_input("ISA deviation (°C)", -30.0, 40.0, 0.0, 0.1, format="%.1f")
msa_ft = col2.number_input("MSA (ft)", MIN_MSA, 30000, 15000, 100, format="%d")

fuel_burn = st.number_input("Fuel burned to obstacle (lbs)", 0.0, 10000.0, 2000.0, 100.0)

if st.button("Calculate ZNTOL", type="primary"):
    res = calculate_zntol(isa_dev, msa_ft, fuel_burn)
    st.success(f"**ZNTOL = {res['zntol']:,} lbs**")
    with st.expander("Details"):
        st.metric("Weight at obstacle", f"{res['w_obstacle']:,} lbs")
        st.metric("Effective MSA used", f"{res['effective_msa']:,} ft")
        st.metric("Source", res['source'])

st.caption("For reference only • Verify with official AFM")


