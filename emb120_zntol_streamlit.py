# emb120_zntol_streamlit.py
# Uses curve fitting (quadratic poly) to reconcile low/high AFM data for smooth, accurate results across altitudes

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

# ────────────────────────────────────────────────
# Calculation with curve fit
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

    # Cold cap
    if isa_dev <= -5:
        w_obstacle_max = 25000.0
        source = "cold cap"
    else:
        # Clamp ISA (use nearest available data)
        isa_clamp = np.clip(isa_dev, 0, 20)  # Low data range; high has up to 30 but we clamp for fit
        source = "curve fit" if isa_dev == isa_clamp else f"curve fit (clamped ISA {isa_clamp}°C)"

        # Collect all known points for this ISA from low and high
        points = {}  # MSA: weight

        # From low
        if isa_clamp in low_data:
            points.update(low_data[isa_clamp])

        # From high
        row_idx = np.where(high_isa_grid == isa_clamp)[0]
        if len(row_idx) > 0:
            for col_idx, msa_high in enumerate(high_msa_grid):
                val = high_tow[row_idx[0], col_idx]
                if msa_high in points:
                    # Average overlap
                    points[msa_high] = (points[msa_high] + val) / 2
                else:
                    points[msa_high] = val

        if not points:
            return {"error": "No data available for this ISA deviation"}

        # Sort points and fit quadratic curve (degree 2 for smooth reconciliation)
        msas = sorted(points.keys())
        weights = np.array([points[m] for m in msas])
        fit = np.polyfit(msas, weights, deg=2)  # Quadratic: ax^2 + bx + c
        poly = np.poly1d(fit)

        # Compute w at effective_msa using the curve
        w_obstacle_max = poly(effective_msa)

    # Caps
    w_obstacle_max = min(max(w_obstacle_max, 4600), STRUCTURAL_MTOW)  # Min from high table edge

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
st.caption("Curve-fitted high/low MSA AFM data • Smooth accuracy across altitudes")

with st.sidebar:
    st.header("Instructions")
    st.markdown("Enter values at the highest enroute obstacle. Uses quadratic curve fit to reconcile low/high data for better integrity at overlaps and higher altitudes.")
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

with st.expander("Assumptions"):
    st.markdown("""
    - Quadratic curve fit (poly deg 2) to all low/high AFM data points (averages overlaps)
    - Effective MSA = entered - 1,000 ft if >6,000 ft (clearance adjustment)
    - Cold ISA ≤ -5 °C capped at 25,000 lbs
    - ISA clamped to 0–20 °C for curve data availability
    - Structural MTOW cap = 26,433 lbs
    - No wind/anti-ice/wet adjustments
    """)

st.caption("For reference only • Verify with official AFM")

