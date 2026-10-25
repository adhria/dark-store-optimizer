import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import model

st.set_page_config(page_title="Dark Store Optimizer - Chennai", layout="wide")
st.title("Dark Store Location Optimizer - Chennai")
st.caption("Profit-maximizing facility location with a max store-customer distance constraint (OR-Tools / SCIP).")

SWEEP_KS = [2, 5, 10, 20, 25]
# Offline run: 10,000 customers, 50 candidates, 5 km
PRECOMPUTED = {2: 281261, 5: 651526, 10: 747677, 20: 747677, 25: 747677}

with st.sidebar:
    st.header("Inputs")
    n_cand = st.number_input("Dark store candidates", 5, 100, 50, 5)
    n_cust = st.number_input("Number of customers", 1000, 20000, 10000, 1000)
    max_dist = st.slider("Max store-customer distance (km)", 1.0, 15.0, 5.0, 0.5)
    max_stores = st.number_input("Max stores allowed to open (K)", 1, 100, 10, 1)
    run = st.button("Run optimization", type="primary")


@st.cache_data(show_spinner=False)
def run_model(n_cust, n_cand, max_dist, max_stores):
    return model.solve(int(n_cust), int(n_cand), float(max_dist), int(max_stores))


tab1, tab2 = st.tabs(["Optimization results", "Profit vs. store limit"])

with tab1:
    if run:
        with st.spinner("Solving... this can take up to a minute."):
            st.session_state["res"] = run_model(n_cust, n_cand, max_dist, max_stores)
            st.session_state["ran"] = True

    if not st.session_state.get("ran"):
        st.info("Set the inputs in the sidebar and click **Run optimization**.")
    elif st.session_state["res"] is None:
        st.error("No feasible solution found. Try a larger max distance.")
    else:
        res = st.session_state["res"]
        st.caption(f"Solver status: {res['status']}")

        # ---- KPI table ----
        kpi_df = pd.DataFrame(
            [(k, f"{v:,}") for k, v in res["kpis"].items()], columns=["KPI", "Value"]
        )
        st.subheader("KPIs")
        st.dataframe(kpi_df, hide_index=True, use_container_width=True)

        # ---- Map ----
        glat, glon, gw = res["g_lat"], res["g_lon"], res["g_w"]
        slat, slon = res["s_lat"], res["s_lon"]
        served = np.zeros(len(gw))
        load, lines = {}, []
        for c, i, f in res["flows"]:
            served[c] += f
            load[i] = load.get(i, 0.0) + f
            lines.append({"slon": glon[c], "slat": glat[c], "tlon": slon[i], "tlat": slat[i]})
        pct = np.clip(served / gw, 0, 1) * 100

        cells = pd.DataFrame({
            "lat": glat, "lon": glon, "radius": 60 + 20 * np.sqrt(gw),
            "tip": [f"Demand cell: {w:.0f} customers, {p:.0f}% served" for w, p in zip(gw, pct)],
        })
        cells["color"] = [[46, 160, 67, 150] if p >= 50 else [220, 50, 47, 170] for p in pct]

        opened = set(res["opened"])
        st_tip, st_col, st_rad = [], [], []
        for i in range(len(slat)):
            if i in opened:
                l = load.get(i, 0.0)
                st_tip.append(f"Store {i} (OPEN): {l:.0f} customers, "
                              f"{100 * l / model.CAPACITY:.0f}% utilisation")
                st_col.append([0, 90, 200, 240]); st_rad.append(450)
            else:
                st_tip.append(f"Candidate {i} (not opened)")
                st_col.append([120, 120, 120, 170]); st_rad.append(200)
        stores = pd.DataFrame({"lat": slat, "lon": slon, "tip": st_tip,
                               "color": st_col, "radius": st_rad})

        layers = []
        if lines:
            layers.append(pdk.Layer("LineLayer", pd.DataFrame(lines),
                                    get_source_position="[slon, slat]",
                                    get_target_position="[tlon, tlat]",
                                    get_color=[0, 90, 200, 50], get_width=1))
        layers.append(pdk.Layer("ScatterplotLayer", cells, get_position="[lon, lat]",
                                get_radius="radius", get_fill_color="color", pickable=True))
        layers.append(pdk.Layer("ScatterplotLayer", stores, get_position="[lon, lat]",
                                get_radius="radius", get_fill_color="color", pickable=True))

        st.subheader("Solution map")
        st.pydeck_chart(pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=13.02, longitude=80.2, zoom=9.5),
            tooltip={"text": "{tip}"},
        ))
        st.caption("Blue = open stores, grey = unopened candidates, green = served demand cells, "
                   "red = unserved cells, lines = assignments.")

with tab2:
    st.subheader("Profit vs. maximum number of open stores")
    st.write("10,000 customers, 50 candidate locations, using the max distance from the sidebar.")

    if st.button("Run sweep live (about 2 minutes)"):
        bar = st.progress(0.0)
        profits = []
        for n, K in enumerate(SWEEP_KS):
            r = run_model(10000, 50, max_dist, K)
            profits.append(r["kpis"]["Net profit (Rs)"] if r else 0)
            bar.progress((n + 1) / len(SWEEP_KS))
        st.session_state["sweep"] = (max_dist, dict(zip(SWEEP_KS, profits)))

    sweep = st.session_state.get("sweep")
    if sweep and sweep[0] == max_dist:
        data, src = sweep[1], "live run"
    elif max_dist == 5.0:
        data, src = PRECOMPUTED, "precomputed for 5 km"
    else:
        data, src = None, None

    if data:
        df = pd.DataFrame({"K": list(data.keys()), "Net profit (Rs)": list(data.values())})
        st.line_chart(df.set_index("K"))
        st.dataframe(df, hide_index=True)
        st.caption(f"Source: {src}. Profit flattens once extra stores no longer cover their fixed cost.")
    else:
        st.info("Click the button above to run the sweep for this max distance.")

with st.expander("Assumptions"):
    st.markdown(
        f"- Profit per customer: Rs {model.PROFIT_PER_CUST}/month; delivery cost: Rs {model.COST_PER_KM}/km; "
        f"fixed cost: Rs {model.FIXED_COST:,}/store/month; capacity: {model.CAPACITY} customers/store\n"
        "- Customers are synthetic, clustered around 10 Chennai areas; distances are straight-line (haversine)\n"
        f"- Customers are aggregated into {model.CELL_KM} km grid cells, and a cell's demand may be split across stores\n"
        "- Distance constraint: a store can only serve customers within the max distance"
    )
