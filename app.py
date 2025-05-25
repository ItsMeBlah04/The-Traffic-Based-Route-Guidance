import streamlit as st
import pandas as pd
import pydeck as pdk
from datetime import date

from route_planner import RoutePlanner

# ========== Configuration ==========
st.set_page_config(layout="wide")
st.title("🚦 Traffic-Based Route Planner")

# ========== Predefined Options ==========
locations = {
    "auburn_rd": (-37.816467, 145.046181),
    "balwyn_rd": (-37.790271, 145.085768),
    "barkers_rd": (-37.812583, 145.019117),
    "belmore_rd": (-37.803403, 145.102324),
    "bridge_rd": (-37.819946, 145.015970),
    "bulleen_rd": (-37.768636, 145.075474),
    "burke_rd": (-37.837812, 145.058885),
    "burnley_st": (-37.820401, 145.008140),
    "burwood_hwy": (-37.849210, 145.111692),
    "burwood_rd": (-37.823965, 145.045676),
    "camberwell_rd": (-37.839999, 145.070625),
    "canterbury_rd": (-37.832443, 145.074052),
    "chandler_hwy": (-37.779252, 145.026420),
    "charles_st": (-37.793395, 145.067466),
    "church_st": (-37.809895, 145.021694),
    "cotham_rd": (-37.806728, 145.057320),
    "denmark_st": (-37.807896, 145.028983),
    "doncaster_rd": (-37.787331, 145.124513),
    "earl_st": (-37.794403, 145.067796),
    "eastern_fwy": (-37.786401, 145.046902),
    "eastern_fwy_w_bd_ramps": (-37.788047, 145.030670),
    "glenferrie_rd": (-37.819881, 145.035978),
    "harp_rd": (-37.799904, 145.055413),
    "harp_st": (-37.799552, 145.054585),
    "high street_rd": (-37.865180, 145.140190),
    "high_st": (-37.805203, 145.022546),
    "highbury_rd": (-37.857936, 145.130618),
    "kilby_rd": (-37.802740, 145.043198),
    "madden_gv": (-37.804433, 145.066160),
    "maroondah_hwy": (-37.410222, 145.706653),
    "mont albert_rd": (-37.816831, 145.067648),
    "offramp_eastern_fwy": (-37.782087, 145.077826),
    "power_st": (-37.820284, 145.024237),
    "princess_st": (-37.800093, 145.038160),
    "rathmines_rd": (-37.805895, 145.065944),
    "riversdale_rd": (-37.831616, 145.058930),
    "s.e.arterial": (-37.843206, 145.037970),
    "severn_st": (-37.793724, 145.070231),
    "seymour_gv": (-37.806379, 145.064778),
    "stanhope_gv": (-37.804687, 145.065622),
    "studley park_rd": (-37.800768, 145.023242),
    "swan_st": (-37.825627, 144.999371),
    "thompsons_rd": (-37.778083, 145.074740),
    "tooronga_rd": (-37.849327, 145.057409),
    "trafalgar_rd": (-37.833693, 145.063387),
    "union_rd": (-37.816900, 145.066580),
    "valerie_st": (-37.793705, 145.066707),
    "victoria_st": (-37.813373, 145.008224),
    "walmer_st": (-37.812332, 145.024268),
    "warrigal_rd": (-37.840671, 145.088500),
    "whitehorse_rd": (-37.809963, 145.112181),
    "wills_st": (-37.794165, 145.068872)
}

times = [f"{h:02}:{m:02}" for h in range(24) for m in [0, 15, 30, 45]]
model_options = ["gru", "lstm", "cnn_lstm"]
path_options = ['bfs', 'dfs', 'gbfs', 'df_limit', 'ucs', 'a_star']

# ========== UI Layout ==========
left, right = st.columns([1, 2])

with left:
    st.subheader("🧭 Select Route Options")

    start_loc = st.selectbox("Start Location", list(locations.keys()))
    end_loc = st.selectbox("Goal Location", list(locations.keys()), index=1)
    selected_date = st.date_input("Select Date", value=date(2006, 10, 1))
    selected_time = st.selectbox("Select Time", times)
    model_type = st.selectbox("Model", model_options, index=2)
    path_type = st.selectbox("Path Finding", path_options, index=0)

    run_button = st.button("🚀 Run Route Estimation")

route_planner = RoutePlanner()

try:
    locations_path, coords, est_time, cost, flow, speeds = route_planner.route_estimate(
        origin=start_loc.lower(),
        destination=end_loc.lower(),
        date=selected_date.strftime("%d/%m/%Y"),
        time=selected_time,
        model_type=model_type,
        path_finder_type=path_type
    )

    # Prepare IconLayer (unchanged)
    icon_urls = {
        "start": "https://raw.githubusercontent.com/Concept211/Google-Maps-Markers/master/images/marker_red.png",
        "end": "https://raw.githubusercontent.com/Concept211/Google-Maps-Markers/master/images/marker_green.png",
        "middle": "https://raw.githubusercontent.com/Concept211/Google-Maps-Markers/master/images/marker_blue.png"
    }

    df_coords = pd.DataFrame([
        {
            "lon": lon,
            "lat": lat,
            "label": label,
            "icon_data": {
                "url": (
                    icon_urls["start"] if i == 0
                    else icon_urls["end"] if i == len(coords) - 1
                    else icon_urls["middle"]
                ),
                "width": 64,
                "height": 64,
                "anchorY": 64
            }
        }
        for i, ((lat, lon), label) in enumerate(zip(coords, ["Start"] + [""] * (len(coords) - 2) + ["End"]))
    ])

    # 📌 Show summary on the left
    with left:
        st.subheader("📋 Route Summary")
        st.success(f"📍 Route from `{start_loc}` to `{end_loc}`")
        st.markdown(f"""
        - 📆 **Date**: `{selected_date.strftime("%d/%m/%Y")}`
        - ⏰ **Time**: `{selected_time}`
        - 🧠 **Model**: `{model_type}`
        - 🧭 **Path Type**: `{path_type}`  
        - 🚦 **Estimated Flow**: `{flow:.2f}`
        - 🚗 **Estimated Speeds**: {speeds[0]:.1f} km/h (Congested), {speeds[1]:.1f} km/h (Free-Flow)
        - ⏳ **Estimated Travel Time**: `{est_time:.1f} mins`
        - 🧮 **Total Distance**: `{cost:.2f} km`
        """)
        st.markdown("### 🛣️ Route Path:")
        st.markdown(" → ".join(f"`{loc}`" for loc in locations_path))

    # 🗺️ Show full-width map below
    with right:
        st.subheader("🗺️ Route Preview")
        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/streets-v11",
                initial_view_state=pdk.ViewState(
                    latitude=sum([c[0] for c in coords]) / len(coords),
                    longitude=sum([c[1] for c in coords]) / len(coords),
                    zoom=11,
                    pitch=0,
                ),
                layers=[
                    pdk.Layer(
                        "IconLayer",
                        data=df_coords,
                        get_icon="icon_data",
                        get_position='[lon, lat]',
                        get_size=4,
                        size_scale=15,
                        pickable=True
                    ),
                    pdk.Layer(
                        "PathLayer",
                        data=[{"path": [(lon, lat) for lat, lon in coords]}],
                        get_path="path",
                        get_color=[0, 100, 255],
                        get_width=5,
                    )
                ],
                tooltip={"text": "{label}"}
            ),
            height=800  
        )
    st.success("✅ Route estimation completed successfully!")
except Exception as e:
    st.error(f"❌ Error: {e}")