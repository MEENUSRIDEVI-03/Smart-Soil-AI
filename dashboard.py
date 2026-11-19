import os
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "soil_dataset.csv")
COLUMNS = ["time", "moisture", "delta", "rate"]

st.set_page_config(
    page_title="Smart Soil AI | Soil Monitor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
:root { color-scheme: light; }
.stApp { background: #f5f8f2; color: #203126; }
header[data-testid="stHeader"] { background: rgba(245,248,242,.92); }
.block-container { max-width: 1440px; padding-top: 1.7rem; padding-bottom: 3rem; }
h1,h2,h3 { color: #203b2b !important; letter-spacing: -0.025em; }
p, label, span { color: #405347; }
.hero {
  background: linear-gradient(115deg,#e5f1df 0%,#f5f8ed 55%,#e9f3e7 100%);
  border: 1px solid #dce8d7; border-radius: 22px; padding: 25px 30px;
  margin-bottom: 22px;
}
.eyebrow { color:#47734f; font-size: .78rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
.hero-title { color:#1e3c2b; font-size:2.05rem; line-height:1.15; font-weight:800; margin:.35rem 0; }
.hero-sub { color:#607264; font-size:1rem; margin:0; }
.pill { display:inline-block; padding:6px 11px; border-radius:99px; background:#fff; border:1px solid #d9e6d5; color:#47734f; font-size:.78rem; font-weight:650; }
.metric-card {
 background:#fff; border:1px solid #e1e9df; border-radius:17px; padding:17px 19px;
 min-height:115px; box-shadow:0 2px 10px rgba(37,66,43,.035);
}
.metric-label { color:#738176; font-size:.82rem; font-weight:650; }
.metric-value { color:#203b2b; font-size:1.75rem; font-weight:800; margin-top:5px; }
.metric-note { color:#849187; font-size:.76rem; margin-top:2px; }
.section-card { background:#fff; border:1px solid #e1e9df; border-radius:18px; padding:20px 22px; }
.status-good { background:#e8f5e9; color:#27643b; border:1px solid #cce7d0; }
.status-mid { background:#fff5dc; color:#8b6418; border:1px solid #f1dfad; }
.status-poor { background:#fff0e9; color:#a64b2e; border:1px solid #f2d2c4; }
.status-badge { display:inline-block; padding:7px 13px; border-radius:99px; font-weight:750; font-size:.86rem; }
div.stButton > button[kind="primary"] { background:#36784a; border:0; border-radius:11px; min-height:2.8rem; }
div.stButton > button { border-radius:10px; border-color:#d8e3d5; }
div[data-testid="stSlider"] [data-baseweb="slider"] { padding-top:.6rem; }
div[data-testid="stAlert"] { border-radius:12px; }
hr { border-color:#e2e9df; }
.small-muted { color:#829084; font-size:.8rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            if set(COLUMNS).issubset(df.columns):
                df = df[COLUMNS].copy()
                df["time"] = pd.to_datetime(df["time"], errors="coerce")
                for c in ["moisture", "delta", "rate"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                return df.dropna().sort_values("time").reset_index(drop=True)
        except Exception:
            pass
    return pd.DataFrame(columns=COLUMNS)

def append_reading(value):
    old = load_data()
    now = datetime.now()
    previous = float(old.iloc[-1]["moisture"]) if not old.empty else float(value)
    previous_time = old.iloc[-1]["time"] if not old.empty else now
    seconds = max((now - previous_time).total_seconds(), 1.0)
    delta = float(value) - previous
    row = pd.DataFrame([[now, int(value), delta, delta / seconds]], columns=COLUMNS)
    row.to_csv(DATA_FILE, mode="a", header=not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0, index=False)
    load_data.clear()

def classify(value, rate):
    if 400 <= value <= 800 and abs(rate) <= 2:
        return "Healthy", "good", "Moisture is within the configured target band and currently stable."
    if 200 <= value <= 900 and abs(rate) <= 10:
        return "Monitor", "mid", "The reading is within the wider operating band. Keep monitoring the soil."
    return "Needs attention", "poor", "The reading is outside the configured healthy range or is changing quickly."

def estimated_soil(rate):
    if abs(rate) <= 2: return "Clay-like*"
    if abs(rate) <= 10: return "Loam-like*"
    return "Sandy-like*"

data = load_data()
latest = data.iloc[-1] if not data.empty else None
current_value = int(latest["moisture"]) if latest is not None else 600
current_rate = float(latest["rate"]) if latest is not None else 0.0
status, status_key, status_text = classify(current_value, current_rate)
moisture_pct = round(current_value / 1023 * 100)

st.markdown("""
<div class="hero">
  <div class="eyebrow">SMART AGRICULTURE · SOFTWARE PROTOTYPE</div>
  <div class="hero-title">🌱 Smart Soil AI</div>
  <p class="hero-sub">Soil monitoring dashboard & interactive sensor simulator</p>
  <div style="margin-top:15px"><span class="pill">● Simulator mode</span> &nbsp; <span class="pill">No physical hardware required</span></div>
</div>
""", unsafe_allow_html=True)

# Status summary
c1, c2, c3, c4 = st.columns(4)
metrics = [
    (c1, "SOIL MOISTURE INDEX", f"{moisture_pct}%", f"Raw sensor value: {current_value}/1023"),
    (c2, "RAW SENSOR READING", f"{current_value}", "Simulated analog input"),
    (c3, "CHANGE RATE", f"{current_rate:.2f}", "Sensor units per second"),
    (c4, "RECORDED SAMPLES", f"{len(data)}", "Saved simulation readings"),
]
for col, label, value, note in metrics:
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.02, 1.45], gap="large")

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🎛️ Sensor simulator")
    st.caption("Adjust the virtual sensor value, then run a sample to update the dashboard.")
    with st.form("simulation_form"):
        selected = st.slider("Soil moisture sensor (raw)", 0, 1023, current_value, help="This is a simulated 10-bit analog reading. The demo mapping treats higher values as wetter soil.")
        st.markdown(f"**Selected input:** `{selected} / 1023`")
        st.progress(selected / 1023)
        submitted = st.form_submit_button("▶ Run simulation", type="primary", use_container_width=True)
    if submitted:
        append_reading(selected)
        st.session_state["notice"] = f"Simulation sample {selected} recorded."
        st.rerun()
    if st.session_state.get("notice"):
        st.success(st.session_state.pop("notice"))
    st.markdown("**Quick test scenarios**")
    q1,q2 = st.columns(2)
    presets = [(q1, "🌵 Dry soil", 150), (q2, "🌿 Mid range", 600),
               (q1, "💧 High moisture", 850), (q2, "🫧 Very wet", 950)]
    for col, label, value in presets:
        with col:
            if st.button(f"{label} · {value}", key=f"preset_{value}", use_container_width=True):
                append_reading(value)
                st.session_state["notice"] = f"{label} sample recorded ({value})."
                st.rerun()
    st.markdown('<p class="small-muted">Demo scale follows the supplied project examples: 150 = dry, 600 = mid-range, 850 = high moisture.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🌿 Current soil condition")
    badge_class = {"good":"status-good","mid":"status-mid","poor":"status-poor"}[status_key]
    st.markdown(f'<span class="status-badge {badge_class}">{status}</span>', unsafe_allow_html=True)
    st.write(status_text)
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=moisture_pct,
        number={"suffix":"%", "font":{"size":36,"color":"#203b2b"}},
        title={"text":"Moisture index (scaled from raw input)", "font":{"size":14,"color":"#607264"}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#829084"},
               "bar":{"color":"#4c9360"},
               "bgcolor":"#f4f7f1","borderwidth":0,
               "steps":[{"range":[0,30],"color":"#f8e7df"},{"range":[30,70],"color":"#f7f0d8"},{"range":[70,100],"color":"#e2f0df"}]}
    ))
    gauge.update_layout(height=235, margin=dict(l=15,r=15,t=45,b=5), paper_bgcolor="rgba(0,0,0,0)", font_color="#203b2b")
    st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar":False})
    soil = estimated_soil(current_rate)
    st.markdown(f"**Rule-based soil estimate:** {soil}")
    st.caption("*Illustrative estimate retained from the original prototype; moisture change rate alone cannot determine soil texture.")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
st.subheader("📈 Reading history")
if data.empty:
    st.info("No samples recorded yet. Choose a scenario or adjust the slider and run your first simulation.")
else:
    history = data.tail(150).copy()
    fig = px.line(history, x="time", y="moisture", markers=True,
                  labels={"time":"Time","moisture":"Raw moisture reading"},
                  title="Soil moisture over simulation time")
    fig.update_traces(line_color="#4c9360", marker_color="#4c9360", line_width=3)
    fig.update_layout(height=330, margin=dict(l=10,r=10,t=55,b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
                      font_color="#405347", xaxis=dict(showgrid=False),
                      yaxis=dict(gridcolor="#edf1eb", range=[0,1023]))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    with st.expander("View recorded samples"):
        st.dataframe(data.tail(100).sort_values("time", ascending=False), use_container_width=True, hide_index=True)

st.write("")
st.subheader("🤖 AI moisture trend prediction")
if len(data) >= 30:
    model_data = data.copy()
    model_data["time_num"] = model_data["time"].astype("int64") // 10**9
    X = model_data[["time_num","moisture","delta","rate"]].iloc[:-1]
    y = model_data["moisture"].shift(-1).iloc[:-1]
    model = RandomForestRegressor(n_estimators=80, random_state=42)
    model.fit(X, y)
    row = data.iloc[-1]
    x_now = pd.DataFrame([[int(datetime.now().timestamp()), float(row["moisture"]), float(row["delta"]), float(row["rate"])]],
                         columns=["time_num","moisture","delta","rate"])
    prediction = float(model.predict(x_now)[0])
    p1,p2 = st.columns([1,2])
    with p1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">PREDICTED NEXT READING</div><div class="metric-value">{prediction:.0f}</div><div class="metric-note">Experimental Random Forest output</div></div>', unsafe_allow_html=True)
    with p2:
        st.info("The model is trained on the readings recorded in this simulator. This is an experimental demonstration, not a calibrated agronomic forecast.")
else:
    remaining = 30 - len(data)
    st.info(f"Record {remaining} more samples to unlock the Random Forest demonstration. You can use the quick scenarios to build a sample history.")

st.divider()
foot1, foot2 = st.columns([3,1])
with foot1:
    last_time = str(latest["time"]) if latest is not None else "No samples yet"
    st.caption(f"Last sample: {last_time} · Data source: manual simulation · Local CSV history")
with foot2:
    if st.button("Reset simulation history", use_container_width=True):
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
        load_data.clear()
        st.rerun()

st.caption("Prototype note: all values on this page are simulated. No live ESP-12E or physical sensor data is being received.")
