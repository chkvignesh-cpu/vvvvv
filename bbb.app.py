import numpy as np
import pandas as pd
import streamlit as st
import model as mdl
from alerts import AlertEngine
from news2 import LEVEL_COLORS, score_vitals
from simulator import NAMES, VITALS, Patient

st.set_page_config(page_title="PulseGuard AI", page_icon="🫀", layout="wide")

MAX_ROWS = 6000
LABELS = {"hr": "Heart rate (bpm)", "spo2": "SpO2 (%)", "sbp": "Systolic BP (mmHg)", "temp": "Temp (°C)", "rr": "Resp. rate (/min)"}

@st.cache_resource(show_spinner="Training risk model on synthetic data...")
def get_model():
    return mdl.load_or_train()

def advance():
    ss = st.session_state
    ss.tick += 1
    for p in ss.patients:
        v = p.step()
        s = score_vitals(v)
        ss.rows.append({"patient": p.pid, "t": ss.tick, "name": p.name, "bed": p.bed, **v, "score": s["total"], "level": s["level"], "red": s["red"]})
    if len(ss.rows) > MAX_ROWS:
        ss.rows = ss.rows[-MAX_ROWS:]

def init_state(n, seed):
    rng = np.random.default_rng(seed)
    doomed = set(rng.choice(n, size=min(3, n), replace=False).tolist())
    ss = st.session_state
    ss.patients = [Patient(i, NAMES[i % len(NAMES)], f"B{i + 1:02d}", rng, onset=int(rng.integers(18, 40)) if i in doomed else None) for i in range(n)]
    ss.rows, ss.tick = [], 0
    ss.engine = AlertEngine()
    for _ in range(12):
        advance()

st.sidebar.title("🫀 PulseGuard AI")
n_patients = st.sidebar.slider("Patients on ward", 6, 20, 12)
running = st.sidebar.toggle("Live simulation", True)
interval = st.sidebar.slider("Seconds per reading", 1, 10, 2)
st.sidebar.subheader("Alerts")
alerts_on = st.sidebar.toggle("Send alerts", True)
channels = st.sidebar.multiselect("Channels", ["SMS", "Telegram"], default=["SMS"])
cooldown = st.sidebar.slider("Cooldown per patient (s)", 30, 600, 120)

if "patients" not in st.session_state or st.session_state.get("n") != n_patients:
    st.session_state.n = n_patients
    init_state(n_patients, seed=7)

st.session_state.engine.cooldown_s = cooldown
st.sidebar.subheader("Demo controls")
target = st.sidebar.selectbox("Patient", [f"{p.bed} - {p.name}" for p in st.session_state.patients])
c1, c2 = st.sidebar.columns(2)
idx = [f"{p.bed} - {p.name}" for p in st.session_state.patients].index(target)

if c1.button("Deteriorate"):
    st.session_state.patients[idx].deteriorating = True
if c2.button("Recover"):
    st.session_state.patients[idx].deteriorating = False
if st.sidebar.button("Reset ward"):
    init_state(n_patients, seed=int(np.random.default_rng().integers(1, 10_000)))

pipe, metrics = get_model()

with st.sidebar.expander("Model performance (synthetic test set)"):
    st.write(f"ROC-AUC **{metrics['roc_auc']}**, recall **{metrics['recall_escalation']}**, " f"precision **{metrics['precision_escalation']}**")
    st.caption(f"Predicts escalation to NEWS2 >= 5 within {metrics['horizon_readings']} readings.")
    st.bar_chart(pd.Series(metrics["feature_weights"], name="weight"))

def live_panel():
    ss = st.session_state
    if running:
        advance()
    df = pd.DataFrame(ss.rows)
    risk = mdl.predict_latest(pipe, df[df.t >= ss.tick - 6])
    now = df[df.t == ss.tick].copy()
    now["risk"] = now["patient"].map(risk)
    now = now.sort_values(["score", "risk"], ascending=False)
    
    if alerts_on and channels:
        for r in now.itertuples():
            vit = f"HR {r.hr:.0f}, SpO2 {r.spo2:.0f}%, BP {r.sbp:.0f}, Temp {r.temp:.1f}, RR {r.rr:.0f}"
            if r.level == "HIGH":
                ss.engine.notify(f"high-{r.patient}", r.name, "HIGH RISK", f"PulseGuard ALERT: {r.name} ({r.bed}) NEWS2={r.score} HIGH. {vit}", channels)
            elif r.risk >= 0.75 and r.level != "HIGH":
                ss.engine.notify(f"early-{r.patient}", r.name, "EARLY WARNING", f"PulseGuard EARLY WARNING: {r.name} ({r.bed}) trending worse, " f"escalation risk {r.risk:.0%}, NEWS2={r.score}. {vit}", channels)
                
    st.title("PulseGuard AI - Ward Early-Warning Desk")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("HIGH", int((now.level == "HIGH").sum()))
    m2.metric("MEDIUM", int((now.level == "MEDIUM").sum()))
    m3.metric("LOW", int((now.level == "LOW").sum()))
    m4.metric("Reading #", ss.tick)
    
    st.subheader("Active tracking desk")
    table = now[["bed", "name", "hr", "spo2", "sbp", "temp", "rr", "score", "level", "risk"]].rename(columns={
        "bed": "Bed", "name": "Patient", "hr": "HR", "spo2": "SpO2", "sbp": "SBP", "temp": "Temp", "rr": "RR", "score": "NEWS2", "level": "Level", "risk": "Escalation risk"})
    styled = (table.style
              .apply(lambda r: [f"background-color: {LEVEL_COLORS[r['Level']]}; color: #111"] * len(r), axis=1)
              .format({"Escalation risk": "{:.0%}", "HR": "{:.0f}", "SpO2": "{:.0f}", "SBP": "{:.0f}", "Temp": "{:.1f}", "RR": "{:.0f}"}))
    st.dataframe(styled, hide_index=True, height=min(60 + 36 * len(table), 520))
    st.caption("Level: LOW 0-4, MEDIUM 5-6 or any single vital scoring 3, HIGH 7+. Escalation risk = model's probability of reaching NEWS2 >= 5 soon (100% if already there).")
    
    st.subheader("Patient detail")
    names = [f"{b} - {n}" for b, n in zip(table["Bed"], table["Patient"])]
    pick = st.selectbox("Select patient", names, key="detail_pick")
    pid = now[(now.bed == pick.split(" - ")[0])]["patient"].iloc[0]
    hist = df[df.patient == pid].tail(60).set_index("t")
    cols = st.columns(3)
    for i, v in enumerate(VITALS + ["score"]):
        with cols[i % 3]:
            st.caption(LABELS.get(v, "NEWS2 score"))
            st.line_chart(hist[[v]], height=140)
            
    st.subheader("Alert log")
    if ss.engine.log:
        st.dataframe(pd.DataFrame(ss.engine.log[::-1]), hide_index=True)
    else:
        st.info("No alerts yet. Use 'Deteriorate' in the sidebar to trigger one.")

st.fragment(run_every=(interval if running else None))(live_panel)()
st.caption("Synthetic data only. Decision-support prototype, not a medical device.")
