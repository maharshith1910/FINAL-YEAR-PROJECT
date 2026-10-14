"""
dashboard/app.py
----------------
Streamlit dashboard for the Smart Crime Detection system.

Features:
  ✅ Upload audio file → real-time window scoring timeline
  ✅ Live threat score gauge
  ✅ Incident alert log with severity badges
  ✅ Model comparison (CNN vs CNN-LSTM metrics from JSON)
  ✅ Confusion matrix & PR curve viewer
  ✅ CSV export of alert log
  ✅ Threshold sweep slider

Run:
    streamlit run dashboard/app.py
"""

import sys
import json
import time
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.alert_logger import AlertLogger
from inference.predictor import ScreamPredictor, extract_mel
import librosa

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "SafeWatch AI — Smart Crime Detection",
    page_icon   = "🚨",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Dark gradient background */
  .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 50%, #0a0e1a 100%); }

  /* Metric cards */
  .metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(12px);
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(99,102,241,0.15); }

  /* Alert severity badges */
  .badge-high   { background:#dc2626; color:white; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; }
  .badge-medium { background:#d97706; color:white; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; }
  .badge-low    { background:#059669; color:white; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; }
  .badge-pending{ background:#6366f1; color:white; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; }

  /* Alert fire animation */
  @keyframes pulse-red { 0%,100%{box-shadow:0 0 0 0 rgba(220,38,38,.4)} 50%{box-shadow:0 0 0 12px rgba(220,38,38,0)} }
  .alert-active { animation: pulse-red 1.5s infinite; border-color: #dc2626 !important; }

  /* Section headers */
  .section-header {
    font-size: 1.1rem; font-weight: 700; color: #a5b4fc;
    border-bottom: 1px solid rgba(165,180,252,0.2);
    padding-bottom: 8px; margin-bottom: 16px; letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  /* Sidebar */
  .css-1d391kg { background: rgba(10,14,26,0.9); }

  /* Hide streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }

  /* Custom scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0a0e1a; }
  ::-webkit-scrollbar-thumb { background: #4f46e5; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── Constants & paths ─────────────────────────────────────────────────────────
METRICS_DIR    = Path("outputs/metrics")
CHECKPOINT_DIR = Path("models/checkpoints")
SAMPLE_RATE    = 22050
WINDOW_SEC     = 1.0
HOP_SEC        = 0.5

# ─── Session state init ────────────────────────────────────────────────────────
if "predictor"      not in st.session_state: st.session_state.predictor      = None
if "results"        not in st.session_state: st.session_state.results        = []
if "alert_logger"   not in st.session_state: st.session_state.alert_logger   = AlertLogger()
if "model_type"     not in st.session_state: st.session_state.model_type     = "cnn_lstm"
if "threshold"      not in st.session_state: st.session_state.threshold      = 0.65
if "consec"         not in st.session_state: st.session_state.consec         = 3


# ─── Helpers ───────────────────────────────────────────────────────────────────

def load_predictor(model_type, threshold, consec):
    ckpt = CHECKPOINT_DIR / f"{model_type}_best.pth"
    st.session_state.predictor = ScreamPredictor(
        model_type          = model_type,
        threshold           = threshold,
        consecutive_windows = consec,
        location_label      = st.session_state.get("location", "Camera A"),
    )
    return st.session_state.predictor


def load_metrics(model_type: str) -> dict | None:
    path = METRICS_DIR / f"{model_type}_test_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def color_for_prob(p: float) -> str:
    if p >= 0.75: return "#dc2626"
    if p >= 0.50: return "#d97706"
    return "#059669"


def make_gauge(value: float, title: str = "Threat Level") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = value * 100,
        delta = {"reference": 50, "valueformat": ".1f"},
        title = {"text": title, "font": {"size": 16, "color": "#a5b4fc", "family": "Inter"}},
        number= {"suffix": "%", "font": {"size": 28, "color": "white"}},
        gauge = {
            "axis":  {"range": [0, 100], "tickcolor": "#4f46e5", "tickfont": {"color": "#6b7280"}},
            "bar":   {"color": color_for_prob(value), "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0,  50], "color": "rgba(5,150,105,0.15)"},
                {"range": [50, 75], "color": "rgba(217,119,6,0.15)"},
                {"range": [75,100], "color": "rgba(220,38,38,0.15)"},
            ],
            "threshold": {
                "line":  {"color": "#f59e0b", "width": 3},
                "thickness": 0.8,
                "value": st.session_state.threshold * 100
            }
        }
    ))
    fig.update_layout(
        height=220, margin=dict(t=30, b=0, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def make_timeline(results: list) -> go.Figure:
    if not results:
        return go.Figure()

    ts    = [r.timestamp_sec for r in results]
    probs = [r.scream_prob   for r in results]
    alert = [r.is_alert      for r in results]
    colors = [color_for_prob(p) for p in probs]

    fig = go.Figure()

    # Shaded alert regions
    for i, a in enumerate(alert):
        if a:
            x0 = ts[i] - HOP_SEC / 2
            x1 = ts[i] + HOP_SEC / 2
            fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(220,38,38,0.2)",
                          line_width=0, layer="below")

    # Threshold line
    fig.add_hline(
        y = st.session_state.threshold, line_dash="dot",
        line_color="#f59e0b", line_width=1.5,
        annotation_text=f"Threshold {st.session_state.threshold:.2f}",
        annotation_font_color="#f59e0b"
    )

    # Probability trace
    fig.add_trace(go.Scatter(
        x=ts, y=probs,
        mode="lines+markers",
        name="P(scream)",
        line=dict(color="#6366f1", width=2.5, shape="spline"),
        marker=dict(size=6, color=colors, line=dict(color="white", width=1)),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.08)",
        hovertemplate="t=%{x:.2f}s<br>P(scream)=%{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        height=280,
        xaxis=dict(title="Time (seconds)", color="#9ca3af", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="P(scream)", range=[0, 1], color="#9ca3af", gridcolor="rgba(255,255,255,0.05)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=16, b=40, l=50, r=20),
        legend=dict(font=dict(color="#9ca3af")),
        font=dict(color="#9ca3af"),
    )
    return fig


def render_metrics_card(metrics: dict | None, model_name: str):
    if metrics is None:
        st.info(f"No test results yet for **{model_name}**. Run `python training/evaluate.py --model {model_name}`")
        return

    cols = st.columns(6)
    items = [
        ("Accuracy",  f"{metrics['accuracy']*100:.1f}%",  "📊"),
        ("Precision", f"{metrics['precision']*100:.1f}%", "🎯"),
        ("Recall",    f"{metrics['recall']*100:.1f}%",    "🔍"),
        ("F1 Score",  f"{metrics['f1']*100:.1f}%",        "⚖️"),
        ("FPR",       f"{metrics['false_positive_rate']*100:.1f}%", "⚠️"),
        ("Latency",   f"{metrics['avg_latency_ms']:.1f}ms", "⚡"),
    ]
    for col, (label, val, icon) in zip(cols, items):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center">
              <div style="font-size:1.5rem">{icon}</div>
              <div style="font-size:1.5rem; font-weight:800; color:white">{val}</div>
              <div style="font-size:0.75rem; color:#6b7280; margin-top:4px">{label}</div>
            </div>""", unsafe_allow_html=True)

    if metrics.get("note"):
        st.caption(f"ℹ️ {metrics['note']}")


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0">
      <div style="font-size:2.5rem">🚨</div>
      <div style="font-size:1.2rem; font-weight:800; color:white">SafeWatch AI</div>
      <div style="font-size:0.75rem; color:#6b7280; margin-top:4px">Smart Crime Detection</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="section-header">⚙️ Model Settings</div>', unsafe_allow_html=True)

    model_type = st.selectbox(
        "Detection Model",
        options=["cnn_lstm", "cnn"],
        format_func=lambda x: "CNN-LSTM (Temporal)" if x == "cnn_lstm" else "CNN (Baseline)",
        key="model_type",
    )

    location = st.text_input("📍 Location Label", value="Camera A", key="location")

    threshold = st.slider(
        "Alert Threshold",
        min_value=0.3, max_value=0.95, value=0.65, step=0.05,
        key="threshold",
        help="P(scream) must exceed this value per window"
    )

    consec = st.slider(
        "Consecutive Windows",
        min_value=1, max_value=8, value=3,
        key="consec",
        help="How many consecutive windows above threshold before alert fires"
    )

    if st.button("🔄 Load / Reload Model", use_container_width=True, type="primary"):
        with st.spinner("Loading model checkpoint…"):
            load_predictor(model_type, threshold, consec)
        st.success("Model loaded!")

    st.divider()
    st.markdown('<div class="section-header">📋 Alert Stats</div>', unsafe_allow_html=True)

    stats = st.session_state.alert_logger.stats()
    st.markdown(f"""
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px">
      <div class="metric-card" style="text-align:center">
        <div style="font-size:1.5rem;color:white;font-weight:800">{stats['total']}</div>
        <div style="font-size:0.7rem;color:#6b7280">Total Alerts</div>
      </div>
      <div class="metric-card" style="text-align:center">
        <div style="font-size:1.5rem;color:#dc2626;font-weight:800">{stats['high']}</div>
        <div style="font-size:0.7rem;color:#6b7280">HIGH</div>
      </div>
      <div class="metric-card" style="text-align:center">
        <div style="font-size:1.5rem;color:#d97706;font-weight:800">{stats['medium']}</div>
        <div style="font-size:0.7rem;color:#6b7280">MEDIUM</div>
      </div>
      <div class="metric-card" style="text-align:center">
        <div style="font-size:1.5rem;color:#6366f1;font-weight:800">{stats['pending']}</div>
        <div style="font-size:0.7rem;color:#6b7280">Pending</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Main content tabs ─────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🎙️ Live Detection",
    "📊 Model Performance",
    "🚨 Incident Log",
    "ℹ️ About",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Detection
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <h2 style="color:white; font-weight:800; margin-bottom:4px">
      🎙️ Real-Time Threat Detection
    </h2>
    <p style="color:#6b7280; margin-bottom:24px">
      Upload an audio file to analyse. Each second-long window is scored by
      the model and flagged if the scream probability exceeds the threshold
      for N consecutive windows.
    </p>
    """, unsafe_allow_html=True)

    # ── Check model loaded
    if st.session_state.predictor is None:
        st.warning("⚠️ No model loaded. Click **Load / Reload Model** in the sidebar first.")
        st.info("If checkpoints are missing, run:\n```\npython training/train.py --model cnn_lstm\n```")

    col_upload, col_gauge = st.columns([2, 1])

    with col_upload:
        st.markdown('<div class="section-header">📁 Upload Audio</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop a .wav file here",
            type=["wav", "mp3", "flac", "ogg"],
            label_visibility="collapsed"
        )

        if uploaded and st.session_state.predictor:
            # Save to temp file
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            with st.spinner(f"🔍 Analysing {uploaded.name} …"):
                results = st.session_state.predictor.process_file(tmp_path)
                st.session_state.results = results

            n_alerts = sum(1 for r in results if r.is_alert)
            max_prob  = max((r.scream_prob for r in results), default=0.0)

            if n_alerts > 0:
                st.markdown(f"""
                <div class="metric-card alert-active" style="border:1px solid #dc2626;margin-bottom:16px;padding:16px">
                  <span style="font-size:1.5rem">🚨</span>
                  <span style="color:#dc2626;font-weight:800;font-size:1.1rem;margin-left:8px">
                    {n_alerts} DISTRESS EVENT(S) DETECTED — Requires human verification
                  </span>
                </div>""", unsafe_allow_html=True)
            else:
                st.success(f"✅ No threats detected in {len(results)} windows. Max P(scream) = {max_prob:.3f}")

            # Timeline
            st.markdown('<div class="section-header">📈 Detection Timeline</div>', unsafe_allow_html=True)
            st.plotly_chart(make_timeline(results), use_container_width=True)

            # Per-window table (collapsible)
            with st.expander("🔢 Raw Window Scores"):
                rows = [{"Time (s)": f"{r.timestamp_sec:.2f}",
                         "P(scream)": f"{r.scream_prob:.4f}",
                         "Status": r.label} for r in results]
                df_r = pd.DataFrame(rows)
                st.dataframe(df_r, use_container_width=True, height=200)

        elif uploaded and st.session_state.predictor is None:
            st.error("Load a model checkpoint first.")

    with col_gauge:
        st.markdown('<div class="section-header">🌡️ Threat Level</div>', unsafe_allow_html=True)
        results = st.session_state.results
        current_prob = max((r.scream_prob for r in results), default=0.0) if results else 0.0
        st.plotly_chart(make_gauge(current_prob), use_container_width=True)

        if results:
            alerts = [r for r in results if r.is_alert]
            st.markdown(f"""
            <div style="margin-top:8px">
              <div style="color:#9ca3af;font-size:0.8rem">Windows analysed</div>
              <div style="color:white;font-weight:700;font-size:1.2rem">{len(results)}</div>
              <div style="color:#9ca3af;font-size:0.8rem;margin-top:8px">Alert events fired</div>
              <div style="color:#dc2626;font-weight:700;font-size:1.2rem">{len(alerts)}</div>
              <div style="color:#9ca3af;font-size:0.8rem;margin-top:8px">Peak confidence</div>
              <div style="color:#f59e0b;font-weight:700;font-size:1.2rem">{current_prob:.3f}</div>
            </div>""", unsafe_allow_html=True)

        st.caption("⚠️ All detections are flagged as *potential* distress events requiring human verification.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <h2 style="color:white; font-weight:800; margin-bottom:4px">📊 Model Performance</h2>
    <p style="color:#6b7280; margin-bottom:24px">
      All metrics are computed on the held-out test set. No data fabrication.
    </p>
    """, unsafe_allow_html=True)

    for model_name, display_name in [("cnn", "Baseline CNN"), ("cnn_lstm", "CNN-LSTM (Temporal)")]:
        st.markdown(f'<div class="section-header">🔬 {display_name}</div>', unsafe_allow_html=True)
        metrics = load_metrics(model_name)
        render_metrics_card(metrics, model_name)

        # Show plots if available
        if metrics:
            col_cm, col_pr = st.columns(2)
            with col_cm:
                cm_path = METRICS_DIR / f"{model_name}_confusion_matrix.png"
                if cm_path.exists():
                    st.image(str(cm_path), caption="Confusion Matrix", use_column_width=True)
            with col_pr:
                pr_path = METRICS_DIR / f"{model_name}_pr_curve.png"
                if pr_path.exists():
                    st.image(str(pr_path), caption="Precision-Recall Curve", use_column_width=True)

            # Loss curves
            curve_path = METRICS_DIR / f"{model_name}_curves.png"
            if curve_path.exists():
                st.image(str(curve_path), caption="Training Curves", use_column_width=True)

            # Confusion matrix breakdown
            if metrics.get("confusion_matrix"):
                cm = metrics["confusion_matrix"]
                if len(cm) == 2:
                    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
                    cols = st.columns(4)
                    for col, (label, val, color) in zip(cols, [
                        ("True Negative",  tn, "#059669"),
                        ("False Positive", fp, "#dc2626"),
                        ("False Negative", fn, "#d97706"),
                        ("True Positive",  tp, "#6366f1"),
                    ]):
                        with col:
                            st.markdown(f"""
                            <div class="metric-card" style="text-align:center">
                              <div style="font-size:1.4rem;font-weight:800;color:{color}">{val}</div>
                              <div style="font-size:0.7rem;color:#6b7280">{label}</div>
                            </div>""", unsafe_allow_html=True)

        st.divider()

    # Comparison table
    st.markdown('<div class="section-header">📋 Side-by-Side Comparison</div>', unsafe_allow_html=True)
    cnn_m    = load_metrics("cnn")
    lstm_m   = load_metrics("cnn_lstm")

    if cnn_m and lstm_m:
        comp = pd.DataFrame({
            "Metric":    ["Accuracy", "Precision", "Recall", "F1", "FPR", "Latency (ms)"],
            "CNN":       [cnn_m["accuracy"],  cnn_m["precision"],  cnn_m["recall"],
                          cnn_m["f1"],         cnn_m["false_positive_rate"], cnn_m["avg_latency_ms"]],
            "CNN-LSTM":  [lstm_m["accuracy"], lstm_m["precision"], lstm_m["recall"],
                          lstm_m["f1"],        lstm_m["false_positive_rate"], lstm_m["avg_latency_ms"]],
        })
        st.dataframe(comp, use_container_width=True, hide_index=True)
    else:
        st.info("Run evaluation for both models to see comparison:\n"
                "```\npython training/evaluate.py --model cnn\n"
                "python training/evaluate.py --model cnn_lstm\n```")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Incident Log
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <h2 style="color:white; font-weight:800; margin-bottom:4px">🚨 Incident Log</h2>
    <p style="color:#6b7280; margin-bottom:24px">
      All alerts logged by the system. Each event requires human verification
      before any action is taken.
    </p>
    """, unsafe_allow_html=True)

    df_alerts = st.session_state.alert_logger.load_all()

    col_f1, col_f2, col_refresh = st.columns([2, 2, 1])
    with col_f1:
        sev_filter = st.multiselect("Filter by Severity", ["HIGH", "MEDIUM", "LOW"],
                                    default=["HIGH", "MEDIUM", "LOW"])
    with col_f2:
        status_filter = st.multiselect("Filter by Status", ["PENDING", "VERIFIED", "FALSE_ALARM"],
                                       default=["PENDING", "VERIFIED", "FALSE_ALARM"])
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            df_alerts = st.session_state.alert_logger.load_all()

    if not df_alerts.empty:
        mask = (df_alerts["severity"].isin(sev_filter) &
                df_alerts["verification_status"].isin(status_filter))
        df_filtered = df_alerts[mask]

        if not df_filtered.empty:
            # Severity bar chart
            sev_counts = df_filtered["severity"].value_counts().reset_index()
            sev_counts.columns = ["severity", "count"]
            color_map = {"HIGH": "#dc2626", "MEDIUM": "#d97706", "LOW": "#059669"}
            fig_bar = px.bar(
                sev_counts, x="severity", y="count", color="severity",
                color_discrete_map=color_map,
                title="Alert Distribution by Severity"
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9ca3af"), showlegend=False,
                title_font=dict(color="white"), margin=dict(t=40)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Table with colour-coded rows
            st.dataframe(
                df_filtered[["alert_id","timestamp_iso","location","confidence",
                             "severity","model_used","verification_status","note"]],
                use_container_width=True, height=300,
            )

            # CSV download
            csv_data = df_filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Alert Log (CSV)",
                data       = csv_data,
                file_name  = "alert_log_export.csv",
                mime       = "text/csv",
                use_container_width = True,
            )
        else:
            st.info("No alerts match the current filters.")
    else:
        st.info("No alerts logged yet. Upload and analyse an audio file in the **Live Detection** tab.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — About
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <h2 style="color:white; font-weight:800">ℹ️ About SafeWatch AI</h2>

    <div style="color:#d1d5db; line-height:1.8; max-width:700px">

    <h3 style="color:#a5b4fc">🎯 Project Overview</h3>
    <p>
    SafeWatch AI is a deep learning-based audio threat detection system that identifies
    potential distress events (screams) in real time using Mel spectrogram analysis.
    </p>

    <h3 style="color:#a5b4fc">🧠 Architecture</h3>
    <ul>
      <li><strong>Baseline CNN:</strong> 4 convolutional blocks on 128-band log-Mel spectrograms</li>
      <li><strong>CNN-LSTM:</strong> CNN feature extractor + 2-layer LSTM over consecutive windows</li>
      <li><strong>Smoothing:</strong> N consecutive windows above threshold before alert fires</li>
      <li><strong>Feature:</strong> 1-second windows, 50% overlap, 22,050 Hz sample rate</li>
    </ul>

    <h3 style="color:#a5b4fc">📚 Datasets</h3>
    <ul>
      <li>Screams: Kaggle / Hugging Face human scream audio collections</li>
      <li>Non-scream: UrbanSound8K (traffic, construction, music, speech)</li>
      <li>Augmentation: SpecAugment (time/freq masking), Gaussian noise</li>
    </ul>

    <h3 style="color:#a5b4fc">⚠️ Ethical Notice</h3>
    <p style="border-left:3px solid #dc2626; padding-left:12px">
    All detections are flagged as <strong>potential distress events requiring human verification</strong>.
    No personal identification is performed. This system is an assistive tool for security
    personnel, not an autonomous decision-making system. Results must be reviewed by
    qualified human operators before any action is taken.
    </p>

    <h3 style="color:#a5b4fc">👥 Team</h3>
    <p>Final Year Project — Deep Learning Human Threat Detection<br>
    <strong>Maharshith</strong> and team</p>
    </div>
    """, unsafe_allow_html=True)
