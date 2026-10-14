"""
dashboard/app.py  ·  SafeWatch AI  ·  Deep Learning Threat Detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOVABLE-INSPIRED DESIGN — neon-teal / crimson glassmorphism
Run:  streamlit run dashboard/app.py
"""

import sys, json, time, tempfile, math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from inference.alert_logger import AlertLogger
    from inference.predictor    import ScreamPredictor, extract_mel
    from inference.forensic_engine import AcousticForensicEngine, ForensicReport
    import librosa
    DEPS_OK = True
except ImportError as e:
    DEPS_OK = False
    _IMPORT_ERR = str(e)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SafeWatch AI — Smart Crime Detection & Monitoring System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  (Lovable-inspired: neon teal + crimson + deep slate)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
  font-family: 'Space Grotesk', sans-serif;
  color: #e2e8f0;
}
.stApp {
  background: radial-gradient(ellipse at 20% 10%, #0d1f2d 0%, #060e18 40%, #050c14 100%);
  min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #071525 0%, #050d1a 100%) !important;
  border-right: 1px solid rgba(0,229,204,0.1) !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── Tab bar ── */
[data-testid="stTabs"] > div:first-child {
  border-bottom: 1px solid rgba(0,229,204,0.15) !important;
  gap: 4px;
}
[data-testid="stTabs"] button {
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.04em !important;
  color: #64748b !important;
  border-radius: 8px 8px 0 0 !important;
  padding: 8px 18px !important;
  transition: all .2s !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: #00e5cc !important;
  background: rgba(0,229,204,0.07) !important;
  border-bottom: 2px solid #00e5cc !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #060e18; }
::-webkit-scrollbar-thumb { background: #00e5cc44; border-radius: 4px; }

/* ── Metric card ── */
.sw-card {
  background: linear-gradient(135deg,rgba(0,229,204,.06),rgba(0,229,204,.02));
  border: 1px solid rgba(0,229,204,.14);
  border-radius: 16px;
  padding: 18px 22px;
  backdrop-filter: blur(16px);
  transition: transform .2s, box-shadow .25s;
  position: relative;
  overflow: hidden;
}
.sw-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg,transparent,rgba(0,229,204,.4),transparent);
}
.sw-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 32px rgba(0,229,204,.12);
}

/* ── Alert card (pulsing red) ── */
@keyframes pulse-border {
  0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,.5); }
  50%      { box-shadow: 0 0 0 10px rgba(239,68,68,.0); }
}
.sw-alert-banner {
  background: linear-gradient(135deg,rgba(239,68,68,.15),rgba(239,68,68,.05));
  border: 1px solid rgba(239,68,68,.5);
  border-radius: 14px;
  padding: 16px 20px;
  animation: pulse-border 1.8s infinite;
  margin-bottom: 18px;
}

/* ── Section header ── */
.sw-section {
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #00e5cc;
  border-bottom: 1px solid rgba(0,229,204,.18);
  padding-bottom: 8px;
  margin-bottom: 18px;
  margin-top: 6px;
}

/* ── Badge ── */
.badge {
  display: inline-block;
  padding: 3px 11px;
  border-radius: 20px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .06em;
  text-transform: uppercase;
}
.badge-high   { background: rgba(239,68,68,.25);  color: #f87171; border:1px solid rgba(239,68,68,.4); }
.badge-medium { background: rgba(251,146,60,.2);  color: #fb923c; border:1px solid rgba(251,146,60,.4);}
.badge-low    { background: rgba(52,211,153,.2);  color: #34d399; border:1px solid rgba(52,211,153,.4);}
.badge-ok     { background: rgba(0,229,204,.15);  color: #00e5cc; border:1px solid rgba(0,229,204,.35);}

/* ── Logo text ── */
.sw-logo-title {
  font-size: 1.35rem; font-weight: 700;
  background: linear-gradient(135deg, #00e5cc, #22d3ee);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sw-logo-sub {
  font-size: 0.7rem; color: #475569; letter-spacing: .08em;
  text-transform: uppercase; margin-top: 3px;
}

/* ── Stat number ── */
.stat-big   { font-size: 1.9rem; font-weight: 700; color: #f8fafc; line-height: 1; }
.stat-label { font-size: 0.68rem; color: #64748b; margin-top: 5px; letter-spacing:.05em; }

/* ── Code mono ── */
.mono { font-family: 'JetBrains Mono', monospace; }

/* ── Divider ── */
.sw-divider {
  height: 1px;
  background: linear-gradient(90deg,transparent,rgba(0,229,204,.2),transparent);
  margin: 20px 0;
}

/* ── Status dot ── */
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }
.dot-live  { display:inline-block;width:8px;height:8px;border-radius:50%;background:#34d399;animation:blink 1.4s infinite;margin-right:6px; }
.dot-idle  { display:inline-block;width:8px;height:8px;border-radius:50%;background:#475569;margin-right:6px; }
.dot-alert { display:inline-block;width:8px;height:8px;border-radius:50%;background:#ef4444;animation:blink .8s infinite;margin-right:6px; }

/* ── Hero banner ── */
.hero {
  background: linear-gradient(135deg,rgba(0,229,204,.08),rgba(34,211,238,.04));
  border: 1px solid rgba(0,229,204,.12);
  border-radius: 20px;
  padding: 28px 32px;
  margin-bottom: 28px;
  position: relative;
  overflow: hidden;
}
.hero::after {
  content:'';
  position:absolute;
  right:-60px; top:-60px;
  width:200px; height:200px;
  background: radial-gradient(circle, rgba(0,229,204,.12) 0%, transparent 70%);
  pointer-events:none;
}

/* ── Streamlit overrides ── */
.stSlider > div > div > div { background: rgba(0,229,204,.3) !important; }
.stSlider > div > div > div > div { background: #00e5cc !important; }
div[data-testid="stSelectbox"] > div { background: rgba(0,229,204,.05) !important;
  border: 1px solid rgba(0,229,204,.2) !important; border-radius: 10px !important; }
div[data-testid="stTextInput"] > div { background: rgba(0,229,204,.05) !important;
  border: 1px solid rgba(0,229,204,.2) !important; border-radius: 10px !important; }
.stButton > button {
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important;
  border-radius: 10px !important;
  transition: all .2s !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #00e5cc, #22d3ee) !important;
  color: #050c14 !important;
  border: none !important;
}
.stButton > button[kind="primary"]:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 24px rgba(0,229,204,.4) !important;
}
.stFileUploader > div {
  background: rgba(0,229,204,.04) !important;
  border: 2px dashed rgba(0,229,204,.25) !important;
  border-radius: 14px !important;
}
.stDataFrame { border-radius: 12px !important; }

</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
METRICS_DIR    = ROOT / "outputs" / "metrics"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
SAMPLE_RATE    = 22050
WINDOW_SEC     = 1.0
HOP_SEC        = 0.5

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
_avail_model = "cnn"
if (CHECKPOINT_DIR / "cnn_lstm_best.pth").exists():
    _avail_model = "cnn_lstm"
elif (CHECKPOINT_DIR / "cnn_best.pth").exists():
    _avail_model = "cnn"

_defaults = dict(
    predictor       = None,
    forensic_engine = None,
    forensic_report = None,
    results         = [],
    alert_logger    = None,
    model_type      = _avail_model,
    threshold       = 0.50,
    consec          = 2,
    location        = "Node 01 — North Alleyway CCTV-04",
    dispatch_msg    = None,
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.alert_logger is None and DEPS_OK:
    st.session_state.alert_logger = AlertLogger()

if st.session_state.forensic_engine is None and DEPS_OK:
    try:
        st.session_state.forensic_engine = AcousticForensicEngine()
    except Exception:
        st.session_state.forensic_engine = None


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_predictor():
    if not DEPS_OK:
        return False
    try:
        st.session_state.predictor = ScreamPredictor(
            model_type          = st.session_state.model_type,
            threshold           = st.session_state.threshold,
            consecutive_windows = st.session_state.consec,
            location_label      = st.session_state.location,
        )
        return True
    except Exception as e:
        st.error(f"❌ Could not load model: {e}\n\nRun training first:  `python training/train.py --model cnn_lstm`")
        return False

# Auto-initialize predictor on first load if checkpoint exists
if st.session_state.predictor is None and DEPS_OK:
    load_predictor()


def load_metrics(model_type: str) -> dict | None:
    path = METRICS_DIR / f"{model_type}_test_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def prob_color(p: float) -> str:
    if p >= 0.75: return "#ef4444"
    if p >= 0.50: return "#fb923c"
    return "#34d399"


def pct(v): return f"{v*100:.1f}%"


# ──────────────────────────────────────────────────────────────────────────────
# PLOTLY FIGURES
# ──────────────────────────────────────────────────────────────────────────────

PLOT_BG = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
AXIS_STYLE = dict(color="#475569", gridcolor="rgba(0,229,204,0.06)", zeroline=False)


def make_gauge(value: float) -> go.Figure:
    color = prob_color(value)
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = value * 100,
        number= {"suffix": "%", "font": {"size": 32, "color": "white", "family": "Space Grotesk"}},
        title = {"text": "THREAT SCORE", "font": {"size": 10, "color": "#00e5cc",
                 "family": "Space Grotesk"}, },
        gauge = {
            "axis":  {"range": [0, 100], "tickcolor": "#1e293b",
                      "tickfont": {"color": "#475569", "size": 9}},
            "bar":   {"color": color, "thickness": 0.22},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  50], "color": "rgba(52,211,153,0.08)"},
                {"range": [50, 75], "color": "rgba(251,146,60,0.08)"},
                {"range": [75,100], "color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {
                "line":  {"color": "#f59e0b", "width": 2},
                "thickness": 0.75,
                "value": st.session_state.threshold * 100,
            },
        }
    ))
    fig.update_layout(**PLOT_BG, height=200, margin=dict(t=28, b=0, l=10, r=10))
    return fig


def make_timeline(results: list) -> go.Figure:
    if not results:
        fig = go.Figure()
        fig.update_layout(**PLOT_BG, height=260,
                          xaxis={**AXIS_STYLE, "title":""},
                          yaxis={**AXIS_STYLE, "title":""},
                          annotations=[{"text":"Upload an audio file to begin analysis",
                                        "xref":"paper","yref":"paper","x":.5,"y":.5,
                                        "showarrow":False,"font":{"color":"#334155","size":14}}])
        return fig

    ts     = [r.timestamp_sec for r in results]
    probs  = [r.scream_prob   for r in results]
    alerts = [r.is_alert      for r in results]
    colors = [prob_color(p)   for p in probs]

    fig = go.Figure()

    # Alert shading
    for i, a in enumerate(alerts):
        if a:
            fig.add_vrect(x0=ts[i]-HOP_SEC/2, x1=ts[i]+HOP_SEC/2,
                          fillcolor="rgba(239,68,68,0.12)", line_width=0, layer="below")

    # Threshold
    fig.add_hline(y=st.session_state.threshold, line_dash="dot",
                  line_color="#f59e0b", line_width=1.5,
                  annotation_text=f"Threshold {st.session_state.threshold:.2f}",
                  annotation_font_color="#f59e0b", annotation_font_size=10)

    # Main prob trace
    fig.add_trace(go.Scatter(
        x=ts, y=probs, mode="lines+markers", name="P(scream)",
        line=dict(color="#00e5cc", width=2.5, shape="spline"),
        marker=dict(size=5, color=colors, line=dict(color="rgba(0,0,0,0.3)", width=1)),
        fill="tozeroy", fillcolor="rgba(0,229,204,0.05)",
        hovertemplate="<b>t=%{x:.2f}s</b><br>P(scream)=%{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        **PLOT_BG, height=260,
        margin=dict(t=10, b=40, l=52, r=12),
        xaxis={**AXIS_STYLE, "title": "Time (seconds)"},
        yaxis={**AXIS_STYLE, "title": "P(scream)", "range": [0, 1]},
        legend=dict(font=dict(color="#64748b", size=10), bgcolor="rgba(0,0,0,0)"),
        font=dict(color="#64748b", family="Space Grotesk"),
    )
    return fig


def make_comparison_radar(cnn_m: dict, lstm_m: dict) -> go.Figure:
    cats = ["Accuracy", "Precision", "Recall", "F1", "Low FPR"]
    def _vals(m):
        return [m["accuracy"], m["precision"], m["recall"],
                m["f1"], 1 - m["false_positive_rate"]]

    fig = go.Figure()
    for m, name, color in [(cnn_m,"CNN","#22d3ee"), (lstm_m,"CNN-LSTM","#00e5cc")]:
        v = _vals(m)
        fig.add_trace(go.Scatterpolar(
            r=v + [v[0]], theta=cats + [cats[0]], name=name,
            line=dict(color=color, width=2),
            fill="toself", fillcolor=color.replace("#","rgba(").replace("e","e,")+",.08)" if False else f"{color}18",
        ))

    fig.update_layout(
        **PLOT_BG, height=300,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], color="#334155",
                            gridcolor="rgba(0,229,204,0.1)", tickfont=dict(size=8)),
            angularaxis=dict(color="#64748b", gridcolor="rgba(0,229,204,0.1)"),
        ),
        legend=dict(font=dict(color="#64748b"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=20, b=20, l=50, r=50),
        font=dict(color="#64748b", family="Space Grotesk"),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 0 16px">
      <div style="display:flex;align-items:center;gap:10px">
        <div style="font-size:2rem">🛡️</div>
        <div>
          <div class="sw-logo-title">SafeWatch AI</div>
          <div class="sw-logo-sub">Threat Detection System</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Status indicator
    loaded = st.session_state.predictor is not None
    dot    = "dot-live" if loaded else "dot-idle"
    status = "Model Active" if loaded else "No Model Loaded"
    st.markdown(f'<div style="font-size:.75rem;margin-bottom:12px">'
                f'<span class="{dot}"></span><span style="color:#64748b">{status}</span></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="sw-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-section">⚙️ Model Config</div>', unsafe_allow_html=True)

    model_type = st.selectbox(
        "Architecture",
        options=["cnn_lstm", "cnn"],
        format_func=lambda x: "CNN-LSTM (Temporal)" if x == "cnn_lstm" else "CNN (Baseline)",
        key="model_type",
    )

    location = st.text_input("📍 Location Label", key="location")

    threshold = st.slider("Alert Threshold", 0.30, 0.95, key="threshold", step=0.05,
                          help="Minimum P(scream) to trigger alert")

    consec = st.slider("Consecutive Windows", 1, 8, key="consec",
                       help="Windows above threshold before alert fires")

    if st.button("⚡ Load / Reload Model", use_container_width=True, type="primary"):
        with st.spinner("Initialising model checkpoint…"):
            ok = load_predictor()
        if ok:
            st.success("✅ Model loaded successfully!")

    if not DEPS_OK:
        st.warning(f"Missing dependency: {_IMPORT_ERR}\n\nRun:  `pip install -r requirements.txt`")

    st.markdown('<div class="sw-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-section">📊 Session Stats</div>', unsafe_allow_html=True)

    # Stats grid
    al = st.session_state.alert_logger
    stats = al.stats() if al else {"total": 0, "high": 0, "medium": 0, "pending": 0}

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
      <div class="sw-card" style="padding:12px 14px;text-align:center">
        <div class="stat-big">{stats['total']}</div>
        <div class="stat-label">TOTAL ALERTS</div>
      </div>
      <div class="sw-card" style="padding:12px 14px;text-align:center">
        <div class="stat-big" style="color:#ef4444">{stats['high']}</div>
        <div class="stat-label">HIGH SEVERITY</div>
      </div>
      <div class="sw-card" style="padding:12px 14px;text-align:center">
        <div class="stat-big" style="color:#fb923c">{stats['medium']}</div>
        <div class="stat-label">MEDIUM</div>
      </div>
      <div class="sw-card" style="padding:12px 14px;text-align:center">
        <div class="stat-big" style="color:#00e5cc">{stats['pending']}</div>
        <div class="stat-label">PENDING</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sw-divider"></div>
    <div style="font-size:.65rem;color:#334155;text-align:center;line-height:1.6">
      All detections are <em>potential distress events</em><br>requiring human verification.
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab_detect, tab_perf, tab_log, tab_about = st.tabs([
    "🛡️  Crime Surveillance & Detect",
    "📈  Model Proofs & Metrics",
    "🚨  Incident Command Center",
    "ℹ️   System Architecture",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CRIME SURVEILLANCE & DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_detect:

    # Hero
    st.markdown("""
    <div class="hero">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">
        <span style="font-size:1.6rem">🛡️</span>
        <span style="font-size:1.25rem;font-weight:700;color:#f8fafc">Smart Crime Detection & Monitoring System</span>
      </div>
      <div style="color:#64748b;font-size:.85rem;max-width:700px">
        Continuous acoustic surveillance detecting human screams and violent distress events in real-time.
        Powered by deep convolutional neural networks and spatial-temporal LSTM architectures.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Surveillance Node Status Grid
    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px">
      <div class="sw-card" style="padding:12px 14px;border-left:3px solid #00e5cc">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span style="font-size:0.75rem;font-weight:700;color:#f8fafc">NODE 01: North Alleyway</span>
          <span class="badge badge-high" style="font-size:8px">ACTIVE SENSOR</span>
        </div>
        <div style="font-size:0.68rem;color:#64748b;margin-top:4px">CCTV-04 · Acoustic Mic #1</div>
      </div>
      <div class="sw-card" style="padding:12px 14px;border-left:3px solid #34d399">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span style="font-size:0.75rem;font-weight:700;color:#f8fafc">NODE 02: Parking Bay B</span>
          <span class="badge badge-low" style="font-size:8px">SECURE</span>
        </div>
        <div style="font-size:0.68rem;color:#64748b;margin-top:4px">Underground Level -1</div>
      </div>
      <div class="sw-card" style="padding:12px 14px;border-left:3px solid #34d399">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span style="font-size:0.75rem;font-weight:700;color:#f8fafc">NODE 03: Transit West</span>
          <span class="badge badge-low" style="font-size:8px">SECURE</span>
        </div>
        <div style="font-size:0.68rem;color:#64748b;margin-top:4px">Concourse Platform A</div>
      </div>
      <div class="sw-card" style="padding:12px 14px;border-left:3px solid #34d399">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span style="font-size:0.75rem;font-weight:700;color:#f8fafc">NODE 04: ATM Perimeter</span>
          <span class="badge badge-low" style="font-size:8px">SECURE</span>
        </div>
        <div style="font-size:0.68rem;color:#64748b;margin-top:4px">Vault Corridor South</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not DEPS_OK:
        st.error(f"⚠️ Cannot run detection — dependency missing: `{_IMPORT_ERR}`\n\n"
                 "```\npip install -r requirements.txt\n```")
        st.stop()

    if st.session_state.predictor is None:
        load_predictor()

    col_upload, col_gauge = st.columns([3, 1], gap="large")

    with col_upload:
        st.markdown('<div class="sw-section">🎙️ Surveillance Audio Feed & Scenarios</div>', unsafe_allow_html=True)

        input_mode = st.radio(
            "Input Mode",
            ["⚡ Pre-Recorded Surveillance Audio Scenarios (1-Click Test)", "📁 Upload Custom Audio File"],
            horizontal=True,
            label_visibility="collapsed",
        )

        audio_to_process = None
        audio_name = None

        if input_mode.startswith("⚡"):
            sample_options = {
                "🚨 Active Crime / Distress Scream (Scenario #1)": "data/raw/scream/---1_cCGK4M_out.wav",
                "🚨 Active Crime / Distress Scream (Scenario #2)": "data/raw/scream/-20uudT97E0_out.wav",
                "🚗 Vehicle Horn / Bus Honk (Acoustic Outlier)": "data/test_samples/vehicle_horn_outlier.wav",
                "🚨 Emergency Vehicle Siren (Stationary Harmonic)": "data/test_samples/siren_outlier.wav",
                "🚙 High-Friction Tire Screech (Outlier)": "data/test_samples/tire_screech_outlier.wav",
                "🟢 Normal Urban Alleyway Ambience (No Threat)": "data/raw/non_scream/--PJHxphWEs_out.wav",
                "🟢 Normal Street Activity & Footsteps (No Threat)": "data/raw/non_scream/-28U1_qW0sU_out.wav",
            }
            selected_scenario = st.selectbox("Select Test Scenario:", list(sample_options.keys()))
            sample_rel = sample_options[selected_scenario]
            sample_path = ROOT / sample_rel

            if sample_path.exists():
                st.audio(str(sample_path), format="audio/wav")
                c_btn, _ = st.columns([1, 1])
                with c_btn:
                    if st.button("▶️ Run Crime Threat Analysis", type="primary", use_container_width=True):
                        audio_to_process = str(sample_path)
                        audio_name = selected_scenario
            else:
                st.warning(f"Audio file not found at {sample_path}")
        else:
            uploaded = st.file_uploader(
                "Drop an audio file (.wav, .mp3, .flac, .ogg)",
                type=["wav", "mp3", "flac", "ogg"],
                label_visibility="collapsed",
            )
            if uploaded:
                suffix = Path(uploaded.name).suffix
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded.read())
                    audio_to_process = tmp.name
                    audio_name = uploaded.name
                st.audio(uploaded)

        if audio_to_process and st.session_state.predictor:
            with st.spinner(f"🔍 Analyzing Acoustic Feed [{audio_name}] …"):
                results = st.session_state.predictor.process_file(audio_to_process)
                st.session_state.results = results
                st.session_state.dispatch_msg = None

                # Run Bio-Acoustic Forensic Outlier Analysis
                if st.session_state.forensic_engine:
                    max_p = max((r.scream_prob for r in results), default=0.0)
                    with st.spinner("🤖 Evaluating Bio-Acoustic Physics & Outliers via Gemini Forensic AI …"):
                        st.session_state.forensic_report = st.session_state.forensic_engine.analyze(
                            audio_to_process,
                            cnn_prob=max_p,
                            location=st.session_state.location
                        )

        if st.session_state.results:
            results   = st.session_state.results
            n_alerts  = sum(1 for r in results if r.is_alert)
            max_prob  = max((r.scream_prob for r in results), default=0.0)
            dur_s     = results[-1].timestamp_sec if results else 0

            # ── Banner
            if n_alerts > 0:
                st.markdown(f"""
                <div class="sw-alert-banner">
                  <span class="dot-alert"></span>
                  <span style="color:#f87171;font-weight:700;font-size:1rem">
                    ⚠️ {n_alerts} VIOLENT CRIME / DISTRESS ALERT(S) CONFIRMED
                  </span>
                  <div style="color:#94a3b8;font-size:.78rem;margin-top:4px">
                    Peak Threat Probability: {max_prob:.1%} · Duration: {dur_s:.1f}s ·
                    Location: {st.session_state.location} · Immediate Security Verification Required.
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:rgba(52,211,153,.08);border:1px solid rgba(52,211,153,.25);
                     border-radius:12px;padding:14px 18px;margin-bottom:16px">
                  <span style="color:#34d399;font-weight:700">✅ Zone Secure — No Criminal Distress Detected</span>
                  <span style="color:#64748b;font-size:.82rem;margin-left:10px">
                    {len(results)} windows analyzed · Max P(threat) = {max_prob:.3f}
                  </span>
                </div>""", unsafe_allow_html=True)

            # ── Summary cards
            c1, c2, c3, c4 = st.columns(4)
            for col, (label, val, color, icon) in zip(
                [c1, c2, c3, c4], [
                    ("Acoustic Windows", len(results),      "#e2e8f0", "📦"),
                    ("Crime Alerts",     n_alerts,           "#ef4444", "🚨"),
                    ("Peak P(distress)", f"{max_prob:.3f}",  "#fb923c", "📊"),
                    ("Duration",         f"{dur_s:.1f}s",    "#00e5cc", "⏱️"),
                ]
            ):
                with col:
                    st.markdown(f"""
                    <div class="sw-card" style="text-align:center;padding:14px">
                      <div style="font-size:1.1rem">{icon}</div>
                      <div style="font-size:1.4rem;font-weight:700;color:{color}">{val}</div>
                      <div class="stat-label">{label}</div>
                    </div>""", unsafe_allow_html=True)

            # ── AI Forensic Intelligence Dossier & Outlier Filter
            if st.session_state.get("forensic_report"):
                rep = st.session_state.forensic_report
                fp = rep.fingerprint or {}
                
                # Determine styling
                if rep.classification == "GENUINE_HUMAN_SCREAM":
                    border_color = "rgba(239, 68, 68, 0.45)"
                    header_badge = ('<span class="badge badge-high" style="font-size:0.8rem;padding:4px 10px">'
                                    '🚨 VERIFIED THREAT: HUMAN VOCAL DISTRESS</span>')
                    verdict_icon = "⚠️"
                elif "HORN" in rep.classification:
                    border_color = "rgba(0, 229, 204, 0.45)"
                    header_badge = ('<span class="badge badge-ok" style="font-size:0.8rem;padding:4px 10px">'
                                    '🛡️ OUTLIER FILTERED: VEHICLE HORN / BUS HONK</span>')
                    verdict_icon = "🚗"
                elif "SCREECH" in rep.classification:
                    border_color = "rgba(56, 189, 248, 0.45)"
                    header_badge = ('<span class="badge badge-ok" style="font-size:0.8rem;padding:4px 10px">'
                                    '🛡️ OUTLIER FILTERED: TIRE SCREECH / FRICTION</span>')
                    verdict_icon = "🚙"
                elif "SIREN" in rep.classification:
                    border_color = "rgba(251, 146, 60, 0.45)"
                    header_badge = ('<span class="badge badge-medium" style="font-size:0.8rem;padding:4px 10px">'
                                    '🛡️ OUTLIER FILTERED: EMERGENCY SIREN</span>')
                    verdict_icon = "🚨"
                else:
                    border_color = "rgba(52, 211, 153, 0.35)"
                    header_badge = ('<span class="badge badge-low" style="font-size:0.8rem;padding:4px 10px">'
                                    '✅ NORMAL AMBIENT: CLEAR</span>')
                    verdict_icon = "🌿"

                engine_badge = ("🤖 Gemini 2.5 Flash Agent" if getattr(st.session_state.forensic_engine, "client", None)
                                else "⚡ Local Bio-Acoustic Physics Engine")

                st.markdown(f"""
                <div class="sw-card" style="margin-top:20px;border:1px solid {border_color};background:rgba(7, 21, 37, 0.75)">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;border-bottom:1px solid rgba(0,229,204,0.12);padding-bottom:10px">
                    <div style="display:flex;align-items:center;gap:10px">
                      <span style="font-size:1.2rem">{verdict_icon}</span>
                      <span style="font-weight:700;font-size:1.05rem;color:#f8fafc">AI Acoustic Forensic Dossier & Outlier Filter</span>
                    </div>
                    <div>
                      {header_badge}
                      <span style="font-size:0.68rem;color:#64748b;margin-left:8px;background:rgba(255,255,255,0.05);padding:3px 8px;border-radius:6px">{engine_badge}</span>
                    </div>
                  </div>
                  
                  <div style="background:rgba(0,0,0,0.25);border-radius:10px;padding:12px 16px;margin-bottom:14px;border-left:3px solid {border_color}">
                    <div style="font-size:0.75rem;font-weight:700;color:#cbd5e1;margin-bottom:4px">FORENSIC RATIONALE (Acoustic Physics & Non-Linear Dynamics)</div>
                    <div style="font-size:0.84rem;color:#e2e8f0;line-height:1.5">{rep.forensic_rationale}</div>
                  </div>

                  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px">
                    <div style="background:rgba(0,229,204,0.04);border:1px solid rgba(0,229,204,0.1);border-radius:8px;padding:10px;text-align:center">
                      <div style="font-size:0.68rem;color:#64748b">FUNDAMENTAL F0</div>
                      <div style="font-size:1.05rem;font-weight:700;color:#00e5cc;font-family:'JetBrains Mono'">{fp.get('f0_mean_hz', 0)} Hz</div>
                      <div style="font-size:0.62rem;color:#475569">Pitch Baseline</div>
                    </div>
                    <div style="background:rgba(0,229,204,0.04);border:1px solid rgba(0,229,204,0.1);border-radius:8px;padding:10px;text-align:center">
                      <div style="font-size:0.68rem;color:#64748b">PITCH MOD (JITTER)</div>
                      <div style="font-size:1.05rem;font-weight:700;color:{'#ef4444' if fp.get('f0_modulation_pct',0)>8 else '#34d399'};font-family:'JetBrains Mono'">{fp.get('f0_modulation_pct', 0)}%</div>
                      <div style="font-size:0.62rem;color:#475569">{'>8% = Vocal Chaos' if fp.get('f0_modulation_pct',0)>8 else '<7% = Stationary Tone'}</div>
                    </div>
                    <div style="background:rgba(0,229,204,0.04);border:1px solid rgba(0,229,204,0.1);border-radius:8px;padding:10px;text-align:center">
                      <div style="font-size:0.68rem;color:#64748b">HARMONIC/NOISE (HNR)</div>
                      <div style="font-size:1.05rem;font-weight:700;color:{'#34d399' if fp.get('hnr_db',0)>=18 else '#fb923c'};font-family:'JetBrains Mono'">{fp.get('hnr_db', 0)} dB</div>
                      <div style="font-size:0.62rem;color:#475569">{'>18dB = Machine Horn' if fp.get('hnr_db',0)>=18 else '<10dB = Vocal Turbulence'}</div>
                    </div>
                    <div style="background:rgba(0,229,204,0.04);border:1px solid rgba(0,229,204,0.1);border-radius:8px;padding:10px;text-align:center">
                      <div style="font-size:0.68rem;color:#64748b">CHAOS INDEX</div>
                      <div style="font-size:1.05rem;font-weight:700;color:{'#ef4444' if fp.get('chaos_index',0)>=0.3 else '#00e5cc'};font-family:'JetBrains Mono'">{fp.get('chaos_index', 0)}</div>
                      <div style="font-size:0.62rem;color:#475569">Vocal Bifurcation Metric</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Incident Response Actions
            st.markdown('<div class="sw-section" style="margin-top:20px">🚔 Incident Response Actions</div>', unsafe_allow_html=True)
            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            with col_b1:
                if st.button("🚨 Dispatch Patrol", use_container_width=True, type="primary"):
                    st.session_state.dispatch_msg = "🚔 SECURITY PATROL DISPATCHED to Location! Unit #402 en route."
            with col_b2:
                if st.button("📞 Emergency 911 / 112", use_container_width=True):
                    st.session_state.dispatch_msg = "📞 LAW ENFORCEMENT DISPATCH CODE TRANSMITTED: #EMERGENCY-AC-911"
            with col_b3:
                if st.button("✅ Verify Threat", use_container_width=True):
                    st.session_state.dispatch_msg = "✅ Incident verified by operator and logged to evidentiary record."
            with col_b4:
                if st.button("❌ Dismiss Alarm", use_container_width=True):
                    st.session_state.dispatch_msg = "❌ Event flagged as false alarm. Baseline ambient profile updated."

            if st.session_state.get("dispatch_msg"):
                st.success(st.session_state.dispatch_msg)

        # ── Timeline
        st.markdown('<div class="sw-section" style="margin-top:20px">📈 Detection Timeline</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(make_timeline(st.session_state.results), use_container_width=True)

        # ── Raw table
        if st.session_state.results:
            with st.expander("🔢 Raw Window Scores"):
                rows = [{"t (s)": f"{r.timestamp_sec:.2f}",
                         "P(scream)": f"{r.scream_prob:.4f}",
                         "Alert": "🚨" if r.is_alert else "—",
                         "Label": r.label}
                        for r in st.session_state.results]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, height=200)

    with col_gauge:
        st.markdown('<div class="sw-section">🌡️ Threat Level</div>', unsafe_allow_html=True)
        res = st.session_state.results
        peak = max((r.scream_prob for r in res), default=0.0)
        st.plotly_chart(make_gauge(peak), use_container_width=True)

        # Status pill
        if peak >= 0.75:
            pill_c, pill_t = "#ef4444", "CRITICAL"
        elif peak >= 0.50:
            pill_c, pill_t = "#fb923c", "WARNING"
        elif peak > 0:
            pill_c, pill_t = "#34d399", "CLEAR"
        else:
            pill_c, pill_t = "#334155", "IDLE"

        st.markdown(f"""
        <div style="text-align:center;margin-top:6px">
          <span style="background:{pill_c}22;border:1px solid {pill_c}66;
                color:{pill_c};font-size:.7rem;font-weight:700;
                padding:5px 16px;border-radius:20px;letter-spacing:.08em">
            {pill_t}
          </span>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:18px;display:grid;gap:8px">
          <div class="sw-card" style="padding:10px 14px">
            <div class="stat-label">MODEL</div>
            <div style="color:#00e5cc;font-weight:600;font-size:.85rem;margin-top:3px">
              {"CNN-LSTM" if st.session_state.model_type=="cnn_lstm" else "Baseline CNN"}
            </div>
          </div>
          <div class="sw-card" style="padding:10px 14px">
            <div class="stat-label">THRESHOLD</div>
            <div style="color:#f8fafc;font-weight:600;font-size:.85rem;margin-top:3px">
              {st.session_state.threshold:.2f}
            </div>
          </div>
          <div class="sw-card" style="padding:10px 14px">
            <div class="stat-label">LOCATION</div>
            <div style="color:#f8fafc;font-weight:600;font-size:.78rem;margin-top:3px">
              {st.session_state.location}
            </div>
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.markdown("""
    <div class="hero">
      <div style="font-size:1.2rem;font-weight:700;color:#f8fafc;margin-bottom:4px">
        📈 Model Performance Dashboard
      </div>
      <div style="color:#64748b;font-size:.83rem">
        All metrics computed on the held-out test set. No figures are fabricated.
        Run <code style="background:rgba(0,229,204,.1);padding:1px 6px;border-radius:4px">
        python training/evaluate.py --model cnn_lstm</code> to populate.
      </div>
    </div>
    """, unsafe_allow_html=True)

    cnn_m  = load_metrics("cnn")
    lstm_m = load_metrics("cnn_lstm")

    def render_model_block(m: dict | None, model_key: str, display: str):
        st.markdown(f'<div class="sw-section">🔬 {display}</div>', unsafe_allow_html=True)
        if m is None:
            st.info(f"No test results for **{display}** yet.  "
                    f"Run: `python training/evaluate.py --model {model_key}`")
            return

        cols = st.columns(6)
        items = [
            ("Accuracy",  pct(m["accuracy"]),                   "#00e5cc", "📊"),
            ("Precision", pct(m["precision"]),                  "#22d3ee", "🎯"),
            ("Recall",    pct(m["recall"]),                     "#38bdf8", "🔍"),
            ("F1 Score",  pct(m["f1"]),                         "#818cf8", "⚖️"),
            ("FPR",       pct(m["false_positive_rate"]),        "#f87171", "⚠️"),
            ("Latency",   f"{m['avg_latency_ms']:.1f}ms",       "#fb923c", "⚡"),
        ]
        for col, (lbl, val, color, icon) in zip(cols, items):
            with col:
                st.markdown(f"""
                <div class="sw-card" style="text-align:center;padding:14px 10px">
                  <div style="font-size:1.1rem">{icon}</div>
                  <div style="font-size:1.35rem;font-weight:700;color:{color};margin-top:4px">{val}</div>
                  <div class="stat-label">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        # Plots
        col_cm, col_pr = st.columns(2)
        with col_cm:
            cm_path = METRICS_DIR / f"{model_key}_confusion_matrix.png"
            if cm_path.exists():
                st.image(str(cm_path), caption="Confusion Matrix", use_column_width=True)
        with col_pr:
            pr_path = METRICS_DIR / f"{model_key}_pr_curve.png"
            if pr_path.exists():
                st.image(str(pr_path), caption="Precision-Recall Curve", use_column_width=True)

        curve_path = METRICS_DIR / f"{model_key}_curves.png"
        if curve_path.exists():
            st.image(str(curve_path), caption="Training Loss & Accuracy", use_column_width=True)

        if m.get("confusion_matrix"):
            cm = m["confusion_matrix"]
            if len(cm) == 2:
                tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
                c1, c2, c3, c4 = st.columns(4)
                for col, (lbl, val, color) in zip([c1,c2,c3,c4], [
                    ("True Negative",  tn, "#34d399"),
                    ("False Positive", fp, "#ef4444"),
                    ("False Negative", fn, "#fb923c"),
                    ("True Positive",  tp, "#818cf8"),
                ]):
                    with col:
                        st.markdown(f"""
                        <div class="sw-card" style="text-align:center;padding:12px">
                          <div style="font-size:1.5rem;font-weight:700;color:{color}">{val}</div>
                          <div class="stat-label">{lbl}</div>
                        </div>""", unsafe_allow_html=True)

        if m.get("note"):
            st.caption(f"ℹ️ {m['note']}")

    render_model_block(cnn_m,  "cnn",      "Baseline CNN")
    st.markdown('<div class="sw-divider"></div>', unsafe_allow_html=True)
    render_model_block(lstm_m, "cnn_lstm", "CNN-LSTM (Temporal)")

    # ── Radar comparison
    if cnn_m and lstm_m:
        st.markdown('<div class="sw-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sw-section">📊 Head-to-Head Comparison</div>', unsafe_allow_html=True)

        col_rad, col_tbl = st.columns([1, 1], gap="large")
        with col_rad:
            st.plotly_chart(make_comparison_radar(cnn_m, lstm_m), use_container_width=True)
        with col_tbl:
            comp = pd.DataFrame({
                "Metric":   ["Accuracy", "Precision", "Recall", "F1", "FPR", "Latency (ms)"],
                "CNN":      [pct(cnn_m["accuracy"]),  pct(cnn_m["precision"]),
                             pct(cnn_m["recall"]),    pct(cnn_m["f1"]),
                             pct(cnn_m["false_positive_rate"]), f"{cnn_m['avg_latency_ms']:.1f}"],
                "CNN-LSTM": [pct(lstm_m["accuracy"]), pct(lstm_m["precision"]),
                             pct(lstm_m["recall"]),   pct(lstm_m["f1"]),
                             pct(lstm_m["false_positive_rate"]), f"{lstm_m['avg_latency_ms']:.1f}"],
            })
            st.dataframe(comp, use_container_width=True, hide_index=True, height=250)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — INCIDENTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_log:
    st.markdown("""
    <div class="hero">
      <div style="font-size:1.2rem;font-weight:700;color:#f8fafc;margin-bottom:4px">
        🚨 Incident Command Center
      </div>
      <div style="color:#64748b;font-size:.83rem">
        All logged alerts require human verification before any action is taken.
      </div>
    </div>
    """, unsafe_allow_html=True)

    al = st.session_state.alert_logger
    df_alerts = al.load_all() if al else pd.DataFrame()

    col_f1, col_f2, col_refresh = st.columns([2, 2, 1])
    with col_f1:
        sev_filter = st.multiselect(
            "Severity", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"])
    with col_f2:
        status_filter = st.multiselect(
            "Status", ["PENDING", "VERIFIED", "FALSE_ALARM"],
            default=["PENDING", "VERIFIED", "FALSE_ALARM"])
    with col_refresh:
        if st.button("🔄", use_container_width=True, help="Refresh alerts"):
            df_alerts = al.load_all() if al else pd.DataFrame()

    if not df_alerts.empty:
        mask = (df_alerts["severity"].isin(sev_filter) &
                df_alerts["verification_status"].isin(status_filter))
        df_f = df_alerts[mask]

        if not df_f.empty:
            # Severity bar
            counts = df_f["severity"].value_counts().reset_index()
            counts.columns = ["severity", "count"]
            cmap = {"HIGH": "#ef4444", "MEDIUM": "#fb923c", "LOW": "#34d399"}
            fig_bar = px.bar(counts, x="severity", y="count", color="severity",
                             color_discrete_map=cmap)
            fig_bar.update_layout(**PLOT_BG, height=200, showlegend=False,
                                  margin=dict(t=10,b=30,l=40,r=10),
                                  xaxis={**AXIS_STYLE}, yaxis={**AXIS_STYLE},
                                  font=dict(color="#64748b",family="Space Grotesk"))
            fig_bar.update_traces(marker_line_width=0)
            st.plotly_chart(fig_bar, use_container_width=True)

            display_cols = [c for c in
                            ["alert_id","timestamp_iso","location","confidence",
                             "severity","model_used","verification_status","note"]
                            if c in df_f.columns]
            st.dataframe(df_f[display_cols], use_container_width=True, height=280)

            csv_data = df_f.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Export CSV", data=csv_data,
                               file_name="safewatch_alerts.csv", mime="text/csv",
                               use_container_width=True)
        else:
            st.info("No alerts match the selected filters.")
    else:
        st.info("No incidents logged yet. Upload and analyse audio in the **Detect** tab.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    col_a, col_b = st.columns([3, 2], gap="large")

    with col_a:
        st.markdown("""
        <div style="color:#f8fafc">

        <h2 style="font-weight:700;background:linear-gradient(135deg,#00e5cc,#22d3ee);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px">
          SafeWatch AI
        </h2>
        <div style="color:#64748b;font-size:.85rem;margin-bottom:24px">
          Deep Learning Human Threat Detection · Final Year Project
        </div>

        <div class="sw-section">🎯 Project Overview</div>
        <p style="color:#94a3b8;font-size:.87rem;line-height:1.75">
          SafeWatch AI is a real-time audio threat detection system that classifies
          human screams using Mel spectrogram deep learning. It is designed for
          deployment in security monitoring, healthcare, and emergency response.
        </p>

        <div class="sw-section" style="margin-top:20px">🧠 Architecture</div>
        <div style="display:grid;gap:10px">
        """, unsafe_allow_html=True)

        arch_items = [
            ("Baseline CNN", "4 convolutional blocks on 128-band log-Mel spectrograms (1s windows)"),
            ("CNN-LSTM",     "CNN feature extractor + 2-layer LSTM over 5 consecutive windows"),
            ("Smoothing",    "Alert fires only after N consecutive windows exceed threshold"),
            ("Feature",      "22,050 Hz SR · 1s windows · 50% hop · 128 mel bins · log-scale"),
        ]
        for title, desc in arch_items:
            st.markdown(f"""
            <div class="sw-card" style="padding:12px 16px;margin-bottom:6px">
              <span style="color:#00e5cc;font-weight:600;font-size:.85rem">{title}</span>
              <div style="color:#64748b;font-size:.78rem;margin-top:3px">{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="sw-section" style="margin-top:20px">📚 Datasets</div>
        <div style="display:grid;gap:8px">
        """, unsafe_allow_html=True)

        datasets = [
            ("Human Scream Audio (Kaggle)", "862 scream + 2,631 non-scream clips · MIT License · 44.1kHz WAV"),
            ("ESC-50",                      "2,000 environmental audio clips · background noise diversity"),
            ("Augmentation",                "SpecAugment time/freq masking · Gaussian noise · pitch shift"),
        ]
        for name, desc in datasets:
            st.markdown(f"""
            <div class="sw-card" style="padding:10px 14px;margin-bottom:4px">
              <span style="color:#22d3ee;font-weight:600;font-size:.82rem">{name}</span>
              <div style="color:#64748b;font-size:.75rem;margin-top:2px">{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="sw-section" style="margin-top:20px">⚠️ Ethical Notice</div>
        <div style="background:rgba(239,68,68,.06);border-left:3px solid #ef4444;
             border-radius:0 10px 10px 0;padding:14px 18px;
             color:#94a3b8;font-size:.82rem;line-height:1.7">
          All detections are flagged as <strong style="color:#f87171">potential distress events
          requiring human verification</strong>. No personal identification is performed.
          This is an assistive tool for security personnel, not an autonomous decision system.
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="sw-section">👥 Team</div>
        """, unsafe_allow_html=True)

        members = [
            ("Maharshith",  "Lead · Model Architecture & Training"),
            ("Teammate 2",  "Data Pipeline & Feature Engineering"),
            ("Teammate 3",  "Dashboard · Evaluation & Report"),
        ]
        for name, role in members:
            st.markdown(f"""
            <div class="sw-card" style="padding:14px 16px;margin-bottom:8px">
              <div style="color:#f8fafc;font-weight:600">{name}</div>
              <div style="color:#64748b;font-size:.75rem;margin-top:2px">{role}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="sw-section" style="margin-top:20px">🛠️ Tech Stack</div>
        """, unsafe_allow_html=True)

        stack = [
            ("PyTorch 2.x",    "Model training & inference"),
            ("torchaudio",     "Audio loading & resampling"),
            ("librosa",        "Mel spectrogram extraction"),
            ("audiomentations","Audio augmentation pipeline"),
            ("Streamlit",      "Interactive dashboard"),
            ("Plotly",         "Interactive visualizations"),
        ]
        for tech, desc in stack:
            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:space-between;
                 padding:8px 14px;border-radius:8px;background:rgba(0,229,204,.04);
                 border:1px solid rgba(0,229,204,.08);margin-bottom:4px">
              <span style="color:#00e5cc;font-size:.8rem;font-weight:600">{tech}</span>
              <span style="color:#475569;font-size:.72rem">{desc}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="sw-section" style="margin-top:20px">🔗 Quick Commands</div>
        <div style="background:rgba(0,229,204,.04);border:1px solid rgba(0,229,204,.1);
             border-radius:12px;padding:14px 16px;font-family:'JetBrains Mono',monospace;
             font-size:.72rem;color:#94a3b8;line-height:2">
          # Install deps<br>
          <span style="color:#00e5cc">pip</span> install -r requirements.txt<br><br>
          # Train models<br>
          <span style="color:#00e5cc">python</span> training/train.py --model cnn --epochs 30<br>
          <span style="color:#00e5cc">python</span> training/train.py --model cnn_lstm --epochs 30<br><br>
          # Evaluate<br>
          <span style="color:#00e5cc">python</span> training/evaluate.py --model cnn<br>
          <span style="color:#00e5cc">python</span> training/evaluate.py --model cnn_lstm<br><br>
          # Launch dashboard<br>
          <span style="color:#00e5cc">streamlit</span> run dashboard/app.py
        </div>
        """, unsafe_allow_html=True)
