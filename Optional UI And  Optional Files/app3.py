# ---------------- SYSTEM SETUP ----------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import pandas as pd
import math

from feature_extractor import extract_features, FEATURE_COLS

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NeuroGraph – Parkinson's Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- GLOBAL CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #080e1a;
    color: #c8d8e8;
}
.stApp {
    background: radial-gradient(ellipse at 20% 0%, #0d2137 0%, #080e1a 60%),
                radial-gradient(ellipse at 80% 100%, #091a2e 0%, #080e1a 60%);
    min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1400px; }

.neo-card {
    background: linear-gradient(135deg, #0d1f35 0%, #0a1826 100%);
    border: 1px solid #1a3a5c;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    box-shadow: 0 4px 40px rgba(0,180,255,0.05), inset 0 1px 0 rgba(255,255,255,0.04);
    margin-bottom: 1.2rem;
}

.stButton > button {
    background: linear-gradient(135deg, #0066cc, #0044aa) !important;
    color: #ffffff !important;
    border: 1px solid #0088ff !important;
    border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 20px rgba(0,136,255,0.2) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0088ff, #0055cc) !important;
    box-shadow: 0 0 30px rgba(0,136,255,0.45) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stFileUploader"] {
    border: 1.5px dashed #1a4a7a !important;
    border-radius: 12px !important;
    background: rgba(0,80,160,0.04) !important;
    padding: 1rem !important;
}

hr { border-color: #1a3050 !important; }

.metric-box {
    background: rgba(0,80,160,0.08);
    border: 1px solid #1a3a5c;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-label {
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    color: #5a8aaa;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

.streamlit-expanderHeader {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #4a8aaa !important;
    letter-spacing: 0.06em !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #080e1a; }
::-webkit-scrollbar-thumb { background: #1a4a7a; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#080e1a",
    "axes.facecolor":    "#080e1a",
    "axes.edgecolor":    "#1a3a5c",
    "axes.labelcolor":   "#7aaac8",
    "xtick.color":       "#5a8aaa",
    "ytick.color":       "#5a8aaa",
    "text.color":        "#c8d8e8",
    "grid.color":        "#0e2540",
    "grid.linewidth":    0.6,
    "font.family":       "monospace",
})


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_path = "models/rf_model.pkl"
    if not os.path.exists(model_path):
        st.error("Model not found. Run `python train_model.py` first.")
        st.stop()
    return joblib.load(model_path)

model = load_model()


# ── Helpers ──────────────────────────────────────────────────────────────────
def severity_info(prob):
    if prob < 0.33:
        return "HEALTHY / MILD", "#4ade80", "#0a2e1a", "#1a7a40", "●"
    elif prob < 0.66:
        return "MODERATE", "#fbbf24", "#2e2000", "#a06010", "◆"
    else:
        return "SEVERE", "#f87171", "#2e0a0a", "#8b1a1a", "▲"


# ── Chart: Gauge ─────────────────────────────────────────────────────────────
def make_gauge(prob):
    fig, ax = plt.subplots(figsize=(5, 3.2), subplot_kw=dict(polar=False))
    fig.patch.set_alpha(0)
    ax.set_aspect("equal")
    ax.axis("off")

    zones = [
        (0,    0.33, "#0a2e1a", "#1a7a40"),
        (0.33, 0.66, "#2e2000", "#a06010"),
        (0.66, 1.00, "#2e0a0a", "#8b1a1a"),
    ]
    for start, end, bg, edge in zones:
        t1 = 180 - start * 180
        t2 = 180 - end   * 180
        arc = mpatches.Wedge(
            center=(0.5, 0.15), r=0.46, theta1=t2, theta2=t1,
            width=0.13, facecolor=bg, edgecolor=edge, linewidth=0.8,
            transform=ax.transAxes
        )
        ax.add_patch(arc)

    _, color, _, _, _ = severity_info(prob)
    ax.add_patch(mpatches.Wedge(
        center=(0.5, 0.15), r=0.46,
        theta1=180 - prob * 180, theta2=180,
        width=0.13, facecolor=color, alpha=0.88,
        transform=ax.transAxes
    ))

    angle_rad = math.pi * (1 - prob)
    nx = 0.5 + 0.30 * math.cos(angle_rad)
    ny = 0.15 + 0.30 * math.sin(angle_rad)
    ax.annotate("", xy=(nx, ny), xytext=(0.5, 0.15),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5,
                                mutation_scale=16))
    ax.add_patch(plt.Circle((0.5, 0.15), 0.028, color="#1a3a5c",
                             transform=ax.transAxes, zorder=5))

    ax.text(0.5, 0.60, f"{prob*100:.1f}%", ha="center", va="center",
            transform=ax.transAxes, fontsize=24, fontweight="bold", color=color)
    ax.text(0.5, 0.76, "PARKINSON'S PROBABILITY", ha="center", va="center",
            transform=ax.transAxes, fontsize=7, color="#5a8aaa", fontstyle="normal")

    for x, lbl in [(0.05, "LOW"), (0.5, "MED"), (0.95, "HIGH")]:
        ax.text(x, 0.00, lbl, ha="center", transform=ax.transAxes,
                fontsize=6.5, color="#3a6a8a")

    fig.tight_layout(pad=0)
    return fig


# ── Chart: Radar ─────────────────────────────────────────────────────────────
def make_radar(feats):
    labels = ["RMS", "Max\nET-HT", "Min\nET-HT", "Std Dev\nET-HT",
              "Mean\nResp Time", "Max\nHold Time", "Min\nHold Time", "Std\nHold Time", "Direction\nChanges"]
    values = [feats[c] for c in FEATURE_COLS]
    ranges = [(0,6000),(0,10000),(0,40000),(0,0.1),
              (0,50),(0,200),(0,0.15),(0,2500),(0,0.35)]
    norm = [min(max((v-lo)/(hi-lo if hi!=lo else 1),0),1)
            for v,(lo,hi) in zip(values, ranges)]

    N      = len(labels)
    angles = [n / N * 2 * math.pi for n in range(N)] + [0]
    norm   = norm + norm[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
    fig.patch.set_alpha(0)
    ax.set_facecolor("#080e1a")
    ax.set_rlim(0, 1)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8, color="#5a8aaa")
    ax.spines["polar"].set_color("#1a3a5c")
    ax.grid(color="#0e2540", linewidth=0.8)

    ax.plot(angles, norm, color="#38bdf8", linewidth=2)
    ax.fill(angles, norm, color="#38bdf8", alpha=0.15)
    for a, v in zip(angles[:-1], norm[:-1]):
        ax.plot(a, v, "o", color="#38bdf8", markersize=5, zorder=4)

    ax.set_title("Feature Profile", fontsize=10, color="#7aaac8", pad=14)
    fig.tight_layout(pad=0.5)
    return fig


# ── Chart: Horizontal bars ────────────────────────────────────────────────────
def make_bar_chart(feats):
    names  = ["RMS","Max ET-HT","Min ET-HT","Std Dev ET-HT",
              "Mean Response Time","Max Hold Time","Min Hold Time","Std Hold Time","Direction Changes"]
    values = [feats[c] for c in FEATURE_COLS]
    ranges = [(0,6000),(0,10000),(0,40000),(0,0.1),
              (0,50),(0,200),(0,0.15),(0,2500),(0,0.35)]
    norm = [min(max((v-lo)/(hi-lo if hi!=lo else 1),0),1)
            for v,(lo,hi) in zip(values, ranges)]

    colors = ["#38bdf8" if n<0.5 else "#fbbf24" if n<0.75 else "#f87171"
              for n in norm]

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    fig.patch.set_alpha(0)
    bars = ax.barh(names, norm, color=colors, height=0.52, edgecolor="none")
    for bar, raw in zip(bars, values):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{raw:.1f}", va="center", fontsize=7, color="#7aaac8")
    ax.set_xlim(0, 1.28)
    ax.set_xlabel("Normalised Intensity", fontsize=8)
    ax.set_title("Feature Intensities", fontsize=10, color="#7aaac8", pad=10)
    ax.invert_yaxis()
    ax.axvline(0.5,  color="#1a3a5c", linewidth=1, linestyle="--")
    ax.axvline(0.75, color="#2a2a1a", linewidth=1, linestyle="--")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(pad=0.8)
    return fig


# ── Chart: Per-image feature contribution donut ───────────────────────────────
def make_donut(mdl, feats):
    rf  = mdl.named_steps["clf"]
    imp = rf.feature_importances_
    full_names = ["RMS","Max ET-HT","Min ET-HT","Std Dev ET-HT",
                  "Mean Response Time","Max Hold Time","Min Hold Time",
                  "Std Hold Time","Direction Changes"]
    ranges = [(0,6000),(0,10000),(0,40000),(0,0.1),
              (0,50),(0,200),(0,0.15),(0,2500),(0,0.35)]
    values = [feats[c] for c in FEATURE_COLS]
    norm   = [min(max((v-lo)/(hi-lo if hi!=lo else 1),0),1)
              for v,(lo,hi) in zip(values, ranges)]
    contrib = np.array([i * n for i, n in zip(imp, norm)])
    total   = contrib.sum()
    if total == 0:
        contrib = imp.copy(); total = contrib.sum()
    contrib = contrib / total
    idx = np.argsort(contrib)[::-1]
    pal = ["#38bdf8","#818cf8","#34d399","#fbbf24","#f87171",
           "#a78bfa","#60a5fa","#fb923c","#4ade80"]
    fig, ax = plt.subplots(figsize=(5, 4.8))
    fig.patch.set_alpha(0)
    wedges, _, autotexts = ax.pie(
        contrib[idx], labels=None, autopct="%1.0f%%", startangle=90,
        colors=pal, wedgeprops=dict(width=0.45, edgecolor="#080e1a", linewidth=2),
        pctdistance=0.78,
    )
    for at in autotexts:
        at.set_fontsize(7.5); at.set_color("#080e1a"); at.set_fontweight("bold")
    ax.legend(wedges, [full_names[i] for i in idx],
              loc="lower center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, fontsize=7, framealpha=0, labelcolor="#7aaac8")
    ax.set_title("Feature Contribution (This Image)", fontsize=10,
                 color="#7aaac8", pad=10)
    fig.tight_layout(pad=0.5)
    return fig


# ── Chart: Per-image accuracy breakdown ──────────────────────────────────────
def make_accuracy_panel(mdl, feats, prob):
    rf     = mdl.named_steps["clf"]
    scaler = mdl.named_steps["scaler"]
    X_sc   = scaler.transform(np.array([[feats[c] for c in FEATURE_COLS]]))

    prob_healthy = 1 - prob
    tree_votes   = np.array([t.predict(X_sc)[0] for t in rf.estimators_])
    votes_pd      = int(tree_votes.sum())
    votes_healthy = len(tree_votes) - votes_pd

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.8))
    fig.patch.set_alpha(0)

    # Stacked probability bar
    ax1 = axes[0]
    ax1.barh([""], [prob_healthy * 100], color="#4ade80", height=0.45)
    ax1.barh([""], [prob * 100], left=[prob_healthy * 100],
             color="#f87171", height=0.45)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Probability %", fontsize=8)
    ax1.set_title("Class Probabilities", fontsize=9, color="#7aaac8", pad=8)
    ax1.axvline(50, color="#1a3a5c", linewidth=1, linestyle="--")
    if prob_healthy > 0.15:
        ax1.text(prob_healthy * 50, 0, f"{prob_healthy*100:.1f}%",
                 ha="center", va="center", fontsize=8,
                 color="#080e1a", fontweight="bold")
    if prob > 0.15:
        ax1.text(prob_healthy*100 + prob*50, 0, f"{prob*100:.1f}%",
                 ha="center", va="center", fontsize=8,
                 color="#080e1a", fontweight="bold")
    ax1.text(25,  -0.45, "Healthy", ha="center", fontsize=7.5, color="#4ade80")
    ax1.text(75,  -0.45, "PD",      ha="center", fontsize=7.5, color="#f87171")

    # Tree vote donut
    ax2 = axes[1]
    ax2.pie([votes_healthy, votes_pd],
            colors=["#4ade80","#f87171"],
            wedgeprops=dict(width=0.42, edgecolor="#080e1a", linewidth=1.5),
            startangle=90)
    ax2.set_title("Tree Votes", fontsize=9, color="#7aaac8", pad=8)
    ax2.text(0, 0.1,  f"{votes_pd}", ha="center", va="center",
             fontsize=11, color="#f87171", fontweight="bold")
    ax2.text(0, -0.25, "PD votes", ha="center", fontsize=7, color="#5a8aaa")
    ax2.text(-1.6, -1.4,
             f"🟢 Healthy: {votes_healthy}  🔴 PD: {votes_pd}  / {len(tree_votes)} trees",
             fontsize=7, color="#5a8aaa")

    fig.tight_layout(pad=0.6)
    return fig


# ── Chart: Session history ────────────────────────────────────────────────────
def make_history(history):
    fig, ax = plt.subplots(figsize=(6, 2.2))
    fig.patch.set_alpha(0)
    xs = list(range(1, len(history) + 1))
    ys = [h * 100 for h in history]
    ax.plot(xs, ys, color="#1a3a5c", linewidth=1.5, zorder=1)
    ax.fill_between(xs, 0, ys, color="#38bdf8", alpha=0.07)
    for x, y, h in zip(xs, ys, history):
        c = severity_info(h)[1]
        ax.scatter(x, y, color=c, s=60, zorder=3, edgecolors="#080e1a", linewidth=1)
    ax.axhline(33, color="#1a7a40", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(66, color="#a06010", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylim(0, 100)
    ax.set_xlim(0.5, max(len(history)+0.5, 3))
    ax.set_xticks(xs); ax.set_xticklabels([f"#{x}" for x in xs], fontsize=7.5)
    ax.set_ylabel("PD Prob %", fontsize=8)
    ax.set_title("Session History", fontsize=10, color="#7aaac8", pad=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(pad=0.5)
    return fig


# ────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ────────────────────────────────────────────────────────────────────────────
if "history"    not in st.session_state: st.session_state.history    = []
if "last_feats" not in st.session_state: st.session_state.last_feats = None
if "last_prob"  not in st.session_state: st.session_state.last_prob  = None


# ────────────────────────────────────────────────────────────────────────────
#  HEADER
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:1.5rem 0 0.5rem;'>
  <div style='font-family:Syne,sans-serif; font-size:2.8rem; font-weight:800;
              letter-spacing:-0.01em; color:#e8f4ff;'>🧠 NeuroGraph</div>
  <div style='font-size:0.78rem; letter-spacing:0.18em; color:#3a7aaa;
              text-transform:uppercase; margin-top:0.3rem;'>
    Parkinson's Detection &nbsp;·&nbsp; Handwriting Analysis &nbsp;·&nbsp; v2.0
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
#  STATS STRIP
# ────────────────────────────────────────────────────────────────────────────
total  = len(st.session_state.history)
avg_p  = np.mean(st.session_state.history) * 100 if total else 0.0
high_r = sum(1 for h in st.session_state.history if h >= 0.66)

for col, val, lbl in zip(
    st.columns(4),
    [str(total), f"{avg_p:.1f}%", str(high_r), "RF · 300 TREES"],
    ["ANALYSES RUN", "AVG PD SCORE", "HIGH-RISK COUNT", "MODEL"]
):
    col.markdown(f"""
    <div class='metric-box'>
      <div class='metric-val'>{val}</div>
      <div class='metric-label'>{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
#  MAIN LAYOUT
# ────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.65], gap="large")

# ── LEFT: upload ──────────────────────────────────────────────────────────────
with left:
    st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:Syne,sans-serif; font-size:0.95rem; font-weight:700;
                letter-spacing:0.08em; color:#7aaac8; margin-bottom:1rem;'>
    ◈ &nbsp; INPUT SAMPLE
    </div>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload spiral image", type=["png","jpg","jpeg"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        raw = np.frombuffer(uploaded_file.read(), np.uint8)
        preview = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                 caption="Uploaded sample", width=320)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    analyze_btn = st.button("⚡  RUN ANALYSIS", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Feature table
    if st.session_state.last_feats:
        with st.expander("🔬  Raw Feature Values"):
            fdf = pd.DataFrame({
                "Feature": FEATURE_COLS,
                "Value": [round(st.session_state.last_feats[c], 4) for c in FEATURE_COLS]
            })
            st.dataframe(fdf, hide_index=True, use_container_width=True)

    # History chart
    if st.session_state.history:
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family:Syne,sans-serif; font-size:0.9rem; font-weight:700;
                    letter-spacing:0.08em; color:#7aaac8; margin-bottom:0.8rem;'>
        ◈ &nbsp; SESSION HISTORY
        </div>""", unsafe_allow_html=True)
        st.pyplot(make_history(st.session_state.history), use_container_width=True)
        if st.button("Clear history", use_container_width=True):
            st.session_state.history    = []
            st.session_state.last_prob  = None
            st.session_state.last_feats = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ── RIGHT: results ────────────────────────────────────────────────────────────
with right:
    # Run analysis
    if uploaded_file and analyze_btn:
        uploaded_file.seek(0)
        raw    = np.frombuffer(uploaded_file.read(), np.uint8)
        img    = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        feats  = extract_features(img)
        if feats is None:
            st.error("⚠️ Could not extract features. Try a clearer spiral drawing.")
        else:
            X    = np.array([[feats[c] for c in FEATURE_COLS]])
            prob = float(model.predict_proba(X)[0][1])
            st.session_state.last_feats = feats
            st.session_state.last_prob  = prob
            st.session_state.history.append(prob)
            st.rerun()

    prob = st.session_state.last_prob

    if prob is not None:
        feats                           = st.session_state.last_feats
        label, color, bg_col, brd, icon = severity_info(prob)

        # Severity banner
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,{bg_col} 0%,#080e1a 100%);
                    border:1.5px solid {brd}; border-radius:14px;
                    padding:1.4rem 2rem; margin-bottom:1rem;
                    display:flex; align-items:center; gap:1.5rem;'>
          <div style='font-size:2.6rem; line-height:1;'>{icon}</div>
          <div>
            <div style='font-family:Syne,sans-serif; font-size:1.9rem;
                        font-weight:800; color:{color}; line-height:1;'>
              {label}
            </div>
            <div style='font-size:0.72rem; color:{brd};
                        letter-spacing:0.14em; margin-top:0.35rem;'>
              PARKINSON'S ASSESSMENT RESULT
            </div>
          </div>
          <div style='margin-left:auto; text-align:right;'>
            <div style='font-family:Syne,sans-serif; font-size:2.5rem;
                        font-weight:800; color:{color};'>{prob*100:.1f}%</div>
            <div style='font-size:0.68rem; color:#3a6a8a; letter-spacing:0.1em;'>
              CONFIDENCE
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Row 1 – Gauge + Radar
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='neo-card' style='padding:1rem;'>", unsafe_allow_html=True)
            st.pyplot(make_gauge(prob), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='neo-card' style='padding:1rem;'>", unsafe_allow_html=True)
            st.pyplot(make_radar(feats), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Row 2 – Bars + Donut
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("<div class='neo-card' style='padding:1rem;'>", unsafe_allow_html=True)
            st.pyplot(make_bar_chart(feats), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c4:
            st.markdown("<div class='neo-card' style='padding:1rem;'>", unsafe_allow_html=True)
            st.pyplot(make_donut(model, feats), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Row 3 – Accuracy panel (full width)
        st.markdown("<div class='neo-card' style='padding:1rem;'>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family:Syne,sans-serif; font-size:0.85rem; font-weight:700;
                    letter-spacing:0.08em; color:#7aaac8; margin-bottom:0.6rem;'>
        ◈ &nbsp; PREDICTION CONFIDENCE BREAKDOWN
        </div>""", unsafe_allow_html=True)
        st.pyplot(make_accuracy_panel(model, feats, prob), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Empty state — no charts shown
        st.markdown("""
        <div style='background:linear-gradient(135deg,#0d1f35,#0a1826);
                    border:1.5px dashed #1a3a5c; border-radius:14px;
                    padding:5rem 2rem; text-align:center; margin-top:0.2rem;'>
          <div style='font-size:3rem; margin-bottom:1rem; opacity:0.35;'>🧬</div>
          <div style='font-family:Syne,sans-serif; font-size:0.95rem;
                      color:#2a5a7a; letter-spacing:0.1em;'>
            UPLOAD AN IMAGE &amp; CLICK RUN ANALYSIS
          </div>
          <div style='font-size:0.7rem; color:#1a3a5c;
                      margin-top:0.6rem; letter-spacing:0.06em;'>
            Gauge · Radar · Feature Bars · Contribution Donut · Accuracy Breakdown
          </div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:1.2rem 0;
            border-top:1px solid #0e2540; margin-top:1.5rem;'>
  <span style='font-size:0.7rem; color:#2a5a7a; letter-spacing:0.14em;'>
    AI-ASSISTED SCREENING TOOL &nbsp;·&nbsp; NOT A MEDICAL DIAGNOSIS
    &nbsp;·&nbsp; NEUROGRAPH v2.0
  </span>
</div>
""", unsafe_allow_html=True)