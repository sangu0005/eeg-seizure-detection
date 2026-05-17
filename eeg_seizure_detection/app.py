"""
EEG Epileptic Seizure Detection — Streamlit Application
Trained Model : SVM (RBF kernel) on CHB-MIT Full Dataset
Features      : Time-domain (prediction) + GLCM Image Descriptors (analysis)
Author        : SSIT 2025-26
"""

import os
import warnings
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import mne
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler

mne.set_log_level("ERROR")

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_ROWS    = 48    # EEG segment reshaped → 48 × 64 grayscale image
IMG_COLS    = 64
GLCM_LEVELS = 32   # gray-level quantisation for GLCM

FEAT_NAMES     = ["Mean", "Std Dev", "Skewness", "Kurtosis", "ZCR", "RMS"]
IMG_DESC_NAMES = [
    "Contrast", "Energy", "Homogeneity",
    "GLCM Entropy", "Correlation",
    "Hist Entropy", "Mean Intensity", "Std Intensity",
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EEG Seizure Detector",
    page_icon="🧠",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 18px 24px;
        border: 1px solid #2d3250;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; margin: 0; }
    .metric-label { font-size: 0.85rem; color: #9ca3af; margin: 0; }
    .seizure  { color: #f87171; }
    .normal   { color: #4ade80; }
    .neutral  { color: #60a5fa; }
    .warning-box {
        background: #2d2010;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 10px 16px;
        margin: 8px 0;
    }
    .success-box {
        background: #0d2b1a;
        border: 1px solid #4ade80;
        border-radius: 8px;
        padding: 10px 16px;
        margin: 8px 0;
    }
    .desc-card {
        background: #161b2e;
        border-radius: 10px;
        padding: 14px 18px;
        border: 1px solid #2d3250;
        margin-bottom: 6px;
    }
    .desc-name  { font-size: 0.78rem; color: #94a3b8; margin: 0; }
    .desc-value { font-size: 1.3rem; font-weight: 700; margin: 0; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  CORE FUNCTIONS
# =============================================================================

def extract_time_features(segment: np.ndarray) -> list:
    """6 time-domain features — identical to training notebook."""
    std = np.std(segment)
    if std < 1e-10:
        return [float(np.mean(segment)), 0.0, 0.0, 0.0, 0.0, 0.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        skewn = float(skew(segment))
        kurt  = float(kurtosis(segment))
    mean = float(np.mean(segment))
    zcr  = float(np.sum(np.diff(np.sign(segment)) != 0) / (len(segment) - 1))
    rms  = float(np.sqrt(np.mean(segment ** 2)))
    return [mean, std, skewn, kurt, zcr, rms]


def segment_to_image(segment: np.ndarray,
                     rows: int = IMG_ROWS,
                     cols: int = IMG_COLS) -> np.ndarray:
    """
    Reshape 1-D EEG segment to 2-D grayscale image (uint8).
    Rows = time sub-windows, Cols = amplitude samples per sub-window.
    Extra samples discarded; short segments zero-padded.
    """
    needed = rows * cols
    if len(segment) >= needed:
        s = segment[:needed]
    else:
        s = np.pad(segment, (0, needed - len(segment)))
    mn, mx = s.min(), s.max()
    img = ((s - mn) / (mx - mn + 1e-9) * 255).astype(np.uint8)
    return img.reshape(rows, cols)


def glcm_features(img: np.ndarray) -> dict:
    """
    Gray-Level Co-occurrence Matrix (GLCM) features + histogram descriptors.

    GLCM: horizontal co-occurrence (angle=0, distance=1).
    Descriptors: Contrast, Energy, Homogeneity, GLCM Entropy, Correlation,
                 Hist Entropy, Mean Intensity, Std Intensity.
    """
    lvl = GLCM_LEVELS
    mn, mx = img.min(), img.max()
    if mx == mn:
        q = np.zeros_like(img, dtype=np.int32)
    else:
        q = ((img.astype(float) - mn) / (mx - mn) * (lvl - 1)).astype(np.int32)
        q = np.clip(q, 0, lvl - 1)

    # Build normalised GLCM
    glcm = np.zeros((lvl, lvl), dtype=np.float64)
    np.add.at(glcm, (q[:, :-1].ravel(), q[:, 1:].ravel()), 1)
    total = glcm.sum()
    if total > 0:
        glcm /= total

    I, J = np.meshgrid(np.arange(lvl), np.arange(lvl), indexing="ij")

    contrast    = float(np.sum(glcm * (I - J) ** 2))
    energy      = float(np.sum(glcm ** 2))
    homogeneity = float(np.sum(glcm / (1.0 + np.abs(I - J))))

    gf = glcm.ravel()
    gf = gf[gf > 1e-12]
    glcm_entropy = float(-np.sum(gf * np.log2(gf)))

    mu_i  = float(np.sum(I * glcm))
    mu_j  = float(np.sum(J * glcm))
    sig_i = float(np.sqrt(np.sum(glcm * (I - mu_i) ** 2) + 1e-9))
    sig_j = float(np.sqrt(np.sum(glcm * (J - mu_j) ** 2) + 1e-9))
    correlation = float(
        np.sum(glcm * (I - mu_i) * (J - mu_j)) / (sig_i * sig_j)
    )

    hist, _ = np.histogram(img.ravel(), bins=32, range=(0, 255))
    hp = hist / (hist.sum() + 1e-9)
    hp = hp[hp > 0]
    hist_entropy = float(-np.sum(hp * np.log2(hp)))

    return {
        "Contrast"      : round(contrast, 4),
        "Energy"        : round(energy, 6),
        "Homogeneity"   : round(homogeneity, 4),
        "GLCM Entropy"  : round(glcm_entropy, 4),
        "Correlation"   : round(correlation, 4),
        "Hist Entropy"  : round(hist_entropy, 4),
        "Mean Intensity": round(float(img.mean()), 2),
        "Std Intensity" : round(float(img.std()), 2),
    }


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def load_edf(edf_bytes: bytes):
    """Write bytes to temp file, read with MNE, return (data, sfreq, ch_names)."""
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(edf_bytes)
        tmp_path = tmp.name
    try:
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        data, sfreq, ch_names = raw.get_data(), raw.info["sfreq"], raw.ch_names
    finally:
        os.unlink(tmp_path)
    return data, sfreq, ch_names


@st.cache_data(show_spinner=False)
def full_analysis(edf_bytes: bytes,
                  model_path: str,
                  scaler_path: Optional[str],
                  channel_idx: int,
                  seg_len: int):
    """
    Full pipeline: EDF → segments → time features → SVM predict
                              → grayscale images → GLCM descriptors.
    Returns (df, sfreq, ch_names, scaler_note, segments_raw, images, img_descs)
    """
    data, sfreq, ch_names = load_edf(edf_bytes)

    if channel_idx >= data.shape[0]:
        st.error(f"Channel {channel_idx} out of range "
                 f"(file has {data.shape[0]} channels).")
        return None

    model       = load_model(model_path)
    seg_samples = int(seg_len * sfreq)
    signal      = data[channel_idx]
    n_segments  = len(signal) // seg_samples

    time_feats, segments_raw, images, img_descs = [], [], [], []

    for i in range(n_segments):
        seg  = signal[i * seg_samples : (i + 1) * seg_samples]
        tf   = extract_time_features(seg)
        time_feats.append(tf if np.all(np.isfinite(tf)) else [0.0] * 6)
        segments_raw.append(seg)
        img  = segment_to_image(seg)
        images.append(img)
        img_descs.append(glcm_features(img))

    X = np.array(time_feats)

    if scaler_path and os.path.exists(scaler_path):
        scaler      = joblib.load(scaler_path)
        X_scaled    = scaler.transform(X)
        scaler_note = "training scaler"
    else:
        scaler      = StandardScaler()
        X_scaled    = scaler.fit_transform(X)
        scaler_note = "per-file StandardScaler (no scaler.pkl found)"

    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]

    rows = [{
        "Segment"        : i + 1,
        "Start (s)"      : i * seg_len,
        "End (s)"        : (i + 1) * seg_len,
        "Confidence (%)": round(probs[i] * 100, 2),
        "_label"         : int(preds[i]),
        "_prob"          : float(probs[i]),
    } for i in range(n_segments)]

    df = pd.DataFrame(rows)
    return df, sfreq, ch_names, scaler_note, segments_raw, images, img_descs


# =============================================================================
#  PLOTTING HELPERS
# =============================================================================

DARK = dict(
    template      = "plotly_dark",
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(14,17,23,0.8)",
    margin        = dict(l=10, r=10, t=30, b=40),
)


def plot_grayscale(img: np.ndarray, title: str,
                   border_color: str = "#60a5fa") -> go.Figure:
    """Render 2-D uint8 array as a grey heatmap with a coloured border."""
    fig = go.Figure(go.Heatmap(
        z          = img[::-1],
        colorscale = "gray",
        zmin=0, zmax=255,
        showscale  = False,
        hovertemplate="Row %{y} | Col %{x}<br>Intensity: %{z}<extra></extra>",
    ))
    fig.update_layout(
        **DARK,
        title  = dict(text=title, font=dict(size=12), x=0.5),
        height = 230,
        xaxis  = dict(showticklabels=False, showgrid=False),
        yaxis  = dict(showticklabels=False, showgrid=False),
        shapes = [dict(
            type="rect", xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color=border_color, width=2.5),
        )],
    )
    return fig


def plot_descriptor_radar(normal_means: dict, seizure_means: dict) -> go.Figure:
    """Radar chart comparing normalised mean descriptors for Normal vs Seizure."""
    labels = list(normal_means.keys())
    nv = list(normal_means.values())
    sv = list(seizure_means.values())
    nn, sn = [], []
    for a, b in zip(nv, sv):
        lo, hi = min(a, b), max(a, b)
        rng = hi - lo if hi != lo else 1.0
        nn.append((a - lo) / rng)
        sn.append((b - lo) / rng)

    fig = go.Figure()
    for vals, name, color, fill in [
        (nn, "Normal",  "#4ade80", "rgba(74,222,128,0.12)"),
        (sn, "Seizure", "#f87171", "rgba(248,113,113,0.12)"),
    ]:
        fig.add_trace(go.Scatterpolar(
            r         = vals + [vals[0]],
            theta     = labels + [labels[0]],
            fill      = "toself",
            name      = name,
            line      = dict(color=color, width=2),
            fillcolor = fill,
        ))
    fig.update_layout(
        **DARK,
        polar  = dict(
            bgcolor    = "rgba(14,17,23,0.8)",
            radialaxis = dict(visible=True, range=[0, 1],
                              showticklabels=False),
            angularaxis= dict(tickfont=dict(size=10)),
        ),
        legend = dict(orientation="h", x=0.28, y=-0.1),
        height = 360,
        title  = dict(text="Image Descriptor Comparison (normalised)",
                      x=0.5, font=dict(size=13)),
    )
    return fig


def plot_descriptor_boxplots(desc_df: pd.DataFrame,
                              labels: np.ndarray) -> go.Figure:
    """2×4 box-plot grid for all descriptors split by class."""
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=IMG_DESC_NAMES,
        vertical_spacing=0.22,
        horizontal_spacing=0.07,
    )
    pos = [(r + 1, c + 1) for r in range(2) for c in range(4)]
    for idx, desc in enumerate(IMG_DESC_NAMES):
        row, col = pos[idx]
        for lbl, name, color in [(0, "Normal", "#4ade80"), (1, "Seizure", "#f87171")]:
            vals = desc_df[desc][labels == lbl].values
            fig.add_trace(
                go.Box(y=vals, name=name, marker_color=color,
                       showlegend=(idx == 0), boxmean=True,
                       line=dict(width=1.5)),
                row=row, col=col,
            )
    fig.update_layout(
        title="Descriptor Boxplots",
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig


def plot_descriptor_timeline(desc_df: pd.DataFrame,
                              start_times: np.ndarray,
                              labels: np.ndarray,
                              descriptor: str) -> go.Figure:
    """Scatter of a single descriptor over the recording, coloured by class."""
    fig = go.Figure()
    for lbl, name, color in [(0, "Normal", "#4ade80"), (1, "Seizure", "#f87171")]:
        m = labels == lbl
        fig.add_trace(go.Scatter(
            x=start_times[m], y=desc_df[descriptor].values[m],
            mode="markers", name=name,
            marker=dict(color=color, size=7, symbol="circle"),
        ))
    fig.update_layout(
        **DARK,
        height      = 260,
        xaxis_title = "Time (s)",
        yaxis_title = descriptor,
        legend      = dict(orientation="h", y=1.12),
        title       = dict(text=f"{descriptor} across the recording",
                           x=0.5, font=dict(size=12)),
    )
    return fig


# =============================================================================
#  SIDEBAR
# =============================================================================

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/"
        "EEG_cap.jpg/320px-EEG_cap.jpg",
        use_container_width=True,
    )
    st.title("⚙️ Settings")

    model_path  = st.text_input("Model path",             value="models/svm_model.pkl")
    scaler_path = st.text_input("Scaler path (optional)", value="models/scaler.pkl")
    channel_idx = st.number_input("EEG channel index", min_value=0, value=0, step=1)
    seg_len     = st.slider("Segment length (s)", 4, 30, 12, 2)
    threshold   = st.slider("Seizure probability threshold", 0.10, 0.99, 0.50, 0.01)

    st.markdown("---")
    st.markdown("**Model info**")
    if os.path.exists(model_path):
        _m = load_model(model_path)
        st.success(f"✅ {_m.kernel.upper()} kernel")
        st.caption(f"C={_m.C:.3f} | γ={_m.gamma:.4f}")
        st.caption(f"Support vectors: {sum(_m.n_support_):,}")
    else:
        st.error("Model not found.")

    st.markdown("---")
    st.markdown("**Image descriptor info**")
    st.caption(f"Image shape : {IMG_ROWS} × {IMG_COLS} px")
    st.caption(f"GLCM levels : {GLCM_LEVELS}")
    st.caption("Orientation : horizontal (angle=0°, d=1)")
    st.markdown("---")
    st.caption("SSIT 2025-26 | EEG Seizure Detection")


# =============================================================================
#  MAIN AREA
# =============================================================================

st.title("🧠 EEG Epileptic Seizure Detector")
st.markdown("*SVM classifier · GLCM Image Descriptors · CHB-MIT Dataset*")

tab_predict, tab_image, tab_about = st.tabs([
    "🔍 Predict", "🖼️ Image Analysis", "ℹ️ About"
])


# =============================================================================
#  TAB 1 — PREDICT
# =============================================================================
with tab_predict:
    col_up, col_demo = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader("Upload an EDF file", type=["edf"],
                                    key="up1")
    with col_demo:
        st.markdown("#### Or use the included file")
        use_demo = st.button("▶ Run on chb20_02.edf",
                             use_container_width=True, key="d1")

    edf_bytes, source_name = None, None
    if uploaded:
        edf_bytes, source_name = uploaded.read(), uploaded.name
    elif use_demo:
        demo = "chb20_02.edf"
        if os.path.exists(demo):
            with open(demo, "rb") as f:
                edf_bytes = f.read()
            source_name = demo
        else:
            st.error(f"Demo file not found: {demo}")

    if edf_bytes is not None:
        if not os.path.exists(model_path):
            st.error(f"⛔ Model not found: `{model_path}`")
            st.stop()

        with st.spinner("Analysing EEG recording…"):
            result = full_analysis(
                edf_bytes, model_path,
                scaler_path if os.path.exists(scaler_path) else None,
                channel_idx, seg_len,
            )
        if result is None:
            st.stop()

        df, sfreq, ch_names, scaler_note, segs_raw, _, _ = result

        df["_label_t"] = (df["_prob"] >= threshold).astype(int)
        df["Prediction"] = df["_label_t"].map({1: "🔴 SEIZURE", 0: "🟢 Normal"})

        seizure_segs   = int(df["_label_t"].sum())
        total_segs     = len(df)
        recording_time = total_segs * seg_len

        # Banner
        if "per-file" in scaler_note:
            st.markdown("""<div class='warning-box'>
            ⚠️ <b>No scaler.pkl</b> — per-file normalisation used as fallback.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='success-box'>
            ✅ Training scaler loaded from <code>{scaler_path}</code>
            </div>""", unsafe_allow_html=True)

        # Metrics row
        st.markdown(f"### Results — `{source_name}`")
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, val, lbl, cls in [
            (c1, str(total_segs), "Total Segments", "neutral"),
            (c2, f"{recording_time//60:.0f}m {recording_time%60:.0f}s",
             "Recording", "neutral"),
            (c3, str(seizure_segs), "Seizure Segments",
             "seizure" if seizure_segs > 0 else "normal"),
            (c4, f"{seizure_segs/total_segs*100:.1f}%", "Seizure Ratio",
             "seizure" if seizure_segs > 0 else "normal"),
            (c5, f"{df['_prob'].max()*100:.1f}%", "Peak Confidence",
             "seizure" if seizure_segs > 0 else "normal"),
        ]:
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                  <p class='metric-value {cls}'>{val}</p>
                  <p class='metric-label'>{lbl}</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Probability timeline
        st.markdown("#### Seizure Probability Timeline")
        fig_tl = go.Figure()
        for _, row in df[df["_label_t"] == 1].iterrows():
            fig_tl.add_vrect(
                x0=row["Start (s)"], x1=row["End (s)"],
                fillcolor="rgba(248,113,113,0.15)", line_width=0,
            )
        fig_tl.add_hline(
            y=threshold, line_dash="dash", line_color="#f59e0b", line_width=1.5,
            annotation_text=f"Threshold {threshold:.0%}",
            annotation_position="top right",
            annotation_font_color="#f59e0b",
        )
        fig_tl.add_trace(go.Scatter(
            x=df["Start (s)"], y=df["_prob"],
            mode="lines+markers",
            line=dict(color="#60a5fa", width=2),
            marker=dict(
                color=df["_prob"].apply(
                    lambda p: "#f87171" if p >= threshold else "#4ade80"),
                size=6,
            ),
            hovertemplate=(
                "<b>Segment %{customdata[0]}</b><br>"
                "Time: %{customdata[1]}s–%{customdata[2]}s<br>"
                "Prob: %{y:.2%}<extra></extra>"
            ),
            customdata=df[["Segment", "Start (s)", "End (s)"]].values,
        ))
        fig_tl.update_layout(
            **DARK, height=350, xaxis_title="Time (s)",
            yaxis_title="Seizure Probability",
            yaxis=dict(range=[0, 1.05], tickformat=".0%"),
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_tl, use_container_width=True)

        # Time-feature heatmap
        st.markdown("#### Time-Feature Heatmap")
        tf_matrix = np.array([extract_time_features(s) for s in segs_raw])
        tf_norm   = (tf_matrix - tf_matrix.mean(0)) / (tf_matrix.std(0) + 1e-9)
        fig_hm = go.Figure(go.Heatmap(
            z=tf_norm.T, x=df["Start (s)"], y=FEAT_NAMES,
            colorscale="RdBu_r", zmin=-3, zmax=3,
            colorbar=dict(title="Z-score", thickness=12),
        ))
        fig_hm.update_layout(**DARK, height=230, xaxis_title="Time (s)")
        st.plotly_chart(fig_hm, use_container_width=True)

        # Data table
        st.markdown("#### Segment-level Predictions")
        cf1, cf2 = st.columns([2, 1])
        with cf1:
            show_only = st.checkbox("Show seizure segments only", value=False)
        with cf2:
            st.download_button(
                "⬇ Download CSV",
                df[["Segment","Start (s)","End (s)",
                    "Confidence (%)","Prediction"]].to_csv(index=False),
                "predictions.csv", "text/csv",
                use_container_width=True,
            )
        show_df = df[df["_label_t"] == 1] if show_only else df
        st.dataframe(
            show_df[["Segment","Start (s)","End (s)",
                     "Confidence (%)","Prediction"]],
            use_container_width=True, height=320,
        )

        with st.expander("EDF file metadata"):
            st.write(f"**Sampling rate:** {sfreq} Hz")
            st.write(f"**Channels ({len(ch_names)}):** {', '.join(ch_names)}")
            ch_lbl = (ch_names[channel_idx] if channel_idx < len(ch_names)
                      else "out of range")
            st.write(f"**Active channel:** {channel_idx} → `{ch_lbl}`")
            st.write(f"**Scaler:** {scaler_note}")


# =============================================================================
#  TAB 2 — IMAGE ANALYSIS
# =============================================================================
with tab_image:
    st.markdown(f"""
Each EEG segment is reshaped into a **{IMG_ROWS}×{IMG_COLS} grayscale image**.  
**GLCM** (Gray-Level Co-occurrence Matrix) descriptors capture texture differences
between normal brain activity and ictal (seizure) events.
""")

    col_up2, col_demo2 = st.columns([2, 1])
    with col_up2:
        uploaded2 = st.file_uploader("Upload an EDF file", type=["edf"], key="up2")
    with col_demo2:
        st.markdown("#### Or use the included file")
        use_demo2 = st.button("▶ Run on chb20_02.edf",
                              use_container_width=True, key="d2")

    edf_bytes2, source2 = None, None
    if uploaded2:
        edf_bytes2, source2 = uploaded2.read(), uploaded2.name
    elif use_demo2:
        demo = "chb20_02.edf"
        if os.path.exists(demo):
            with open(demo, "rb") as f:
                edf_bytes2 = f.read()
            source2 = demo
        else:
            st.error(f"Demo file not found: {demo}")

    if edf_bytes2 is not None:
        if not os.path.exists(model_path):
            st.error(f"⛔ Model not found: `{model_path}`")
            st.stop()

        with st.spinner("Generating grayscale images & GLCM descriptors…"):
            result2 = full_analysis(
                edf_bytes2, model_path,
                scaler_path if os.path.exists(scaler_path) else None,
                channel_idx, seg_len,
            )
        if result2 is None:
            st.stop()

        df2, sfreq2, ch_names2, _, segs2, imgs2, descs2 = result2
        df2["_label_t"] = (df2["_prob"] >= threshold).astype(int)
        labels_arr  = df2["_label_t"].values
        start_times = df2["Start (s)"].values
        desc_df     = pd.DataFrame(descs2)
        sz_idxs     = np.where(labels_arr == 1)[0]
        nm_idxs     = np.where(labels_arr == 0)[0]

        # ── Section 1: Single segment viewer ─────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔬 Single Segment Viewer")
        n_segs = len(imgs2)

        s_col, b_col = st.columns([4, 1])
        with s_col:
            seg_idx = st.slider("Select segment", 1, n_segs, 1, 1,
                                key="seg_sl")
        with b_col:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Jump to 1st seizure", use_container_width=True) and len(sz_idxs):
                seg_idx = int(sz_idxs[0]) + 1

        idx      = seg_idx - 1
        img      = imgs2[idx]
        desc     = descs2[idx]
        pred_lbl = labels_arr[idx]
        pred_str = "🔴 SEIZURE" if pred_lbl == 1 else "🟢 Normal"
        border   = "#f87171"   if pred_lbl == 1 else "#4ade80"
        t0_s     = start_times[idx]
        t1_s     = t0_s + seg_len
        prob_pct = df2["_prob"].iloc[idx] * 100

        left, right = st.columns([1, 1])

        with left:
            st.plotly_chart(
                plot_grayscale(
                    img,
                    f"Seg {seg_idx} | {t0_s}s–{t1_s}s | "
                    f"{pred_str} ({prob_pct:.1f}%)",
                    border,
                ),
                use_container_width=True,
            )
            st.caption(
                f"Image shape: {IMG_ROWS}×{IMG_COLS} px  ·  "
                f"Pixel range: {img.min()}–{img.max()}  ·  "
                f"Pixel mean: {img.mean():.1f}"
            )

        with right:
            st.markdown("**GLCM & Histogram Descriptors**")
            d_cols = st.columns(2)
            for ci, (dname, dval) in enumerate(desc.items()):
                with d_cols[ci % 2]:
                    st.markdown(f"""
                    <div class='desc-card'>
                      <p class='desc-name'>{dname}</p>
                      <p class='desc-value' style='color:#e2e8f0'>{dval}</p>
                    </div>""", unsafe_allow_html=True)

        with st.expander("Raw EEG waveform of this segment"):
            t_axis = np.linspace(t0_s, t1_s, len(segs2[idx]))
            fig_w  = go.Figure(go.Scatter(
                x=t_axis, y=segs2[idx],
                mode="lines", line=dict(color=border, width=1),
            ))
            fig_w.update_layout(
                title="Waveform",
                margin=dict(l=10, r=10, t=20, b=20)
            )
            st.plotly_chart(fig_w, use_container_width=True)

        # ── Section 2: Normal vs Seizure comparison ───────────────────────────
        st.markdown("---")
        st.markdown("### 🔁 Normal vs Seizure — Image Comparison")

        if len(sz_idxs) == 0:
            st.info("No seizure segments detected at the current threshold.")
        else:
            nm_pick = int(nm_idxs[len(nm_idxs) // 2])
            sz_pick = int(sz_idxs[0])

            col_nm, col_sz = st.columns(2)
            with col_nm:
                st.markdown("#### Normal segment")
                st.plotly_chart(
                    plot_grayscale(
                        imgs2[nm_pick],
                        f"Seg {nm_pick+1} | {start_times[nm_pick]}s | 🟢 Normal",
                        "#4ade80",
                    ),
                    use_container_width=True,
                )
                st.caption("Smooth, low-contrast, regular texture")

            with col_sz:
                st.markdown("#### Seizure segment")
                st.plotly_chart(
                    plot_grayscale(
                        imgs2[sz_pick],
                        f"Seg {sz_pick+1} | {start_times[sz_pick]}s | 🔴 Seizure",
                        "#f87171",
                    ),
                    use_container_width=True,
                )
                st.caption("High-contrast, irregular, chaotic texture")

            # Pixel histogram comparison
            st.markdown("#### Pixel Intensity Histograms")
            fig_ph = go.Figure()
            for pick, name, color in [
                (nm_pick, "Normal",  "#4ade80"),
                (sz_pick, "Seizure", "#f87171"),
            ]:
                fig_ph.add_trace(go.Histogram(
                    x=imgs2[pick].ravel(), nbinsx=32,
                    name=name, opacity=0.7,
                    marker_color=color,
                    histnorm="probability",
                ))
            fig_ph.update_layout(
                **DARK, height=230, barmode="overlay",
                xaxis_title="Pixel Intensity (0–255)",
                yaxis_title="Probability",
                legend=dict(orientation="h", y=1.12),
                title=dict(text="Intensity distributions", x=0.5,
                           font=dict(size=12)),
            )
            st.plotly_chart(fig_ph, use_container_width=True)

        # ── Section 3: Descriptor statistics ─────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 Image Descriptor Statistics")

        if len(sz_idxs) > 0 and len(nm_idxs) > 0:
            normal_means  = desc_df.iloc[nm_idxs].mean().to_dict()
            seizure_means = desc_df.iloc[sz_idxs].mean().to_dict()

            ra_col, tb_col = st.columns([1, 1])
            with ra_col:
                st.plotly_chart(
                    plot_descriptor_radar(normal_means, seizure_means),
                    use_container_width=True,
                )
            with tb_col:
                st.markdown("**Mean descriptor values**")
                cmp_df = pd.DataFrame({
                    "Descriptor"    : list(normal_means.keys()),
                    "Normal (mean)" : [round(v, 4) for v in normal_means.values()],
                    "Seizure (mean)": [round(v, 4) for v in seizure_means.values()],
                })
                cmp_df["Δ (Sz – Nm)"] = (
                    cmp_df["Seizure (mean)"] - cmp_df["Normal (mean)"]
                ).round(4)
                st.dataframe(cmp_df, use_container_width=True, height=310)

            st.plotly_chart(
                plot_descriptor_boxplots(desc_df, labels_arr),
                use_container_width=True,
            )

        # ── Section 4: Descriptor timeline ───────────────────────────────────
        st.markdown("---")
        st.markdown("### 📈 Descriptor Timeline")
        desc_choice = st.selectbox(
            "Choose descriptor to plot over time:",
            IMG_DESC_NAMES, index=0, key="dtl",
        )
        st.plotly_chart(
            plot_descriptor_timeline(desc_df, start_times,
                                     labels_arr, desc_choice),
            use_container_width=True,
        )

        # ── Section 5: Full descriptor download ──────────────────────────────
        with st.expander("📋 Full image descriptor table (all segments)"):
            full_df = desc_df.copy()
            full_df.insert(0, "Segment",  np.arange(1, len(full_df) + 1))
            full_df.insert(1, "Start (s)", start_times)
            full_df.insert(2, "Label",
                           pd.Series(labels_arr).map({0: "Normal", 1: "Seizure"}))
            st.dataframe(full_df, use_container_width=True, height=350)
            st.download_button(
                "⬇ Download image descriptors CSV",
                full_df.to_csv(index=False),
                "image_descriptors.csv", "text/csv",
                use_container_width=True,
            )


# =============================================================================
#  TAB 3 — ABOUT
# =============================================================================
with tab_about:
    st.markdown(f"""
## About this application

Epileptic seizure detection from EEG `.edf` files, combining an SVM classifier
with GLCM image-descriptor analysis.

---

### 🔍 Predict tab — how it works
1. EDF split into non-overlapping **{12}-second windows** (configurable in sidebar).
2. Six time-domain features extracted per window:
   **Mean, Std Dev, Skewness, Kurtosis, Zero-Crossing Rate, RMS**.
3. Features standardised and fed to the SVM → seizure probability per segment.

---

### 🖼️ Image Analysis tab — how it works

Each 1-D EEG segment (e.g. 3072 samples at 256 Hz × 12 s) is reshaped into a
**{IMG_ROWS}×{IMG_COLS} grayscale image** by tiling the signal into {IMG_ROWS}
rows of {IMG_COLS} amplitude samples each, then normalising to 0–255.

Eight descriptors are computed from each image:

| Descriptor | What it captures |
|---|---|
| **Contrast** | Intensity variation between adjacent pixels |
| **Energy** | Texture uniformity (high = smooth) |
| **Homogeneity** | Proximity of GLCM values to the diagonal |
| **GLCM Entropy** | Randomness of the co-occurrence matrix |
| **Correlation** | Linear dependency of grey-level pairs |
| **Hist Entropy** | Complexity of the pixel intensity distribution |
| **Mean Intensity** | Average pixel brightness |
| **Std Intensity** | Pixel brightness variability |

> **Descriptors are for visual/interpretability analysis only.**  
> The SVM prediction uses the 6 time-domain features above.  
> GLCM features can be used to train or extend a future model.

---

### Model details
| Parameter | Value |
|---|---|
| Kernel | RBF |
| C | 1.329 |
| γ | 0.635 |
| Training data | CHB-MIT Scalp EEG (all subjects) |
| Class balancing | SMOTE |
| Image shape | {IMG_ROWS} × {IMG_COLS} px |
| GLCM grey levels | {GLCM_LEVELS} |

---

### File layout
```
eeg_seizure_app/
├── app.py
├── requirements.txt
├── models/
│   ├── svm_model.pkl   ← provided ✅
│   └── scaler.pkl      ← add from training Cell 14 for best accuracy
└── chb20_02.edf        ← demo recording ✅
```

CHB-MIT Scalp EEG Database — PhysioNet
""")
