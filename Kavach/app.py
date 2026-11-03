# -------------------------------------------------------------
# app.py – Streamlit dashboard for Kavach EdgeInspect
# -------------------------------------------------------------
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import soundfile as sf

# -----------------------------------------------------------------
# Ensure project root is on the Python path
# -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -----------------------------------------------------------------
# Project‑wide configuration
# -----------------------------------------------------------------
from config import (
    CLASS_NAMES,
    DATA_DIR,
    MODEL_PATH,
    QUANTIZED_MODEL_PATH,
    BASELINE_PATH,
    SAMPLE_RATE,
    CLIP_DURATION,
    N_MELS,
    HOP_LENGTH,
)

from preprocess import load_audio, audio_to_logmel, preprocess_audio
from inference import EdgeInference
from baseline import create_baseline, load_baseline, baseline_distance


# -----------------------------------------------------------------
# Helper – compute dataset statistics (file count, total duration,
# total size) for a given folder containing *.wav recordings.
# -----------------------------------------------------------------
def get_dataset_stats(folder: Path) -> dict:
    """Return count, total duration (seconds) and size (MiB) for *.wav files."""
    if not folder.exists():
        return {"count": 0, "duration": 0.0, "size_mb": 0.0}
    wav_files = list(folder.glob("*.wav"))
    total_seconds = 0.0
    total_bytes = 0
    for f in wav_files:
        try:
            info = sf.info(str(f))
            total_seconds += info.frames / info.samplerate
        except Exception:
            pass
        total_bytes += f.stat().st_size
    return {
        "count": len(wav_files),
        "duration": total_seconds,
        "size_mb": total_bytes / (1024 * 1024),
    }


# -----------------------------------------------------------------
# Streamlit page configuration – emojis removed
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Kavach EdgeInspect",
    layout="wide",
)
# -------------------------------------------------
# Banner image at top
# -------------------------------------------------
banner_path = Path(r"C:\Users\Sajai S\.gemini\antigravity-ide\brain\9792dfc3-bc13-41eb-8745-f630697c3f6a\kavach_banner_1790136349090.jpg")
if banner_path.exists():
    st.image(str(banner_path), use_container_width=True)

# -----------------------------------------------------------------
# Dark Snapdragon theme — Google Fonts + glassmorphism + gradients
# -----------------------------------------------------------------
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Orbitron:wght@600;700;900&display=swap" rel="stylesheet">
    <style>
    /* ── Global dark canvas ─────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {
        background: #080d1a !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1629 0%, #0a0e1a 100%) !important;
        border-right: 1px solid rgba(108, 58, 255, 0.25) !important;
    }
    [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #a78bfa !important; }

    /* ── Typography ─────────────────────────────────────── */
    * { font-family: 'Inter', sans-serif !important; }

    /* ── Main header (Orbitron) ─────────────────────────── */
    .main-header {
        font-family: 'Orbitron', monospace !important;
        font-size: 2.6rem;
        font-weight: 900;
        background: linear-gradient(135deg, #6c3aff 0%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.2rem;
        font-weight: 400;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    /* ── Glassmorphism cards ────────────────────────────── */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(108, 58, 255, 0.25);
        border-radius: 14px;
        padding: 18px 22px;
        backdrop-filter: blur(12px);
        margin-bottom: 1rem;
        transition: border-color 0.3s ease;
    }
    .glass-card:hover { border-color: rgba(108, 58, 255, 0.55); }

    /* ── Disclaimer ─────────────────────────────────────── */
    .disclaimer-box {
        background: rgba(251, 191, 36, 0.08);
        border-left: 3px solid #f59e0b;
        padding: 12px 18px;
        color: #fde68a;
        font-size: 0.88rem;
        font-weight: 500;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }

    /* ── Provider pills ─────────────────────────────────── */
    .provider-pill {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        margin: 6px 0;
        text-transform: uppercase;
    }
    .provider-cpu {
        background: rgba(148, 163, 184, 0.12);
        color: #94a3b8;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }
    .provider-qnn {
        background: linear-gradient(135deg, rgba(108,58,255,0.25), rgba(0,212,255,0.25));
        color: #00d4ff;
        border: 1px solid rgba(0, 212, 255, 0.5);
        box-shadow: 0 0 14px rgba(0, 212, 255, 0.2);
    }

    /* ── Streamlit widget overrides ─────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #6c3aff, #00d4ff) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        transition: opacity 0.2s ease, transform 0.15s ease !important;
    }
    .stButton > button:hover {
        opacity: 0.88 !important;
        transform: translateY(-1px) !important;
    }
    .stRadio label, .stSelectbox label {
        color: #94a3b8 !important;
        font-size: 0.9rem !important;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(108,58,255,0.2) !important;
        border-radius: 12px !important;
        padding: 14px !important;
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-family: 'Orbitron', monospace !important;
        font-size: 1.45rem !important;
    }
    [data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #6c3aff, #00d4ff) !important;
        border-radius: 4px !important;
    }
    hr { border-color: rgba(108,58,255,0.2) !important; }
    .stMarkdown table {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 8px;
        overflow: hidden;
    }
    .stMarkdown th {
        background: rgba(108,58,255,0.2) !important;
        color: #a78bfa !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        font-size: 0.78rem;
        letter-spacing: 0.06em;
    }
    .stMarkdown td { color: #cbd5e1 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# Page title & subtitle
# -------------------------------------------------------------
st.markdown('<div class="main-header">Kavach EdgeInspect</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Offline Snapdragon NPU &nbsp;|&nbsp; Motor Condition Triage</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# Permanent required disclaimer
# -------------------------------------------------------------
st.markdown(
    """
    <div class="disclaimer-box">
        Prototype condition triage &mdash; not a certified industrial fault-diagnosis system.
        Results must be verified by a qualified engineer before acting.
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------
# Load the inference engine (cached)
# -------------------------------------------------------------
@st.cache_resource
def get_engine():
    try:
        return EdgeInference()
    except Exception as err:
        return err


engine_or_err = get_engine()


# -------------------------------------------------------------
# Sidebar – system & edge engine status (no emojis)
# -------------------------------------------------------------
with st.sidebar:
    st.header("Edge Engine Status")
    if isinstance(engine_or_err, Exception):
        st.error(f"Inference Engine Unloaded: {engine_or_err}")
        st.info("Export model with: `python scripts/export_onnx.py`")
        current_provider = "Unavailable"
    else:
        current_provider = engine_or_err.provider
        st.write("**Inference Provider:**")
        if current_provider == "QNN HTP":
            st.markdown(
                '<div class="provider-pill provider-qnn">QNN HTP</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="provider-pill provider-cpu">CPU fallback</div>',
                unsafe_allow_html=True,
            )
        st.caption(f"Active Model: `{Path(engine_or_err.model_path).name}`")

    st.markdown("---")
    st.subheader("Reference Datasets")

    # ---------------------------------------------------------
    # Detailed dataset statistics
    # ---------------------------------------------------------
    normal_stats = get_dataset_stats(DATA_DIR / "normal")
    imbalance_stats = get_dataset_stats(DATA_DIR / "imbalance_rubbing")
    maintenance_stats = get_dataset_stats(DATA_DIR / "maintenance_check")

    st.markdown(
        f"""
        | Class                | Files | Total Duration (s) | Size (MiB) |
        |----------------------|-------|--------------------|-----------|
        | Normal               | {normal_stats['count']} | {normal_stats['duration']:.1f} | {normal_stats['size_mb']:.2f} |
        | Imbalance/Rubbing    | {imbalance_stats['count']} | {imbalance_stats['duration']:.1f} | {imbalance_stats['size_mb']:.2f} |
        | Maintenance Check    | {maintenance_stats['count']} | {maintenance_stats['duration']:.1f} | {maintenance_stats['size_mb']:.2f} |
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("Healthy Baseline")
    saved_baseline = load_baseline()
    if saved_baseline is not None:
        st.success(f"Baseline active ({BASELINE_PATH.name})")
    else:
        st.warning("No healthy baseline found.")


# -------------------------------------------------------------
# 1️⃣ Audio input
# -------------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Audio input")
    input_mode = st.radio(
        "Audio Source",
        ["Select sample WAV", "Upload WAV"],
        horizontal=True,
    )

    selected_audio_path = None

    if input_mode == "Select sample WAV":
        sample_files = []
        if DATA_DIR.exists():
            sample_files = sorted(list(DATA_DIR.rglob("*.wav")))
        if sample_files:
            sample_labels = [f"{p.parent.name}/{p.name}" for p in sample_files]
            chosen_label = st.selectbox("Choose sample recording:", sample_labels)
            selected_audio_path = sample_files[sample_labels.index(chosen_label)]
        else:
            st.info("No sample WAV files currently located in `data/` subdirectories.")
    else:
        uploaded_file = st.file_uploader("Upload WAV audio file (16 kHz, mono)", type=["wav"])
        if uploaded_file is not None:
            temp_dir = Path(tempfile.gettempdir()) / "kavach_audio"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            selected_audio_path = temp_path

    if selected_audio_path is not None and Path(selected_audio_path).exists():
        st.audio(str(selected_audio_path), format="audio/wav")
        st.caption(f"Active Audio File: `{Path(selected_audio_path).name}`")

# -------------------------------------------------------------
# 2️⃣ Healthy baseline
# -------------------------------------------------------------
with col2:
    st.subheader("2. Healthy baseline")
    st.write(
        "Calibrate the median acoustic signature using reference healthy recordings from `data/normal/`."
    )

    available_normal = (
        list(DATA_DIR.glob("normal/*.wav")) if (DATA_DIR / "normal").exists() else []
    )
    if available_normal:
        st.write(f"Found **{len(available_normal)}** healthy recordings in `data/normal/`.")
        if st.button("Calibrate / Update Baseline"):
            try:
                create_baseline(available_normal)
                st.success(
                    "Healthy baseline calculated and saved to `models/healthy_baseline.json`!"
                )
                st.rerun()
            except Exception as e:
                st.error(f"Failed to calculate baseline: {e}")
    else:
        st.caption("No recordings in `data/normal/` yet. Add healthy WAV files to enable baseline generation.")

    current_b = load_baseline()
    if current_b is not None:
        st.info(f"Healthy baseline profile loaded ({len(current_b)} features).")
    else:
        st.caption("No healthy baseline saved yet.")


st.markdown("---")

# -------------------------------------------------------------
# 3️⃣ Run inspection
# -------------------------------------------------------------
st.subheader("3. Run inspection")

inspect_clicked = st.button("Run Inspection", type="primary", use_container_width=True)

if inspect_clicked:
    if selected_audio_path is None or not Path(selected_audio_path).exists():
        st.error("Please upload a WAV file or select a sample WAV first.")
    elif isinstance(engine_or_err, Exception):
        st.error(f"Inference engine unavailable: {engine_or_err}")
    else:
        with st.spinner("Executing edge model inference..."):
            try:
                res = engine_or_err.predict(selected_audio_path)

                # Baseline distance computation
                b_dist = None
                base_vec = load_baseline()
                if base_vec is not None:
                    feat = preprocess_audio(selected_audio_path).flatten()
                    b_dist = baseline_distance(feat, base_vec)

                st.session_state["inspection_result"] = res
                st.session_state["baseline_dist"] = b_dist
                st.session_state["inspected_path"] = str(selected_audio_path)
            except Exception as e:
                st.error(f"Inference execution failed: {e}")

# -------------------------------------------------------------
# 4️⃣ Result
# -------------------------------------------------------------
if "inspection_result" in st.session_state:
    res = st.session_state["inspection_result"]
    b_dist = st.session_state.get("baseline_dist")
    predicted_class = res["class_name"]
    conf = res["confidence"]
    latency = res["latency_ms"]
    probs = res["probabilities"]

    st.markdown("---")
    st.subheader("4. Result")

    r_col1, r_col2, r_col3, r_col4 = st.columns([1.5, 1, 1, 1])

    badge_bg = "#16A34A" if predicted_class == "Normal" else "#EA580C" if predicted_class == "Imbalance/Rubbing Pattern" else "#DC2626"

    with r_col1:
        st.markdown(
            f"""
            <div style="background-color: {badge_bg}; color: white; padding: 14px 18px; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">Triage Classification</div>
                <div style="font-size: 1.35rem; font-weight: 700; margin-top: 4px;">{predicted_class}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r_col2:
        st.metric("Confidence", f"{conf * 100:.1f} %")
    with r_col3:
        st.metric("Latency", f"{latency:.2f} ms")
    with r_col4:
        if b_dist is not None:
            st.metric("Healthy Baseline Distance", f"{b_dist:.3f}")
        else:
            st.metric("Healthy Baseline Distance", "N/A")

    st.write("**Class Probabilities:**")
    p_cols = st.columns(3)
    for i, cname in enumerate(CLASS_NAMES):
        with p_cols[i]:
            st.write(f"**{cname}**")
            st.progress(float(probs[i]))
            st.caption(f"{probs[i] * 100:.2f}%")

# -------------------------------------------------------------
# 5️⃣ Spectrogram
# -------------------------------------------------------------
active_spec_path = None
if "inspected_path" in st.session_state and Path(st.session_state["inspected_path"]).exists():
    active_spec_path = st.session_state["inspected_path"]
elif selected_audio_path is not None and Path(selected_audio_path).exists():
    active_spec_path = selected_audio_path

st.markdown("---")
st.subheader("5. Spectrogram")

if active_spec_path is not None:
    try:
        raw_audio = load_audio(active_spec_path)
        logmel = audio_to_logmel(raw_audio)  # (64, 376)

        fig, ax = plt.subplots(figsize=(10, 3.2), dpi=120)
        cax = ax.imshow(
            logmel,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[0, CLIP_DURATION, 0, N_MELS],
        )
        ax.set_title("Log-Mel Spectrogram (64 Mels, 16 kHz, 6.0s Duration)", fontsize=11, fontweight="600")
        ax.set_xlabel("Time (seconds)", fontsize=9)
        ax.set_ylabel("Mel Frequency Bins", fontsize=9)
        cbar = fig.colorbar(cax, ax=ax, format="%+2.1f dB")
        cbar.ax.tick_params(labelsize=8)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Error rendering log-mel spectrogram: {e}")
else:
    st.info("Upload or select a WAV audio file to visualize the log-mel spectrogram.")

# -------------------------------------------------------------
# 6️⃣ Edge inference information
# -------------------------------------------------------------
st.markdown("---")
st.subheader("6. Edge inference information")

e_col1, e_col2, e_col3 = st.columns(3)

with e_col1:
    st.markdown("**Inference Provider:**")
    if not isinstance(engine_or_err, Exception):
        if engine_or_err.provider == "QNN HTP":
            st.markdown('<div class="provider-pill provider-qnn">QNN HTP</div>', unsafe_allow_html=True)
            st.caption("Qualcomm Hexagon Tensor Processor (Snapdragon NPU)")
        else:
            st.markdown('<div class="provider-pill provider-cpu">CPU fallback</div>', unsafe_allow_html=True)
            st.caption("Standard CPUExecutionProvider")
    else:
        st.write("Unavailable")

with e_col2:
    st.markdown("**Edge Model Architecture:**")
    st.write("`MotorConditionCNN` (Fixed Batch=1, 1×1×64×376)")
    if not isinstance(engine_or_err, Exception):
        st.caption(f"Loaded: `{Path(engine_or_err.model_path).name}`")

with e_col3:
    st.markdown("**Inference Latency:**")
    if "inspection_result" in st.session_state:
        st.write(f"`{st.session_state['inspection_result']['latency_ms']:.2f} ms`")
    else:
        st.write("Awaiting inspection run")
    st.caption("Measured directly on edge runtime")
