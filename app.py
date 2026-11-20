"""
Moodly – Feel the Sentiment (Streamlit Application).

Serves the trained PyTorch SentimentLSTM model with exact training-time
preprocessing, interactive confidence visualizations, technical token
diagnostics, and session-scoped prediction history.

Theme: Magenta + Light Pink  |  Display font: DM Serif Display
Body font: Source Sans 3     |  Numbers / code: DM Mono
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import (
    EMBEDDING_DIM,
    HIDDEN_DIM,
    MAX_SEQ_LEN,
    VOCAB_SIZE,
    PredictionResult,
    SentimentLSTM,
    load_trained_model,
    load_vocabulary,
    predict_sentiment,
)

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pth"
VOCAB_PATH = MODELS_DIR / "vocab.json"

APP_NAME = "Moodly"
APP_TAGLINE = "Feel the sentiment"

# Palette (kept in one place so Plotly + CSS stay in sync)
MAGENTA = "#D6006F"
MAGENTA_DEEP = "#B0125F"
PINK_SOFT = "#FFD6EA"
PINK_LIGHT = "#FFF0F7"
PLUM = "#4A1D3F"

POSITIVE_EXAMPLE = (
    "I absolutely loved this movie. The acting was outstanding and the storyline was fantastic."
)
NEGATIVE_EXAMPLE = (
    "This movie was extremely disappointing. The story was boring and the acting was terrible."
)


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_NAME} – {APP_TAGLINE}",
    page_icon="💖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS (Magenta & Light Pink)
# ---------------------------------------------------------------------------
def inject_custom_css() -> None:
    """Apply responsive custom CSS to elevate the Streamlit interface."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Source+Sans+3:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

        :root {
            --magenta: #D6006F;
            --magenta-deep: #B0125F;
            --magenta-bright: #F2479A;
            --pink-soft: #FFD6EA;
            --pink-light: #FFF0F7;
            --pink-border: #F9C6DE;
            --plum: #4A1D3F;
            --ink: #3B0A25;
            --muted: #8A5A74;
        }

        html, body, [class*="css"] {
            font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--ink);
        }

        h1, h2, h3, h4, h5 {
            font-family: 'DM Serif Display', Georgia, serif !important;
            font-weight: 400 !important;
            letter-spacing: -0.01em;
            color: var(--ink);
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 0%, #FFE0EF 0, transparent 38%),
                radial-gradient(circle at 100% 18%, #FFD1E8 0, transparent 32%),
                #FFF8FC;
        }

        [data-testid="stHeader"] { background: transparent; }

        .main .block-container {
            max-width: 1120px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* ---------------- Sidebar ---------------- */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #FFF5FA 0%, #FFE3F0 100%);
            border-right: 1px solid var(--pink-border);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            margin: 0.5rem 0 0.25rem 0;
        }

        .brand-ring {
            width: 52px;
            height: 52px;
            border-radius: 50%;
            border: 2px solid var(--magenta);
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'DM Serif Display', serif;
            font-style: italic;
            font-size: 1.7rem;
            color: var(--magenta);
            background: rgba(255, 255, 255, 0.7);
            box-shadow: 0 6px 16px -6px rgba(214, 0, 111, 0.45);
        }

        .brand-name {
            font-family: 'DM Serif Display', serif;
            font-size: 2rem;
            color: var(--magenta-deep);
            line-height: 1;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label {
            padding: 0.45rem 0.75rem;
            border-radius: 12px;
            transition: background 0.15s ease;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(214, 0, 111, 0.08);
        }

        [data-testid="stSidebar"] hr { border-color: var(--pink-border); }

        .sidebar-status-title {
            font-family: 'DM Serif Display', serif;
            font-size: 1.15rem;
            color: var(--ink);
            margin-bottom: 0.4rem;
        }

        /* ---------------- Hero ---------------- */
        .hero-card {
            position: relative;
            overflow: hidden;
            text-align: center;
            color: #FFFFFF;
            background: linear-gradient(160deg, #B0125F 0%, #D6006F 48%, #F2479A 100%);
            border-radius: 26px;
            padding: 3rem 2rem 5.5rem 2rem;
            margin-bottom: 1.9rem;
            box-shadow: 0 22px 40px -18px rgba(214, 0, 111, 0.55);
        }

        /* soft "mountain range" silhouettes, echoing the reference hero */
        .hero-card::before {
            content: "";
            position: absolute;
            left: -4%;
            right: -4%;
            bottom: 0;
            height: 58%;
            background: rgba(255, 255, 255, 0.13);
            clip-path: polygon(0 100%, 0 64%, 11% 42%, 21% 60%, 37% 16%, 51% 54%, 63% 30%, 77% 62%, 90% 38%, 100% 56%, 100% 100%);
        }

        .hero-card::after {
            content: "";
            position: absolute;
            left: -4%;
            right: -4%;
            bottom: 0;
            height: 36%;
            background: rgba(255, 224, 239, 0.30);
            clip-path: polygon(0 100%, 0 70%, 14% 50%, 30% 72%, 46% 36%, 60% 66%, 74% 44%, 88% 70%, 100% 52%, 100% 100%);
        }

        .hero-inner { position: relative; z-index: 2; }

        .hero-hello {
            font-family: 'DM Serif Display', serif;
            font-style: italic;
            font-size: 1.25rem;
            color: #FFE4F1;
            margin-bottom: 0.1rem;
        }

        .hero-title {
            font-family: 'DM Serif Display', Georgia, serif;
            font-size: clamp(3.2rem, 8vw, 5.4rem);
            font-weight: 400;
            letter-spacing: -0.02em;
            line-height: 1.02;
            margin: 0 0 0.65rem 0;
            color: #FFFFFF;
            text-shadow: 0 6px 24px rgba(74, 29, 63, 0.35);
        }

        .hero-subtitle {
            font-size: 1.12rem;
            color: #FFE9F4;
            margin: 0 auto 1.4rem auto;
            max-width: 560px;
            line-height: 1.55;
        }

        .hero-pills {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 0.65rem;
        }

        /* outlined button look from the reference page */
        .hero-pill {
            border: 1.5px solid rgba(255, 255, 255, 0.75);
            color: #FFFFFF;
            padding: 0.4rem 1.05rem;
            border-radius: 999px;
            font-size: 0.86rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(2px);
        }

        .hero-ring {
            position: absolute;
            z-index: 2;
            left: 20px;
            bottom: 16px;
            width: 54px;
            height: 54px;
            border-radius: 50%;
            border: 2px solid rgba(255, 255, 255, 0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'DM Serif Display', serif;
            font-style: italic;
            font-size: 1.7rem;
            color: #FFFFFF;
        }

        /* ---------------- Headings & captions ---------------- */
        .section-heading {
            font-family: 'DM Serif Display', serif;
            font-size: 1.85rem;
            color: var(--ink);
            margin-bottom: 0.15rem;
            letter-spacing: -0.01em;
        }

        .section-heading::after {
            content: "";
            display: block;
            width: 54px;
            height: 3px;
            border-radius: 999px;
            margin-top: 0.35rem;
            background: linear-gradient(90deg, var(--magenta), var(--magenta-bright));
        }

        .section-caption {
            font-size: 0.98rem;
            color: var(--muted);
            margin: 0.6rem 0 1.1rem 0;
        }

        /* ---------------- Counters ---------------- */
        .counter-row {
            display: flex;
            flex-wrap: wrap;
            gap: 1.25rem;
            font-family: 'DM Mono', monospace;
            font-size: 0.8rem;
            color: var(--muted);
            padding: 0.6rem 1rem;
            background: #FFFFFF;
            border: 1px solid var(--pink-border);
            border-radius: 999px;
            margin-top: 0.4rem;
            margin-bottom: 1rem;
            width: fit-content;
            max-width: 100%;
        }

        .counter-row strong { color: var(--magenta-deep); }

        /* ---------------- Result cards ---------------- */
        .result-card-positive {
            background: linear-gradient(135deg, #FFF0F7 0%, #FFD6EA 100%);
            border: 1px solid #F7A8CF;
            border-radius: 20px;
            padding: 1.6rem 1.85rem;
            margin-top: 1rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 14px 28px -14px rgba(214, 0, 111, 0.4);
        }

        .result-card-negative {
            background: linear-gradient(135deg, #F8F0F6 0%, #E7D3E2 100%);
            border: 1px solid #C9A3BF;
            border-radius: 20px;
            padding: 1.6rem 1.85rem;
            margin-top: 1rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 14px 28px -14px rgba(74, 29, 63, 0.4);
        }

        .result-kicker {
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: var(--muted);
            margin-bottom: 0.25rem;
        }

        .result-headline-pos {
            font-family: 'DM Serif Display', serif;
            font-size: 2.4rem;
            color: var(--magenta);
            margin: 0 0 0.5rem 0;
            line-height: 1.1;
        }

        .result-headline-neg {
            font-family: 'DM Serif Display', serif;
            font-size: 2.4rem;
            color: var(--plum);
            margin: 0 0 0.5rem 0;
            line-height: 1.1;
        }

        .prob-container {
            background: rgba(255, 255, 255, 0.85);
            border: 1px solid var(--pink-border);
            border-radius: 16px;
            padding: 1.1rem 1.25rem;
            margin-top: 0.9rem;
        }

        .prob-label-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 0.35rem;
            color: var(--ink);
        }

        .prob-value-mono {
            font-family: 'DM Mono', monospace;
            font-variant-numeric: tabular-nums;
        }

        .prob-track {
            width: 100%;
            height: 12px;
            background: #FBE3EF;
            border-radius: 999px;
            overflow: hidden;
            margin-bottom: 0.85rem;
        }

        .prob-fill-pos {
            height: 100%;
            background: linear-gradient(90deg, #D6006F 0%, #FF4FA3 100%);
            border-radius: 999px;
        }

        .prob-fill-neg {
            height: 100%;
            background: linear-gradient(90deg, #4A1D3F 0%, #8A4B7A 100%);
            border-radius: 999px;
        }

        /* ---------------- Buttons ---------------- */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #B0125F 0%, #E6007E 100%);
            color: #FFFFFF;
            border: none;
            border-radius: 999px;
            padding: 0.62rem 1.4rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            box-shadow: 0 10px 20px -8px rgba(214, 0, 111, 0.55);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 26px -8px rgba(214, 0, 111, 0.65);
            color: #FFFFFF;
        }

        /* outlined (ghost) buttons – inspired by the "Discover More" button */
        div.stButton > button[kind="secondary"],
        div.stDownloadButton > button {
            background: transparent;
            color: var(--magenta-deep);
            border: 1.5px solid var(--magenta);
            border-radius: 999px;
            padding: 0.55rem 1.1rem;
            font-weight: 600;
            transition: all 0.15s ease;
        }

        div.stButton > button[kind="secondary"]:hover,
        div.stDownloadButton > button:hover {
            background: var(--magenta);
            border-color: var(--magenta);
            color: #FFFFFF;
        }

        div.stButton > button[kind="secondary"]:hover p,
        div.stDownloadButton > button:hover p {
            color: #FFFFFF;
        }

        div.stButton > button:disabled { opacity: 0.45; }

        /* ---------------- Text area ---------------- */
        textarea[aria-label="Enter text for sentiment analysis"] {
            background-color: #FFFFFF !important;
            border: 1.5px solid var(--pink-border) !important;
            border-radius: 18px !important;
            font-size: 1.02rem !important;
            line-height: 1.65 !important;
            color: var(--ink) !important;
            padding: 1rem 1.1rem !important;
        }

        textarea[aria-label="Enter text for sentiment analysis"]:focus {
            border-color: var(--magenta) !important;
            box-shadow: 0 0 0 4px rgba(214, 0, 111, 0.14) !important;
        }

        /* ---------------- Metrics, alerts, expanders ---------------- */
        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid var(--pink-border);
            border-radius: 16px;
            padding: 0.85rem 1rem;
        }

        [data-testid="stMetricValue"] {
            font-family: 'DM Serif Display', serif;
            color: var(--magenta-deep);
        }

        [data-testid="stAlert"] {
            border-radius: 16px;
            background: var(--pink-light);
            border: 1px solid var(--pink-border);
        }

        [data-testid="stAlert"] * { color: #6B0F3C; }

        [data-testid="stExpander"] {
            border: 1px solid var(--pink-border);
            border-radius: 16px;
            background: #FFFFFF;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--pink-border);
            border-radius: 14px;
            overflow: hidden;
        }

        hr { border-color: var(--pink-border) !important; }

        code {
            font-family: 'DM Mono', monospace !important;
            color: var(--magenta-deep);
            background: #FFE4F1;
            border-radius: 6px;
        }

        /* ---------------- About page ---------------- */
        .about-intro {
            background: #FFFFFF;
            border: 1px solid var(--pink-border);
            border-left: 5px solid var(--magenta);
            border-radius: 16px;
            padding: 1.3rem 1.5rem;
            font-size: 1.08rem;
            line-height: 1.65;
            margin-bottom: 1.4rem;
        }

        .about-intro em {
            font-family: 'DM Serif Display', serif;
            color: var(--magenta);
            font-size: 1.15em;
        }

        .spec-card {
            background: #FFFFFF;
            border: 1px solid var(--pink-border);
            border-radius: 18px;
            padding: 1.05rem 1.2rem;
            height: 100%;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        .spec-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 14px 24px -14px rgba(214, 0, 111, 0.5);
        }

        .spec-label {
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--muted);
            margin-bottom: 0.2rem;
        }

        .spec-value {
            font-family: 'DM Serif Display', serif;
            font-size: 1.45rem;
            color: var(--magenta-deep);
            font-variant-numeric: tabular-nums;
        }

        .spec-card.highlight {
            background: linear-gradient(135deg, #D6006F 0%, #F2479A 100%);
            border: none;
        }

        .spec-card.highlight .spec-label { color: #FFE4F1; }
        .spec-card.highlight .spec-value { color: #FFFFFF; }

        .note-card {
            background: linear-gradient(135deg, #FFF0F7 0%, #FFE0EF 100%);
            border: 1px dashed #F08CBD;
            border-radius: 18px;
            padding: 1.2rem 1.5rem;
            line-height: 1.65;
            margin-bottom: 1.5rem;
        }

        .step-card {
            display: flex;
            gap: 1rem;
            align-items: flex-start;
            background: #FFFFFF;
            border: 1px solid var(--pink-border);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.75rem;
        }

        .step-num {
            flex: 0 0 auto;
            width: 38px;
            height: 38px;
            border-radius: 50%;
            background: linear-gradient(135deg, #B0125F, #F2479A);
            color: #FFFFFF;
            font-family: 'DM Serif Display', serif;
            font-size: 1.15rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .step-title {
            font-family: 'DM Serif Display', serif;
            font-size: 1.2rem;
            color: var(--ink);
            margin-bottom: 0.1rem;
        }

        .step-body {
            color: #5C3A4D;
            line-height: 1.55;
            font-size: 0.97rem;
        }

        /* ---------------- Footer ---------------- */
        .app-footer {
            margin-top: 2.75rem;
            padding: 1.25rem 1.5rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #FFE4F1 0%, #FFD0E6 100%);
            border: 1px solid var(--pink-border);
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            font-size: 0.88rem;
            color: var(--muted);
        }

        .footer-brand {
            font-family: 'DM Serif Display', serif;
            font-size: 1.35rem;
            color: var(--magenta-deep);
        }

        .footer-brand em { color: var(--muted); font-size: 0.95rem; }

        @media (max-width: 768px) {
            .hero-card { padding: 2.2rem 1.25rem 5rem 1.25rem; }
            .main .block-container {
                padding-top: 1rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached Model & Vocabulary Loader
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Warming up Moodly's sentiment model...")
def _load_cached_artifacts() -> tuple[SentimentLSTM, dict[str, int], Any]:
    """Load the trained PyTorch model and vocabulary exactly once once files exist."""
    vocab = load_vocabulary(VOCAB_PATH)
    model, device = load_trained_model(MODEL_PATH, vocab_size=VOCAB_SIZE)
    return model, vocab, device


def load_artifacts() -> tuple[SentimentLSTM | None, dict[str, int] | None, Any, str | None]:
    """
    Validate artifact presence first so missing files are never cached, then load once.

    Returns:
        (model, vocab, device, error_message)
    """
    if not VOCAB_PATH.is_file():
        return (
            None,
            None,
            None,
            f"Vocabulary file not found at '{VOCAB_PATH}'. "
            "Place your exported 'vocab.json' inside the 'models/' directory.",
        )
    if not MODEL_PATH.is_file():
        return (
            None,
            None,
            None,
            f"Model weights file not found at '{MODEL_PATH}'. "
            "Place your trained 'model.pth' inside the 'models/' directory.",
        )

    try:
        model, vocab, device = _load_cached_artifacts()
        return model, vocab, device, None
    except Exception as exc:
        _load_cached_artifacts.clear()
        return None, None, None, str(exc)


# ---------------------------------------------------------------------------
# Session State Helpers
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialize required Streamlit session state keys."""
    if "review_input" not in st.session_state:
        st.session_state["review_input"] = ""
    if "latest_result" not in st.session_state:
        st.session_state["latest_result"] = None
    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []


def set_example_text(example_text: str) -> None:
    """Populate the text area with an example review."""
    st.session_state["review_input"] = example_text


def clear_text_input() -> None:
    """Clear the text input area and active prediction display."""
    st.session_state["review_input"] = ""
    st.session_state["latest_result"] = None


def clear_prediction_history() -> None:
    """Clear session-only prediction history."""
    st.session_state["prediction_history"] = []


def shorten_text(text: str, max_chars: int = 85) -> str:
    """Truncate long input text cleanly for table display."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 3]}..."


def append_to_history(result: PredictionResult) -> None:
    """Append a real model prediction to session state (never persisted to disk)."""
    record = {
        "Timestamp": datetime.now().strftime("%H:%M:%S"),
        "Original Text (Shortened)": shorten_text(result.encoded.original_text),
        "Predicted Sentiment": result.sentiment,
        "Positive Probability": f"{result.positive_probability * 100:.2f}%",
        "Negative Probability": f"{result.negative_probability * 100:.2f}%",
        "Tokens Used": result.encoded.used_token_count,
    }
    st.session_state["prediction_history"].insert(0, record)


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------
def render_header() -> None:
    """Render the branded Moodly hero banner."""
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-inner">
                <div class="hero-hello">Hello, mood reader!</div>
                <h1 class="hero-title">{APP_NAME}</h1>
                <p class="hero-subtitle">
                    Understand the emotion behind every review. Drop in some words
                    and let Moodly tell you how they really feel.
                </p>
                <div class="hero-pills">
                    <span class="hero-pill">✦ Deep Learning</span>
                    <span class="hero-pill">✦ Instant Confidence</span>
                    <span class="hero-pill">✦ Private Session</span>
                </div>
            </div>
            <div class="hero-ring">m</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_missing_artifacts_notice(error_message: str) -> None:
    """Display clear placement instructions when model.pth or vocab.json is absent."""
    st.error("Trained model artifacts are not ready for inference.")
    st.markdown(
        f"""
        **Details:** `{error_message}`

        To enable real PyTorch LSTM predictions, place your trained Google Colab files in the `models/` folder:

        ```text
        Moodly/
        └── models/
            ├── model.pth    # Saved SentimentLSTM state_dict
            └── vocab.json   # Word-to-index dictionary (52,692 tokens)
        ```

        Once placed, click **Rerun** or refresh this page. No retraining or mock predictions are performed.
        """
    )


def render_probability_chart(positive_prob: float, negative_prob: float) -> None:
    """Render a Plotly horizontal bar visualization of class probabilities."""
    pos_pct = positive_prob * 100.0
    neg_pct = negative_prob * 100.0
    dominant = "Positive" if positive_prob >= 0.5 else "Negative"

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=["Class Probability"],
            x=[pos_pct],
            name="Positive",
            orientation="h",
            marker=dict(color=MAGENTA),
            text=[f"Positive {pos_pct:.1f}%"],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#FFFFFF", size=13, family="DM Mono, monospace"),
            hovertemplate="Positive Probability: %{x:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=["Class Probability"],
            x=[neg_pct],
            name="Negative",
            orientation="h",
            marker=dict(color=PLUM),
            text=[f"Negative {neg_pct:.1f}%"],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#FFFFFF", size=13, family="DM Mono, monospace"),
            hovertemplate="Negative Probability: %{x:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        barmode="stack",
        height=135,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans 3, sans-serif", color="#3B0A25"),
        xaxis=dict(
            range=[0, 100],
            ticksuffix="%",
            showgrid=True,
            gridcolor="#F9D5E8",
            zeroline=False,
        ),
        yaxis=dict(showticklabels=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="left",
            x=0,
        ),
    )

    st.markdown("#### Confidence Visualization")
    st.caption(
        f"Dominant sentiment: **{dominant}** · These percentages represent the model's "
        "predicted class probabilities from `torch.sigmoid(logit)`, not guaranteed correctness."
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_prediction_result(result: PredictionResult) -> None:
    """Render the prediction result card, probability bars, chart, and technical expander."""
    pos_pct = result.positive_probability * 100.0
    neg_pct = result.negative_probability * 100.0
    is_positive = result.sentiment == "Positive"

    card_class = "result-card-positive" if is_positive else "result-card-negative"
    headline_class = "result-headline-pos" if is_positive else "result-headline-neg"
    emoji = "💖" if is_positive else "🌧️"
    label_upper = result.sentiment.upper()
    pos_color = MAGENTA_DEEP
    neg_color = PLUM

    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="result-kicker">Moodly says it feels</div>
            <div class="{headline_class}">{emoji} {label_upper}</div>
            <div style="font-size: 0.98rem; color: #5C3A4D;">
                Positive Probability: <strong class="prob-value-mono">{pos_pct:.2f}%</strong>
                &nbsp;·&nbsp;
                Negative Probability: <strong class="prob-value-mono">{neg_pct:.2f}%</strong>
            </div>
            <div class="prob-container">
                <div class="prob-label-row">
                    <span>Positive Sentiment Probability</span>
                    <span class="prob-value-mono" style="color: {pos_color};">{pos_pct:.2f}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill-pos" style="width: {min(max(pos_pct, 0.0), 100.0):.2f}%;"></div>
                </div>
                <div class="prob-label-row">
                    <span>Negative Sentiment Probability</span>
                    <span class="prob-value-mono" style="color: {neg_color};">{neg_pct:.2f}%</span>
                </div>
                <div class="prob-track" style="margin-bottom: 0;">
                    <div class="prob-fill-neg" style="width: {min(max(neg_pct, 0.0), 100.0):.2f}%;"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_probability_chart(result.positive_probability, result.negative_probability)

    # Expandable Analysis Information
    enc = result.encoded
    with st.expander("Technical Analysis & Tokenization Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Words (Raw Input)", f"{enc.word_count:,}")
        col2.metric("Characters", f"{enc.char_count:,}")
        col3.metric("Tokens After Preprocessing", f"{enc.preprocessed_token_count:,}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Unknown Words (<UNK>)", f"{enc.unknown_word_count:,}")
        col5.metric(f"Tokens Used by LSTM (Max {MAX_SEQ_LEN})", f"{enc.used_token_count:,}")
        col6.metric("Inference Device", result.device.upper())

        st.markdown("**Original Input Text**")
        st.text(enc.original_text)

        st.markdown("**Cleaned Text (Post-Regex Pipeline)**")
        st.code(enc.cleaned_text if enc.cleaned_text else "<empty -> fallback to [<UNK>]>", language="text")


def render_analyzer_section(
    model: SentimentLSTM | None,
    vocab: dict[str, int] | None,
    device: Any,
    load_error: str | None,
) -> None:
    """Render the main text input, example buttons, inference trigger, and results."""
    st.markdown('<div class="section-heading">How are the words feeling?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Enter an English movie review, customer feedback, or any passage below.</div>',
        unsafe_allow_html=True,
    )

    if load_error is not None:
        render_missing_artifacts_notice(load_error)

    # Example Review Buttons
    ex_col1, ex_col2, _ = st.columns([1.4, 1.4, 2.2])
    with ex_col1:
        st.button(
            "Try a Positive Example",
            on_click=set_example_text,
            args=(POSITIVE_EXAMPLE,),
            use_container_width=True,
        )
    with ex_col2:
        st.button(
            "Try a Negative Example",
            on_click=set_example_text,
            args=(NEGATIVE_EXAMPLE,),
            use_container_width=True,
        )

    user_text = st.text_area(
        label="Enter text for sentiment analysis",
        key="review_input",
        height=170,
        placeholder="Type or paste your movie review, feedback, or any English text here...",
        label_visibility="collapsed",
    )

    # Live Word and Character Counters
    word_count = len(user_text.split()) if user_text.strip() else 0
    char_count = len(user_text)
    st.markdown(
        f"""
        <div class="counter-row">
            <span>Words: <strong>{word_count:,}</strong></span>
            <span>Characters: <strong>{char_count:,}</strong></span>
            <span>Sequence Limit: <strong>{MAX_SEQ_LEN} tokens</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Action Buttons
    btn_col1, btn_col2, _ = st.columns([1.5, 1.0, 2.5])
    with btn_col1:
        analyze_clicked = st.button(
            "Analyze Mood ✦",
            type="primary",
            use_container_width=True,
            disabled=(model is None or vocab is None),
        )
    with btn_col2:
        st.button(
            "Clear Text",
            type="secondary",
            on_click=clear_text_input,
            use_container_width=True,
        )

    if analyze_clicked:
        if not user_text or not user_text.strip():
            st.warning("Please enter some English text before clicking Analyze Mood.")
        elif model is None or vocab is None:
            st.error("Model artifacts are unavailable. Check the models/ directory.")
        else:
            try:
                with st.spinner("Reading the mood..."):
                    result = predict_sentiment(
                        text=user_text,
                        model=model,
                        vocab=vocab,
                        device=device,
                    )
                st.session_state["latest_result"] = result
                append_to_history(result)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    if st.session_state["latest_result"] is not None:
        render_prediction_result(st.session_state["latest_result"])


def render_history_section(compact: bool = False) -> None:
    """Render session-scoped prediction history table, CSV download, and clear button."""
    st.markdown('<div class="section-heading">Prediction History</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Session-only predictions stored in memory. Nothing is saved permanently.</div>',
        unsafe_allow_html=True,
    )

    history: list[dict[str, Any]] = st.session_state.get("prediction_history", [])
    if not history:
        st.info("No moods read yet in this session. Analyze a review above to fill this table.")
        return

    df = pd.DataFrame(history)

    if not compact:
        total_count = len(df)
        pos_count = int(np.sum(df["Predicted Sentiment"] == "Positive"))
        neg_count = int(np.sum(df["Predicted Sentiment"] == "Negative"))
        m1, m2, m3 = st.columns(3)
        m1.metric("Session Predictions", f"{total_count:,}")
        m2.metric("Positive Predictions", f"{pos_count:,}")
        m3.metric("Negative Predictions", f"{neg_count:,}")

    display_df = df[
        [
            "Original Text (Shortened)",
            "Predicted Sentiment",
            "Positive Probability",
            "Negative Probability",
        ]
    ]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    dl_col, clr_col, _ = st.columns([1.4, 1.1, 2.5])
    with dl_col:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download as CSV",
            data=csv_bytes,
            file_name="moodly_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with clr_col:
        st.button(
            "Clear History",
            type="secondary",
            on_click=clear_prediction_history,
            use_container_width=True,
        )


def render_about_model_section() -> None:
    """Render the About page: a friendly intro, model facts, scope note, and pipeline steps."""
    st.markdown('<div class="section-heading">About Moodly</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">What powers Moodly, what it learned from, and how it reads your words.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="about-intro">
            Moodly is a compact deep-learning sentiment reader. Give it any English text and
            a PyTorch <em>LSTM</em> network weighs the words in order, then tells you whether
            the overall feeling leans <strong>positive</strong> or <strong>negative</strong>,
            along with how confident it is.
        </div>
        """,
        unsafe_allow_html=True,
    )

    specs = [
        ("Model", "LSTM"),
        ("Framework", "PyTorch"),
        ("Dataset", "IMDb Reviews"),
        ("Training Samples", "25,000"),
        ("Testing Samples", "25,000"),
        ("Embedding Dimension", f"{EMBEDDING_DIM}"),
        ("Hidden Dimension", f"{HIDDEN_DIM}"),
        ("Max Sequence Length", f"{MAX_SEQ_LEN}"),
        ("Vocabulary Size", f"{VOCAB_SIZE:,}"),
        ("Epochs", "10"),
        ("Test Accuracy", "84.09%"),
        ("Classes", "Positive · Negative"),
    ]

    for idx in range(0, len(specs), 3):
        cols = st.columns(3)
        for col, (label, value) in zip(cols, specs[idx : idx + 3]):
            highlight = " highlight" if label == "Test Accuracy" else ""
            with col:
                st.markdown(
                    f"""
                    <div class="spec-card{highlight}" style="margin-bottom: 0.85rem;">
                        <div class="spec-label">{label}</div>
                        <div class="spec-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("#### Good to Know")
    st.markdown(
        """
        <div class="note-card">
            Moodly learned from IMDb movie reviews (Adam optimizer, learning rate 0.001,
            BCEWithLogitsLoss, 10 epochs) and reaches about <strong>84.09%</strong> accuracy on the
            25,000-review test split. It has not been independently validated beyond that test set,
            so results on other kinds of text, such as tweets, product reviews, or sarcasm-heavy
            writing, may vary. Treat each score as the model's confidence, not a guarantee.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### How Moodly Reads Your Text")
    steps = [
        (
            "Clean the text",
            "Lowercases everything, swaps <code>&lt;br&gt;</code> tags for spaces, keeps only "
            "letters a–z, and tidies extra whitespace.",
        ),
        (
            "Turn words into numbers",
            "Splits on whitespace and maps each word using <code>vocab.json</code> "
            "(<code>&lt;PAD&gt;</code> = 0, <code>&lt;UNK&gt;</code> = 1). Empty text falls back to "
            "a single <code>&lt;UNK&gt;</code>.",
        ),
        (
            "Pack the sequence",
            f"Records the real length (capped at {MAX_SEQ_LEN}), pads to {MAX_SEQ_LEN} with "
            "<code>&lt;PAD&gt;</code>, and uses <code>pack_padded_sequence</code> so padding is ignored.",
        ),
        (
            "Run the network",
            f"<code>nn.Embedding({VOCAB_SIZE}, {EMBEDDING_DIM})</code> → "
            f"<code>nn.LSTM({EMBEDDING_DIM}, {HIDDEN_DIM})</code> → "
            f"<code>nn.Linear({HIDDEN_DIM}, 1)</code>.",
        ),
        (
            "Decide the mood",
            "Runs in <code>eval()</code> mode with <code>torch.no_grad()</code>, applies "
            "<code>torch.sigmoid</code>, and calls it Positive when the probability is 0.5 or higher.",
        ),
    ]
    for number, (title, body) in enumerate(steps, start=1):
        st.markdown(
            f"""
            <div class="step-card">
                <div class="step-num">{number}</div>
                <div>
                    <div class="step-title">{title}</div>
                    <div class="step-body">{body}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_footer() -> None:
    """Render the Moodly footer."""
    st.markdown(
        f"""
        <div class="app-footer">
            <span class="footer-brand">{APP_NAME} <em>· {APP_TAGLINE}</em></span>
            <span>Made with 💖 · PyTorch LSTM · Trained on IMDb · Your text stays in this session</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar Navigation & Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    inject_custom_css()
    init_session_state()

    model, vocab, device, load_error = load_artifacts()

    with st.sidebar:
        st.markdown(
            f"""
            <div class="brand">
                <div class="brand-ring">m</div>
                <div class="brand-name">{APP_NAME}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        selected_page = st.radio(
            label="Navigation",
            options=[
                "Home",
                "Sentiment Analyzer",
                "Prediction History",
                "About the Model",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown('<div class="sidebar-status-title">Model Status</div>', unsafe_allow_html=True)
        if load_error is None and device is not None:
            st.success(f"Loaded on `{str(device).upper()}`")
        else:
            st.warning("Awaiting `model.pth` & `vocab.json`")

    render_header()

    if selected_page == "Home":
        render_analyzer_section(model, vocab, device, load_error)
        st.divider()
        render_history_section(compact=True)
    elif selected_page == "Sentiment Analyzer":
        render_analyzer_section(model, vocab, device, load_error)
    elif selected_page == "Prediction History":
        render_history_section(compact=False)
    elif selected_page == "About the Model":
        render_about_model_section()

    render_footer()


if __name__ == "__main__":
    main()