"""
SinCode Web UI — Streamlit interface for the transliteration engine.
"""

import streamlit as st
import time
import os
import base64
from PIL import Image
from sincode_model import BeamSearchDecoder

st.set_page_config(page_title="සිංCode", page_icon="🇱🇰", layout="centered")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _set_background(image_file: str) -> None:
    """Inject a dark-overlay background from a local image."""
    try:
        with open(image_file, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                                  url(data:image/png;base64,{b64});
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        pass


@st.cache_resource
def _load_decoder() -> BeamSearchDecoder:
    """Load the transliteration engine (cached across reruns)."""
    model_name = os.getenv("SICODE_MODEL_NAME")
    dict_path = os.getenv("SICODE_DICTIONARY_PATH", "dictionary.pkl")
    if model_name:
        return BeamSearchDecoder(model_name=model_name, dictionary_path=dict_path)
    return BeamSearchDecoder(dictionary_path=dict_path)


# ─── Layout ──────────────────────────────────────────────────────────────────

_set_background("images/background.png")

with st.sidebar:
    st.image(Image.open("images/SinCodeLogo.jpg"), width=200)
    st.title("සිංCode Project")
    st.info("Prototype")

    st.markdown("### ⚙️ Settings")
    decode_mode = st.radio(
        "Decode Mode",
        options=["greedy", "beam"],
        index=0,
        help=(
            "**Greedy** (recommended) — Faster and more accurate. Picks the "
            "best candidate at each step using real context.\n\n"
            "**Beam** — Explores multiple paths but uses fixed context, "
            "so results are similar with more computation."
        ),
    )

    st.markdown("### 🏗 Architecture")
    st.success(
        "**Hybrid Neuro-Symbolic Engine**\n\n"
        "XLM-R contextual scoring (55%) "
        "+ transliteration fidelity (45%).\n\n"
        "**Common Word Overrides** — "
        "Curated table for high-frequency unambiguous words.\n\n"
        "**Adaptive Code-Switching** — "
        "Preserves English words in mixed input.\n\n"
        "**Contextual Disambiguation** — "
        "Resolves ambiguity via sentence-level probability."
    )
    st.markdown("---")
    st.write("© 2026 Kalana Chandrasekara")

st.title("සිංCode: Context-Aware Transliteration")
st.markdown(
    "Type Singlish sentences below. "
    "The system handles **code-mixing**, **ambiguity**, and **punctuation**."
)

input_text = st.text_area(
    "Input Text", height=100, placeholder="e.g., Singlish sentences type krnna"
)

if st.button("Transliterate", type="primary", use_container_width=True) and input_text:
    try:
        with st.spinner("Processing..."):
            decoder = _load_decoder()
            t0 = time.time()
            result, trace_logs = decoder.decode(input_text, mode=decode_mode)
            elapsed = time.time() - t0

        st.success("Transliteration Complete")
        st.markdown(f"### {result}")
        st.caption(f"Mode: {decode_mode} · Time: {round(elapsed, 2)}s")

        with st.expander("Scoring Breakdown", expanded=True):
            st.caption(
                "MLM = contextual fit · Fid = transliteration fidelity · "
                "Rank = dictionary prior · 🔤 = English"
            )
            for log in trace_logs:
                st.markdown(log)
                st.divider()

    except Exception as e:
        st.error(f"Error: {e}")
