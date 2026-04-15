"""
SinCode v3 — Streamlit demo UI.

Architecture: ByT5-small (seq2seq candidate generation) +
              XLM-RoBERTa (MLM contextual reranking)

Two transliteration modes:
  • Code-Mixed  — ByT5 + MLM; retains English words where contextually apt
  • Full Sinhala — mBart50 sentence-level; transliterates everything to Sinhala
"""

import streamlit as st
from sincode_model import BeamSearchDecoder, SentenceTransliterator

st.set_page_config(page_title="සිංCode v3", page_icon="🇱🇰", layout="centered")

st.title("සිංCode v3")
st.caption("ByT5 seq2seq + XLM-RoBERTa MLM reranking")


@st.cache_resource(show_spinner="Loading models (ByT5 + XLM-RoBERTa)…")
def load_decoder() -> BeamSearchDecoder:
    return BeamSearchDecoder()


@st.cache_resource(show_spinner="Loading mBart50 model…")
def load_transliterator() -> SentenceTransliterator:
    return SentenceTransliterator()


mode = st.radio(
    "Transliteration mode",
    options=["Code-Mixed Output", "Full Sinhala Output"],
    horizontal=True,
    help=(
        "**Code-Mixed**: keeps English technical/borrowed words where natural "
        "(e.g. *buffer*, *bit rate*). "
        "**Full Sinhala**: transliterates every word to Sinhala script "
        "(e.g. *business* → ව්‍යාපාරය)."
    ),
)

sentence = st.text_input(
    "Enter Singlish sentence",
    placeholder="e.g. mema videowe bit rate eka godak wadi nisa buffer wenawa",
)

show_trace = st.checkbox(
    "Show step-by-step trace",
    value=False,
    disabled=(mode == "Full Sinhala Output"),
    help="Trace is only available in Code-Mixed mode.",
)

if st.button("Transliterate", type="primary") and sentence.strip():
    if mode == "Full Sinhala Output":
        with st.spinner("Transliterating (mBart50)…"):
            transliterator = load_transliterator()
            result = transliterator.transliterate(sentence.strip())

        st.markdown("### Result")
        st.success(result)

    else:
        with st.spinner("Transliterating…"):
            decoder = load_decoder()
            result, trace_logs = decoder.decode(sentence.strip())

        st.markdown("### Result")
        st.success(result)

        if show_trace:
            st.markdown("### Trace")
            for log in trace_logs:
                st.markdown(log)
