"""
SinCode Web UI — Streamlit interface for the transliteration engine.
"""

import streamlit as st
import time
import os
import csv
import html as html_lib
import base64
from datetime import datetime
from pathlib import Path
from PIL import Image
from sincode_model import BeamSearchDecoder

FEEDBACK_FILE = Path("feedback.csv")

st.set_page_config(page_title="සිංCode", page_icon="🇱🇰", layout="centered")


# ─── Helpers ─────────────────────────────────────────────────────────────────

@st.cache_data
def _background_css(image_file: str) -> str:
    """Return the CSS string for the background image (cached after first read)."""
    try:
        with open(image_file, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return (
            f"<style>.stApp {{background-image: linear-gradient(rgba(0,0,0,0.7),"
            f"rgba(0,0,0,0.7)),url(data:image/png;base64,{b64});"
            f"background-size:cover;background-position:center;"
            f"background-attachment:fixed;}}</style>"
        )
    except FileNotFoundError:
        return ""


def _set_background(image_file: str) -> None:
    css = _background_css(image_file)
    if css:
        st.markdown(css, unsafe_allow_html=True)


@st.cache_data
def _load_logo(image_file: str):
    return Image.open(image_file)


def _save_feedback(input_sentence: str, original_output: str, corrected_output: str) -> None:
    """Append a full-sentence correction to the feedback CSV."""
    with FEEDBACK_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["timestamp", "input_sentence", "original_output", "corrected_output"])
        writer.writerow([datetime.now().isoformat(), input_sentence, original_output, corrected_output])


@st.cache_resource
def _load_decoder() -> BeamSearchDecoder:
    """Load the transliteration engine (cached across reruns)."""
    model_name = os.getenv("SINCODE_MODEL_NAME")
    dict_path = os.getenv("SINCODE_DICTIONARY_PATH", "dictionary.pkl")
    if model_name:
        return BeamSearchDecoder(model_name=model_name, dictionary_path=dict_path)
    return BeamSearchDecoder(dictionary_path=dict_path)


# ─── Layout ──────────────────────────────────────────────────────────────────

_set_background("images/background.png")

with st.sidebar:
    st.image(_load_logo("images/SinCodeLogo.jpg"), width=200)
    st.title("සිංCode Project")
    st.info("6COSC023C.Y Final Project")

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
            if decode_mode == "greedy":
                result, trace_logs, diagnostics = decoder.greedy_decode_with_diagnostics(input_text)
            else:
                result, trace_logs, diagnostics = decoder.decode_with_diagnostics(input_text)
            elapsed = time.time() - t0

        # Store results in session state for interactive word swapping
        selected = [d.selected_candidate for d in diagnostics]
        st.session_state["diagnostics"] = diagnostics
        st.session_state["output_words"] = selected
        st.session_state["original_words"] = list(selected)
        st.session_state["input_sentence"] = input_text
        st.session_state["trace_logs"] = trace_logs
        st.session_state["elapsed"] = elapsed
        st.session_state["correction_mode"] = False
        st.session_state["correction_submitted_for"] = None

    except Exception as e:
        st.error(f"Error: {e}")

# ─── Render output (persists across reruns for word swapping) ─────────────

if "output_words" in st.session_state and st.session_state["output_words"]:
    diagnostics = st.session_state["diagnostics"]
    output_words = st.session_state["output_words"]
    original_words = st.session_state.get("original_words", list(output_words))
    trace_logs = st.session_state["trace_logs"]
    elapsed = st.session_state["elapsed"]

    current_result = " ".join(output_words)
    original_result = " ".join(original_words)
    has_changes = output_words != original_words

    st.success("Transliteration Complete")

    # Output display with native copy button (st.code has built-in clipboard support)
    safe_display = html_lib.escape(current_result)
    st.markdown(
        f'<span style="font-size:1.4em;font-weight:700;">{safe_display}</span>',
        unsafe_allow_html=True,
    )
    st.code(current_result, language=None)
    st.caption(f"Mode: {decode_mode} · Time: {round(elapsed, 2)}s")

    # ── Correction mode toggle ────────────────────────────────────────
    correction_mode = st.toggle(
        "Correct this translation",
        value=st.session_state.get("correction_mode", False),
        key="correction_toggle",
    )

    if correction_mode:
        st.caption("Use the buttons below to swap alternative transliterations.")

        # ── Inline sentence display (natural text flow, no grid) ─────
        word_spans = []
        for i, diag in enumerate(diagnostics):
            has_alts = len(diag.candidate_breakdown) > 1
            was_changed = output_words[i] != original_words[i]
            w = html_lib.escape(output_words[i])
            if was_changed:
                word_spans.append(
                    f'<span style="color:#68d391;font-weight:700;">{w} ✓</span>'
                )
            elif has_alts:
                word_spans.append(
                    f'<span style="color:#63b3ed;font-weight:700;'
                    f'border-bottom:2px dashed #63b3ed;cursor:default;">{w}</span>'
                )
            else:
                word_spans.append(f'<span style="font-weight:600;">{w}</span>')

        st.markdown(
            '<div style="font-size:1.15em;line-height:2.4;">'
            + " &ensp; ".join(word_spans)
            + "</div>",
            unsafe_allow_html=True,
        )
        # ── Popover buttons only for swappable words ─────────────────
        swappable = [
            (i, diag)
            for i, diag in enumerate(diagnostics)
            if len(diag.candidate_breakdown) > 1
        ]
        if swappable:
            widths = [max(len(output_words[i]), 3) for i, _ in swappable]
            cols = st.columns(widths, gap="small")

            for col, (i, diag) in zip(cols, swappable):
                was_changed = output_words[i] != original_words[i]
                with col:
                    chip = (
                        f":green[**{output_words[i]}**] ✓"
                        if was_changed
                        else f":blue[**{output_words[i]}**]"
                    )
                    with st.popover(chip, use_container_width=True):
                        st.markdown(f"**`{diag.input_word}`** — pick alternative:")
                        for scored in diag.candidate_breakdown[:5]:
                            eng_tag = " 🔤" if scored.is_english else ""
                            is_sel = scored.text == output_words[i]
                            if st.button(
                                f"{'✅ ' if is_sel else ''}{scored.text}{eng_tag}",
                                key=f"alt_{i}_{scored.text}",
                                help=f"Score: {scored.combined_score:.2f}",
                                use_container_width=True,
                                type="primary" if is_sel else "secondary",
                            ):
                                st.session_state["output_words"][i] = scored.text
                                st.rerun()
                        st.markdown("---")
                        custom = st.text_input(
                            "Not listed? Type correct word:",
                            key=f"custom_{i}",
                            placeholder="Type Sinhala word",
                        )
                        if custom and st.button(
                            "Use this", key=f"custom_apply_{i}", use_container_width=True
                        ):
                            st.session_state["output_words"][i] = custom
                            st.rerun()

        # ── Submit correction button (only when changes exist, once per result) ──
        # Guard key: (original sentence, original output) — stable regardless of swaps
        submit_key = (st.session_state["input_sentence"], original_result)
        already_submitted = st.session_state.get("correction_submitted_for") == submit_key
        if has_changes and not already_submitted:
            st.info(f"**Original:** {original_result}\n\n**Corrected:** {current_result}")
            if st.button("Submit Correction", type="primary", use_container_width=True):
                _save_feedback(
                    input_sentence=st.session_state["input_sentence"],
                    original_output=original_result,
                    corrected_output=current_result,
                )
                st.session_state["correction_submitted_for"] = submit_key
                st.session_state["correction_mode"] = False
                st.toast("Correction submitted — thank you!")
                st.rerun()

    # Show outside toggle so it remains visible after submission closes the toggle
    input_sent = st.session_state.get("input_sentence", "")
    if st.session_state.get("correction_submitted_for") == (input_sent, original_result):
        st.success("Correction already submitted.")

    with st.expander("Scoring Breakdown", expanded=False):
        st.caption(
            "MLM = contextual fit · Fid = transliteration fidelity · "
            "Rank = dictionary prior · 🔤 = English"
        )
        st.markdown("\n\n---\n\n".join(trace_logs))
