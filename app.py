"""
SinCode v3 — Streamlit demo UI.

Architecture: ByT5-small (seq2seq candidate generation) +
              XLM-RoBERTa (MLM contextual reranking)

Two transliteration modes:
  - Code-Mixed  — ByT5 + MLM; retains English words where contextually apt
  - Full Sinhala — mBart50 sentence-level; transliterates everything to Sinhala
"""

from __future__ import annotations

import os

import streamlit as st

from feedback_store import FeedbackStore, format_feedback_error
from sincode_model import BeamSearchDecoder, SentenceTransliterator


st.set_page_config(page_title="සිංCode", page_icon="🇱🇰", layout="centered")


def _secret_or_env(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name)
        if value is not None:
            return str(value)
    except Exception:
        # Compatible with Streamlit versions where secrets backend differs
        # or no secrets file is configured.
        pass
    return os.getenv(name, default)


@st.cache_resource(show_spinner="Loading models (ByT5 + XLM-RoBERTa)…")
def load_decoder() -> BeamSearchDecoder:
    return BeamSearchDecoder()


@st.cache_resource(show_spinner="Loading mBart50 model…")
def load_transliterator() -> SentenceTransliterator:
    return SentenceTransliterator()


@st.cache_resource
def load_feedback_store() -> FeedbackStore:
    return FeedbackStore(
        supabase_url=_secret_or_env("SUPABASE_URL"),
        supabase_anon_key=_secret_or_env("SUPABASE_ANON_KEY"),
        supabase_service_key=_secret_or_env("SUPABASE_SERVICE_ROLE_KEY"),
        table_name=_secret_or_env("SUPABASE_FEEDBACK_TABLE", "feedback_submissions"),
    )


st.title("සිංCode")

with st.sidebar:
    st.markdown("### About")
    st.write(
        "A model-driven Singlish to Sinhala transliteration interface with "
        "optional user feedback for future model refinement."
    )
    store = load_feedback_store()
    st.caption(f"Feedback storage: {store.backend_label}")

mode = st.radio(
    "Transliteration mode",
    options=["Code-Mixed Output", "Full Sinhala Output"],
    horizontal=True,
    help=(
        "**Code-Mixed**: keeps English technical or borrowed words where natural "
        "(e.g. *buffer*, *bit rate*). "
        "**Full Sinhala**: transliterates every word to Sinhala script "
        "(e.g. *business* -> ව්‍යාපාරය)."
    ),
)

if mode == "Full Sinhala Output":
    st.caption("mBart50 sentence-level transliteration")
else:
    st.caption("ByT5 seq2seq + XLM-RoBERTa MLM reranking")

sentence = st.text_area(
    "Enter Singlish sentence",
    height=120,
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
        trace_logs: list[str] = []
    else:
        with st.spinner("Transliterating…"):
            decoder = load_decoder()
            decode_out = decoder.decode(sentence.strip())
            if isinstance(decode_out, tuple) and len(decode_out) == 3:
                result, trace_logs, word_candidates = decode_out
            else:
                # Backward compatibility for cached/older decoders returning (result, trace_logs)
                result, trace_logs = decode_out
                word_candidates = [(w, []) for w in str(result).split()]

    st.session_state["last_input"] = sentence.strip()
    st.session_state["last_mode"] = mode
    st.session_state["last_result"] = result
    st.session_state["last_trace_logs"] = trace_logs
    st.session_state["last_word_candidates"] = word_candidates if mode == "Code-Mixed Output" else []
    st.session_state["last_output_words"] = [w for w, _ in word_candidates] if word_candidates else result.split()
    st.session_state["feedback_submitted_for"] = None

if st.session_state.get("last_result"):
    current_input = st.session_state.get("last_input", "")
    original_result = st.session_state["last_result"]
    current_mode = st.session_state.get("last_mode", mode)
    word_candidates = st.session_state.get("last_word_candidates", [])

    # output_words may be mutated by alternative selection — keep in session state
    if "last_output_words" not in st.session_state:
        st.session_state["last_output_words"] = original_result.split()
    output_words: list[str] = st.session_state["last_output_words"]
    current_result = " ".join(output_words)

    st.markdown("### Result")
    st.success(current_result)
    st.caption(f"Mode: {current_mode}")

    # ── Alternative selection (Code-Mixed only) ───────────────────────────────
    if current_mode == "Code-Mixed Output" and word_candidates:
        swappable = [
            (i, sel, alts)
            for i, (sel, alts) in enumerate(word_candidates)
            if len(alts) > 1
        ]
        if swappable:
            st.markdown("##### Alternative transliterations")
            st.caption("Click a word to pick an alternative.")
            widths = [max(len(output_words[i]), 3) for i, _, _ in swappable]
            cols = st.columns(widths, gap="small")
            for col, (i, _sel, alts) in zip(cols, swappable):
                current_word = output_words[i]
                was_changed = current_word != word_candidates[i][0]
                chip = (
                    f":green[**{current_word}**]"
                    if was_changed
                    else f":blue[**{current_word}**]"
                )
                with col:
                    with st.popover(chip, use_container_width=True):
                        st.markdown(f"**Alternatives for `{current_word}`:**")
                        for alt in alts:
                            is_sel = alt == current_word
                            if st.button(
                                f"{'✅ ' if is_sel else ''}{alt}",
                                key=f"alt_{i}_{alt}",
                                use_container_width=True,
                                type="primary" if is_sel else "secondary",
                            ):
                                st.session_state["last_output_words"][i] = alt
                                st.session_state["corrected_output"] = " ".join(
                                    st.session_state["last_output_words"]
                                )
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
                            st.session_state["last_output_words"][i] = custom
                            st.session_state["corrected_output"] = " ".join(
                                st.session_state["last_output_words"]
                            )
                            st.rerun()

    if current_mode == "Code-Mixed Output" and show_trace:
        trace_logs = st.session_state.get("last_trace_logs", [])
        if trace_logs:
            st.markdown("### Trace")
            for log in trace_logs:
                st.markdown(log)

    with st.expander("Submit a correction", expanded=False):
        st.caption(
            "If the output is incorrect, submit a corrected sentence. "
            "These feedback samples can be reviewed and used in future fine-tuning."
        )
        corrected_output = st.text_area(
            "Corrected output",
            value=current_result,
            height=120,
            key="corrected_output",
        )
        feedback_comment = st.text_area(
            "Optional note",
            placeholder="Example: In this sentence, this word should stay in English.",
            key="feedback_comment",
        )

        submit_key = (current_input, original_result, corrected_output)
        already_submitted = st.session_state.get("feedback_submitted_for") == submit_key

        if already_submitted:
            st.success("Correction submitted.")
        elif st.button("Submit Feedback", use_container_width=True):
            try:
                load_feedback_store().save_submission(
                    input_sentence=current_input,
                    original_output=original_result,
                    corrected_output=corrected_output.strip(),
                    user_comment=feedback_comment,
                    decode_mode=current_mode,
                )
                st.session_state["feedback_submitted_for"] = submit_key
                st.success("Feedback submitted successfully.")
            except Exception as exc:
                st.error(f"Could not submit feedback: {format_feedback_error(exc)}")
