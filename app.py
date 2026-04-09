"""
SinCode Web UI — Streamlit interface for the transliteration engine.
"""

import streamlit as st
import time
import os
import hmac
import html as html_lib
import base64
from PIL import Image
from feedback_store import FeedbackStore, format_feedback_error
from sincode_model import BeamSearchDecoder

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


def _secret_or_env(name: str, default: str = "") -> str:
    if name in st.secrets:
        return str(st.secrets[name])
    return os.getenv(name, default)


@st.cache_resource
def _load_feedback_store() -> FeedbackStore:
    return FeedbackStore(
        supabase_url=_secret_or_env("SUPABASE_URL"),
        supabase_anon_key=_secret_or_env("SUPABASE_ANON_KEY"),
        supabase_service_key=_secret_or_env("SUPABASE_SERVICE_ROLE_KEY"),
        table_name=_secret_or_env("SUPABASE_FEEDBACK_TABLE", "feedback_submissions"),
    )


def _admin_credentials_configured() -> bool:
    return bool(_secret_or_env("ADMIN_USERNAME") and _secret_or_env("ADMIN_PASSWORD"))


def _authenticate_admin(username: str, password: str) -> bool:
    expected_username = _secret_or_env("ADMIN_USERNAME")
    expected_password = _secret_or_env("ADMIN_PASSWORD")
    return bool(
        expected_username
        and expected_password
        and hmac.compare_digest(username, expected_username)
        and hmac.compare_digest(password, expected_password)
    )


def _save_feedback(
    input_sentence: str,
    original_output: str,
    corrected_output: str,
    user_comment: str,
    decode_mode: str,
) -> None:
    _load_feedback_store().save_submission(
        input_sentence=input_sentence,
        original_output=original_output,
        corrected_output=corrected_output,
        user_comment=user_comment,
        decode_mode=decode_mode,
    )


def _render_admin_access(store: FeedbackStore) -> bool:
    st.sidebar.markdown("### 📨 Feedback")
    st.sidebar.caption(f"Storage: {store.backend_label}")

    with st.sidebar.expander("Admin Access", expanded=st.session_state.get("admin_authenticated", False)):
        if st.session_state.get("admin_authenticated", False):
            st.success("Admin session active")
            open_panel = st.toggle(
                "Open feedback review panel",
                value=st.session_state.get("show_admin_panel", False),
                key="show_admin_panel_toggle",
            )
            st.session_state["show_admin_panel"] = open_panel
            if st.button("Log Out", use_container_width=True):
                st.session_state["admin_authenticated"] = False
                st.session_state["show_admin_panel"] = False
                st.rerun()
            return st.session_state.get("show_admin_panel", False)

        if not _admin_credentials_configured():
            st.info("Set ADMIN_USERNAME and ADMIN_PASSWORD in Hugging Face Secrets to enable review access.")
            return False

        username = st.text_input("Username", key="admin_username")
        password = st.text_input("Password", type="password", key="admin_password")
        if st.button("Admin Login", use_container_width=True):
            if _authenticate_admin(username, password):
                st.session_state["admin_authenticated"] = True
                st.session_state["show_admin_panel"] = True
                st.rerun()
            st.error("Invalid admin credentials.")
    return st.session_state.get("show_admin_panel", False)


def _render_admin_panel(store: FeedbackStore) -> None:
    st.title("Feedback Review")
    st.caption("Review submitted corrections, approve useful examples, and export them later for future retraining.")

    try:
        all_rows = store.list_submissions(review_status=None, limit=500)
    except Exception as exc:
        st.error(f"Could not load feedback records: {format_feedback_error(exc)}")
        return

    pending_count = sum(1 for row in all_rows if row.get("review_status") == "pending")
    approved_count = sum(1 for row in all_rows if row.get("review_status") == "approved")
    rejected_count = sum(1 for row in all_rows if row.get("review_status") == "rejected")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Pending", pending_count)
    metric_cols[1].metric("Approved", approved_count)
    metric_cols[2].metric("Rejected", rejected_count)

    filter_cols = st.columns([1, 1, 2])
    status_filter = filter_cols[0].selectbox(
        "Status",
        options=["pending", "approved", "rejected", "all"],
        index=0,
    )
    row_limit = filter_cols[1].selectbox("Rows", options=[25, 50, 100, 200], index=1)
    search_term = filter_cols[2].text_input("Search", placeholder="Search input, output, or note")

    filtered_rows = all_rows
    if status_filter != "all":
        filtered_rows = [row for row in filtered_rows if row.get("review_status") == status_filter]

    if search_term:
        needle = search_term.casefold()
        filtered_rows = [
            row
            for row in filtered_rows
            if needle in row.get("input_sentence", "").casefold()
            or needle in row.get("original_output", "").casefold()
            or needle in row.get("corrected_output", "").casefold()
            or needle in row.get("user_comment", "").casefold()
            or needle in row.get("admin_notes", "").casefold()
        ]

    filtered_rows = filtered_rows[:row_limit]

    if not filtered_rows:
        st.info("No feedback matches the current filters.")
        return

    for row in filtered_rows:
        with st.container(border=True):
            meta_cols = st.columns([2, 1, 1])
            meta_cols[0].caption(f"Submitted: {row.get('created_at', 'unknown')}")
            meta_cols[1].caption(f"Mode: {row.get('decode_mode') or 'n/a'}")
            meta_cols[2].caption(f"Status: {row.get('review_status', 'pending')}")

            st.markdown("**Input (Singlish)**")
            st.code(row.get("input_sentence", ""), language=None)
            st.markdown("**Model Output**")
            st.code(row.get("original_output", ""), language=None)
            st.markdown("**User Correction**")
            st.code(row.get("corrected_output", ""), language=None)

            if row.get("user_comment"):
                st.markdown("**User Note**")
                st.write(row["user_comment"])

            notes_key = f"admin_notes_{row['id']}"
            notes_value = st.text_area(
                "Admin Notes",
                value=row.get("admin_notes", ""),
                key=notes_key,
                height=80,
            )

            action_cols = st.columns(3)
            if action_cols[0].button("Approve", key=f"approve_{row['id']}", use_container_width=True):
                try:
                    store.update_submission_status(str(row["id"]), "approved", notes_value)
                    st.toast("Feedback approved.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not update feedback: {format_feedback_error(exc)}")
            if action_cols[1].button("Reject", key=f"reject_{row['id']}", use_container_width=True):
                try:
                    store.update_submission_status(str(row["id"]), "rejected", notes_value)
                    st.toast("Feedback rejected.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not update feedback: {format_feedback_error(exc)}")
            if action_cols[2].button("Mark Pending", key=f"pending_{row['id']}", use_container_width=True):
                try:
                    store.update_submission_status(str(row["id"]), "pending", notes_value)
                    st.toast("Feedback returned to pending.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not update feedback: {format_feedback_error(exc)}")


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

feedback_store = _load_feedback_store()

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

    if not feedback_store.is_remote_enabled:
        st.warning("Feedback storage is offline. Set Supabase secrets to enable submissions.")

show_admin_panel = _render_admin_access(feedback_store)

if show_admin_panel:
    _render_admin_panel(feedback_store)
    st.stop()

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
        st.session_state["feedback_comment"] = ""

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
            feedback_comment = st.text_area(
                "Optional note for reviewers",
                key="feedback_comment",
                placeholder="Example: The word 'kalaya' should mean time in this context.",
            )
            if st.button("Submit Correction", type="primary", use_container_width=True):
                try:
                    _save_feedback(
                        input_sentence=st.session_state["input_sentence"],
                        original_output=original_result,
                        corrected_output=current_result,
                        user_comment=feedback_comment,
                        decode_mode=decode_mode,
                    )
                    st.session_state["correction_submitted_for"] = submit_key
                    st.session_state["correction_mode"] = False
                    st.toast("Correction submitted for review — thank you!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not submit feedback: {format_feedback_error(exc)}")

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
