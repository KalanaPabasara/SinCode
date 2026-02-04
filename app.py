import streamlit as st
import time
from sincode_model import BeamSearchDecoder
from PIL import Image
import base64

st.set_page_config(page_title="සිංCode Prototype", page_icon="🇱🇰", layout="centered")
def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as f:
            data = f.read()
        b64_data = base64.b64encode(data).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url(data:image/png;base64,{b64_data});
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        pass 

@st.cache_resource
def load_system():
    decoder = BeamSearchDecoder()
    return decoder

background_path = "SinCode/images/background.png"
add_bg_from_local(background_path)

with st.sidebar:
    logo = Image.open("SinCode/images/SinCodeLogo.jpg")
    st.image(logo, width=200)
    st.title("සිංCode Project")
    st.info("Prototype")
    st.markdown("### 🏗 Architecture")
    st.success("""
    **Hybrid Neuro-Symbolic Engine**
    Combines rule-based speed with Deep Learning (XLM-R) context awareness.

    **Adaptive Code-Switching**
    Intelligently detects and preserves English contexts.

    **Contextual Disambiguation**
    Resolves Singlish ambiguity using sentence-level probability.
    """)

    st.markdown("---")
    st.write("© 2026 Kalana Chandrasekara")

st.title("සිංCode: Context-Aware Transliteration")
st.markdown("Type Singlish sentences below. The system handles **code-mixing**, **ambiguity**, and **punctuation**.")

input_text = st.text_area("Input Text", height=100, placeholder="e.g., Singlish sentences type krnna")

if st.button("Transliterate", type="primary", use_container_width=True) and input_text:
    try:
        with st.spinner("Processing..."):
            decoder = load_system()
            start_time = time.time()
            result, trace_logs = decoder.decode(input_text)
            end_time = time.time()

        st.success("Transliteration Complete")
        st.markdown(f"### {result}")
        st.caption(f"Time: {round(end_time - start_time, 2)}s")

        with st.expander("See How It Works (Debug Info)", expanded=True):
            st.write("Below shows the candidate scoring for each word step:")
            for log in trace_logs:
                st.markdown(log)
                st.divider()

    except Exception as e:
        st.error(f"Error: {e}")
