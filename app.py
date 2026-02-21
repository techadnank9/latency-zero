import logging
import os

import streamlit as st

from config import get_settings
from execution import execute_tool
from hybrid_router import cactus_transcribe, generate_hybrid
from tools import TOOL_SCHEMAS


logging.basicConfig(level=logging.INFO)
settings = get_settings()

st.set_page_config(page_title="Latency Zero", page_icon="⚡", layout="wide")

if "local_count" not in st.session_state:
    st.session_state.local_count = 0
if "cloud_count" not in st.session_state:
    st.session_state.cloud_count = 0
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "cloud_enabled" not in st.session_state:
    st.session_state.cloud_enabled = False


def _toggle_cloud() -> None:
    st.session_state.cloud_enabled = not st.session_state.cloud_enabled


st.title("Latency Zero — Hybrid Voice Agent")
st.caption("Low Latency • Local First • Intelligent Fallback")
_, control_col = st.columns([3, 3])
with control_col:
    st.markdown("### Cloud Fallback")
    label = "Disable Cloud" if st.session_state.cloud_enabled else "Enable Cloud"
    button_type = "secondary" if st.session_state.cloud_enabled else "primary"
    st.markdown(
        """
        <style>
        div[data-testid="stButton"] > button[kind="primary"] {
            background-color: #16a34a;
            border-color: #16a34a;
            color: white;
            font-size: 1.0rem;
            font-weight: 700;
            padding: 0.45rem 0.8rem;
            min-height: 2.3rem;
        }
        div[data-testid="stButton"] > button[kind="secondary"] {
            background-color: #dc2626;
            border-color: #dc2626;
            color: white;
            font-size: 1.0rem;
            font-weight: 700;
            padding: 0.45rem 0.8rem;
            min-height: 2.3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.button(
        label,
        key="cloud_toggle_btn",
        use_container_width=True,
        type=button_type,
        on_click=_toggle_cloud,
    )
    st.write(f"Current: **{'Enabled' if st.session_state.cloud_enabled else 'Disabled'}**")

st.subheader("Voice Input")
audio = st.audio_input("🎤 Speak command")
typed_command = st.text_input("Or type a command (fallback)", placeholder="Add task buy groceries")

transcript = ""
transcribe_error = ""
if audio is not None:
    with st.spinner("Transcribing locally..."):
        audio_suffix = os.path.splitext(getattr(audio, "name", "") or "")[1] or ".wav"
        transcript, transcribe_error = cactus_transcribe(
            audio.getvalue(),
            settings.transcribe_model,
            suffix=audio_suffix,
        )

if not transcript and typed_command.strip():
    transcript = typed_command.strip()

st.subheader("Transcript")
if transcript:
    st.write(f'Transcript: "{transcript}"')
else:
    st.write('Transcript: ""')

should_route = (audio is not None) or bool(typed_command.strip())

if should_route:
    if not transcript:
        st.warning("No transcript detected. Please try again.")
        if transcribe_error:
            with st.expander("Transcription debug"):
                st.code(transcribe_error)
    else:
        with st.spinner("Running hybrid router..."):
            result = generate_hybrid(
                transcript,
                models={
                    "local": settings.local_model,
                    "cloud": settings.cloud_model,
                    "cloud_enabled": st.session_state.cloud_enabled,
                },
                tools=TOOL_SCHEMAS,
            )

        st.session_state.last_result = result

        routing = result.get("routing", "LOCAL")
        if routing == "LOCAL":
            st.session_state.local_count += 1
        else:
            st.session_state.cloud_count += 1

        tool_name = result.get("tool_name")
        args = result.get("args", {})

        execution_result = "No tool executed."
        if tool_name:
            execution_result = execute_tool(tool_name, args)

        st.subheader("Routing Dashboard")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Routing Decision", routing)
        c2.metric("Confidence", f"{result.get('confidence', 0.0):.2f}")
        c3.metric("Latency", f"{result.get('latency', 0.0):.2f}s")
        c4.metric("Cloud Used", "Yes" if routing == "CLOUD" else "No")
        with st.expander("Routing Logic"):
            st.json(result.get("routing_meta", {}))

        st.subheader("Tool Execution")
        st.write(f"Tool Called: `{tool_name or 'None'}`")
        st.json(args if isinstance(args, dict) else {})
        st.success(execution_result)

        st.subheader("Local Model Output")
        local_out = result.get("local_model_output")
        if local_out:
            st.json(local_out)
        else:
            st.write("No local model output available.")

        if result.get("response_text"):
            st.info(result["response_text"])

st.subheader("Metrics")
local_count = st.session_state.local_count
cloud_count = st.session_state.cloud_count
total = local_count + cloud_count
edge_usage = (local_count / total * 100.0) if total else 0.0

m1, m2, m3 = st.columns(3)
m1.metric("Local Executions Counter", local_count)
m2.metric("Cloud Fallback Counter", cloud_count)
m3.metric("Edge Usage %", f"{edge_usage:.1f}%")
