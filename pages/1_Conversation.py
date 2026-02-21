import os
import time
import uuid

import streamlit as st
from streamlit_chat import message

from config import get_settings
from execution import execute_tool
from hybrid_router import cactus_transcribe, generate_hybrid
from tools import TOOL_SCHEMAS


settings = get_settings()

st.set_page_config(page_title="Latency Zero - Conversation", page_icon="💬", layout="wide")
st.markdown(
    """
    <style>
    .lz-compose-wrap {
      border-radius: 16px;
      border: 2px solid #ff5a66;
      background: #2a2d38;
      padding: 10px 12px;
      margin-top: 6px;
    }
    .lz-compose-wrap:focus-within {
      box-shadow: 0 0 0 3px rgba(255, 90, 102, 0.18);
    }
    .lz-compose-wrap .stTextInput input {
      border: none !important;
      box-shadow: none !important;
      background: transparent !important;
      font-size: 1.1rem !important;
      color: #d7d9df !important;
      min-height: 48px !important;
      padding-left: 0.2rem !important;
    }
    .lz-compose-wrap .stTextInput input::placeholder {
      color: #aeb3bf !important;
      opacity: 1 !important;
    }
    .lz-mic-btn button {
      border: none !important;
      background: transparent !important;
      box-shadow: none !important;
      min-height: 40px !important;
      width: 40px !important;
      color: #b7bcc8 !important;
      font-size: 1.25rem !important;
    }
    .lz-send-btn button {
      border-radius: 14px !important;
      min-height: 40px !important;
      width: 40px !important;
      background: #3a3e4d !important;
      border: none !important;
      color: #aeb3bf !important;
      font-size: 1.1rem !important;
      padding: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Conversation")

if "cloud_enabled" not in st.session_state:
    st.session_state.cloud_enabled = False
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "pending_user_text" not in st.session_state:
    st.session_state.pending_user_text = ""
if "last_voice_token" not in st.session_state:
    st.session_state.last_voice_token = ""
if "chat_prompt" not in st.session_state:
    st.session_state.chat_prompt = ""
if "set_prompt_next" not in st.session_state:
    st.session_state.set_prompt_next = ""


def _toggle_cloud() -> None:
    st.session_state.cloud_enabled = not st.session_state.cloud_enabled
_, c2 = st.columns([9.2, 0.8], vertical_alignment="top")
with c2:
    cloud_label = "🔴☁" if st.session_state.cloud_enabled else "🟢☁"
    st.button(cloud_label, key="cloud_btn", help="Toggle cloud fallback", on_click=_toggle_cloud)

# Render chat history in st-chat format
for idx, item in enumerate(st.session_state.chat_messages):
    if item["role"] == "user":
        message(item["text"], is_user=True, key=f"u_{idx}", avatar_style="initials")
    else:
        message(item["text"], is_user=False, key=f"a_{idx}", avatar_style="bottts")
        if item.get("meta"):
            with st.expander(f"Details #{idx+1}"):
                st.json(item["meta"])

# Native fixed-bottom composer with integrated mic + send UI.
if st.session_state.set_prompt_next:
    st.session_state.chat_prompt = st.session_state.set_prompt_next
    st.session_state.set_prompt_next = ""

prompt = st.chat_input("Say or record something", accept_audio=True, key="chat_prompt")
user_text = ""

if prompt:
    # Streamlit may return either a plain string or an object with text/audio.
    if isinstance(prompt, str):
        user_text = prompt.strip()
    else:
        text = (getattr(prompt, "text", "") or "").strip()
        audio = getattr(prompt, "audio", None)
        if text:
            user_text = text
        elif audio is not None:
            token = f"{getattr(audio, 'name', '')}:{len(audio.getvalue())}"
            if token != st.session_state.last_voice_token:
                st.session_state.last_voice_token = token
                suffix = os.path.splitext(getattr(audio, "name", "") or "")[1] or ".wav"
                transcript, transcribe_error = cactus_transcribe(
                    audio.getvalue(), settings.transcribe_model, suffix=suffix
                )
                if transcript:
                    # Put transcript into input draft; user can edit and send manually.
                    st.session_state.set_prompt_next = transcript.strip()
                    st.toast("Transcript added to input")
                    st.rerun()
                elif transcribe_error:
                    st.warning("Transcription failed")
                    with st.expander("Transcription debug"):
                        st.code(transcribe_error)

# Backward compatibility for previously queued text.
if not user_text:
    user_text = st.session_state.pending_user_text.strip()
if user_text:
    st.session_state.pending_user_text = ""
    st.session_state.chat_messages.append({"role": "user", "text": user_text})

    result = generate_hybrid(
        user_text,
        models={
            "local": settings.local_model,
            "cloud": settings.cloud_model,
            "cloud_enabled": st.session_state.cloud_enabled,
        },
        tools=TOOL_SCHEMAS,
    )

    tool_name = result.get("tool_name")
    args = result.get("args", {}) if isinstance(result.get("args", {}), dict) else {}
    tool_result = execute_tool(tool_name, args) if tool_name else "No tool executed."

    assistant_text = (
        f"{result.get('response_text', '')}\n\n"
        f"Routing: {result.get('routing', 'LOCAL')} | "
        f"Confidence: {result.get('confidence', 0.0):.2f} | "
        f"Latency: {result.get('latency', 0.0):.2f}s\n"
        f"Reasons: {', '.join(result.get('routing_meta', {}).get('reasons', [])) or 'local_preferred'}\n"
        f"Tool: {tool_name or 'None'} | Execution: {tool_result}"
    )

    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "text": assistant_text,
            "meta": result,
            "id": str(uuid.uuid4()),
            "ts": time.time(),
        }
    )
    st.rerun()
