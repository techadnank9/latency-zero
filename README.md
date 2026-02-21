# Latency Zero — Hybrid Voice Action Agent

Production-style hackathon MVP for a local-first hybrid voice action agent.

## Features

- Voice input from Streamlit (`st.audio_input`)
- Local transcription via Cactus CLI (`cactus transcribe`)
- Hybrid router: local-first with Gemini fallback
- Tool execution engine with argument validation
- Transparent routing dashboard and live metrics
- Failure-safe behavior (no hard crashes)

## Project Structure

```text
latency-zero/
├── app.py
├── hybrid_router.py
├── tools.py
├── execution.py
├── config.py
├── requirements.txt
└── README.md
```

## Install

```bash
cd latency-zero
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e /Users/adnan/Documents/cactus/python
```

Set environment variables:

```bash
export GEMINI_API_KEY="your_key"
export HF_TOKEN="your_hf_token"  # needed for cactus transcribe/model downloads
export CACTUS_BIN="/Users/adnan/Documents/cactus/venv/bin/cactus"  # recommended
export CONFIDENCE_THRESHOLD="0.75"   # optional
export LOCAL_MODEL="google/functiongemma-270m-it"  # optional
export CLOUD_MODEL="gemini-1.5-flash"  # optional
export TRANSCRIBE_MODEL="openai/whisper-small"  # optional
```

## Run

```bash
streamlit run app.py
```

## Demo Commands

- `Add task buy groceries` → local route → `add_task`
- `Send message John hello` → local route → `send_message`
- `Schedule meeting tomorrow 2pm` → `create_event`
- `Explain quantum physics` → cloud fallback

## Architecture Overview

```text
Voice Input -> Local Transcribe -> Hybrid Router -> Local Tool Call Validate
                                          | valid
                                          v
                                   Execute Tool -> UI
                                          |
                                     invalid/low confidence
                                          v
                                   Gemini Fallback -> Execute -> UI
```

## Notes

- Local routing is prioritized for short commands (< 8 words).
- Fallback is triggered if confidence is low, tool call is invalid, args are bad JSON, tool is unknown, or the query is reasoning-heavy.
- If cloud key is missing or cloud fails, the app returns a safe error message instead of crashing.
