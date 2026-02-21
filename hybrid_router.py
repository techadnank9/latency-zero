import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from google.genai.errors import ClientError

from config import get_settings


LOGGER = logging.getLogger(__name__)
_CLOUD_BACKOFF_UNTIL = 0.0
_LOCAL_MODEL_HANDLE = None
_LOCAL_MODEL_PATH = None
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+?\|>")
NOISE_PATTERNS = [
    re.compile(r"^\s*👋\s*goodbye!?\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\[[^\]]+\]\s*$"),  # metadata/control tag lines e.g. [cloud_handoff:true]
    re.compile(r"^\s*cloud_handoff\s*[:=]\s*(true|false)\s*$", re.IGNORECASE),
    re.compile(r"^\s*transcribing\s*:\s*.+$", re.IGNORECASE),
    re.compile(r"^\s*/.+\.(wav|mp3|m4a|ogg|webm)\s*$", re.IGNORECASE),
    re.compile(r"^\s*setting up cactus", re.IGNORECASE),
    re.compile(r"^\s*how to use the cactus", re.IGNORECASE),
    re.compile(r"^\s*cactus (auth|run|transcribe|download|convert|build|test|clean)\b", re.IGNORECASE),
    re.compile(r"^\s*examples?:", re.IGNORECASE),
    re.compile(r"^\s*[-=]{5,}\s*$"),
]


def _word_count(text: str) -> int:
    return len([w for w in text.strip().split() if w])


def _is_reasoning_heavy(text: str) -> bool:
    lower = text.lower()
    heavy_keywords = [
        "explain",
        "why",
        "analyze",
        "compare",
        "quantum",
        "architecture",
        "derive",
        "reason",
    ]
    return any(k in lower for k in heavy_keywords) or _word_count(text) > 20


def _is_action_request(text: str) -> bool:
    lower = text.lower().strip()
    action_markers = [
        "add task",
        "send message",
        "create event",
        "schedule",
        "set mode",
        "remind",
        "message ",
        "task ",
        "event ",
    ]
    return any(m in lower for m in action_markers)


def _looks_like_refusal(text: str) -> bool:
    lower = (text or "").lower()
    phrases = [
        "i cannot",
        "i can't",
        "i do not have access",
        "my capabilities are limited",
        "outside my capabilities",
        "i apologize, but",
    ]
    return any(p in lower for p in phrases)


def _looks_like_capability_disclaimer(text: str) -> bool:
    lower = (text or "").lower()
    phrases = [
        "my current capabilities",
        "capabilities are limited",
        "within my supported",
        "provided tools",
        "tool limitations",
        "i cannot assist",
    ]
    return any(p in lower for p in phrases)


def _tool_names(tools: List[Dict[str, Any]]) -> set:
    return {t.get("name", "") for t in tools}


def _tool_required_fields(tools: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    required: Dict[str, List[str]] = {}
    for tool in tools:
        name = str(tool.get("name", "")).strip()
        params = tool.get("parameters", {}) if isinstance(tool.get("parameters"), dict) else {}
        req = params.get("required", []) if isinstance(params, dict) else []
        required[name] = [str(x) for x in req if isinstance(x, str)]
    return required


def _is_valid_tool_call(name: Optional[str], args: Any, tools: List[Dict[str, Any]]) -> Tuple[bool, str]:
    if not name:
        return False, "no_tool_call"
    if name not in _tool_names(tools):
        return False, "unknown_tool"
    if not isinstance(args, dict):
        return False, "args_not_object"

    required_map = _tool_required_fields(tools)
    missing = [key for key in required_map.get(name, []) if not str(args.get(key, "")).strip()]
    if missing:
        return False, f"missing_required:{','.join(missing)}"
    return True, "ok"


def _complexity_score(text: str) -> float:
    words = _word_count(text)
    lower = text.lower()
    score = 0.0
    if words >= 16:
        score += 0.35
    if words >= 28:
        score += 0.25
    if any(k in lower for k in ["explain", "analyze", "compare", "strategy", "architecture", "step by step"]):
        score += 0.35
    if any(k in lower for k in ["why", "tradeoff", "optimize", "design"]):
        score += 0.15
    return max(0.0, min(1.0, score))


def _safe_json_dict(value: Any) -> Tuple[bool, Dict[str, Any]]:
    if isinstance(value, dict):
        return True, value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return True, parsed
        except json.JSONDecodeError:
            return False, {}
    return False, {}


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def _weights_dir_for_model(model_id: str, settings) -> Path:
    if settings.cactus_cwd:
        base = Path(settings.cactus_cwd)
    elif settings.cactus_bin:
        base = Path(settings.cactus_bin).resolve().parents[2]
    else:
        base = Path("/Users/adnan/Documents/cactus")
    model_dir = model_id.split("/")[-1].lower()
    return base / "weights" / model_dir


def _get_local_model_handle():
    global _LOCAL_MODEL_HANDLE, _LOCAL_MODEL_PATH
    settings = get_settings()
    weights_dir = _weights_dir_for_model(settings.local_model, settings)
    if _LOCAL_MODEL_HANDLE is not None and _LOCAL_MODEL_PATH == str(weights_dir):
        return _LOCAL_MODEL_HANDLE, ""
    try:
        from src.cactus import cactus_get_last_error, cactus_init
    except Exception as err:
        return None, f"Local model import failed: {err}"

    handle = cactus_init(str(weights_dir))
    if not handle:
        try:
            from src.cactus import cactus_get_last_error

            return None, cactus_get_last_error() or f"Failed to init local model at {weights_dir}"
        except Exception:
            return None, f"Failed to init local model at {weights_dir}"
    _LOCAL_MODEL_HANDLE = handle
    _LOCAL_MODEL_PATH = str(weights_dir)
    return handle, ""


def _model_variants(name: str) -> List[str]:
    base = (name or "").strip()
    if not base:
        return []
    if base.startswith("models/"):
        return [base, base.replace("models/", "", 1)]
    return [base, f"models/{base}"]


def _discover_cloud_models(client: genai.Client) -> List[str]:
    names: List[str] = []
    try:
        for model in client.models.list():
            name = getattr(model, "name", "") or ""
            if "gemini" in name.lower():
                names.append(name)
    except Exception as err:
        LOGGER.warning("Could not list Gemini models: %s", err)
    return names


def _resolve_cactus_bin() -> str:
    settings = get_settings()
    configured = getattr(settings, "cactus_bin", "") if hasattr(settings, "cactus_bin") else ""
    if configured:
        return configured
    return shutil.which("cactus") or "cactus"


def cactus_transcribe(audio_bytes: bytes, model: str, suffix: str = ".wav") -> Tuple[str, str]:
    if not audio_bytes:
        return "", "No audio bytes received."

    if not suffix.startswith("."):
        suffix = ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        try:
            settings = get_settings()
            cmd = [_resolve_cactus_bin(), "transcribe", model, "--file", tmp.name]
            hf_token = os.getenv("HF_TOKEN", "").strip()
            if hf_token:
                cmd.extend(["--token", hf_token])
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
                cwd=settings.cactus_cwd or None,
            )
        except Exception as err:
            LOGGER.warning("cactus_transcribe failed: %s", err)
            return "", f"Local transcription failed: {err}"

    if proc.returncode != 0:
        if "unauthenticated requests to the HF Hub" in (proc.stderr or proc.stdout):
            return "", "Missing HF auth for transcription. Set HF_TOKEN and pre-download whisper model."
        LOGGER.warning(
            "cactus transcribe non-zero exit (%s): %s",
            proc.returncode,
            (proc.stderr or proc.stdout).strip(),
        )
        err = proc.stderr.strip() or proc.stdout.strip() or "Unknown cactus transcribe error."
        err = f"{err}\nCommand: {' '.join(cmd)}"
        return "", err

    cleaned_lines: List[str] = []
    for raw in proc.stdout.splitlines():
        clean = ANSI_RE.sub("", raw).strip()
        clean = SPECIAL_TOKEN_RE.sub("", clean).strip()
        if clean and clean not in {"[0m", "0m"}:
            cleaned_lines.append(clean)

    lines = [ln for ln in cleaned_lines if not any(p.search(ln) for p in NOISE_PATTERNS)]
    if not lines:
        return "", "Cactus transcribe returned no usable text."

    for ln in reversed(lines):
        match = re.search(r"transcription\s*[:\-]\s*(.+)$", ln, flags=re.IGNORECASE)
        if match:
            text = SPECIAL_TOKEN_RE.sub("", match.group(1)).strip()
            if text:
                return text, ""

    # Fall back only to lines that look like natural language transcript,
    # not CLI status text.
    for ln in reversed(lines):
        if re.search(r"[A-Za-z]{2,}", ln) and len(ln.split()) >= 2:
            text = SPECIAL_TOKEN_RE.sub("", ln).strip()
            if text:
                return text, ""

    return "", "Could not extract transcript from Cactus output."


def cactus_complete(
    user_text: str,
    tools: List[Dict[str, Any]],
    temperature: float = 0.1,
    force_tools: bool = True,
) -> Dict[str, Any]:
    start = time.perf_counter()
    _ = (temperature, force_tools, tools)
    handle, err = _get_local_model_handle()
    if not handle:
        return {
            "confidence": 0.0,
            "function_call": None,
            "response_text": f"Local model unavailable: {err}",
            "latency": time.perf_counter() - start,
        }

    try:
        from src.cactus import cactus_complete as cactus_ffi_complete
    except Exception as import_err:
        return {
            "confidence": 0.0,
            "function_call": None,
            "response_text": f"Local model import error: {import_err}",
            "latency": time.perf_counter() - start,
        }

    action_mode = _is_action_request(user_text)
    if action_mode:
        planner_messages = [
            {
                "role": "system",
                "content": (
                    "You are an action planner. "
                    "If the user asks for an actionable command, call exactly one tool with valid JSON args. "
                    "If no tool matches, do not call a tool and provide a short natural-language response."
                ),
            },
            {"role": "user", "content": user_text},
        ]
        raw = cactus_ffi_complete(
            handle,
            planner_messages,
            tools=tools,
            temperature=0.1,
            max_tokens=180,
            force_tools=True,
            tool_rag_top_k=4,
            confidence_threshold=0.0,
        )
    else:
        chat_messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful local AI assistant. "
                    "Answer directly in natural language and be concise."
                ),
            },
            {"role": "user", "content": user_text},
        ]
        raw = cactus_ffi_complete(
            handle,
            chat_messages,
            temperature=0.35,
            max_tokens=180,
            confidence_threshold=0.0,
        )
    try:
        outer = json.loads(raw)
    except Exception:
        outer = {}

    model_conf = float(outer.get("confidence", 0.0)) if isinstance(outer, dict) else 0.0
    response_text = str(outer.get("response") or "") if isinstance(outer, dict) else ""
    function_call = None
    if isinstance(outer, dict):
        fcalls = outer.get("function_calls") or []
        if isinstance(fcalls, list) and fcalls and isinstance(fcalls[0], dict):
            first = fcalls[0]
            name = first.get("name")
            arguments = first.get("arguments", {})
            if isinstance(arguments, str):
                ok, parsed = _safe_json_dict(arguments)
                arguments = parsed if ok else {}
            if isinstance(name, str) and isinstance(arguments, dict):
                function_call = {"name": name, "args": arguments}

    if not response_text:
        response_text = "I’m here. Ask me anything."

    if (not action_mode) and _looks_like_capability_disclaimer(response_text):
        # Retry once for conversational quality when the model slips into
        # meta capability disclaimers on simple prompts.
        retry_messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise assistant in a normal chat. "
                    "Give the best direct answer to the user's message. "
                    "Avoid capability disclaimers unless explicitly requested."
                ),
            },
            {"role": "user", "content": user_text},
        ]
        retry_raw = cactus_ffi_complete(
            handle,
            retry_messages,
            temperature=0.35,
            max_tokens=140,
            confidence_threshold=0.0,
        )
        try:
            retry_outer = json.loads(retry_raw)
            retry_resp = str(retry_outer.get("response") or "").strip()
            retry_conf = float(retry_outer.get("confidence", model_conf))
            if retry_resp and not _looks_like_capability_disclaimer(retry_resp):
                response_text = retry_resp
                model_conf = max(model_conf, retry_conf)
        except Exception:
            pass

    if (not action_mode) and (_looks_like_refusal(response_text) or _looks_like_capability_disclaimer(response_text)):
        rewrite_messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite your last answer as a direct, useful response to the user. "
                    "Do not mention capabilities, policies, tools, or limitations. "
                    "If uncertain, give practical guidance and ask one concise follow-up question."
                ),
            },
            {"role": "user", "content": user_text},
        ]
        rewrite_raw = cactus_ffi_complete(
            handle,
            rewrite_messages,
            temperature=0.4,
            max_tokens=180,
            confidence_threshold=0.0,
        )
        try:
            rewrite_outer = json.loads(rewrite_raw)
            rewrite_resp = str(rewrite_outer.get("response") or "").strip()
            rewrite_conf = float(rewrite_outer.get("confidence", model_conf))
            if rewrite_resp and not (
                _looks_like_refusal(rewrite_resp) or _looks_like_capability_disclaimer(rewrite_resp)
            ):
                response_text = rewrite_resp
                model_conf = max(model_conf, rewrite_conf)
        except Exception:
            pass

    return {
        "confidence": model_conf,
        "function_call": function_call,
        "response_text": response_text,
        "latency": time.perf_counter() - start,
        "raw_local": raw,
        "action_mode": action_mode,
    }


def _cloud_fallback(user_text: str, tools: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    settings = get_settings()
    start = time.perf_counter()
    global _CLOUD_BACKOFF_UNTIL

    if not settings.gemini_api_key:
        return {
            "routing": "CLOUD",
            "confidence": 0.0,
            "latency": time.perf_counter() - start,
            "tool_name": None,
            "args": {},
            "response_text": "Cloud fallback unavailable: missing GEMINI_API_KEY.",
        }

    now = time.time()
    if now < _CLOUD_BACKOFF_UNTIL:
        retry_in = int(_CLOUD_BACKOFF_UNTIL - now)
        return {
            "routing": "CLOUD",
            "confidence": 0.0,
            "latency": time.perf_counter() - start,
            "tool_name": None,
            "args": {},
            "response_text": f"Cloud fallback temporarily paused due to quota limits. Retry in ~{retry_in}s.",
        }

    try:
        client = genai.Client(api_key=settings.gemini_api_key)

        fdecls = [
            types.FunctionDeclaration(
                name=t["name"],
                description=t.get("description", ""),
                parameters_json_schema=t.get("parameters", {}),
            )
            for t in tools
        ]
        requested = _model_variants(model_name)
        common_fallbacks = [
            "gemini-2.0-flash",
            "models/gemini-2.0-flash",
            "gemini-1.5-flash-8b",
            "models/gemini-1.5-flash-8b",
            "gemini-1.5-pro",
            "models/gemini-1.5-pro",
        ]
        discovered = _discover_cloud_models(client)
        candidates: List[str] = []
        for name in requested + common_fallbacks + discovered:
            if name and name not in candidates:
                candidates.append(name)

        response = None
        used_model = None
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                response = client.models.generate_content(
                    model=candidate,
                    contents=user_text,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        tools=[types.Tool(function_declarations=fdecls)],
                    ),
                )
                used_model = candidate
                break
            except ClientError as err:
                last_error = err
                if "NOT_FOUND" in str(err) or "not found" in str(err).lower():
                    continue
                raise

        if response is None:
            raise RuntimeError(f"No usable Gemini model found. Last error: {last_error}")

        tool_name = None
        args: Dict[str, Any] = {}
        text_out = ""

        candidates = getattr(response, "candidates", None) or []
        if candidates and getattr(candidates[0], "content", None):
            parts = getattr(candidates[0].content, "parts", None) or []
            for part in parts:
                fc = getattr(part, "function_call", None)
                if fc:
                    tool_name = fc.name
                    args = dict(fc.args) if fc.args else {}
                    break

        text_out = getattr(response, "text", "") or "Cloud response generated."

        return {
            "routing": "CLOUD",
            "confidence": 0.85 if tool_name else 0.55,
            "latency": time.perf_counter() - start,
            "tool_name": tool_name,
            "args": args,
            "response_text": text_out or f"Cloud response generated via {used_model}.",
        }
    except ClientError as err:
        message = str(err)
        if "RESOURCE_EXHAUSTED" in message or "429" in message:
            _CLOUD_BACKOFF_UNTIL = time.time() + 60
            return {
                "routing": "CLOUD",
                "confidence": 0.0,
                "latency": time.perf_counter() - start,
                "tool_name": None,
                "args": {},
                "response_text": "Cloud quota exceeded (429). Using local-only mode for ~60s; retry after quota reset.",
            }
        LOGGER.warning("Gemini client error: %s", err)
        return {
            "routing": "CLOUD",
            "confidence": 0.0,
            "latency": time.perf_counter() - start,
            "tool_name": None,
            "args": {},
            "response_text": f"Cloud fallback failed: {err}",
        }
    except Exception as err:
        LOGGER.warning("Gemini fallback failed: %s", err)
        return {
            "routing": "CLOUD",
            "confidence": 0.0,
            "latency": time.perf_counter() - start,
            "tool_name": None,
            "args": {},
            "response_text": f"Cloud fallback failed: {err}",
        }


def generate_hybrid(user_text: str, models: Dict[str, str], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    settings = get_settings()
    route_start = time.perf_counter()

    if not user_text or not user_text.strip():
        return {
            "routing": "LOCAL",
            "confidence": 0.0,
            "latency": 0.0,
            "tool_name": None,
            "args": {},
            "response_text": "Empty transcript.",
        }

    prefer_local = True
    cloud_enabled = bool(models.get("cloud_enabled", True))
    local_result = cactus_complete(
        user_text=user_text,
        tools=tools,
        temperature=0.1,
        force_tools=True,
    )

    confidence = float(local_result.get("confidence", 0.0))
    function_call = local_result.get("function_call")
    has_function = isinstance(function_call, dict) and function_call.get("name")
    tool_name = function_call.get("name") if has_function else None
    raw_args = function_call.get("args") if has_function else {}
    args_ok, args = _safe_json_dict(raw_args)
    valid_tool = tool_name in _tool_names(tools) if tool_name else False

    action_mode = bool(local_result.get("action_mode", _is_action_request(user_text)))
    complexity = _complexity_score(user_text)
    local_latency = float(local_result.get("latency", 0.0))
    local_bad_text = _looks_like_refusal(local_result.get("response_text", "")) or _looks_like_capability_disclaimer(
        local_result.get("response_text", "")
    )
    valid_call, call_reason = _is_valid_tool_call(tool_name, args if args_ok else raw_args, tools)

    reasons: List[str] = []
    if action_mode and not valid_call:
        reasons.append(call_reason)
    if confidence < settings.confidence_threshold:
        reasons.append("low_confidence")
    if complexity >= 0.7:
        reasons.append("high_complexity")
    if _is_reasoning_heavy(user_text):
        reasons.append("reasoning_heavy")
    if local_bad_text:
        reasons.append("low_quality_local_text")
    if local_latency > 1.5:
        reasons.append("local_slow")

    # Local-first policy:
    # - non-action chat stays local unless low-quality + high complexity
    # - action requests require valid tool call or fallback
    must_fallback = False
    if action_mode:
        must_fallback = ("low_confidence" in reasons) or ("no_tool_call" in reasons) or ("unknown_tool" in reasons) or (
            "missing_required" in " ".join(reasons)
        )
    else:
        must_fallback = local_bad_text and cloud_enabled

    if not must_fallback and (prefer_local or confidence >= settings.confidence_threshold):
        return {
            "routing": "LOCAL",
            "confidence": confidence,
            "latency": time.perf_counter() - route_start,
            "tool_name": tool_name,
            "args": args,
            "response_text": local_result.get("response_text", "Local route selected."),
            "local_model_output": local_result,
            "routing_meta": {
                "action_mode": action_mode,
                "complexity": complexity,
                "reasons": reasons,
                "valid_tool_call": valid_call,
                "local_latency": local_latency,
            },
        }

    if not cloud_enabled:
        local_text = (local_result.get("response_text") or "").strip()
        if local_text:
            msg = local_text
        else:
            msg = "Cloud fallback is disabled. Stayed in local-only mode."
        return {
            "routing": "LOCAL",
            "confidence": confidence,
            "latency": time.perf_counter() - route_start,
            "tool_name": tool_name if valid_tool else None,
            "args": args if args_ok else {},
            "response_text": msg,
            "local_model_output": local_result,
            "routing_meta": {
                "action_mode": action_mode,
                "complexity": complexity,
                "reasons": reasons + ["cloud_disabled"],
                "valid_tool_call": valid_call,
                "local_latency": local_latency,
            },
        }

    cloud = _cloud_fallback(user_text, tools, models.get("cloud", settings.cloud_model))
    cloud["latency"] = time.perf_counter() - route_start
    cloud["local_model_output"] = local_result
    cloud["routing_meta"] = {
        "action_mode": action_mode,
        "complexity": complexity,
        "reasons": reasons,
        "valid_tool_call": valid_call,
        "local_latency": local_latency,
    }
    return cloud
