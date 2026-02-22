
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    start_time = time.time()

    model_candidates = [
        "gemini-2.5-flash",
        "models/gemini-2.5-flash",
        "gemini-2.0-flash",
        "models/gemini-2.0-flash",
        "gemini-1.5-flash-8b",
        "models/gemini-1.5-flash-8b",
    ]

    gemini_response = None
    last_err = None
    for model_name in model_candidates:
        try:
            gemini_response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(tools=gemini_tools),
            )
            break
        except Exception as err:
            last_err = err
            if "NOT_FOUND" in str(err) or "no longer available" in str(err).lower():
                continue
            raise

    if gemini_response is None:
        raise RuntimeError(f"No available Gemini model found. Last error: {last_err}")

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Hybrid local-first strategy with deterministic routing logic.
    Keeps interface compatible with benchmark.py.
    """

    def _last_user_text(msgs):
        for m in reversed(msgs):
            if m.get("role") == "user":
                return str(m.get("content", "")).strip().lower()
        return ""

    def _is_multi_intent(text):
        markers = [" and ", ", and ", " also ", " then ", " plus "]
        return any(tok in text for tok in markers)

    def _expected_intents(text):
        # Base 1 intent; add one per conjunction marker hit.
        markers = [" and ", ", and ", " also ", " then ", " plus "]
        extra = sum(text.count(m) for m in markers)
        return max(1, 1 + extra)

    def _intent_tool_hints(text):
        hint_map = {
            "get_weather": ["weather", "temperature", "forecast", "rain", "sunny"],
            "set_alarm": ["alarm", "wake me", "wake", "am", "pm"],
            "send_message": ["message", "text", "sms", "send"],
            "create_reminder": ["remind", "reminder"],
            "search_contacts": ["contact", "contacts", "find", "look up", "search"],
            "play_music": ["play", "music", "song", "playlist", "beats", "jazz", "classical"],
            "set_timer": ["timer", "countdown", "minutes"],
        }
        hinted = set()
        for tool_name, keywords in hint_map.items():
            if any(k in text for k in keywords):
                hinted.add(tool_name)
        return hinted

    def _tool_meta(available_tools):
        names = set()
        required_by_name = {}
        for t in available_tools:
            name = t.get("name")
            if not isinstance(name, str):
                continue
            names.add(name)
            params = t.get("parameters", {}) if isinstance(t.get("parameters"), dict) else {}
            req = params.get("required", [])
            required_by_name[name] = [r for r in req if isinstance(r, str)]
        return names, required_by_name

    def _is_plausible_arg(name, args):
        if name == "set_alarm":
            hour = args.get("hour")
            minute = args.get("minute")
            if not isinstance(hour, int) or not isinstance(minute, int):
                return False
            return 0 <= hour <= 23 and 0 <= minute <= 59
        if name == "set_timer":
            minutes = args.get("minutes")
            return isinstance(minutes, int) and minutes > 0
        return True

    def _validate_local(result, available_tools):
        tool_names, required_by_name = _tool_meta(available_tools)
        calls = result.get("function_calls", [])
        if not isinstance(calls, list):
            return True, 0  # invalid structure

        valid_call_count = 0
        for call in calls:
            if not isinstance(call, dict):
                return True, valid_call_count
            name = call.get("name")
            args = call.get("arguments")
            if name not in tool_names:
                return True, valid_call_count
            if not isinstance(args, dict):
                return True, valid_call_count
            for req_key in required_by_name.get(name, []):
                value = args.get(req_key)
                if value is None:
                    return True, valid_call_count
                if isinstance(value, str) and not value.strip():
                    return True, valid_call_count
            if not _is_plausible_arg(name, args):
                return True, valid_call_count
            valid_call_count += 1
        return False, valid_call_count

    def _acceptance_threshold(tool_count, multi_intent):
        t = 0.66
        if tool_count >= 4:
            t -= 0.06
        if tool_count >= 6:
            t -= 0.04
        if multi_intent:
            t -= 0.06
        return max(0.46, min(0.72, t))

    def _should_accept_local(result, text, available_tools):
        conf = float(result.get("confidence", 0.0))
        calls = result.get("function_calls", [])
        n_calls = len(calls) if isinstance(calls, list) else 0
        n_tools = len(available_tools)
        multi_intent = _is_multi_intent(text)
        local_invalid, _ = _validate_local(result, available_tools)
        if local_invalid:
            return False

        tool_names = {t.get("name") for t in available_tools if isinstance(t.get("name"), str)}
        predicted_names = [c.get("name") for c in calls if isinstance(c, dict)]
        predicted_set = {n for n in predicted_names if isinstance(n, str)}
        hinted = _intent_tool_hints(text) & tool_names
        if hinted and predicted_set and hinted.isdisjoint(predicted_set):
            return False

        # High-confidence clear single-intent local fast path.
        if (not multi_intent) and n_tools <= 2 and n_calls == 1 and conf >= 0.74:
            return True
        if (not multi_intent) and hinted and len(hinted) == 1 and predicted_set == hinted and conf >= 0.70:
            return True

        t = _acceptance_threshold(n_tools, multi_intent)
        if multi_intent:
            expected = min(_expected_intents(text), max(1, n_tools))
            return n_calls >= max(2, expected) and conf >= (t - 0.02)
        return n_calls >= 1 and conf >= t

    def _choose_better(a, b, available_tools):
        _, a_valid = _validate_local(a, available_tools)
        _, b_valid = _validate_local(b, available_tools)
        if b_valid != a_valid:
            return b if b_valid > a_valid else a
        a_conf = float(a.get("confidence", 0.0))
        b_conf = float(b.get("confidence", 0.0))
        if b_conf != a_conf:
            return b if b_conf > a_conf else a
        a_time = float(a.get("total_time_ms", 0.0))
        b_time = float(b.get("total_time_ms", 0.0))
        return b if b_time < a_time else a

    def _should_force_cloud(result, text, available_tools):
        # Explicit cloud-forcing policy for ambiguous/complex cases.
        conf = float(result.get("confidence", 0.0))
        tool_count = len(available_tools)
        words = len([w for w in text.split() if w.strip()])
        calls = result.get("function_calls", [])
        n_calls = len(calls) if isinstance(calls, list) else 0
        multi_intent = _is_multi_intent(text)
        local_invalid, valid_count = _validate_local(result, available_tools)

        if local_invalid:
            return True
        if multi_intent and valid_count < max(2, _expected_intents(text)):
            return True
        if multi_intent and conf < 0.83:
            return True
        if words <= 8 and tool_count >= 3 and conf < 0.86:
            return True
        if words <= 10 and tool_count >= 4 and conf < 0.88:
            return True
        if n_calls == 0 and conf < 0.95:
            return True
        return False

    user_text = _last_user_text(messages)

    # Pass 1: local
    local_1 = generate_cactus(messages, tools)
    local_elapsed = float(local_1.get("total_time_ms", 0.0))
    best_local = dict(local_1)

    # Pass 2: one local retry only when first pass is weak.
    if not _should_accept_local(best_local, user_text, tools):
        local_2 = generate_cactus(messages, tools)
        local_elapsed += float(local_2.get("total_time_ms", 0.0))
        best_local = _choose_better(local_1, local_2, tools)

    has_cloud_key = bool(os.environ.get("GEMINI_API_KEY"))
    force_cloud = _should_force_cloud(best_local, user_text, tools)

    # Graceful local-only mode when cloud credentials are unavailable.
    if not has_cloud_key:
        best_local = dict(best_local)
        best_local["total_time_ms"] = local_elapsed
        best_local["source"] = "on-device"
        return best_local

    if force_cloud or not _should_accept_local(best_local, user_text, tools):
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = float(best_local.get("confidence", 0.0))
        cloud["total_time_ms"] = float(cloud.get("total_time_ms", 0.0)) + local_elapsed
        return cloud

    best_local = dict(best_local)
    best_local["total_time_ms"] = local_elapsed
    best_local["source"] = "on-device"
    return best_local


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
