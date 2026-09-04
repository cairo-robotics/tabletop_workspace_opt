#!/usr/bin/env python3
"""Predict one CASPER-lite intent from a prompt read on stdin.

The script is designed to be used by `casper_lite_evaluator.py --backend command`.
It prints exactly one JSON object on stdout:

  {"intent_id": "...", "confidence": 0.0, "reason": "..."}
"""

import argparse
import base64
import json
import mimetypes
import os
import re
import sys


def configure_stdio_utf8():
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def clean_api_key(value):
    raw = str(value or "").strip()
    return "".join(ch for ch in raw if 33 <= ord(ch) <= 126)


SYSTEM_PROMPT = """You are a shared-autonomy intent inference module.
Choose exactly one candidate_id from the prompt.
Return only JSON that follows the requested schema.
Do not invent candidate ids."""


def extract_candidate_ids(prompt):
    ids = []
    seen = set()
    for match in re.finditer(r"candidate_id=([A-Za-z0-9_.:-]+)", prompt):
        candidate_id = match.group(1).strip()
        if candidate_id and candidate_id not in seen:
            seen.add(candidate_id)
            ids.append(candidate_id)
    return ids


def extract_task_hint(prompt):
    match = re.search(r"Task type hint:\s*([^\n]+)", prompt)
    return match.group(1).strip().lower() if match else ""


def extract_image_paths(prompt):
    paths = []
    seen = set()
    patterns = [
        r"Image path(?:\s+\d+)?:\s*([^\n]+)",
        r"image_path(?:\s+\d+)?:\s*([^\n]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, prompt, flags=re.IGNORECASE):
            path = match.group(1).strip()
            if not path or path.lower() == "unknown":
                continue
            expanded = os.path.expanduser(path)
            if expanded not in seen:
                seen.add(expanded)
                paths.append(expanded)
    return paths


def image_path_to_data_url(path):
    if not path:
        return ""
    if not os.path.exists(path):
        raise RuntimeError("image path does not exist: {}".format(path))
    mime_type = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return "data:{};base64,{}".format(mime_type, encoded)


def parse_json_response(text, candidate_ids):
    raw = str(text or "").strip()
    payload = {}
    try:
        payload = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
            except Exception:
                payload = {}

    intent_id = ""
    confidence = 0.0
    reason = ""
    if isinstance(payload, dict):
        intent_id = str(
            payload.get("intent_id")
            or payload.get("candidate_id")
            or payload.get("intent")
            or ""
        ).strip()
        try:
            confidence = float(payload.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        reason = str(payload.get("reason") or "").strip()

    if intent_id not in candidate_ids:
        lowered = {candidate_id.lower(): candidate_id for candidate_id in candidate_ids}
        intent_id = lowered.get(intent_id.lower(), "")

    confidence = max(0.0, min(1.0, confidence))
    return {"intent_id": intent_id, "confidence": confidence, "reason": reason}


def heuristic_predict(prompt, candidate_ids):
    """Local sanity-check provider for testing command-mode wiring."""
    text = prompt.lower()
    task_hint = extract_task_hint(prompt)

    wanted_object = ""
    for slot in ("left", "center", "right"):
        slot_match = re.search(r"\b{}=([A-Za-z0-9_:-]+)".format(slot), prompt)
        if slot_match and slot in text:
            wanted_object = slot_match.group(1)
            break

    if not wanted_object:
        for obj in ("whole_milk", "oat_milk", "soy_milk"):
            if obj.replace("_", " ") in text or obj.split("_", 1)[0] in text:
                wanted_object = obj
                break

    if not task_hint:
        if "pour" in text:
            task_hint = "pour"
        elif any(word in text for word in ("pick", "grab", "take", "choose")):
            task_hint = "pickup"

    best_id = candidate_ids[0] if candidate_ids else ""
    best_score = -1
    candidate_blocks = re.findall(
        r"candidate_id=([A-Za-z0-9_.:-]+)\s+object=([A-Za-z0-9_.:-]+)\s+grasp_type=([A-Za-z0-9_.:-]+)\s+task_suitability=([A-Za-z0-9_.:-]+)",
        prompt,
    )
    for candidate_id, object_name, _grasp_type, task_suitability in candidate_blocks:
        score = 0
        if wanted_object and object_name == wanted_object:
            score += 2
        if task_hint and task_suitability == task_hint:
            score += 1
        if score > best_score:
            best_score = score
            best_id = candidate_id

    confidence = 0.5 if best_score <= 0 else min(0.95, 0.45 + 0.2 * best_score)
    return {
        "intent_id": best_id,
        "confidence": confidence,
        "reason": "heuristic provider for command-mode testing",
    }


def openai_predict(prompt, candidate_ids, args):
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "OpenAI Python SDK is not installed. Install with `pip install openai`."
        ) from exc

    raw_api_key = os.environ.get(args.api_key_env)
    if not raw_api_key:
        raise RuntimeError("{} is not set".format(args.api_key_env))
    api_key = clean_api_key(raw_api_key)
    if not api_key:
        raise RuntimeError("{} is empty after removing invalid characters".format(args.api_key_env))

    if not candidate_ids:
        raise RuntimeError("no candidate_id entries found in prompt")

    client = OpenAI(api_key=api_key)
    content = [{"type": "input_text", "text": prompt}]
    if not args.no_image:
        for image_path in extract_image_paths(prompt):
            content.append(
                {
                    "type": "input_image",
                    "image_url": image_path_to_data_url(image_path),
                    "detail": args.image_detail,
                }
            )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "intent_id": {
                "type": "string",
                "enum": candidate_ids,
                "description": "The selected candidate_id.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence in the selected intent.",
            },
            "reason": {
                "type": "string",
                "description": "A short explanation for debugging.",
            },
        },
        "required": ["intent_id", "confidence", "reason"],
    }

    kwargs = {
        "model": args.model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "max_output_tokens": int(args.max_output_tokens),
    }
    if args.temperature is not None:
        kwargs["temperature"] = float(args.temperature)

    # Current Responses API structured output uses text.format. Some older SDK
    # builds accepted response_format, so keep a fallback for lab machines.
    try:
        response = client.responses.create(
            text={
                "format": {
                    "type": "json_schema",
                    "name": "casper_lite_intent",
                    "schema": schema,
                    "strict": True,
                }
            },
            **kwargs
        )
    except TypeError:
        response = client.responses.create(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "casper_lite_intent",
                    "schema": schema,
                    "strict": True,
                },
            },
            **kwargs
        )

    output_text = getattr(response, "output_text", "")
    if not output_text:
        output_text = str(response)
    parsed = parse_json_response(output_text, candidate_ids)
    if not parsed["intent_id"]:
        raise RuntimeError("model response did not contain a valid candidate_id: {}".format(output_text))
    return parsed


def deepseek_predict(prompt, candidate_ids, args):
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "OpenAI Python SDK is not installed. Install with `pip install openai`."
        ) from exc

    if not os.environ.get(args.api_key_env):
        raise RuntimeError("{} is not set".format(args.api_key_env))

    if not candidate_ids:
        raise RuntimeError("no candidate_id entries found in prompt")

    model = args.model or "deepseek-v4-pro"
    client = OpenAI(api_key=os.environ.get(args.api_key_env), base_url=args.deepseek_base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                    + '\nOutput valid json only, for example: '
                    + '{"intent_id":"soy_side","confidence":0.82,"reason":"short reason"}'
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=int(args.max_output_tokens),
        extra_body={"thinking": {"type": "disabled"}},
    )
    output_text = response.choices[0].message.content or ""
    parsed = parse_json_response(output_text, candidate_ids)
    if not parsed["intent_id"]:
        raise RuntimeError("model response did not contain a valid candidate_id: {}".format(output_text))
    return parsed


def main():
    configure_stdio_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=("openai", "deepseek", "heuristic"),
        default="openai",
        help="Prediction provider. Use heuristic only to test plumbing.",
    )
    parser.add_argument("--model", default="", help="Model name. Defaults to gpt-5 for OpenAI, deepseek-v4-pro for DeepSeek.")
    parser.add_argument("--api-key-env", default="", help="Environment variable containing API key.")
    parser.add_argument("--deepseek-base-url", default="https://api.deepseek.com", help="DeepSeek OpenAI-compatible base URL.")
    parser.add_argument("--max-output-tokens", type=int, default=180, help="OpenAI response token limit.")
    parser.add_argument("--temperature", type=float, default=None, help="Optional model temperature.")
    parser.add_argument(
        "--image-detail",
        choices=("low", "auto", "high"),
        default="low",
        help="OpenAI image detail level. low is cheaper and usually enough for CASPER-lite logs.",
    )
    parser.add_argument("--no-image", action="store_true", help="Do not attach Image path as image input.")
    args = parser.parse_args()

    prompt = sys.stdin.read()
    candidate_ids = extract_candidate_ids(prompt)
    if not args.api_key_env:
        args.api_key_env = "DEEPSEEK_API_KEY" if args.provider == "deepseek" else "OPENAI_API_KEY"
    if not args.model and args.provider == "openai":
        args.model = "gpt-5"

    try:
        if args.provider == "heuristic":
            result = heuristic_predict(prompt, candidate_ids)
        elif args.provider == "deepseek":
            result = deepseek_predict(prompt, candidate_ids, args)
        else:
            result = openai_predict(prompt, candidate_ids, args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "intent_id": "",
                    "confidence": 0.0,
                    "reason": "prediction_failed: {}".format(exc),
                },
                sort_keys=True,
            )
        )
        return 1

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
