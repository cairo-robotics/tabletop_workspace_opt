"""Minimal self-contained OpenAI-compatible VLM client.

Talks to a local vLLM server (chat completions). Credentials come from an
environment variable only (any placeholder works for keyless local servers).
Responses are expected to be a single strict-JSON object; one markdown code
fence and one leading closed <think>...</think> block are tolerated.
"""
import base64
import concurrent.futures
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests

_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_THINK = re.compile(r"^\s*<think>(.*?)</think>", re.DOTALL)


def encode_image_b64(bgr_image) -> str:
    """BGR ndarray -> base64 JPEG string."""
    import cv2
    ok, buf = cv2.imencode(".jpg", bgr_image,
                           [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("failed to JPEG-encode image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def parse_strict_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse one JSON object; tolerate a leading closed <think> block and a
    single markdown fence. Returns None on anything else (prose, garbage,
    truncated think blocks)."""
    if not text or not text.strip():
        return None
    text = _THINK.sub("", text.strip(), count=1).strip()
    fences = _FENCE.findall(text)
    if len(fences) == 1 and not _FENCE.sub("", text, count=1).strip():
        text = fences[0].strip()
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


class VLMClient:
    def __init__(self, base_url: str, model: str,
                 api_key_env: str = "VLM_API_KEY",
                 temperature: float = 0.6,
                 max_tokens: int = 256,
                 request_timeout_s: float = 20.0):
        base_url = (base_url or "").rstrip("/")
        if "://" not in base_url:
            base_url = "http://" + base_url
        self.base_url = base_url
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = float(request_timeout_s)
        self._api_key = os.environ.get(api_key_env) or "none"
        self._session = requests.Session()

    def chat_completion(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """One request; returns the response text or None on failure."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        try:
            response = self._session.post(
                self.base_url + "/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer %s" % self._api_key},
                timeout=self.timeout,
            )
        except requests.RequestException:
            return None
        if response.status_code != 200:
            return None
        try:
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    def vote_batch(self, messages: List[Dict[str, Any]], k: int
                   ) -> List[Dict[str, Any]]:
        """K identical parallel requests (vLLM batches them); returns the
        successfully parsed JSON dicts (may be fewer than k)."""
        results: List[Dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=k) as pool:
            futures = [pool.submit(self.chat_completion, messages)
                       for _ in range(k)]
            for future in concurrent.futures.as_completed(futures):
                parsed = parse_strict_json(future.result())
                if parsed is not None:
                    results.append(parsed)
        return results


def render_messages_text(messages: List[Dict[str, Any]]) -> str:
    """Human-readable flattening of chat messages for prompt inspection.

    Text parts are shown inline; image parts become placeholders (the actual
    pixels are inspected via the published/saved image, not the base64 blob).
    """
    lines: List[str] = []
    for message in messages:
        lines.append("===== %s =====" % str(message.get("role", "?")).upper())
        content = message.get("content")
        if isinstance(content, str):
            lines.append(content)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    lines.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    lines.append("[IMAGE: %d base64 chars — see saved/"
                                 "published frame]" % len(url))
        lines.append("")
    return "\n".join(lines)


def image_content_part(bgr_image) -> Dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64,%s" % encode_image_b64(bgr_image)
        },
    }


def text_content_part(text: str) -> Dict[str, str]:
    return {"type": "text", "text": text}
