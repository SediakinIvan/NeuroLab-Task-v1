from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from src.config import GigaChatConfig


class GigaChatClient:
    def __init__(self, config: GigaChatConfig, logs_dir: Path) -> None:
        self.config = config
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self.logs_dir / "gigachat_requests.jsonl"
        self._access_token: str | None = None
        self._access_token_expires_at: float = 0.0
        self._oauth_error: RuntimeError | None = None

    @property
    def is_ready(self) -> bool:
        return bool(self.config.enabled and self.config.resolved_api_key())

    def _append_log(self, payload: dict[str, Any]) -> None:
        with self._log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _safe_preview(self, text: str, max_len: int = 180) -> str:
        compact = " ".join(text.split())
        if len(compact) <= max_len:
            return compact
        return f"{compact[:max_len]}..."

    def chat_json(self, *, prompt: str, request_id: str | None = None) -> dict[str, Any]:
        req_id = request_id or str(uuid4())
        api_key = self.config.resolved_api_key()
        if not api_key:
            raise RuntimeError("GigaChat API key is not configured.")

        url = f"{self.config.base_url.rstrip('/')}{self.config.chat_endpoint}"
        bearer_token = self._get_bearer_token(api_key=api_key)
        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
            "Idempotency-Key": req_id,
        }
        models_to_try = [self.config.model, *self.config.model_fallbacks]
        models_to_try = [m for i, m in enumerate(models_to_try) if m and m not in models_to_try[:i]]

        timeout = httpx.Timeout(float(self.config.timeout_seconds))
        errors: list[str] = []
        for model_name in models_to_try:
            body = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a precise data analyst. Respond with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.config.temperature,
            }
            for attempt in range(self.config.max_retries + 1):
                started = time.time()
                try:
                    with httpx.Client(timeout=timeout, verify=self.config.verify_ssl) as client:
                        response = client.post(url, headers=headers, json=body)
                    elapsed_ms = int((time.time() - started) * 1000)

                    if response.status_code == 404:
                        raise ValueError(f"Model '{model_name}' is unavailable.")
                    if response.status_code in {429, 500, 502, 503, 504}:
                        raise httpx.HTTPStatusError(
                            f"Retryable status code: {response.status_code}",
                            request=response.request,
                            response=response,
                        )

                    response.raise_for_status()
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    parsed = self._parse_json_content(content)
                    self._append_log(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "request_id": req_id,
                            "attempt": attempt + 1,
                            "status": "success",
                            "status_code": response.status_code,
                            "elapsed_ms": elapsed_ms,
                            "model": model_name,
                            "request_preview": self._safe_preview(prompt) if self.config.log_raw_text else None,
                        }
                    )
                    return parsed
                except (httpx.RequestError, httpx.HTTPStatusError, KeyError, ValueError, json.JSONDecodeError) as exc:
                    err = f"{model_name}: {exc}"
                    errors.append(err)
                    if "unavailable" in str(exc).lower():
                        break
                    wait_s = self.config.backoff_seconds * (2**attempt)
                    if attempt < self.config.max_retries:
                        time.sleep(wait_s)
                    else:
                        break

        self._append_log(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": req_id,
                "status": "failed",
                "errors": errors,
                "request_preview": self._safe_preview(prompt) if self.config.log_raw_text else None,
            }
        )
        raise RuntimeError(f"GigaChat request failed after retries: {errors}")

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            # Remove possible language marker
            if text.startswith("json"):
                text = text[4:].strip()
        # Try direct JSON parse first.
        try:
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError("Model JSON response must be an object.")
            return obj
        except json.JSONDecodeError:
            pass

        # Fallback: parse first JSON object range.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model response.")
        obj = json.loads(text[start : end + 1])
        if not isinstance(obj, dict):
            raise ValueError("Model JSON response must be an object.")
        return obj

    def _get_bearer_token(self, api_key: str) -> str:
        if not self.config.use_oauth:
            return api_key

        if self._oauth_error is not None:
            raise self._oauth_error

        now = time.time()
        if self._access_token and now < (self._access_token_expires_at - 60):
            return self._access_token

        try:
            token_payload = self._request_oauth_token(api_key=api_key)
        except RuntimeError as exc:
            self._oauth_error = exc
            raise
        access_token = token_payload.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise RuntimeError("OAuth token response does not include access_token.")

        expires_at_ms = token_payload.get("expires_at")
        if isinstance(expires_at_ms, (int, float)):
            self._access_token_expires_at = float(expires_at_ms) / 1000.0
        else:
            self._access_token_expires_at = now + 1800
        self._access_token = access_token
        return access_token

    def _request_oauth_token(self, api_key: str) -> dict[str, Any]:
        url = f"{self.config.oauth_base_url.rstrip('/')}{self.config.oauth_endpoint}"
        headers = {
            "Authorization": f"Basic {api_key}",
            "RqUID": str(uuid4()),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        data = {"scope": self.config.scope}
        timeout = httpx.Timeout(float(self.config.timeout_seconds))

        errors: list[str] = []
        for attempt in range(self.config.max_retries + 1):
            try:
                with httpx.Client(timeout=timeout, verify=self.config.verify_ssl) as client:
                    response = client.post(url, headers=headers, data=data)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise httpx.HTTPStatusError(
                        f"Retryable status code: {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("OAuth response must be an object.")
                return payload
            except (httpx.RequestError, httpx.HTTPStatusError, ValueError, json.JSONDecodeError) as exc:
                errors.append(str(exc))
                if attempt < self.config.max_retries:
                    time.sleep(self.config.backoff_seconds * (2**attempt))
                else:
                    raise RuntimeError(f"GigaChat OAuth failed after retries: {errors}") from exc

        raise RuntimeError("Unreachable OAuth retry state.")
