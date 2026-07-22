from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
_SCHEMA_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class OpenRouterError(RuntimeError):
    """Base error raised by the constrained OpenRouter integration."""


class ModelNotAllowedError(OpenRouterError):
    """Raised before I/O when a caller requests a model outside the allowlist."""


class StructuredResponseError(OpenRouterError):
    """Raised when an endpoint response cannot satisfy the requested contract."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChatMessage(_StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class OpenRouterSettings(_StrictModel):
    """Connection policy.

    A model is intentionally absent from this object. Every operation must name a
    model, and that model must be present in ``allowed_models``.
    """

    api_key: SecretStr
    allowed_models: frozenset[str]
    endpoint: str = OPENROUTER_CHAT_COMPLETIONS_URL
    timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    max_tokens: int = Field(default=4096, ge=1, le=64_000)
    max_attempts: int = Field(default=3, ge=1, le=10)
    retry_backoff_seconds: float = Field(default=1.0, ge=0, le=60)
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "none"
    app_name: str | None = None
    app_url: str | None = None

    @field_validator("api_key")
    @classmethod
    def _nonempty_api_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("api_key must not be empty")
        return value

    @field_validator("allowed_models")
    @classmethod
    def _explicit_nonempty_allowlist(cls, value: frozenset[str]) -> frozenset[str]:
        cleaned = frozenset(model.strip() for model in value if model.strip())
        if not cleaned:
            raise ValueError("allowed_models must contain at least one explicit model id")
        return cleaned


class CompletionProvenance(_StrictModel):
    model: str
    endpoint: str
    response_id: str | None
    prompt_template_version: str
    schema_name: str
    payload_sha256: str
    schema_sha256: str
    response_sha256: str
    source_sha256: dict[str, str]
    created_at: datetime
    zdr_enforced: Literal[True] = True
    strict_json_schema: Literal[True] = True


OutputT = TypeVar("OutputT", bound=BaseModel)


class StructuredCompletion(_StrictModel, Generic[OutputT]):
    output: OutputT
    provenance: CompletionProvenance
    usage: dict[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=lambda item: repr(item))
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")


def canonical_json(value: Any) -> str:
    """Return a stable UTF-8 JSON representation suitable for provenance hashes."""

    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _coerce_messages(messages: Sequence[ChatMessage | Mapping[str, Any]]) -> list[ChatMessage]:
    if not messages:
        raise ValueError("messages must not be empty")
    return [message if isinstance(message, ChatMessage) else ChatMessage.model_validate(message) for message in messages]


def _extract_content(response_body: Mapping[str, Any]) -> Any:
    try:
        message = response_body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise StructuredResponseError("OpenRouter response has no choices[0].message") from exc

    if isinstance(message, Mapping) and message.get("parsed") is not None:
        return message["parsed"]
    if not isinstance(message, Mapping) or "content" not in message:
        raise StructuredResponseError("OpenRouter response message has no content")

    content = message["content"]
    if isinstance(content, Mapping):
        return dict(content)
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        blocks: list[str] = []
        for block in content:
            if isinstance(block, str):
                blocks.append(block)
            elif isinstance(block, Mapping) and isinstance(block.get("text"), str):
                blocks.append(block["text"])
        content = "".join(blocks)
    if not isinstance(content, str):
        choice = response_body.get("choices", [{}])[0]
        message_keys = sorted(str(key) for key in message) if isinstance(message, Mapping) else []
        finish_reason = choice.get("finish_reason") if isinstance(choice, Mapping) else None
        raise StructuredResponseError(
            "Structured response content must be a JSON string or object "
            f"(content_type={type(content).__name__}, message_keys={message_keys}, "
            f"finish_reason={finish_reason!r})"
        )
    stripped = content.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise StructuredResponseError("Structured response content is not valid JSON") from exc


class OpenRouterClient:
    """Small policy-enforcing client for OpenRouter structured outputs.

    ``http_client`` is injectable so callers can use ``httpx.MockTransport`` in
    tests and a separately configured/pool-aware client in production.
    """

    def __init__(
        self,
        settings: OpenRouterSettings,
        *,
        http_client: httpx.Client | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self._client = http_client or httpx.Client()
        self._owns_client = http_client is None
        self._clock = clock
        self._sleeper = sleeper

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OpenRouterClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def complete_structured(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
        response_model: type[OutputT],
        prompt_template_version: str,
        schema_name: str | None = None,
        source_payloads: Mapping[str, Any] | None = None,
    ) -> StructuredCompletion[OutputT]:
        if model not in self.settings.allowed_models:
            raise ModelNotAllowedError(
                f"Model {model!r} is not allowlisted; allowed models are "
                f"{sorted(self.settings.allowed_models)!r}"
            )
        if not prompt_template_version.strip():
            raise ValueError("prompt_template_version must not be empty")

        resolved_schema_name = schema_name or response_model.__name__
        if not _SCHEMA_NAME.fullmatch(resolved_schema_name):
            raise ValueError("schema_name must contain 1-64 letters, digits, underscores, or hyphens")

        normalized_messages = _coerce_messages(messages)
        schema = response_model.model_json_schema()
        request_payload: dict[str, Any] = {
            "model": model,
            "messages": [message.model_dump(mode="json") for message in normalized_messages],
            "temperature": 0,
            "max_tokens": self.settings.max_tokens,
            "reasoning_effort": self.settings.reasoning_effort,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": resolved_schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
            "provider": {
                "zdr": True,
                "require_parameters": True,
            },
        }

        headers = {
            "Authorization": f"Bearer {self.settings.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        if self.settings.app_url:
            headers["HTTP-Referer"] = self.settings.app_url
        if self.settings.app_name:
            headers["X-Title"] = self.settings.app_name

        response_body: Any = None
        last_error: Exception | None = None
        for attempt in range(1, self.settings.max_attempts + 1):
            try:
                response = self._client.post(
                    self.settings.endpoint,
                    headers=headers,
                    json=request_payload,
                    timeout=self.settings.timeout_seconds,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                response_body = response.json()
                break
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code == 429 or exc.response.status_code >= 500
                )
                if not retryable or attempt >= self.settings.max_attempts:
                    break
                retry_after = (
                    exc.response.headers.get("Retry-After")
                    if isinstance(exc, httpx.HTTPStatusError)
                    else None
                )
                try:
                    delay = float(retry_after) if retry_after is not None else None
                except ValueError:
                    delay = None
                self._sleeper(
                    delay
                    if delay is not None
                    else self.settings.retry_backoff_seconds * (2 ** (attempt - 1))
                )
        if response_body is None:
            raise OpenRouterError(f"OpenRouter request failed: {last_error}") from last_error
        if not isinstance(response_body, Mapping):
            raise StructuredResponseError("OpenRouter response body must be a JSON object")

        try:
            output = response_model.model_validate(_extract_content(response_body))
        except StructuredResponseError:
            raise
        except Exception as exc:
            raise StructuredResponseError(
                f"Response does not satisfy {resolved_schema_name}: {exc}"
            ) from exc

        sources = source_payloads or {}
        provenance = CompletionProvenance(
            model=model,
            endpoint=self.settings.endpoint,
            response_id=str(response_body["id"]) if response_body.get("id") is not None else None,
            prompt_template_version=prompt_template_version,
            schema_name=resolved_schema_name,
            payload_sha256=payload_sha256(request_payload),
            schema_sha256=payload_sha256(schema),
            response_sha256=payload_sha256(response_body),
            source_sha256={name: payload_sha256(value) for name, value in sorted(sources.items())},
            created_at=self._clock(),
        )
        usage = response_body.get("usage")
        return StructuredCompletion(
            output=output,
            provenance=provenance,
            usage=dict(usage) if isinstance(usage, Mapping) else {},
        )
