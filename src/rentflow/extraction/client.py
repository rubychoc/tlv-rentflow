"""
OpenAI client wrapper for the extraction engine.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token counts from one successful OpenAI API call."""
    prompt_tokens: int
    completion_tokens: int
    # Subset of prompt_tokens served from the prompt cache (0 if caching not active).
    cached_tokens: int
    model: str

    @property
    def uncached_prompt_tokens(self) -> int:
        return self.prompt_tokens - self.cached_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

# Provenance fields that live at the group (shared) level.
_GROUP_PROV_FIELDS = [
    "budget_nis", "move_in_date", "has_pets", "household_size", "preferred_language",
]

# Provenance fields that live on each individual applicant.
_PERSON_PROV_FIELDS = [
    "employment_status", "age", "gender", "name", "phone",
]

# Per-person object schema used as the `items` type in the `applicants` array.
_PERSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "employment_status", "age", "gender", "name", "phone",
        *[f"{f}_prov" for f in _PERSON_PROV_FIELDS],
    ],
    "properties": {
        "employment_status": {
            "anyOf": [
                {"type": "string", "enum": ["employed", "self_employed", "student", "unemployed"]},
                {"type": "null"},
            ],
        },
        "age": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "gender": {
            "anyOf": [
                {"type": "string", "enum": ["male", "female", "other"]},
                {"type": "null"},
            ],
        },
        "name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "phone": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        **{
            f"{f}_prov": {"anyOf": [{"type": "string"}, {"type": "null"}]}
            for f in _PERSON_PROV_FIELDS
        },
    },
}

# Top-level group schema.
_TENANT_GROUP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "budget_nis", "move_in_date", "has_pets", "household_size", "preferred_language",
        "applicants",
        *[f"{f}_prov" for f in _GROUP_PROV_FIELDS],
    ],
    "properties": {
        "budget_nis": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "move_in_date": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "has_pets": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
        "household_size": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "preferred_language": {
            "anyOf": [
                {"type": "string", "enum": ["he", "en"]},
                {"type": "null"},
            ],
        },
        "applicants": {
            "type": "array",
            "items": _PERSON_SCHEMA,
        },
        **{
            f"{f}_prov": {"anyOf": [{"type": "string"}, {"type": "null"}]}
            for f in _GROUP_PROV_FIELDS
        },
    },
}


class ExtractionClient:
    """
    Thin wrapper around the OpenAI chat completions API.

    The client owns the network concern (retries, timeouts, auth).
    The engine owns business logic (prompt assembly, response parsing).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1-mini",
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._client = OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=0)
        # Set after every successful extract_raw() call; None before the first call.
        self.last_usage: TokenUsage | None = None

    def extract_raw(self, system_prompt: str, user_text: str) -> dict[str, Any]:
        """
        Sends the prompt + tenant text to OpenAI and returns the parsed JSON dict.

        Retries on transient errors with exponential backoff.

        Raises:
            RuntimeError: if all retries are exhausted or a non-retryable error occurs.
        """
        backoff = 1.0
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "TenantGroup",
                            "strict": True,
                            "schema": _TENANT_GROUP_SCHEMA,
                        },
                    },
                    temperature=0.5,
                )

                raw_json = response.choices[0].message.content
                if not raw_json:
                    raise RuntimeError("OpenAI returned an empty response body.")

                usage = response.usage
                cached = 0
                if usage and usage.prompt_tokens_details:
                    cached = usage.prompt_tokens_details.cached_tokens or 0
                if usage:
                    self.last_usage = TokenUsage(
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        cached_tokens=cached,
                        model=self._model,
                    )

                return json.loads(raw_json)

            except (RateLimitError, APITimeoutError) as exc:
                last_error = exc
                logger.warning(
                    "Attempt %d/%d failed (%s). Retrying in %.1fs.",
                    attempt, self._max_retries, type(exc).__name__, backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

            except APIError as exc:
                if exc.status_code is not None and exc.status_code < 500:
                    raise RuntimeError(f"Non-retryable OpenAI error: {exc}") from exc
                last_error = exc
                logger.warning(
                    "Attempt %d/%d — API error %s. Retrying in %.1fs.",
                    attempt, self._max_retries, exc.status_code, backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

            except json.JSONDecodeError as exc:
                raise RuntimeError(f"OpenAI returned invalid JSON: {exc}") from exc

        raise RuntimeError(
            f"OpenAI extraction failed after {self._max_retries} attempts. "
            f"Last error: {last_error}"
        )
