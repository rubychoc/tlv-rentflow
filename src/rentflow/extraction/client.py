"""
OpenAI client wrapper for the extraction engine.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

logger = logging.getLogger(__name__)

# All screening fields that get a paired _prov provenance key.
_PROVENANCE_FIELDS = [
    "budget_nis", "move_in_date",
    "employment_status", "has_pets", "num_roommates",
    "age", "gender",
    "name", "phone", "preferred_language",
]

_TENANT_PROFILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "budget_nis", "move_in_date",
        "employment_status", "has_pets", "num_roommates",
        "age", "gender",
        "name", "phone", "preferred_language",
        *[f"{f}_prov" for f in _PROVENANCE_FIELDS],
    ],
    "properties": {
        "budget_nis": {
            "anyOf": [{"type": "integer"}, {"type": "null"}],
        },
        "move_in_date": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
        },
        "employment_status": {
            "anyOf": [
                {"type": "string", "enum": ["employed", "self_employed", "student", "unemployed"]},
                {"type": "null"},
            ],
        },
        "has_pets": {
            "anyOf": [{"type": "boolean"}, {"type": "null"}],
        },
        "num_roommates": {
            "anyOf": [{"type": "integer"}, {"type": "null"}],
        },
        "age": {
            "anyOf": [{"type": "integer"}, {"type": "null"}],
        },
        "gender": {
            "anyOf": [
                {"type": "string", "enum": ["male", "female", "other"]},
                {"type": "null"},
            ],
        },
        "name": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
        },
        "phone": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
        },
        "preferred_language": {
            "anyOf": [
                {"type": "string", "enum": ["he", "en"]},
                {"type": "null"},
            ],
        },
        **{
            f"{f}_prov": {"anyOf": [{"type": "string"}, {"type": "null"}]}
            for f in _PROVENANCE_FIELDS
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
                            "name": "TenantProfile",
                            "strict": True,
                            "schema": _TENANT_PROFILE_SCHEMA,
                        },
                    },
                    temperature=0,
                )

                raw_json = response.choices[0].message.content
                if not raw_json:
                    raise RuntimeError("OpenAI returned an empty response body.")

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
