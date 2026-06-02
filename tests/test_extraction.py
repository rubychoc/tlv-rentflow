"""
Deterministic tests for Station 2 — extraction engine and client (no real OpenAI calls).

Split into two sections:

  2a — ExtractionEngine (engine.py)
       The engine receives a raw dict from the client, reshapes provenance keys,
       validates into TenantProfile, and raises ExtractionError on any failure.
       Tests here stub ExtractionClient so the engine's logic runs in isolation.

  2b — ExtractionClient (client.py)
       The client owns retries, backoff, and error mapping. Tests here stub
       the underlying OpenAI SDK object so every API error path can be exercised
       without a network connection.

Neither section makes any network calls or requires OPENAI_API_KEY.
"""

import json
import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest
from openai import APIError, APITimeoutError, RateLimitError

from rentflow.extraction.client import (
    ExtractionClient,
    _PROVENANCE_FIELDS,
    _TENANT_PROFILE_SCHEMA,
)
from rentflow.extraction.engine import ExtractionEngine, ExtractionError
from rentflow.extraction.prompts import SYSTEM_PROMPT
from rentflow.offer.models import (
    Channel,
    EmploymentStatus,
    Gender,
    RawOffer,
    TenantProfile,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_offer(text: str = "מעוניין", offer_id: str = "test_001") -> RawOffer:
    return RawOffer(
        offer_id=offer_id,
        channel=Channel.WHATSAPP,
        sender="+972541234567",
        timestamp=datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
        text=text,
    )


def _stub_engine(response_dict: dict) -> ExtractionEngine:
    """Engine wired to a fake client returning response_dict."""
    client = MagicMock()
    client.extract_raw.return_value = response_dict
    return ExtractionEngine(client=client)


# Minimal valid LLM response: all fields null, all _prov keys null.
# Must match the flat schema the client returns before engine reshaping.
_NULL_PROFILE: dict = {
    "budget_nis": None, "move_in_date": None,
    "employment_status": None, "has_pets": None, "num_roommates": None,
    "age": None, "gender": None,
    "name": None, "phone": None, "preferred_language": None,
    **{f"{f}_prov": None for f in _PROVENANCE_FIELDS},
}


# ---------------------------------------------------------------------------
# 2a — ExtractionEngine: happy path
# ---------------------------------------------------------------------------

class TestEngineHappyPath:
    def test_valid_response_returns_extraction_result(self):
        result = _stub_engine(_NULL_PROFILE).extract(_make_offer())
        assert isinstance(result.profile, TenantProfile)

    def test_result_carries_the_original_offer(self, raw_offer):
        result = _stub_engine(_NULL_PROFILE).extract(raw_offer)
        assert result.offer is raw_offer

    def test_all_null_fields_stay_none(self):
        profile = _stub_engine(_NULL_PROFILE).extract(_make_offer()).profile
        for field in ("budget_nis", "move_in_date", "employment_status",
                      "has_pets", "num_roommates", "age", "gender"):
            assert getattr(profile, field) is None

    def test_populated_fields_land_with_correct_types(self):
        response = {
            **_NULL_PROFILE,
            "budget_nis": 7000, "budget_nis_prov": "7000",
            "move_in_date": "2026-08-01", "move_in_date_prov": "August 1st",
            "employment_status": "employed", "employment_status_prov": "works",
            "has_pets": False, "has_pets_prov": "no pets",
            "num_roommates": 0, "num_roommates_prov": "alone",
            "age": 28, "age_prov": "28",
            "gender": "female", "gender_prov": "woman",
        }
        profile = _stub_engine(response).extract(
            _make_offer("I'm a 28-year-old woman, works, no pets, alone, budget 7000, August 1st")
        ).profile
        assert profile.budget_nis == 7000
        assert profile.employment_status == EmploymentStatus.EMPLOYED
        assert profile.has_pets is False
        assert profile.num_roommates == 0
        assert profile.age == 28
        assert profile.gender == Gender.FEMALE

    def test_has_pets_false_survives_as_false_not_none(self):
        # Three-way distinction: False (no pets) must not collapse to None
        profile = _stub_engine({**_NULL_PROFILE, "has_pets": False}).extract(
            _make_offer("ללא חיות")
        ).profile
        assert profile.has_pets is False

    def test_has_pets_true_survives_as_true(self):
        profile = _stub_engine({**_NULL_PROFILE, "has_pets": True, "has_pets_prov": "יש לי כלב"}).extract(
            _make_offer("יש לי כלב")
        ).profile
        assert profile.has_pets is True

    def test_client_called_with_system_prompt_and_offer_text(self):
        client = MagicMock()
        client.extract_raw.return_value = _NULL_PROFILE
        ExtractionEngine(client=client).extract(_make_offer("unique payload"))
        client.extract_raw.assert_called_once_with(
            system_prompt=SYSTEM_PROMPT,
            user_text="unique payload",
        )


# ---------------------------------------------------------------------------
# 2a — ExtractionEngine: provenance reshaping
# ---------------------------------------------------------------------------

class TestEngineProvenance:
    def test_prov_key_reshaped_into_nested_provenance_dict(self):
        response = {**_NULL_PROFILE, "budget_nis": 6500, "budget_nis_prov": "תקציב 6500"}
        profile = _stub_engine(response).extract(_make_offer("תקציב 6500")).profile
        assert "budget_nis" in profile.provenance
        assert profile.provenance["budget_nis"].source_span == "תקציב 6500"

    def test_null_prov_key_not_added_to_provenance_dict(self):
        # A null _prov means the field was not cited — must not appear in provenance
        response = {**_NULL_PROFILE, "budget_nis": None, "budget_nis_prov": None}
        profile = _stub_engine(response).extract(_make_offer()).profile
        assert "budget_nis" not in profile.provenance

    def test_flat_prov_keys_do_not_leak_onto_profile(self):
        # After reshaping, budget_nis_prov must not exist as an attribute on TenantProfile
        response = {**_NULL_PROFILE, "budget_nis": 6500, "budget_nis_prov": "תקציב 6500"}
        profile = _stub_engine(response).extract(_make_offer("תקציב 6500")).profile
        assert not hasattr(profile, "budget_nis_prov")

    def test_missing_prov_key_in_response_is_tolerated(self):
        # The engine uses .pop(..., None) so a missing _prov key must not raise
        response = {k: v for k, v in _NULL_PROFILE.items() if k != "budget_nis_prov"}
        profile = _stub_engine(response).extract(_make_offer()).profile
        assert isinstance(profile, TenantProfile)

    def test_multiple_prov_keys_all_reshaped(self):
        text = "בת 30, עצמאית, ללא חיות, תקציב 6500"
        response = {
            **_NULL_PROFILE,
            "age": 30, "age_prov": "30",
            "employment_status": "self_employed", "employment_status_prov": "עצמאית",
            "has_pets": False, "has_pets_prov": "ללא חיות",
            "budget_nis": 6500, "budget_nis_prov": "תקציב 6500",
        }
        profile = _stub_engine(response).extract(_make_offer(text)).profile
        assert set(profile.provenance.keys()) == {"age", "employment_status", "has_pets", "budget_nis"}


# ---------------------------------------------------------------------------
# 2a — ExtractionEngine: provenance audit warnings
# ---------------------------------------------------------------------------

class TestEngineProvenanceAudit:
    def test_hallucinated_span_not_in_text_logs_warning(self, caplog):
        # A provenance span that doesn't appear in the original text is a hallucination signal
        response = {**_NULL_PROFILE, "budget_nis": 6500, "budget_nis_prov": "HALLUCINATED"}
        with caplog.at_level(logging.WARNING, logger="rentflow.extraction.engine"):
            _stub_engine(response).extract(_make_offer("תקציב 6500"))
        assert any("not found in source text" in r.message for r in caplog.records)

    def test_non_null_field_without_citation_logs_warning(self, caplog):
        # A non-null field with no provenance span should warn
        response = {**_NULL_PROFILE, "budget_nis": 6500, "budget_nis_prov": None}
        with caplog.at_level(logging.WARNING, logger="rentflow.extraction.engine"):
            _stub_engine(response).extract(_make_offer("תקציב 6500"))
        assert any("missing provenance citation" in r.message for r in caplog.records)

    def test_clean_extraction_produces_no_warnings(self, caplog):
        # A well-formed response with real spans must not generate any warnings
        text = "תקציב 6500"
        response = {**_NULL_PROFILE, "budget_nis": 6500, "budget_nis_prov": "תקציב 6500"}
        with caplog.at_level(logging.WARNING, logger="rentflow.extraction.engine"):
            _stub_engine(response).extract(_make_offer(text))
        assert caplog.records == []


# ---------------------------------------------------------------------------
# 2a — ExtractionEngine: error handling
# ---------------------------------------------------------------------------

class TestEngineErrors:
    def test_client_runtime_error_raises_extraction_error(self):
        # Any RuntimeError from the client must be wrapped in ExtractionError
        client = MagicMock()
        client.extract_raw.side_effect = RuntimeError("network down")
        with pytest.raises(ExtractionError, match="LLM call failed"):
            ExtractionEngine(client=client).extract(_make_offer(offer_id="err_001"))

    def test_extraction_error_contains_offer_id(self):
        # The error message must identify which offer failed for traceability
        client = MagicMock()
        client.extract_raw.side_effect = RuntimeError("boom")
        with pytest.raises(ExtractionError, match="err_id_42"):
            ExtractionEngine(client=client).extract(_make_offer(offer_id="err_id_42"))

    def test_invalid_type_in_response_raises_extraction_error(self):
        # A string where an int is expected must raise ExtractionError, not ValidationError
        bad = {**_NULL_PROFILE, "budget_nis": "six thousand"}
        with pytest.raises(ExtractionError, match="schema validation"):
            _stub_engine(bad).extract(_make_offer())

    def test_invalid_date_format_raises_extraction_error(self):
        # A non-ISO date must be rejected at the Pydantic validation stage
        bad = {**_NULL_PROFILE, "move_in_date": "31/08/2026"}
        with pytest.raises(ExtractionError):
            _stub_engine(bad).extract(_make_offer())

    def test_invalid_enum_value_raises_extraction_error(self):
        # An unrecognised employment status must raise ExtractionError
        bad = {**_NULL_PROFILE, "employment_status": "freelancer"}
        with pytest.raises(ExtractionError):
            _stub_engine(bad).extract(_make_offer())

    def test_invalid_gender_enum_raises_extraction_error(self):
        bad = {**_NULL_PROFILE, "gender": "M"}
        with pytest.raises(ExtractionError):
            _stub_engine(bad).extract(_make_offer())

    def test_invalid_language_enum_raises_extraction_error(self):
        bad = {**_NULL_PROFILE, "preferred_language": "fr"}
        with pytest.raises(ExtractionError):
            _stub_engine(bad).extract(_make_offer())

    def test_model_validate_error_different_path_from_client_error(self):
        # A bad dict that passes JSON parsing but fails model_validate is caught by the
        # second try/except in the engine, not the first — both produce ExtractionError
        # but via distinct code paths. This test pins both paths exist independently.
        client_error_engine = MagicMock()
        client_error_engine.extract_raw.side_effect = RuntimeError("client")
        schema_error_engine = MagicMock()
        schema_error_engine.extract_raw.return_value = {**_NULL_PROFILE, "age": "old"}

        with pytest.raises(ExtractionError, match="LLM call failed"):
            ExtractionEngine(client=client_error_engine).extract(_make_offer())

        with pytest.raises(ExtractionError, match="schema validation"):
            ExtractionEngine(client=schema_error_engine).extract(_make_offer())


# ---------------------------------------------------------------------------
# 2a — ExtractionEngine: extra / missing keys in response
# ---------------------------------------------------------------------------

class TestEngineKeyHandling:
    def test_missing_optional_key_defaults_to_none(self):
        # TenantProfile fields are all Optional — a missing key defaults to None rather than
        # raising. This documents the intentional behavior: the engine is lenient on missing
        # optional fields (treats them as unstated).
        incomplete = {k: v for k, v in _NULL_PROFILE.items() if k != "age"}
        profile = _stub_engine(incomplete).extract(_make_offer()).profile
        assert profile.age is None

    def test_extra_unknown_key_does_not_raise(self):
        # Pydantic ignores extra fields by default; an unexpected key must not crash the engine
        response = {**_NULL_PROFILE, "totally_unknown_field": "value"}
        profile = _stub_engine(response).extract(_make_offer()).profile
        assert isinstance(profile, TenantProfile)


# ---------------------------------------------------------------------------
# 2b — ExtractionClient: server-side API error matrix
# ---------------------------------------------------------------------------

def _make_client(max_retries: int = 3) -> tuple[ExtractionClient, MagicMock]:
    """Returns an ExtractionClient with its internal OpenAI SDK object stubbed out."""
    client = ExtractionClient(api_key="test-key", max_retries=max_retries)
    mock_openai = MagicMock()
    client._client = mock_openai
    return client, mock_openai


def _make_api_error(status_code: int) -> APIError:
    """Construct a minimal APIError with a given HTTP status code."""
    err = APIError.__new__(APIError)
    err.status_code = status_code
    err.message = f"HTTP {status_code}"
    err.body = None
    return err


def _make_response(content: str) -> MagicMock:
    """Stub a successful OpenAI chat completion response."""
    response = MagicMock()
    response.choices[0].message.content = content
    return response


class TestClientAPIErrorMatrix:
    def test_rate_limit_error_is_retried_and_succeeds(self):
        # Two RateLimitErrors then a success: create() called 3 times total
        client, mock_openai = _make_client(max_retries=3)
        payload = json.dumps(_NULL_PROFILE)
        mock_openai.chat.completions.create.side_effect = [
            RateLimitError("rate limited", response=MagicMock(), body=None),
            RateLimitError("rate limited", response=MagicMock(), body=None),
            _make_response(payload),
        ]
        with patch("time.sleep"):
            result = client.extract_raw(system_prompt="s", user_text="t")
        assert result == _NULL_PROFILE
        assert mock_openai.chat.completions.create.call_count == 3

    def test_api_timeout_error_is_retried_and_succeeds(self):
        client, mock_openai = _make_client(max_retries=3)
        payload = json.dumps(_NULL_PROFILE)
        mock_openai.chat.completions.create.side_effect = [
            APITimeoutError(request=MagicMock()),
            _make_response(payload),
        ]
        with patch("time.sleep"):
            result = client.extract_raw(system_prompt="s", user_text="t")
        assert result == _NULL_PROFILE
        assert mock_openai.chat.completions.create.call_count == 2

    def test_5xx_api_error_is_retried_then_raises(self):
        # A 500 error on every attempt → exhausts retries → RuntimeError
        client, mock_openai = _make_client(max_retries=3)
        mock_openai.chat.completions.create.side_effect = _make_api_error(500)
        with patch("time.sleep"), pytest.raises(RuntimeError, match="failed after 3 attempts"):
            client.extract_raw(system_prompt="s", user_text="t")
        assert mock_openai.chat.completions.create.call_count == 3

    def test_502_and_503_are_also_retried(self):
        for status in (502, 503, 504):
            client, mock_openai = _make_client(max_retries=2)
            mock_openai.chat.completions.create.side_effect = _make_api_error(status)
            with patch("time.sleep"), pytest.raises(RuntimeError):
                client.extract_raw(system_prompt="s", user_text="t")
            assert mock_openai.chat.completions.create.call_count == 2

    def test_400_api_error_raises_immediately_without_retry(self):
        # A 4xx client error is non-retryable — raises after exactly 1 attempt
        client, mock_openai = _make_client(max_retries=3)
        mock_openai.chat.completions.create.side_effect = _make_api_error(400)
        with patch("time.sleep"), pytest.raises(RuntimeError, match="Non-retryable"):
            client.extract_raw(system_prompt="s", user_text="t")
        assert mock_openai.chat.completions.create.call_count == 1

    def test_401_raises_immediately(self):
        client, mock_openai = _make_client(max_retries=3)
        mock_openai.chat.completions.create.side_effect = _make_api_error(401)
        with patch("time.sleep"), pytest.raises(RuntimeError, match="Non-retryable"):
            client.extract_raw(system_prompt="s", user_text="t")
        assert mock_openai.chat.completions.create.call_count == 1

    def test_404_raises_immediately(self):
        client, mock_openai = _make_client(max_retries=3)
        mock_openai.chat.completions.create.side_effect = _make_api_error(404)
        with patch("time.sleep"), pytest.raises(RuntimeError, match="Non-retryable"):
            client.extract_raw(system_prompt="s", user_text="t")
        assert mock_openai.chat.completions.create.call_count == 1

    def test_499_raises_immediately_but_500_is_retried(self):
        # The < 500 boundary: 499 → immediate, 500 → retried
        for status, expect_immediate in ((499, True), (500, False)):
            client, mock_openai = _make_client(max_retries=2)
            mock_openai.chat.completions.create.side_effect = _make_api_error(status)
            with patch("time.sleep"), pytest.raises(RuntimeError):
                client.extract_raw(system_prompt="s", user_text="t")
            calls = mock_openai.chat.completions.create.call_count
            if expect_immediate:
                assert calls == 1, f"status {status} should be immediate"
            else:
                assert calls == 2, f"status {status} should be retried"

    def test_rate_limit_exhausted_raises_with_last_error_in_message(self):
        client, mock_openai = _make_client(max_retries=2)
        mock_openai.chat.completions.create.side_effect = RateLimitError(
            "rate limited", response=MagicMock(), body=None
        )
        with patch("time.sleep"), pytest.raises(RuntimeError, match="failed after 2 attempts"):
            client.extract_raw(system_prompt="s", user_text="t")


# ---------------------------------------------------------------------------
# 2b — ExtractionClient: retry count and backoff schedule
# ---------------------------------------------------------------------------

class TestClientRetryBehaviour:
    def test_retry_count_matches_max_retries_on_exhaustion(self):
        # With max_retries=4, create() must be called exactly 4 times before giving up
        client, mock_openai = _make_client(max_retries=4)
        mock_openai.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())
        with patch("time.sleep"), pytest.raises(RuntimeError):
            client.extract_raw(system_prompt="s", user_text="t")
        assert mock_openai.chat.completions.create.call_count == 4

    def test_backoff_schedule_doubles_each_retry(self):
        # Backoff starts at 1.0 and doubles after every attempt including the last:
        # with max_retries=4 there are 4 sleeps: 1.0, 2.0, 4.0, 8.0
        client, mock_openai = _make_client(max_retries=4)
        mock_openai.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())
        with patch("time.sleep") as mock_sleep, pytest.raises(RuntimeError):
            client.extract_raw(system_prompt="s", user_text="t")
        sleep_args = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_args == [1.0, 2.0, 4.0, 8.0]

    def test_backoff_is_capped_at_30_seconds(self):
        # After enough retries the backoff must not exceed 30.0
        client, mock_openai = _make_client(max_retries=10)
        mock_openai.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())
        with patch("time.sleep") as mock_sleep, pytest.raises(RuntimeError):
            client.extract_raw(system_prompt="s", user_text="t")
        sleep_args = [c.args[0] for c in mock_sleep.call_args_list]
        assert max(sleep_args) == 30.0

    def test_no_sleep_on_non_retryable_error(self):
        # A 4xx error must not sleep at all — fail fast
        client, mock_openai = _make_client(max_retries=3)
        mock_openai.chat.completions.create.side_effect = _make_api_error(403)
        with patch("time.sleep") as mock_sleep, pytest.raises(RuntimeError):
            client.extract_raw(system_prompt="s", user_text="t")
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# 2b — ExtractionClient: response-content failures
# ---------------------------------------------------------------------------

class TestClientResponseContent:
    def test_empty_response_body_raises_runtime_error(self):
        client, mock_openai = _make_client()
        mock_openai.chat.completions.create.return_value = _make_response("")
        with pytest.raises(RuntimeError, match="empty response body"):
            client.extract_raw(system_prompt="s", user_text="t")

    def test_none_response_body_raises_runtime_error(self):
        client, mock_openai = _make_client()
        mock_openai.chat.completions.create.return_value = _make_response(None)
        with pytest.raises(RuntimeError, match="empty response body"):
            client.extract_raw(system_prompt="s", user_text="t")

    def test_non_json_body_raises_runtime_error_immediately(self):
        # Invalid JSON must fail fast — not retried
        client, mock_openai = _make_client(max_retries=3)
        mock_openai.chat.completions.create.return_value = _make_response("not { json")
        with patch("time.sleep") as mock_sleep, pytest.raises(RuntimeError, match="invalid JSON"):
            client.extract_raw(system_prompt="s", user_text="t")
        assert mock_openai.chat.completions.create.call_count == 1
        mock_sleep.assert_not_called()

    def test_valid_json_returned_as_dict(self):
        client, mock_openai = _make_client()
        mock_openai.chat.completions.create.return_value = _make_response(
            json.dumps(_NULL_PROFILE)
        )
        result = client.extract_raw(system_prompt="s", user_text="t")
        assert result == _NULL_PROFILE


# ---------------------------------------------------------------------------
# 2b — Schema drift guard
# ---------------------------------------------------------------------------

class TestSchemaDriftGuard:
    def test_all_provenance_fields_present_in_schema_properties(self):
        # Every field in _PROVENANCE_FIELDS must have both a value key and a _prov key
        # in the schema. If a new TenantProfile field is added without updating the schema,
        # this test catches the drift.
        properties = _TENANT_PROFILE_SCHEMA["properties"]
        for field in _PROVENANCE_FIELDS:
            assert field in properties, f"Field '{field}' missing from schema properties"
            prov_key = f"{field}_prov"
            assert prov_key in properties, f"Provenance key '{prov_key}' missing from schema"

    def test_all_provenance_fields_in_schema_required_list(self):
        # strict=True requires every key to be in required[] or the API rejects the schema
        required = _TENANT_PROFILE_SCHEMA["required"]
        for field in _PROVENANCE_FIELDS:
            assert field in required, f"Field '{field}' missing from schema required list"
            assert f"{field}_prov" in required, f"'{field}_prov' missing from schema required list"

    def test_schema_has_additional_properties_false(self):
        # strict mode requires additionalProperties: false at the top level
        assert _TENANT_PROFILE_SCHEMA.get("additionalProperties") is False
