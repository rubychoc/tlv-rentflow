"""
Deterministic tests for Station 2 — extraction engine and client (no real OpenAI calls).

Split into two sections:

  2a — ExtractionEngine (engine.py)
       The engine receives a raw dict from the client, reshapes provenance keys,
       validates into TenantGroup, and raises ExtractionError on any failure.
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
from unittest.mock import MagicMock, patch

import pytest
from openai import APIError, APITimeoutError, RateLimitError

from rentflow.extraction.client import (
    ExtractionClient,
    _GROUP_PROV_FIELDS,
    _PERSON_PROV_FIELDS,
    _TENANT_GROUP_SCHEMA,
)
from rentflow.extraction.engine import ExtractionEngine, ExtractionError
from rentflow.extraction.prompts import SYSTEM_PROMPT
from rentflow.offer.models import (
    Channel,
    EmploymentStatus,
    Gender,
    RawOffer,
    TenantGroup,
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


def _null_person() -> dict:
    """Minimal valid applicant dict: all fields null."""
    return {
        "employment_status": None, "age": None, "gender": None,
        "name": None, "phone": None,
        **{f"{f}_prov": None for f in _PERSON_PROV_FIELDS},
    }


# Minimal valid LLM response: group-level fields all null, one null applicant.
_NULL_GROUP: dict = {
    "budget_nis": None, "move_in_date": None,
    "has_pets": None, "household_size": None,
    "preferred_language": None,
    "applicants": [_null_person()],
    **{f"{f}_prov": None for f in _GROUP_PROV_FIELDS},
}

# Minimal valid non-application response: empty applicants list.
_NON_APP_GROUP: dict = {
    **{f: None for f in ("budget_nis", "move_in_date", "has_pets", "household_size", "preferred_language")},
    "applicants": [],
    **{f"{f}_prov": None for f in _GROUP_PROV_FIELDS},
}


# ---------------------------------------------------------------------------
# 2a — ExtractionEngine: happy path
# ---------------------------------------------------------------------------

class TestEngineHappyPath:
    def test_valid_response_returns_extraction_result(self):
        result = _stub_engine(_NULL_GROUP).extract(_make_offer())
        assert isinstance(result.group, TenantGroup)

    def test_result_carries_the_original_offer(self, raw_offer):
        result = _stub_engine(_NULL_GROUP).extract(raw_offer)
        assert result.offer is raw_offer

    def test_all_null_group_fields_stay_none(self):
        group = _stub_engine(_NULL_GROUP).extract(_make_offer()).group
        for field in ("budget_nis", "move_in_date", "has_pets", "household_size"):
            assert getattr(group, field) is None

    def test_non_application_produces_empty_applicants_list(self):
        group = _stub_engine(_NON_APP_GROUP).extract(_make_offer()).group
        assert group.applicants == []

    def test_populated_group_fields_land_correctly(self):
        response = {
            **_NULL_GROUP,
            "budget_nis": 7000, "budget_nis_prov": "7000",
            "move_in_date": "2026-08-01", "move_in_date_prov": "August 1st",
            "has_pets": False, "has_pets_prov": "no pets",
            "household_size": 1, "household_size_prov": "alone",
        }
        group = _stub_engine(response).extract(
            _make_offer("I'm alone, no pets, budget 7000, August 1st")
        ).group
        assert group.budget_nis == 7000
        assert group.has_pets is False
        assert group.household_size == 1

    def test_populated_person_fields_land_correctly(self):
        person = {
            **_null_person(),
            "employment_status": "employed", "employment_status_prov": "works",
            "age": 28, "age_prov": "28",
            "gender": "female", "gender_prov": "woman",
        }
        response = {**_NULL_GROUP, "applicants": [person]}
        group = _stub_engine(response).extract(
            _make_offer("woman, 28, works")
        ).group
        assert len(group.applicants) == 1
        p = group.applicants[0]
        assert p.employment_status == EmploymentStatus.EMPLOYED
        assert p.age == 28
        assert p.gender == Gender.FEMALE

    def test_has_pets_false_survives_as_false_not_none(self):
        response = {**_NULL_GROUP, "has_pets": False}
        group = _stub_engine(response).extract(_make_offer("ללא חיות")).group
        assert group.has_pets is False

    def test_has_pets_true_survives_as_true(self):
        response = {**_NULL_GROUP, "has_pets": True, "has_pets_prov": "יש לי כלב"}
        group = _stub_engine(response).extract(_make_offer("יש לי כלב")).group
        assert group.has_pets is True

    def test_client_called_with_system_prompt_and_offer_text(self):
        client = MagicMock()
        client.extract_raw.return_value = _NULL_GROUP
        ExtractionEngine(client=client).extract(_make_offer("unique payload"))
        client.extract_raw.assert_called_once_with(
            system_prompt=SYSTEM_PROMPT,
            user_text="unique payload",
        )

    def test_multiple_applicants_all_captured(self):
        person_a = {**_null_person(), "employment_status": "employed", "employment_status_prov": "I work", "age": 31, "age_prov": "31"}
        person_b = {**_null_person(), "employment_status": "self_employed", "employment_status_prov": "she freelances"}
        response = {**_NULL_GROUP, "household_size": 2, "applicants": [person_a, person_b]}
        group = _stub_engine(response).extract(_make_offer("I work, I'm 31. She freelances.")).group
        assert len(group.applicants) == 2
        assert group.applicants[0].age == 31
        assert group.applicants[1].employment_status == EmploymentStatus.SELF_EMPLOYED


# ---------------------------------------------------------------------------
# 2a — ExtractionEngine: provenance reshaping
# ---------------------------------------------------------------------------

class TestEngineProvenance:
    def test_group_prov_key_reshaped_into_nested_provenance_dict(self):
        response = {**_NULL_GROUP, "budget_nis": 6500, "budget_nis_prov": "תקציב 6500"}
        group = _stub_engine(response).extract(_make_offer("תקציב 6500")).group
        assert "budget_nis" in group.provenance
        assert group.provenance["budget_nis"].source_span == "תקציב 6500"

    def test_person_prov_key_reshaped_into_applicant_provenance(self):
        person = {**_null_person(), "age": 28, "age_prov": "age 28"}
        response = {**_NULL_GROUP, "applicants": [person]}
        group = _stub_engine(response).extract(_make_offer("age 28")).group
        assert "age" in group.applicants[0].provenance
        assert group.applicants[0].provenance["age"].source_span == "age 28"

    def test_null_prov_key_not_added_to_provenance_dict(self):
        response = {**_NULL_GROUP, "budget_nis": None, "budget_nis_prov": None}
        group = _stub_engine(response).extract(_make_offer()).group
        assert "budget_nis" not in group.provenance

    def test_flat_prov_keys_do_not_leak_onto_group(self):
        response = {**_NULL_GROUP, "budget_nis": 6500, "budget_nis_prov": "תקציב 6500"}
        group = _stub_engine(response).extract(_make_offer("תקציב 6500")).group
        assert not hasattr(group, "budget_nis_prov")

    def test_flat_prov_keys_do_not_leak_onto_applicant(self):
        person = {**_null_person(), "age": 28, "age_prov": "28"}
        response = {**_NULL_GROUP, "applicants": [person]}
        group = _stub_engine(response).extract(_make_offer("28")).group
        assert not hasattr(group.applicants[0], "age_prov")

    def test_multiple_group_prov_keys_all_reshaped(self):
        text = "no pets, budget 6500, move August 1"
        response = {
            **_NULL_GROUP,
            "has_pets": False, "has_pets_prov": "no pets",
            "budget_nis": 6500, "budget_nis_prov": "budget 6500",
            "move_in_date": "2026-08-01", "move_in_date_prov": "August 1",
        }
        group = _stub_engine(response).extract(_make_offer(text)).group
        assert set(group.provenance.keys()) == {"has_pets", "budget_nis", "move_in_date"}


# ---------------------------------------------------------------------------
# 2a — ExtractionEngine: input truncation
# ---------------------------------------------------------------------------

class TestEngineInputTruncation:
    def test_text_within_limit_is_passed_unchanged(self):
        client = MagicMock()
        client.extract_raw.return_value = _NULL_GROUP
        text = "x" * ExtractionEngine.MAX_INPUT_CHARS
        ExtractionEngine(client=client).extract(_make_offer(text))
        assert client.extract_raw.call_args.kwargs["user_text"] == text

    def test_text_over_limit_is_truncated(self):
        client = MagicMock()
        client.extract_raw.return_value = _NULL_GROUP
        text = "x" * (ExtractionEngine.MAX_INPUT_CHARS + 500)
        ExtractionEngine(client=client).extract(_make_offer(text))
        sent = client.extract_raw.call_args.kwargs["user_text"]
        assert len(sent) == ExtractionEngine.MAX_INPUT_CHARS

    def test_truncation_logs_warning(self, caplog):
        client = MagicMock()
        client.extract_raw.return_value = _NULL_GROUP
        text = "y" * (ExtractionEngine.MAX_INPUT_CHARS + 1)
        with caplog.at_level(logging.WARNING, logger="rentflow.extraction.engine"):
            ExtractionEngine(client=client).extract(_make_offer(text))
        assert any("truncated" in r.message for r in caplog.records)

    def test_text_exactly_at_limit_is_not_truncated(self):
        client = MagicMock()
        client.extract_raw.return_value = _NULL_GROUP
        text = "z" * ExtractionEngine.MAX_INPUT_CHARS
        ExtractionEngine(client=client).extract(_make_offer(text))
        sent = client.extract_raw.call_args.kwargs["user_text"]
        assert len(sent) == ExtractionEngine.MAX_INPUT_CHARS


# ---------------------------------------------------------------------------
# 2a — ExtractionEngine: provenance audit warnings
# ---------------------------------------------------------------------------

class TestEngineProvenanceAudit:
    def test_hallucinated_group_span_logs_warning(self, caplog):
        response = {**_NULL_GROUP, "budget_nis": 6500, "budget_nis_prov": "HALLUCINATED"}
        with caplog.at_level(logging.WARNING, logger="rentflow.extraction.engine"):
            _stub_engine(response).extract(_make_offer("תקציב 6500"))
        assert any("not found in source text" in r.message for r in caplog.records)

    def test_hallucinated_person_span_logs_warning(self, caplog):
        person = {**_null_person(), "age": 28, "age_prov": "FAKE SPAN"}
        response = {**_NULL_GROUP, "applicants": [person]}
        with caplog.at_level(logging.WARNING, logger="rentflow.extraction.engine"):
            _stub_engine(response).extract(_make_offer("age 28"))
        assert any("not found in source text" in r.message for r in caplog.records)

    def test_non_null_group_field_without_citation_logs_warning(self, caplog):
        response = {**_NULL_GROUP, "budget_nis": 6500, "budget_nis_prov": None}
        with caplog.at_level(logging.WARNING, logger="rentflow.extraction.engine"):
            _stub_engine(response).extract(_make_offer("תקציב 6500"))
        assert any("missing provenance" in r.message for r in caplog.records)

    def test_clean_extraction_produces_no_warnings(self, caplog):
        text = "תקציב 6500"
        response = {**_NULL_GROUP, "budget_nis": 6500, "budget_nis_prov": "תקציב 6500"}
        with caplog.at_level(logging.WARNING, logger="rentflow.extraction.engine"):
            _stub_engine(response).extract(_make_offer(text))
        assert caplog.records == []


# ---------------------------------------------------------------------------
# 2a — ExtractionEngine: error handling
# ---------------------------------------------------------------------------

class TestEngineErrors:
    def test_client_runtime_error_raises_extraction_error(self):
        client = MagicMock()
        client.extract_raw.side_effect = RuntimeError("network down")
        with pytest.raises(ExtractionError, match="LLM call failed"):
            ExtractionEngine(client=client).extract(_make_offer(offer_id="err_001"))

    def test_extraction_error_contains_offer_id(self):
        client = MagicMock()
        client.extract_raw.side_effect = RuntimeError("boom")
        with pytest.raises(ExtractionError, match="err_id_42"):
            ExtractionEngine(client=client).extract(_make_offer(offer_id="err_id_42"))

    def test_invalid_type_in_group_raises_extraction_error(self):
        bad = {**_NULL_GROUP, "budget_nis": "six thousand"}
        with pytest.raises(ExtractionError, match="schema validation"):
            _stub_engine(bad).extract(_make_offer())

    def test_invalid_date_format_raises_extraction_error(self):
        bad = {**_NULL_GROUP, "move_in_date": "31/08/2026"}
        with pytest.raises(ExtractionError):
            _stub_engine(bad).extract(_make_offer())

    def test_invalid_enum_value_in_person_raises_extraction_error(self):
        person = {**_null_person(), "employment_status": "freelancer"}
        bad = {**_NULL_GROUP, "applicants": [person]}
        with pytest.raises(ExtractionError):
            _stub_engine(bad).extract(_make_offer())

    def test_invalid_gender_enum_raises_extraction_error(self):
        person = {**_null_person(), "gender": "M"}
        bad = {**_NULL_GROUP, "applicants": [person]}
        with pytest.raises(ExtractionError):
            _stub_engine(bad).extract(_make_offer())

    def test_invalid_language_enum_raises_extraction_error(self):
        bad = {**_NULL_GROUP, "preferred_language": "fr"}
        with pytest.raises(ExtractionError):
            _stub_engine(bad).extract(_make_offer())


# ---------------------------------------------------------------------------
# 2b — ExtractionClient: server-side API error matrix
# ---------------------------------------------------------------------------

def _make_client(max_retries: int = 3) -> tuple[ExtractionClient, MagicMock]:
    client = ExtractionClient(api_key="test-key", max_retries=max_retries)
    mock_openai = MagicMock()
    client._client = mock_openai
    return client, mock_openai


def _make_api_error(status_code: int) -> APIError:
    err = APIError.__new__(APIError)
    err.status_code = status_code
    err.message = f"HTTP {status_code}"
    err.body = None
    return err


def _make_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    return response


class TestClientAPIErrorMatrix:
    def test_rate_limit_error_is_retried_and_succeeds(self):
        client, mock_openai = _make_client(max_retries=3)
        payload = json.dumps(_NULL_GROUP)
        mock_openai.chat.completions.create.side_effect = [
            RateLimitError("rate limited", response=MagicMock(), body=None),
            RateLimitError("rate limited", response=MagicMock(), body=None),
            _make_response(payload),
        ]
        with patch("time.sleep"):
            result = client.extract_raw(system_prompt="s", user_text="t")
        assert result == _NULL_GROUP
        assert mock_openai.chat.completions.create.call_count == 3

    def test_api_timeout_error_is_retried_and_succeeds(self):
        client, mock_openai = _make_client(max_retries=3)
        payload = json.dumps(_NULL_GROUP)
        mock_openai.chat.completions.create.side_effect = [
            APITimeoutError(request=MagicMock()),
            _make_response(payload),
        ]
        with patch("time.sleep"):
            result = client.extract_raw(system_prompt="s", user_text="t")
        assert result == _NULL_GROUP
        assert mock_openai.chat.completions.create.call_count == 2

    def test_5xx_api_error_is_retried_then_raises(self):
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
        client, mock_openai = _make_client(max_retries=4)
        mock_openai.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())
        with patch("time.sleep"), pytest.raises(RuntimeError):
            client.extract_raw(system_prompt="s", user_text="t")
        assert mock_openai.chat.completions.create.call_count == 4

    def test_backoff_schedule_doubles_each_retry(self):
        client, mock_openai = _make_client(max_retries=4)
        mock_openai.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())
        with patch("time.sleep") as mock_sleep, pytest.raises(RuntimeError):
            client.extract_raw(system_prompt="s", user_text="t")
        sleep_args = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_args == [1.0, 2.0, 4.0, 8.0]

    def test_backoff_is_capped_at_30_seconds(self):
        client, mock_openai = _make_client(max_retries=10)
        mock_openai.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())
        with patch("time.sleep") as mock_sleep, pytest.raises(RuntimeError):
            client.extract_raw(system_prompt="s", user_text="t")
        sleep_args = [c.args[0] for c in mock_sleep.call_args_list]
        assert max(sleep_args) == 30.0

    def test_no_sleep_on_non_retryable_error(self):
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
        client, mock_openai = _make_client(max_retries=3)
        mock_openai.chat.completions.create.return_value = _make_response("not { json")
        with patch("time.sleep") as mock_sleep, pytest.raises(RuntimeError, match="invalid JSON"):
            client.extract_raw(system_prompt="s", user_text="t")
        assert mock_openai.chat.completions.create.call_count == 1
        mock_sleep.assert_not_called()

    def test_valid_json_returned_as_dict(self):
        client, mock_openai = _make_client()
        mock_openai.chat.completions.create.return_value = _make_response(
            json.dumps(_NULL_GROUP)
        )
        result = client.extract_raw(system_prompt="s", user_text="t")
        assert result == _NULL_GROUP


# ---------------------------------------------------------------------------
# 2b — Schema drift guard
# ---------------------------------------------------------------------------

class TestSchemaDriftGuard:
    def test_all_group_provenance_fields_present_in_schema(self):
        properties = _TENANT_GROUP_SCHEMA["properties"]
        for field in _GROUP_PROV_FIELDS:
            assert field in properties, f"Field '{field}' missing from schema properties"
            assert f"{field}_prov" in properties, f"'{field}_prov' missing from schema"

    def test_all_person_provenance_fields_present_in_person_schema(self):
        person_props = _TENANT_GROUP_SCHEMA["properties"]["applicants"]["items"]["properties"]
        for field in _PERSON_PROV_FIELDS:
            assert field in person_props, f"Person field '{field}' missing"
            assert f"{field}_prov" in person_props, f"'{field}_prov' missing from person schema"

    def test_all_group_fields_in_required_list(self):
        required = _TENANT_GROUP_SCHEMA["required"]
        for field in _GROUP_PROV_FIELDS:
            assert field in required
            assert f"{field}_prov" in required

    def test_schema_has_additional_properties_false(self):
        assert _TENANT_GROUP_SCHEMA.get("additionalProperties") is False

    def test_person_schema_has_additional_properties_false(self):
        person_schema = _TENANT_GROUP_SCHEMA["properties"]["applicants"]["items"]
        assert person_schema.get("additionalProperties") is False
