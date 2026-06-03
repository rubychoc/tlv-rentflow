"""
Cross-component integration tests — Station 2 mocked, no network calls.

Verifies that the contracts between stations hold when wired together through
the real FastAPI app:

  Listing (Station 0) → Webhook (Station 1) → Pipeline run → Pipeline results

ExtractionEngine.from_env() is patched to return a fake engine so these tests
run without OPENAI_API_KEY. Everything else — HTTP routing, store writes,
scoring math, result caching, and ranking sort — runs for real.

The `_pipeline_results` dict is reset alongside the store on every test so
results from one test can never bleed into the next.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from rentflow.extraction.engine import ExtractionEngine, ExtractionError
from rentflow.ingestion import app as app_module
from rentflow.ingestion.app import app
from rentflow.ingestion.store import InMemoryOfferStore
from rentflow.offer.models import (
    Channel,
    EmploymentStatus,
    Gender,
    RawOffer,
    ScoringCriteria,
    TenantGroup,
    TenantProfile,
)
from rentflow.scoring.vectors import DIM_LABELS, criteria_to_vector, group_to_vector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LISTING = {
    "address": "Rothschild 1, Tel Aviv",
    "rent_nis": 7500,
    "criteria": {
        "lowest_price_nis": 6000,
        "pets_allowed": False,
        "pets_weight": 25,
        "employment_required": True,
        "employment_weight": 15,
        "budget_weight": 10,
        "move_in_weight": 20,
        "occupants_weight": 10,
        "age_weight": 10,
        "gender_weight": 10,
        "approved_threshold": 75,
        "rejected_threshold": 50,
    },
}

OFFER_BODY = {"sender": "+972541234567", "text": "מעוניין בדירה"}


def _null_group() -> TenantGroup:
    return TenantGroup(applicants=[])


def _ideal_group() -> TenantGroup:
    from datetime import date
    return TenantGroup(
        budget_nis=7500,
        has_pets=False,
        move_in_date=date(2026, 8, 31),
        household_size=1,
        applicants=[TenantProfile(
            employment_status=EmploymentStatus.EMPLOYED,
            age=28,
            gender=Gender.FEMALE,
        )],
    )


def _fake_engine(group: TenantGroup) -> MagicMock:
    """Returns a mock ExtractionEngine whose .extract() returns a canned group."""
    engine = MagicMock(spec=ExtractionEngine)
    result = MagicMock()
    result.group = group
    engine.extract.return_value = result
    return engine


def _failing_engine(message: str = "LLM error") -> MagicMock:
    engine = MagicMock(spec=ExtractionEngine)
    engine.extract.side_effect = ExtractionError(message)
    return engine


@pytest.fixture
def client():
    """Fresh store + empty pipeline results for every test."""
    original_store = app_module._store
    original_results = app_module._pipeline_results.copy()
    app_module._store = InMemoryOfferStore()
    app_module._pipeline_results.clear()
    with TestClient(app) as c:
        yield c
    app_module._store = original_store
    app_module._pipeline_results.clear()
    app_module._pipeline_results.update(original_results)


def _post_listing(client, payload=None):
    r = client.post("/listing", json=payload or LISTING)
    assert r.status_code == 201
    return r.json()["listing_id"]


def _post_offer(client, channel="whatsapp", offer_id=None, text=None):
    body = {**OFFER_BODY}
    if offer_id:
        body["offer_id"] = offer_id
    if text:
        body["text"] = text
    r = client.post(f"/webhook/{channel}", json=body)
    assert r.status_code == 202
    return r.json()["offer_id"]


def _run_pipeline(client, offer_id, engine):
    with patch.object(ExtractionEngine, "from_env", return_value=engine):
        return client.post("/pipeline/run", json={"offer_id": offer_id})


# ---------------------------------------------------------------------------
# Listing → webhook gate (Station 0 → Station 1 boundary)
# ---------------------------------------------------------------------------

class TestListingWebhookBoundary:
    def test_webhook_returns_409_before_listing_is_posted(self, client):
        r = client.post("/webhook/whatsapp", json=OFFER_BODY)
        assert r.status_code == 409

    def test_webhook_accepts_offers_immediately_after_listing(self, client):
        _post_listing(client)
        assert client.post("/webhook/whatsapp", json=OFFER_BODY).status_code == 202

    def test_new_listing_closes_then_reopens_gate(self, client):
        _post_listing(client)
        client.post("/webhook/whatsapp", json=OFFER_BODY)
        _post_listing(client)
        assert client.post("/webhook/whatsapp", json=OFFER_BODY).status_code == 202


# ---------------------------------------------------------------------------
# Webhook → store → retrieval (Station 1 round-trip)
# ---------------------------------------------------------------------------

class TestWebhookStoreRetrieval:
    def test_three_offers_across_three_channels_all_stored(self, client):
        _post_listing(client)
        _post_offer(client, "whatsapp", "wa1")
        _post_offer(client, "facebook", "fb1")
        _post_offer(client, "yad2", "y1")
        offers = client.get("/offers").json()
        assert len(offers) == 3

    def test_stored_offer_channel_matches_url_path(self, client):
        _post_listing(client)
        _post_offer(client, "facebook", "fb_ch_test")
        stored = client.get("/offers/fb_ch_test").json()
        assert stored["channel"] == "facebook"

    def test_get_offers_by_id_round_trips_all_three(self, client):
        _post_listing(client)
        for oid in ("id_a", "id_b", "id_c"):
            _post_offer(client, offer_id=oid)
        for oid in ("id_a", "id_b", "id_c"):
            assert client.get(f"/offers/{oid}").status_code == 200


# ---------------------------------------------------------------------------
# Pipeline run — happy path (Station 1 → Station 2 → Station 3)
# ---------------------------------------------------------------------------

class TestPipelineRunSuccess:
    def test_pipeline_run_returns_ok_status(self, client):
        _post_listing(client)
        oid = _post_offer(client, offer_id="p1")
        r = _run_pipeline(client, oid, _fake_engine(_ideal_group()))
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_pipeline_run_caches_group_and_score(self, client):
        _post_listing(client)
        oid = _post_offer(client, offer_id="p2")
        _run_pipeline(client, oid, _fake_engine(_ideal_group()))
        results = client.get("/pipeline/results").json()
        assert len(results) == 1
        assert results[0]["group"] is not None
        assert results[0]["score"] is not None

    def test_pipeline_result_includes_vector_block(self, client):
        _post_listing(client)
        oid = _post_offer(client, offer_id="p3")
        _run_pipeline(client, oid, _fake_engine(_ideal_group()))
        vectors = client.get("/pipeline/results").json()[0]["vectors"]
        assert vectors["dims"] == DIM_LABELS
        assert len(vectors["landlord"]) == len(DIM_LABELS)
        assert len(vectors["tenant"]) == len(DIM_LABELS)

    def test_pipeline_vectors_match_scoring_module_output(self, client):
        _post_listing(client)
        oid = _post_offer(client, offer_id="p4")
        group = _ideal_group()
        _run_pipeline(client, oid, _fake_engine(group))
        result = client.get("/pipeline/results").json()[0]

        criteria = ScoringCriteria(**LISTING["criteria"])
        expected_landlord = [round(v, 4) for v in criteria_to_vector(criteria)]
        expected_tenant = [round(v, 4) for v in group_to_vector(group, criteria, LISTING["rent_nis"])]

        assert result["vectors"]["landlord"] == expected_landlord
        assert result["vectors"]["tenant"] == expected_tenant

    def test_pipeline_run_unknown_offer_id_returns_404(self, client):
        _post_listing(client)
        r = _run_pipeline(client, "ghost_id", _fake_engine(_null_group()))
        assert r.status_code == 404

    def test_pipeline_run_with_known_offer_but_no_listing_returns_409(self, client):
        _post_listing(client)
        _post_offer(client, offer_id="has_offer_no_listing")
        offer = app_module._store.get_offer("has_offer_no_listing")
        fresh_store = InMemoryOfferStore()
        fresh_store._listing = None
        fresh_store._offers = [offer]
        app_module._store = fresh_store
        r = _run_pipeline(client, "has_offer_no_listing", _fake_engine(_null_group()))
        assert r.status_code == 409


# ---------------------------------------------------------------------------
# Pipeline run — extraction failure path
# ---------------------------------------------------------------------------

class TestPipelineRunExtractionError:
    def test_extraction_error_returns_ok_http_but_error_status(self, client):
        _post_listing(client)
        oid = _post_offer(client, offer_id="e1")
        r = _run_pipeline(client, oid, _failing_engine("LLM unavailable"))
        assert r.status_code == 200
        assert r.json()["status"] == "error"
        assert "LLM unavailable" in r.json()["error"]

    def test_extraction_error_cached_with_null_group_and_score(self, client):
        _post_listing(client)
        oid = _post_offer(client, offer_id="e2")
        _run_pipeline(client, oid, _failing_engine())
        results = client.get("/pipeline/results").json()
        assert len(results) == 1
        assert results[0]["group"] is None
        assert results[0]["score"] is None
        assert results[0]["error"] is not None


# ---------------------------------------------------------------------------
# Ranking aggregation (GET /pipeline/results)
# ---------------------------------------------------------------------------

class TestRankingAggregation:
    def test_results_sorted_by_score_descending(self, client):
        _post_listing(client)
        from datetime import date

        high = TenantGroup(
            budget_nis=7500, has_pets=False, move_in_date=date(2026, 8, 31), household_size=1,
            applicants=[TenantProfile(employment_status=EmploymentStatus.EMPLOYED, age=28, gender=Gender.FEMALE)],
        )
        low = TenantGroup(
            budget_nis=6100, has_pets=False, move_in_date=date(2026, 8, 31), household_size=2,
            applicants=[TenantProfile(employment_status=EmploymentStatus.STUDENT, age=45, gender=Gender.MALE)],
        )

        o1 = _post_offer(client, offer_id="rank_low")
        o2 = _post_offer(client, offer_id="rank_high")

        _run_pipeline(client, o1, _fake_engine(low))
        _run_pipeline(client, o2, _fake_engine(high))

        results = client.get("/pipeline/results").json()
        scores = [r["score"]["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_errored_offers_sort_last(self, client):
        _post_listing(client)
        o_ok = _post_offer(client, offer_id="ok_offer")
        o_err = _post_offer(client, offer_id="err_offer")
        _run_pipeline(client, o_ok, _fake_engine(_ideal_group()))
        _run_pipeline(client, o_err, _failing_engine())
        results = client.get("/pipeline/results").json()
        assert results[0]["offer"]["offer_id"] == "ok_offer"
        assert results[-1]["offer"]["offer_id"] == "err_offer"

    def test_tied_scores_broken_by_earlier_received_at(self, client):
        _post_listing(client)
        early_ts = "2026-06-01T08:00:00+00:00"
        late_ts = "2026-06-01T10:00:00+00:00"
        group = _ideal_group()

        r1 = client.post("/webhook/whatsapp",
                         json={"offer_id": "early", "sender": "+1", "text": "x", "timestamp": early_ts})
        r2 = client.post("/webhook/whatsapp",
                         json={"offer_id": "late", "sender": "+2", "text": "x", "timestamp": late_ts})
        assert r1.status_code == 202
        assert r2.status_code == 202

        _run_pipeline(client, "early", _fake_engine(group))
        _run_pipeline(client, "late", _fake_engine(group))

        results = client.get("/pipeline/results").json()
        assert results[0]["offer"]["offer_id"] == "early"

    def test_multiple_runs_same_offer_overwrites_cached_result(self, client):
        _post_listing(client)
        oid = _post_offer(client, offer_id="dup")
        _run_pipeline(client, oid, _fake_engine(_null_group()))
        _run_pipeline(client, oid, _fake_engine(_ideal_group()))
        results = client.get("/pipeline/results").json()
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Lifecycle reset
# ---------------------------------------------------------------------------

class TestLifecycleReset:
    def test_new_listing_clears_pipeline_results(self, client):
        _post_listing(client)
        oid = _post_offer(client, offer_id="before_reset")
        _run_pipeline(client, oid, _fake_engine(_ideal_group()))
        assert len(client.get("/pipeline/results").json()) == 1
        _post_listing(client)
        assert client.get("/pipeline/results").json() == []

    def test_new_listing_clears_offers(self, client):
        _post_listing(client)
        _post_offer(client, offer_id="will_be_gone")
        _post_listing(client)
        assert client.get("/offers").json() == []

    def test_new_listing_clears_both_offers_and_results_atomically(self, client):
        _post_listing(client)
        oid = _post_offer(client, offer_id="both")
        _run_pipeline(client, oid, _fake_engine(_ideal_group()))
        _post_listing(client)
        assert client.get("/offers").json() == []
        assert client.get("/pipeline/results").json() == []
