"""
Tests for the FastAPI ingestion API (ingestion/app.py) — Stations 0 and 1, HTTP layer.

Each test uses a fresh InMemoryOfferStore injected via the app module's _store attribute,
so tests are fully isolated from each other.

Tests are grouped into:
  - Listing endpoints (POST /listing, GET /listing) — Station 0
  - Webhook behaviour (POST /webhook/{channel}) — Station 1
  - Offer retrieval endpoints (GET /offers, GET /offers/{id})
  - Health check (GET /healthz)
"""

import pytest
from fastapi.testclient import TestClient

from rentflow.ingestion import app as app_module
from rentflow.ingestion.app import app
from rentflow.ingestion.store import InMemoryOfferStore

LISTING = {
    "address": "Dizengoff 10, Tel Aviv",
    "rent_nis": 7000,
}

LISTING_STRICT = {
    "address": "Rothschild 1, Tel Aviv",
    "rent_nis": 7500,
    "criteria": {
        "lowest_price_nis": 6000,
        "pets_allowed": False,
        "employment_required": True,
        "budget_weight": 10,
        "pets_weight": 25,
        "employment_weight": 15,
    },
}

OFFER = {"sender": "+972541234567", "text": "מעוניין בדירה, נכנס מיידי."}


@pytest.fixture
def client():
    """Swap in a fresh store for every test; restore afterward."""
    original = app_module._store
    app_module._store = InMemoryOfferStore()
    with TestClient(app) as c:
        yield c
    app_module._store = original


def _post_listing(client, payload=None):
    return client.post("/listing", json=payload or LISTING)


def _post_offer(client, channel="whatsapp", payload=None):
    return client.post(f"/webhook/{channel}", json=payload or OFFER)


# ---------------------------------------------------------------------------
# Station 0 — Listing endpoints
# ---------------------------------------------------------------------------

class TestListingEndpoint:
    def test_post_listing_returns_201_with_id_and_address(self, client):
        r = _post_listing(client)
        assert r.status_code == 201
        body = r.json()
        assert body["status"] == "created"
        assert "listing_id" in body
        assert body["address"] == LISTING["address"]
        assert body["rent_nis"] == LISTING["rent_nis"]

    def test_explicit_listing_id_is_echoed(self, client):
        r = client.post("/listing", json={**LISTING, "listing_id": "my_id"})
        assert r.json()["listing_id"] == "my_id"

    def test_auto_generated_listing_id_has_expected_prefix(self, client):
        r = _post_listing(client)
        assert r.json()["listing_id"].startswith("listing_")

    def test_get_listing_returns_full_listing_including_criteria(self, client):
        _post_listing(client, LISTING_STRICT)
        r = client.get("/listing")
        assert r.status_code == 200
        body = r.json()
        assert body["address"] == LISTING_STRICT["address"]
        assert "criteria" in body

    def test_get_listing_before_any_post_returns_404(self, client):
        assert client.get("/listing").status_code == 404

    def test_second_listing_replaces_first(self, client):
        _post_listing(client, {**LISTING, "address": "First St"})
        _post_listing(client, {**LISTING, "address": "Second St"})
        assert client.get("/listing").json()["address"] == "Second St"

    def test_second_listing_clears_all_prior_offers(self, client):
        _post_listing(client)
        _post_offer(client)
        _post_offer(client)
        assert len(client.get("/offers").json()) == 2
        _post_listing(client)
        assert client.get("/offers").json() == []

    def test_second_listing_clears_pipeline_results(self, client):
        # Pipeline results dict must be wiped when a new listing is posted
        _post_listing(client)
        _post_listing(client)
        assert client.get("/pipeline/results").json() == []

    def test_missing_address_returns_422(self, client):
        assert client.post("/listing", json={"rent_nis": 7000}).status_code == 422

    def test_missing_rent_nis_returns_422(self, client):
        assert client.post("/listing", json={"address": "x"}).status_code == 422

    def test_invalid_weight_over_100_returns_422(self, client):
        payload = {**LISTING, "criteria": {"budget_weight": 101}}
        assert client.post("/listing", json=payload).status_code == 422

    def test_invalid_weight_negative_returns_422(self, client):
        payload = {**LISTING, "criteria": {"pets_weight": -1}}
        assert client.post("/listing", json=payload).status_code == 422

    def test_floor_above_asking_returns_422(self, client):
        # lowest_price_nis must be <= rent_nis by definition
        payload = {**LISTING, "criteria": {"lowest_price_nis": LISTING["rent_nis"] + 1}}
        assert client.post("/listing", json=payload).status_code == 422


# ---------------------------------------------------------------------------
# Station 1 — Webhook
# ---------------------------------------------------------------------------

class TestWebhookGate:
    def test_offer_without_listing_returns_409(self, client):
        r = _post_offer(client)
        assert r.status_code == 409
        assert "listing" in r.json()["detail"].lower()

    def test_offer_with_listing_returns_202(self, client):
        _post_listing(client)
        r = _post_offer(client)
        assert r.status_code == 202
        assert r.json()["status"] == "accepted"

    def test_response_includes_channel_listing_id_and_received_at(self, client):
        _post_listing(client)
        r = _post_offer(client, channel="facebook")
        body = r.json()
        assert body["channel"] == "facebook"
        assert "listing_id" in body
        assert "received_at" in body


class TestWebhookChannelRouting:
    def test_channel_comes_from_url_path_not_body(self, client):
        # The same request body posted to different channel paths must produce different channels
        _post_listing(client)
        r_wa = _post_offer(client, channel="whatsapp")
        r_fb = _post_offer(client, channel="facebook")
        assert r_wa.json()["channel"] == "whatsapp"
        assert r_fb.json()["channel"] == "facebook"

    def test_all_three_valid_channels_accepted(self, client):
        _post_listing(client)
        for channel in ("whatsapp", "facebook", "yad2"):
            r = _post_offer(client, channel=channel)
            assert r.status_code == 202, f"Channel {channel!r} was rejected"

    def test_unknown_channel_returns_422(self, client):
        _post_listing(client)
        assert _post_offer(client, channel="telegram").status_code == 422


class TestWebhookOfferFields:
    def test_auto_generated_offer_id_has_channel_prefix(self, client):
        _post_listing(client)
        r = _post_offer(client, channel="yad2")
        assert r.json()["offer_id"].startswith("yad2_")

    def test_explicit_offer_id_is_preserved(self, client):
        _post_listing(client)
        r = client.post("/webhook/whatsapp", json={**OFFER, "offer_id": "my_offer_99"})
        assert r.json()["offer_id"] == "my_offer_99"

    def test_omitted_timestamp_is_server_stamped(self, client):
        # No timestamp in the body → server fills it in; received_at must be present
        _post_listing(client)
        r = _post_offer(client)
        assert r.json()["received_at"] is not None

    def test_explicit_timestamp_is_preserved(self, client):
        _post_listing(client)
        ts = "2026-06-01T10:00:00+00:00"
        r = client.post("/webhook/whatsapp", json={**OFFER, "timestamp": ts})
        assert r.status_code == 202
        # received_at in response should reflect the submitted timestamp
        assert "2026-06-01" in r.json()["received_at"]

    def test_missing_sender_returns_422(self, client):
        _post_listing(client)
        assert client.post("/webhook/whatsapp", json={"text": "hi"}).status_code == 422

    def test_missing_text_returns_422(self, client):
        _post_listing(client)
        assert client.post("/webhook/whatsapp", json={"sender": "+972001"}).status_code == 422

    def test_stored_offer_has_correct_channel_from_url(self, client):
        # End-to-end: post to /facebook, retrieve via /offers, assert channel on stored object
        _post_listing(client)
        r = client.post("/webhook/facebook", json={**OFFER, "offer_id": "ch_test"})
        assert r.status_code == 202
        stored = client.get("/offers/ch_test").json()
        assert stored["channel"] == "facebook"


# ---------------------------------------------------------------------------
# Offer retrieval endpoints
# ---------------------------------------------------------------------------

class TestOfferRetrieval:
    def test_get_offers_empty_before_any_webhook(self, client):
        _post_listing(client)
        assert client.get("/offers").json() == []

    def test_get_offers_returns_all_accepted(self, client):
        _post_listing(client)
        client.post("/webhook/whatsapp", json={**OFFER, "offer_id": "a"})
        client.post("/webhook/facebook", json={**OFFER, "offer_id": "b"})
        client.post("/webhook/yad2",     json={**OFFER, "offer_id": "c"})
        offers = client.get("/offers").json()
        assert len(offers) == 3

    def test_get_offers_each_has_correct_channel(self, client):
        _post_listing(client)
        client.post("/webhook/whatsapp", json={**OFFER, "offer_id": "wa"})
        client.post("/webhook/facebook", json={**OFFER, "offer_id": "fb"})
        by_id = {o["offer_id"]: o for o in client.get("/offers").json()}
        assert by_id["wa"]["channel"] == "whatsapp"
        assert by_id["fb"]["channel"] == "facebook"

    def test_get_single_offer_by_id_returns_correct_offer(self, client):
        _post_listing(client)
        client.post("/webhook/whatsapp", json={**OFFER, "offer_id": "find_me"})
        r = client.get("/offers/find_me")
        assert r.status_code == 200
        assert r.json()["offer_id"] == "find_me"

    def test_get_nonexistent_offer_returns_404(self, client):
        assert client.get("/offers/ghost").status_code == 404


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_healthz_reports_no_listing_on_fresh_store(self, client):
        r = client.get("/healthz")
        assert r.status_code == 200
        body = r.json()
        assert body["listing_live"] is False
        assert body["listing_id"] is None
        assert body["offers_received"] == 0

    def test_healthz_reports_listing_live_after_post(self, client):
        _post_listing(client)
        body = client.get("/healthz").json()
        assert body["listing_live"] is True
        assert body["listing_id"] is not None

    def test_healthz_offer_count_increments_per_accepted_offer(self, client):
        _post_listing(client)
        _post_offer(client)
        _post_offer(client)
        assert client.get("/healthz").json()["offers_received"] == 2
