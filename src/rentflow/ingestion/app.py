"""
Ingestion API — Station 1 of the TLV-RentFlow pipeline.

Endpoints:

  POST /listing
    The landlord posts their apartment listing here first. The payload includes
    apartment details and their screening criteria (ScoringCriteria). Until
    this is called, the webhook rejects all incoming offers with 409.

  GET  /listing
    Returns the currently active listing, or 404 if none is live.

  POST /webhook/{channel}
    Accepts a raw tenant offer. Returns 409 if no listing is live.

  GET  /healthz
    Health check — shows whether a listing is live and how many offers arrived.

  GET  /offers
    Returns all stored offers for the active listing.

  GET  /offers/{offer_id}
    Returns one offer by ID.
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path as FilePath
from typing import Any

from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ValidationError, model_validator

from rentflow.extraction.engine import ExtractionEngine, ExtractionError
from rentflow.ingestion.store import InMemoryOfferStore, OfferStore
from rentflow.offer.models import Channel, Listing, RawOffer, ScoringCriteria
from rentflow.scoring.engine import ScoringEngine
from rentflow.scoring.vectors import DIM_LABELS, criteria_to_vector, profile_to_vector

app = FastAPI(
    title="TLV-RentFlow Ingestion API",
    description="Receives apartment listings and tenant offers.",
    version="0.2.0",
)

_store: OfferStore = InMemoryOfferStore()

# Pipeline results: offer_id -> {offer, profile, score_result}
# Populated by POST /pipeline/run. Cleared when a new listing is posted.
_pipeline_results: dict[str, dict[str, Any]] = {}

_UI_PATH = FilePath(__file__).parent / "ui.html"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ListingPayload(BaseModel):
    """What the landlord POSTs to create a listing."""
    listing_id: str | None = None
    address: str
    description: str | None = None
    rent_nis: int
    criteria: ScoringCriteria = ScoringCriteria()

    @model_validator(mode="after")
    def floor_must_not_exceed_asking(self) -> "ListingPayload":
        floor = self.criteria.lowest_price_nis
        if floor is not None and floor > self.rent_nis:
            raise ValueError(
                f"lowest_price_nis ({floor}) must be ≤ rent_nis ({self.rent_nis})."
            )
        return self


class ListingResponse(BaseModel):
    status: str = "created"
    listing_id: str
    address: str
    rent_nis: int


class IncomingOfferPayload(BaseModel):
    offer_id: str | None = None
    sender: str
    timestamp: datetime | None = None
    text: str


class AcceptedResponse(BaseModel):
    status: str = "accepted"
    offer_id: str
    channel: str
    listing_id: str
    received_at: datetime


class HealthResponse(BaseModel):
    status: str = "ok"
    listing_live: bool
    listing_id: str | None
    offers_received: int


class PipelineRunRequest(BaseModel):
    """Run extraction + scoring for one offer already in the store."""
    offer_id: str


class PipelineRunResponse(BaseModel):
    offer_id: str
    status: str          # "ok" or "error"
    error: str | None = None


# ---------------------------------------------------------------------------
# Listing endpoints
# ---------------------------------------------------------------------------

@app.post("/listing", response_model=ListingResponse, status_code=201)
async def create_listing(payload: ListingPayload) -> ListingResponse:
    """
    The landlord posts their apartment and screening criteria here.
    Replaces any previously active listing and clears all prior offers.
    """
    listing_id = payload.listing_id or f"listing_{uuid.uuid4().hex[:10]}"
    listing = Listing(
        listing_id=listing_id,
        address=payload.address,
        description=payload.description,
        rent_nis=payload.rent_nis,
        created_at=datetime.now(tz=timezone.utc),
        criteria=payload.criteria,
    )
    _store.set_listing(listing)
    _pipeline_results.clear()
    return ListingResponse(
        listing_id=listing_id,
        address=listing.address,
        rent_nis=listing.rent_nis,
    )


@app.get("/listing", response_model=dict)
async def get_listing() -> dict:
    """Returns the currently active listing."""
    listing = _store.get_listing()
    if listing is None:
        raise HTTPException(status_code=404, detail="No active listing.")
    return listing.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Webhook endpoint (gated on active listing)
# ---------------------------------------------------------------------------

@app.post("/webhook/{channel}", response_model=AcceptedResponse, status_code=202)
async def receive_offer(
    payload: IncomingOfferPayload,
    channel: Channel = Path(..., examples=["whatsapp", "facebook", "yad2"]),
) -> AcceptedResponse:
    """
    Accepts a raw tenant offer for the active listing.
    Returns 409 if no listing has been posted yet.
    """
    listing = _store.get_listing()
    if listing is None:
        raise HTTPException(
            status_code=409,
            detail="No active listing. POST to /listing first.",
        )

    offer_id = payload.offer_id or f"{channel.value}_{uuid.uuid4().hex[:12]}"
    received_at = payload.timestamp or datetime.now(tz=timezone.utc)

    try:
        offer = RawOffer(
            offer_id=offer_id,
            channel=channel,
            sender=payload.sender,
            timestamp=received_at,
            text=payload.text,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    _store.save_offer(offer)

    return AcceptedResponse(
        offer_id=offer_id,
        channel=channel.value,
        listing_id=listing.listing_id,
        received_at=received_at,
    )


# ---------------------------------------------------------------------------
# Health + offer inspection
# ---------------------------------------------------------------------------

@app.get("/healthz", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    listing = _store.get_listing()
    return HealthResponse(
        listing_live=listing is not None,
        listing_id=listing.listing_id if listing else None,
        offers_received=len(_store.all_offers()),
    )


@app.get("/offers", response_model=list[dict])
async def list_offers() -> list[dict]:
    return [o.model_dump(mode="json") for o in _store.all_offers()]


@app.get("/offers/{offer_id}", response_model=dict)
async def get_offer(offer_id: str) -> dict:
    offer = _store.get_offer(offer_id)
    if offer is None:
        raise HTTPException(status_code=404, detail=f"Offer '{offer_id}' not found.")
    return offer.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Pipeline: extraction + scoring
# ---------------------------------------------------------------------------

@app.post("/pipeline/run", response_model=PipelineRunResponse)
async def pipeline_run(body: PipelineRunRequest) -> PipelineRunResponse:
    """
    Run extraction + scoring for one stored offer.
    Requires OPENAI_API_KEY in the environment.
    The result is cached in _pipeline_results and returned by GET /pipeline/results.
    """
    offer = _store.get_offer(body.offer_id)
    if offer is None:
        raise HTTPException(status_code=404, detail=f"Offer '{body.offer_id}' not found.")

    listing = _store.get_listing()
    if listing is None:
        raise HTTPException(status_code=409, detail="No active listing.")

    try:
        engine = ExtractionEngine.from_env()
        result = engine.extract(offer)
    except (EnvironmentError, ExtractionError) as exc:
        _pipeline_results[offer.offer_id] = {
            "offer": offer.model_dump(mode="json"),
            "profile": None,
            "score": None,
            "received_at": offer.timestamp.isoformat(),
            "error": str(exc),
        }
        return PipelineRunResponse(offer_id=offer.offer_id, status="error", error=str(exc))

    scorer = ScoringEngine(listing.criteria, rent_nis=listing.rent_nis)
    score_result = scorer.score(result.profile)

    c_vec = criteria_to_vector(listing.criteria)
    p_vec = profile_to_vector(result.profile, listing.criteria, listing.rent_nis)

    _pipeline_results[offer.offer_id] = {
        "offer": offer.model_dump(mode="json"),
        "profile": result.profile.model_dump(mode="json"),
        "score": score_result.model_dump(mode="json"),
        "received_at": offer.timestamp.isoformat(),
        "vectors": {
            "dims": DIM_LABELS,
            "landlord": [round(v, 4) for v in c_vec],
            "tenant": [round(v, 4) for v in p_vec],
        },
        "error": None,
    }
    return PipelineRunResponse(offer_id=offer.offer_id, status="ok")


@app.get("/pipeline/results", response_model=list[dict])
async def pipeline_results() -> list[dict]:
    """
    Returns all pipeline results sorted by score descending (ranked tenant list).
    Offers that errored during extraction appear last.
    """
    results = list(_pipeline_results.values())
    results.sort(
        key=lambda r: (
            -(r["score"]["score"] if r["score"] else -1),
            r.get("received_at", ""),   # earlier timestamp wins on tie
        ),
    )
    return results


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def ui() -> FileResponse:
    """Serve the landlord UI."""
    if not _UI_PATH.exists():
        raise HTTPException(status_code=404, detail="UI file not found.")
    return FileResponse(_UI_PATH, media_type="text/html")
