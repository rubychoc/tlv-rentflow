# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

TLV-RentFlow is an async pipeline that ingests raw, multilingual (Hebrew/English/slang) tenant messages from simulated WhatsApp, Facebook, and Yad2 sources, extracts structured `TenantProfile` data via OpenAI `gpt-4.1-mini`, scores candidates against landlord criteria using cosine similarity, and evaluates extraction accuracy with a golden dataset. Built as a DriveNets take-home assignment — priorities are robustness, testability, and deterministic handling of non-deterministic LLM output.

## Environment setup

The project uses a local `.venv`. Activate it once per terminal session before running anything:

```bash
cd "tlv-rentflow"
source .venv/bin/activate
```

Install/reinstall the package in editable mode (needed after adding new source files):
```bash
pip install -e .
```

## Running the server

```bash
uvicorn rentflow.ingestion.app:app --reload --port 8000
```

## Running tests (with venv active)

```bash
# All unit tests (fast, no network)
pytest

# Single test file
pytest tests/test_scoring_engine.py

# Single test by name
pytest tests/test_scoring_engine.py -k "test_approved"

# Integration tests (hit real OpenAI API — requires .env with OPENAI_API_KEY)
pytest tests/integration/
```

Unit tests mock the OpenAI client; integration tests make live API calls and require `OPENAI_API_KEY` in `.env`.

## Running scripts (with venv active)

```bash
# Send all fixture messages to the running server
python scripts/send_offers.py

# Stress test (server must be running first)
python scripts/stress_test.py --test concurrency   # 500 simultaneous requests
python scripts/stress_test.py --test memory        # unbounded RAM growth test
python scripts/stress_test.py                      # both
```

## Pipeline architecture

Four stations connected by Pydantic data contracts:

```
RawOffer  →  TenantProfile  →  ScoreResult
(Station 1)   (Station 2)       (Station 3)
Ingestion     Extraction         Scoring
```

**Station 1 — Ingestion** (`src/rentflow/ingestion/`): FastAPI webhook at `POST /webhook/{channel}`. The `channel` path parameter (`whatsapp`/`facebook`/`yad2`) is the authoritative source of platform — not the request body. `InMemoryOfferStore` holds accepted offers for the server's lifetime; it implements the `OfferStore` Protocol so a DB-backed store can be swapped in without touching `app.py`.

**Station 2 — Extraction** (`src/rentflow/extraction/`): Calls OpenAI `gpt-4.1-mini` with structured outputs (forced JSON schema, `strict: True`) to turn `RawOffer.text` into a `TenantProfile`. `provenance` is excluded from the forced schema because OpenAI strict mode forbids free-form dicts (`additionalProperties` must be false everywhere). API key loaded from `.env` via `load_dotenv(override=True)`.

**Station 3 — Scoring** (`src/rentflow/scoring/`): Pure deterministic functions. No LLM, no I/O. Takes a `TenantProfile` + landlord `ScoringCriteria` + `rent_nis`, builds feature vectors, computes cosine similarity, applies a dealbreaker penalty for hard-constraint violations, and returns a 0–100 score + `Approved/Review/Rejected` + per-dimension breakdown.

**Station 4 — Evaluation** (not yet built): Offline harness. Runs the golden dataset through Station 2 and measures per-field precision/recall. Will live in `eval/`.

## Key data contracts (`src/rentflow/offer/models.py`)

- `RawOffer` — the validated incoming message. All fields required. `channel` injected from URL path, not body.
- `TenantProfile` — every extracted field is `Optional` (`None` = not stated, never guessed). Includes `age: int | None` and `gender: Gender | None` in addition to the core screening fields. `provenance: dict[str, Provenance]` maps field names to source substrings for auditing.
- `MoveInReadiness` enum uses buckets (`IMMEDIATE`/`WITHIN_MONTH`/`FLEXIBLE`/`SPECIFIC_DATE`).
- `has_pets` is three-valued: `True` (has pets), `False` (explicitly said none), `None` (never mentioned).
- `gender` is three-valued enum: `MALE`/`FEMALE`/`OTHER` or `None` (not mentioned).

## Scoring model (`src/rentflow/scoring/`)

### `vectors.py` — Feature vectors (pure functions)

Each dimension maps a tenant field to a compatibility value in [0, 1]:
- **budget**: price-band scoring — `≥ rent_nis` → 1.0; `[price_floor, rent_nis)` → linear interpolation; `< price_floor` → 0.0 (dealbreaker); `None` → 1.0 (no floor) or 0.5 (floor set, benefit of the doubt).
- **pets**, **employment**, **move_in**, **occupants**: same 1.0 / 0.0 / 0.5 pattern as before.
- **age**: linear decay from 1.0 within `±age_tolerance` of `preferred_age` to 0.0 at `±2×age_tolerance`. `None` → 0.5.
- **gender**: 1.0 if matches `preferred_gender`, 0.0 if mismatch, 0.5 if `None` or no preference set.

Dimensions are scaled by `sqrt(weight)` so cosine similarity properly reflects the landlord's priorities.

### `engine.py` — ScoringEngine

```
ScoringEngine(criteria, rent_nis).score(profile) → ScoreResult
```

Algorithm:
1. Build `criteria_to_vector` (all dims = 1.0 × sqrt-weight) and `profile_to_vector`.
2. Compute `cosine_similarity` → raw [0, 1].
3. Check `is_dealbreaker` — hard constraints:
   - Tenant has pets but `pets_allowed=False` → ×0.15 penalty.
   - Tenant's stated budget < `lowest_price_nis` → ×0.15 penalty.
4. `score = cosine × multiplier × 100`.
5. Map to `Approved/Review/Rejected` via configurable thresholds.

### `ScoringCriteria` key fields

| Field | Meaning |
|-------|---------|
| `lowest_price_nis` | Private minimum price landlord will accept (y). Must be ≤ `Listing.rent_nis` (x). |
| `preferred_age` + `age_tolerance` | Soft age target; scored by distance. |
| `preferred_gender` | Soft gender preference. |
| `age_pref_public` / `gender_pref_public` | Whether to advertise the preference in the public listing. **No effect on scoring.** |
| `pets_allowed` | Hard constraint when False — violation is a dealbreaker. |
| All other existing fields | Unchanged from original design. |

## Simulation design

Real WhatsApp/Facebook/Yad2 integration is impossible in scope (no public APIs, terms of service). The mock is architecturally correct: `app.py` is identical to what a real integration would hit — only the *ringer* differs. `scripts/send_offers.py` plays the role of the platform's webhook delivery. `data/fixtures.jsonl` contains 15 realistic messages covering Hebrew slang, code-switching, missing fields, non-applicant messages, age/gender signals, and multi-roommate cases.

## Invariants to preserve

- `None` in `TenantProfile` always means "not stated" — never a default or a guess. The extraction prompt, scoring rules, and eval metrics all depend on this three-way distinction.
- Scoring (Station 3) must remain pure functions with no I/O. It is the only part of the pipeline that can be made provably correct.
- The `OfferStore` Protocol boundary in `app.py` must not be broken — `app.py` must never import `InMemoryOfferStore` directly in production paths (only for the module-level default instance).
- `ScoringEngine` requires both `criteria` and `rent_nis` — the public asking price is on `Listing`, not on `ScoringCriteria`, because it is public information that flows through separately from the private screening preferences.
