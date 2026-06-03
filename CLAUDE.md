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
- `TenantGroup` — Station 2's output. Holds the household-level shared facts and a list of per-person `TenantProfile`s. A solo applicant is a group of one. A non-application message produces `applicants=[]` and all-null shared fields.
  - Shared fields: `budget_nis`, `move_in_date`, `has_pets`, `household_size`, `preferred_language`, `provenance`.
  - `household_size` = total occupants (including sender). `None` if not stated.
- `TenantProfile` — per-person data for one applicant. Fields: `employment_status`, `age`, `gender`, `name`, `phone`, `provenance`. Every field is `Optional` (`None` = not stated, never guessed).
- `has_pets` is three-valued (on `TenantGroup`): `True` (household has pets), `False` (explicitly none), `None` (never mentioned).
- `gender` is three-valued enum (on `TenantProfile`): `MALE`/`FEMALE`/`OTHER` or `None` (not mentioned).

## Scoring model (`src/rentflow/scoring/`)

### `vectors.py` — Feature vectors (pure functions)

Seven dimensions, split into shared (group-level) and per-person (averaged across applicants):

**Shared dims** (read from `TenantGroup`):
- **budget**: `≥ rent_nis` → 1.0; `[price_floor, rent_nis)` → linear interpolation; `< price_floor` → 0.0; `None` → 1.0 (implicit acceptance).
- **pets**: `False` and forbidden → 1.0; `True` → 0.0; `None` → 0.05 (_UNKNOWN).
- **move_in**: decay from deadline distance; `None` → 1.0 (implicit acceptance).
- **occupants**: `household_size ≤ max_occupants` → 1.0; over → 0.0; `None` → falls back to `len(applicants)`.

**Per-person dims** (averaged over `TenantGroup.applicants`):
- **employment**: 1.0 employed/self-employed, 0.5 student, 0.0 unemployed, 0.05 unknown.
- **age**: decay from `[min_age, max_age]` range; `None` → 0.05.
- **gender**: 1.0 match, 0.0 mismatch, 0.05 unknown.

Dimensions are scaled by `sqrt(weight)`. Dealbreakers remain strict — one member's violation vetoes the whole group.

### `engine.py` — ScoringEngine

```
ScoringEngine(criteria, rent_nis).score(group: TenantGroup) → ScoreResult
```

Algorithm:
1. Build `criteria_to_vector` (all dims = 1.0 × sqrt-weight) and `group_to_vector`.
2. Compute dot-product ratio: `dot(g, c) / dot(c, c)` → raw [0, 1].
3. Check `is_dealbreaker` — strict hard constraints (any member's violation vetoes the group):
   - Household has pets but `pets_allowed=False` → ×0.01 penalty.
   - Group budget < `lowest_price_nis` → ×0.01 penalty.
   - Move-in misses strict deadline → ×0.01 penalty.
   - `household_size > max_occupants` → ×0.01 penalty.
4. `score = raw_ratio × multiplier × 100`.
5. Map to `Approved/Review/Rejected` via configurable thresholds.

### `ScoringCriteria` key fields

| Field | Meaning |
|-------|---------|
| `lowest_price_nis` | Private minimum price landlord will accept (y). Must be ≤ `Listing.rent_nis` (x). |
| `min_age` / `max_age` | Soft age range; scored by decay outside the range. |
| `preferred_gender` | Soft gender preference. |
| `age_pref_public` / `gender_pref_public` | Whether to advertise the preference in the public listing. **No effect on scoring.** |
| `pets_allowed` | Hard constraint when False — violation is a dealbreaker. |
| `max_occupants` | Hard constraint — `household_size` over this is a dealbreaker. |

## Simulation design

Real WhatsApp/Facebook/Yad2 integration is impossible in scope (no public APIs, terms of service). The mock is architecturally correct: `app.py` is identical to what a real integration would hit — only the *ringer* differs. `scripts/send_offers.py` plays the role of the platform's webhook delivery. `data/fixtures.jsonl` contains 15 realistic messages covering Hebrew slang, code-switching, missing fields, non-applicant messages, age/gender signals, and multi-roommate cases.

## Invariants to preserve

- `None` in `TenantGroup` or `TenantProfile` always means "not stated" — never a default or a guess. The extraction prompt, scoring rules, and eval metrics all depend on this three-way distinction.
- Scoring (Station 3) must remain pure functions with no I/O. It is the only part of the pipeline that can be made provably correct.
- The `OfferStore` Protocol boundary in `app.py` must not be broken — `app.py` must never import `InMemoryOfferStore` directly in production paths (only for the module-level default instance).
- `ScoringEngine` requires both `criteria` and `rent_nis` — the public asking price is on `Listing`, not on `ScoringCriteria`, because it is public information that flows through separately from the private screening preferences.
