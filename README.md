# TLV RentFlow

An async pipeline that ingests tenant messages from WhatsApp or Facebook, extracts structured applicant data via an LLM, and scores candidates against landlord criteria.

## Running locally

**Prerequisites:** Python 3.11+, an OpenAI API key.

```bash
git clone https://github.com/rubychoc/tlv-rentflow.git && cd tlv-rentflow
./start.sh
```

The script sets up the environment, prompts for your API key on first run, and opens the app at `http://localhost:8000`.

## Pipeline

```
Tenant message
      │
      ▼
┌─────────────┐
│  Ingestion  │  FastAPI webhook — receives raw messages from WhatsApp or Facebook
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Extraction │  GPT-4.1-mini — parses multilingual (Hebrew/English/slang) text into
│             │  structured applicant profiles (budget, move-in date, employment, etc.)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Scoring   │  Deterministic — weighted dot product against landlord criteria,
│             │  hard dealbreakers (pets, budget, # occupants), 0–100 score
└──────┬──────┘
       │
       ▼
  Approved / Review / Rejected
```

## Project layout

```
src/rentflow/
  ingestion/   # FastAPI app and webhook handlers
  extraction/  # LLM extraction logic
  scoring/     # Deterministic scoring engine
  offer/       # Shared Pydantic data models
scripts/       # Utilities for sending sample messages and stress testing
data/          # Sample fixture messages
```

---

Tests live in `tests/` — unit tests require no API key; integration tests hit the real OpenAI API.
