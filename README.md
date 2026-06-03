# TLV RentFlow

An async pipeline that ingests tenant messages from WhatsApp, Facebook, and Yad2, extracts structured applicant data via an LLM, and scores candidates against landlord criteria.

## Running locally

**Prerequisites:** Python 3.11+, an OpenAI API key.

```bash
# 1. Clone and set up
git clone <repo-url>
cd tlv-rentflow
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Add your OpenAI key
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Start the server
uvicorn rentflow.ingestion.app:app --reload --port 8000

# 4. (Optional) Send sample tenant messages
python scripts/send_offers.py
```

The API is now live at `http://localhost:8000`.

## Running tests

```bash
pytest                    # unit tests (no API key needed)
pytest tests/integration/ # integration tests (requires .env with OPENAI_API_KEY)
```

## How it works

```
Incoming message  →  Extract TenantProfile (GPT-4.1-mini)  →  Score against criteria  →  Approved / Review / Rejected
```

1. **Ingestion** — FastAPI webhook at `POST /webhook/{channel}` accepts raw tenant messages from any platform.
2. **Extraction** — GPT-4.1-mini parses multilingual (Hebrew/English/slang) messages into structured applicant data.
3. **Scoring** — Pure deterministic scoring using cosine similarity against landlord-defined criteria (budget, pets, move-in date, occupants, age, gender).

## Project layout

```
src/rentflow/
  ingestion/   # FastAPI app and webhook handlers
  extraction/  # LLM extraction logic
  scoring/     # Deterministic scoring engine
  offer/       # Shared Pydantic data models
data/          # Sample fixture messages
scripts/       # Utilities for sending test offers and stress testing
tests/         # Unit and integration tests
```
