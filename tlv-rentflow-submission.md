# TLV-RentFlow — Project Submission

**Ruben Chocron · June 2026**

---

## 1. Executive Summary

### Problem Statement

Tel Aviv's rental market is chaotic in a way that resists standard tooling. Landlords post on Yad2, Facebook Marketplace, and WhatsApp simultaneously, and within hours they receive dozens of unstructured messages in a mix of Hebrew, English, and informal slang. A message like *"היי, אני ועוד שותפה, שתינו עובדות, נכנס תוך חודש, ללא חיות, בת 27"* contains six distinct screening signals — yet a landlord reading it still has to manually compare it against their own criteria (budget floor, pet policy, occupancy limit, preferred move-in date) and decide whether to respond.

The goal was to build a pipeline that automates this screening end-to-end: ingest raw messages from multiple channels, extract structured tenant profiles using an LLM, score each candidate against the landlord's criteria, and surface a ranked list with a per-dimension explanation.

### Goals

1. **Robustness**: handle multilingual input (Hebrew/English/slang), missing fields, non-applicant messages, and multi-roommate scenarios correctly.
2. **Auditability**: every extracted fact traces back to the exact substring that justified it. A landlord should be able to see *why* a score was computed, not just what it was.
3. **Testability**: the non-deterministic part of the system (the LLM) must be fully isolated so that the deterministic parts (scoring, validation, routing) can be tested without API calls.
4. **Honest uncertainty**: `None` in a tenant profile means *not stated* — never a default, never a guess. This three-way distinction (`True`/`False`/`None`) is the central semantic invariant the whole pipeline depends on.

### Architecture

The system is organized as four stations connected by Pydantic data contracts:

```
RawOffer  →  TenantProfile  →  ScoreResult
Station 1     Station 2         Station 3
(Ingestion)   (Extraction)      (Scoring)
```

**Station 1 — Ingestion** (`FastAPI`): A `POST /webhook/{channel}` endpoint accepts raw tenant messages. The channel (`whatsapp`/`facebook`/`yad2`) is authoritative from the URL path, not the body. An `InMemoryOfferStore` backed by a `threading.Lock` holds offers for the server's lifetime. A `POST /listing` endpoint lets landlords publish their apartment and screening criteria; the webhook rejects all offers until a listing is live (HTTP 409). A `POST /pipeline/run` endpoint triggers extraction + scoring on demand for any stored offer and caches the result.

**Station 2 — Extraction** (`gpt-4.1-mini`, structured outputs): Takes a `RawOffer.text` and returns a validated `TenantProfile`. Every extracted field is `Optional` — `None` always means "not stated." The LLM is called with `response_format: json_schema, strict: True` and `temperature: 0` to eliminate format hallucination and reduce output variance. Provenance tracking maps each extracted value back to the exact source substring.

**Station 3 — Scoring** (`pure Python`): Computes a 0–100 compatibility score using a dot-product ratio (fraction of maximum possible points earned). Hard constraints (pets forbidden, budget below private floor, strict move-in deadline, occupant limit exceeded) apply a ×0.01 multiplier as a dealbreaker penalty, applied *after* the vector math so the algebra stays clean. Seven weighted dimensions produce a per-dimension `RuleHit` breakdown for explainability.

**Station 4 — Evaluation** (designed, not yet built): An offline eval harness meant to run the fixture messages through Station 2 and measure per-field precision/recall against a golden dataset. The fixture set (`data/fixtures.jsonl`) covers 24 real-world messages specifically chosen to stress-test the edge cases.

### Outcome

A working end-to-end pipeline with a FastAPI server, ~50 unit and integration tests, and a landlord UI at `GET /`. The scoring engine handles a broad range of real-world cases correctly. The main known limitation is that there is no production-grade database (in-memory only), and the eval harness for measuring LLM extraction quality against golden labels has not been built yet.

---

## 2. Prompts and Development Process

### Initial Problem Framing

The first decision was how to frame the extraction task for the LLM. The naive approach — "extract tenant information from this message" — is insufficiently precise because it leaves several critical ambiguities unspecified:

- What does it mean when a field is absent? (Guess? Default? Leave blank?)
- How should `has_pets` be handled — two-valued or three-valued?
- What counts as a budget? (A tenant writing in implies acceptance of the posted price; only explicit statements like "can do up to 6200" should populate `budget_nis`.)
- How should roommate-reported-per-person budgets be handled?

My initial prompt said:

> "Extract the following fields from the tenant message. If a field is not mentioned, return null."

This was functionally correct but underdetermined. The first iteration of output on the message *"Me and 2 friends, can do max 3500 per person"* returned `budget_nis: 3500` — the per-person figure, not the total. The correct answer is `10500`.

**Iteration 1**: I added a specific rule to the prompt:

> "Be wary of roommates posting price per person — '3500 per person' with 3 roommates means budget_nis=10500."

**Iteration 2**: The `move_in_date` field required the model to know today's date to resolve relative expressions like "immediately" or "within a month." Adding `Today's date for your calculations is {date}` to the prompt (injected at runtime) fixed this. The date is now injected dynamically from the server clock.

**Iteration 3**: Hebrew gender inference was initially missed. Hebrew encodes gender grammatically — *"בן 28"* strongly implies male; *"2 שותפות"* implies the group is female. I added explicit guidance:

> "Notice that in Hebrew, gender is often implied by gendered language. e.g. 'בן 25' strongly implies male, while '2 שותפות' implies female."

This is linguistically non-trivial and required close reading of Hebrew messages to identify the pattern.

**Iteration 4 — Provenance**: I added provenance tracking after realizing there was no way to audit an extraction result. The requirement was that for every non-null field, the model should return the *exact substring* that justified the value. This added 10 `_prov` keys to the schema (one per field). The key design constraint was that OpenAI's `strict: True` mode forbids `additionalProperties: true` anywhere in the schema, which ruled out a clean `{field: source_span}` dict as the provenance container. The solution was to inline the `_prov` keys as flat siblings of the main fields in the JSON response, then reshape them on the Python side (in `ExtractionEngine.extract()`) before passing the dict to Pydantic. This is not elegant, but it satisfies the strict-mode constraint without relaxing it.

**Iteration 5 — Non-application filtering**: Early tests showed the model would try to extract data from messages that clearly weren't applications (e.g., "How much is the rent?" or "Can you send more photos?"). The rule I added:

> "If the message is not a rental application (e.g. asking about price, requesting photos), return all screening fields as null."

This is the correct behavior because such messages go through the same code path as applications — the scoring engine handles them by applying an `_EMPTY_PROFILE_PENALTY` (×0.1) since every scoreable field is `None`.

### Key Design Decisions

**Why `temperature=0`**: LLM non-determinism is the primary source of instability in this pipeline. Setting temperature to 0 doesn't eliminate it entirely (the API is not guaranteed deterministic), but it drastically reduces variance and makes the extraction behavior closer to a deterministic function that can be evaluated against a golden dataset. Combined with structured outputs and an explicit JSON schema, this is the closest thing to "pinning" LLM behavior without fine-tuning.

**Why dot-product ratio instead of cosine similarity**: The scoring algorithm initially used cosine similarity between the tenant vector and the landlord's ideal vector. The problem is that cosine similarity measures *angle*, not *magnitude* — a tenant who answered all questions perfectly but with low weights would score identically to a tenant who answered all questions perfectly with high weights. The dot-product ratio (`dot(p, c) / dot(c, c)`) measures what fraction of the maximum achievable score the tenant actually earned. This gives a true [0, 1] range where every unanswered question costs proportionally to its weight.

**Why the `OfferStore` Protocol boundary**: `app.py` depends on `OfferStore` (a `Protocol`), not `InMemoryOfferStore` (the concrete class). This means the test suite can swap in a fresh store per test without touching app internals, and a database-backed store could be substituted without modifying the API layer. The production path instantiates `InMemoryOfferStore` as a module-level default; only the test layer overrides it.

**Why `lowest_price_nis` lives on `ScoringCriteria`, not `Listing`**: In Israeli rentals, landlords sometimes accept a lower price than they post. The public asking price (`rent_nis`) lives on `Listing` because it's published externally. The private floor (`lowest_price_nis`) lives on `ScoringCriteria` because it's a landlord preference that should not appear in the public listing. The scoring engine takes both as separate parameters, keeping the public/private boundary explicit in the type system.

---

## 3. Critical Reflection on AI Output

### What the AI Got Right

The LLM's extraction quality on most well-formed messages was surprisingly good from the first attempt. Structured outputs with a strict schema meant I never had to write JSON parsing code that handles malformed responses — the model reliably produced schema-conformant output. Null semantics (don't guess) were respected in the majority of cases after rule 1 was stated clearly.

### Where the AI Required Active Correction

**The per-person budget hallucination**: The model's initial behavior on *"3500 per person with 2 friends"* was to return `budget_nis: 3500`. This is a linguistically plausible reading — "3500 is the stated price" — but semantically wrong. The model had no way to know about Israeli rental convention (tenants report their total capacity, not per-person share) without being told. This is an example of domain knowledge that the prompt engineer must supply.

**Over-eager gender inference**: After adding the Hebrew gender rule, early versions of the prompt caused the model to infer gender too aggressively — returning `male` for *"אני עובד בהייטק"* (I work in hi-tech) because "עובד" (the word for "works") is masculine in form, even though it's the default grammatical form used by all genders. I had to add "null if ambiguous" to the rule and include a counter-example in the few-shot section. The final prompt uses *"clearly stated or strongly implied"* as the threshold.

**Provenance hallucination**: The earliest version of provenance tracking had no verification step. I tested it by deliberately providing a message where the model would need to summarize rather than quote directly. The model sometimes returned a paraphrase rather than the exact substring. The engine now checks every provenance span against the original offer text and logs a warning on any mismatch. This is not a hard rejection (the profile is still used), because a warning during extraction is better than a pipeline failure — but it surfaces the hallucination for human review rather than silently accepting it.

**Provenance for null fields**: The model occasionally returned `budget_nis_prov: "tenant did not mention"` rather than `null`. This technically doesn't match the rule ("if a field is null, its `_prov` key must also be null") but wouldn't be caught by schema validation since both are strings. I added a few-shot example showing the correct behavior for a null field, which eliminated this.

### The Limits of Prompt Engineering

The most important thing I learned is that prompt engineering is not a substitute for evaluation. The prompts I've written are tuned against 24 fixtures and a handful of adversarial cases I thought of manually. They almost certainly have failure modes I haven't found yet. This is why Station 4 (the eval harness) was designed from the start: the only way to know whether a prompt change is an improvement is to run it against a labeled golden dataset and measure precision/recall per field. Without that harness, every prompt change is a guess.

The current state is: I have prompts that work well on cases I've observed, but I cannot claim a precision/recall number without the eval harness.

### Engineering Judgment vs. AI Trust

Several outputs that were structurally valid were semantically wrong in ways the schema couldn't catch:

- A message like *"כמה עולה הדירה?"* ("How much is the rent?") returned all nulls — correct behavior — but the model sometimes populated `preferred_language: "he"` because it could infer the language from the message. I made a judgment call to allow this: language detection is reliable even from non-application messages, and it doesn't affect scoring. But it is technically violating rule 9.

- The move_in_date for "immediately" should be today's date. The model returns this correctly *given the injected date*. But during development, I ran scripts without injecting the date and received `move_in_date: "2025-01-01"` (some training-data cutoff date). This was caught by inspection during the demo run and fixed by making the date injection mandatory in the system prompt.

The pattern: **I never trusted a new rule or field addition until I had at least one manually checked test case confirming the expected behavior.** Automated tests verify that the *engine* correctly passes data to the model and reshapes the response — they cannot verify that the *model* will produce the right data for a given real-world message.

---

## 4. Testing Approach

### Strategy Overview

The test suite is organized around one principle: **isolate the non-deterministic boundary**. The LLM is the only part of the system that can behave differently on identical inputs. Everything downstream of the LLM — the engine's response parsing, the scoring math, the API routing — must be deterministic and fully testable without network calls.

| Layer | Kind | What it covers | File(s) |
|---|---|---|---|
| Data models | Unit | Pydantic validation, enum coercion, default semantics | `test_models.py` |
| Scoring vectors | Unit | Per-dimension compat functions, weight zeroing, dealbreaker detection | `test_vectors.py` |
| Scoring engine | Unit | Score aggregation, qualification thresholds, dealbreaker penalty, explainability hits | `test_scoring_engine.py` |
| Extraction engine | Unit (stub) | Prompt assembly, response reshaping, provenance, null semantics, schema rejection | `test_extraction.py` |
| In-memory store | Unit | Thread-safety, listing lifecycle, offer CRUD | `test_store.py` |
| FastAPI webhook | Integration | Endpoint routing, gating logic, channel validation, HTTP status codes | `test_webhook.py` |
| Extraction (live) | Integration | Real OpenAI call on one fixture; skipped without API key | `integration/test_extraction_live.py` |
| Ingestion (live) | Integration | Full round-trip against a running server | `integration/test_ingestion_live.py` |
| Stress | Manual/script | Concurrency (500 simultaneous requests), memory unbounded growth | `scripts/stress_test.py` |

### Unit Tests — Scoring

The scoring engine is a pure function: same inputs, same output, no I/O. This makes it ideal for exhaustive parameterized testing. The most important test is `test_perfect_profile_scores_100` — it provides a profile that satisfies every criterion exactly and asserts the score is 100.0. This test would fail immediately if any vector math regression were introduced. The dealbreaker tests confirm that a ×0.01 penalty pushes a score below the `rejected_threshold` regardless of how strong the other dimensions are.

### Unit Tests — Extraction Engine with Stub

The `ExtractionClient` is replaced in every unit test with a `MagicMock` whose `extract_raw()` returns a pre-built dict. This lets the test suite cover:

- **Schema reshaping**: flat `_prov` keys are correctly assembled into `TenantProfile.provenance`.
- **Null preservation**: `has_pets: False` must survive as `False`, not collapse to `None`. (This is a subtle Python identity check — `assert profile.has_pets is False` fails if the value is `None`.)
- **Provenance hallucination warning**: if the returned span is not a substring of the original text, the engine must log a warning. Tested with `caplog`.
- **Error propagation**: a `RuntimeError` from the client must surface as a clean `ExtractionError`, not a Pydantic traceback.

### Integration Tests — Webhook

FastAPI's `TestClient` runs the full ASGI application without a real server. Each test fixture creates a fresh `InMemoryOfferStore` and swaps it into the module-level `_store` variable, ensuring no shared state between tests. This catches the gating logic (offers rejected without a listing), channel validation (invalid channels return 422), ID auto-generation, and the full CRUD lifecycle.

### Stress Tests

`scripts/stress_test.py` runs two intentionally adversarial tests:

1. **Concurrency storm**: 500 simultaneous requests via `asyncio` / `httpx`. Goal: ensure no 5xx under parallel load. The threading lock in `InMemoryOfferStore` means this should be safe — the test confirms it empirically.
2. **Memory flood**: sends 2000 offers with ~50 KB text payloads each. The expected outcome is that the process eventually runs out of RAM. The test is not a pass/fail assertion — it's an *observability* test. It documents exactly how many offers and how many MB can be stored before the OS kills the process, which informs production deployment decisions.

### What Was Not Tested, and Why

**LLM output quality (precision/recall)**: The extraction engine unit tests verify that the engine correctly handles whatever the LLM returns. They do not verify that the LLM returns the right thing for a given real-world message. This requires a golden dataset eval harness (Station 4), which was not built within this project scope. This is the largest unverified risk: a prompt regression would pass all unit tests and only be caught by manual inspection or an eval run.

**Concurrent extraction**: The pipeline processes one offer at a time on demand (`POST /pipeline/run`). Multiple simultaneous pipeline runs could theoretically interfere through shared `_pipeline_results`. This is not guarded by a lock; it relies on Python's GIL for dict writes. It's accepted as a known limitation for a single-server prototype.

**The landlord UI** (`ui.html`): The HTML/JS frontend has no automated tests. It was manually verified during development but any UI regression would not be caught by the test suite.

**Multi-process / multi-worker safety**: The `InMemoryOfferStore` uses a `threading.Lock` which protects against thread-level races within one process. Running `uvicorn --workers 4` would split state across processes and break everything. This is documented but not tested. Accepted risk: the project is a single-server prototype.

**Retry logic in `ExtractionClient`**: The retry loop (exponential backoff on `RateLimitError` and transient `APIError`) is not tested. Testing it would require injecting exceptions from a mock that fails the first N calls. The logic is straightforward, but omitting it means a regression (e.g., accidentally using the wrong exception type) would not be caught.

**Edge cases in the scoring math**: Several boundary conditions in the vector functions are not covered. For example, `_age_compat` with `min_age=None` and `hi=None` simultaneously hits a `half_range` fallback that uses `hi` as the decay unit — but if `hi` is also `None`, the code would fail. This case cannot arise given the weight-zeroing logic in `_weights()` (age weight is zeroed when both bounds are None), but the relationship is implicit and not tested.

### A Bug That Tests Caught

During development, I changed the dealbreaker penalty from `×0.15` to `×0.01` as I refined the scoring model. At that point, the test `test_dealbreaker_budget_below_floor` asserted `result.score < 20.0` — a threshold that still passed with either multiplier. A sharper test was added:

```python
assert any("DEALBREAKER" in h.reason for h in result.rule_hits)
```

This test is checking the *explainability output*, not just the score, which means it would catch a regression where the score was low for the wrong reason (e.g., empty profile penalty applied instead of dealbreaker). This caught exactly one such case during development, when I refactored the `_build_hits` method and accidentally swapped the condition order, causing some dealbreaker profiles to be tagged as `empty_profile` instead.

### A Bug That Slipped Past Tests

The `move_in_date` injection of today's date is hardcoded in the system prompt string (in `prompts.py`). The string reads `Today's date for your calculations is 2026-06-02`. This works correctly right now. It will silently produce wrong results after any calendar day passes, because the date will be stale. There is no test for this because the prompt is a static module-level constant — the test suite never asserts that the date in the prompt matches the actual current date. A fix would be to template the prompt and inject the date dynamically at call time. This is a known issue accepted as a shortcoming of the current implementation.

---

## 5. Demo

The server is run locally:

```bash
source .venv/bin/activate
uvicorn rentflow.ingestion.app:app --reload --port 8000
```

The full end-to-end flow can be demonstrated via:

- **`GET /`** — Landlord UI: post a listing, simulate offers from any channel, trigger extraction + scoring, view the ranked tenant list with per-dimension breakdowns and provenance.
- **`GET /docs`** — FastAPI Swagger UI: interactive API documentation with all endpoints.
- **`python scripts/send_offers.py`** — Sends all 24 fixture messages from `data/fixtures.jsonl` to the running server.
- **`python scripts/stress_test.py`** — Runs the concurrency and memory stress tests.

As this is a local prototype with no public hosting, a screen recording of the landlord UI showing a full round-trip (post listing → receive offers → run pipeline → view ranked results) is available on request.
