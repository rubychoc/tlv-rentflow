# TLV-RentFlow — Project Submission

**Ruben Chocron · June 2026**

---

## 1. Executive Summary

### Problem Statement

Tel Aviv's rental market is chaotic in a way that resists standard tooling. Landlords post on Yad2, Facebook Marketplace, and WhatsApp simultaneously, and within hours they receive dozens of unstructured messages in a mix of Hebrew, English, and informal slang. A message like *"היי, אני ועוד שותפה, שתינו עובדות, נכנס תוך חודש, ללא חיות, בת 27"* contains six distinct screening signals — yet a landlord reading it still has to manually compare it against their own criteria (budget floor, pet policy, occupancy limit, preferred move-in date) and decide whether to respond.

The goal was to build a pipeline that automates this screening end-to-end: ingest raw messages from multiple channels, extract structured tenant profiles using an LLM, score each candidate household against the landlord's criteria, and surface a ranked list with a per-dimension explanation.

### Goals

1. **Robustness**: handle multilingual input (Hebrew/English/slang), missing fields, non-applicant messages, and multi-applicant households correctly.
2. **Auditability**: every extracted fact traces back to the exact substring that justified it. A landlord should be able to see *why* a score was computed, not just what it was.
3. **Testability**: the non-deterministic part of the system (the LLM) must be fully isolated so that the deterministic parts (scoring, validation, routing) can be tested without API calls.
4. **Honest uncertainty**: `None` in any profile field means *not stated* — never a default, never a guess. This three-way distinction (`True`/`False`/`None`, or value/`None`) is the central semantic invariant the whole pipeline depends on.

### Architecture

The system is organized as four stations connected by Pydantic data contracts:

```
RawOffer  →  TenantGroup  →  ScoreResult
Station 1     Station 2        Station 3
(Ingestion)   (Extraction)     (Scoring)
```

**Station 1 — Ingestion** (`FastAPI`): A `POST /webhook/{channel}` endpoint accepts raw tenant messages. The channel (`whatsapp`/`facebook`/`yad2`) is authoritative from the URL path, not the body. A `threading.Lock`-protected `InMemoryOfferStore` holds offers for the server's lifetime. `POST /listing` lets landlords publish their apartment and screening criteria; the webhook rejects all offers until a listing is live (HTTP 409). `POST /pipeline/run` triggers extraction + scoring on demand for any stored offer.

**Station 2 — Extraction** (`gpt-4.1-mini`, structured outputs): Takes a `RawOffer.text` and returns a validated `TenantGroup`. Input is hard-capped at 1000 characters before the API call to prevent runaway token costs. The LLM is called with `response_format: json_schema, strict: True` to eliminate format hallucination. Provenance tracking maps each extracted value back to the exact source substring, at both the household level and per-person level.

**Station 3 — Scoring** (`pure Python`): Computes a 0–100 compatibility score using a dot-product ratio (fraction of maximum possible points earned). Hard constraints (pets forbidden, budget below private floor, strict move-in deadline, occupant limit exceeded) apply a ×0.01 multiplier as a dealbreaker penalty. Seven weighted dimensions produce a per-dimension `RuleHit` breakdown. Per-person dimensions (employment, age, gender) are **averaged across all applicants** — a household is assessed by the mean of its members.

**Station 4 — Evaluation** (designed, not yet built): An offline eval harness meant to run fixture messages through Station 2 and measure per-field precision/recall against golden labels. The fixture set (`data/fixtures.jsonl`) covers 24 real-world messages.

### Key Data Model: `TenantGroup`

The central architectural decision of the project is the `TenantGroup` / `TenantProfile` split. A single `TenantGroup` represents one rental application, and contains:

- **Shared / household fields**: `budget_nis`, `move_in_date`, `has_pets`, `household_size`, `preferred_language`, `provenance` — facts that belong to the whole household and need only be extracted once.
- **`applicants: list[TenantProfile]`** — one entry per person whose individual details are stated in the message. A solo applicant is a group of one; a couple is a group of two; a non-application message produces `applicants = []`.

This two-tier structure is what allows the pipeline to handle messages like *"Me and my partner, we have 2 cats. I freelance, she's employed. I'm 31."* correctly — two applicants with different employment statuses, a shared pet, and only one known age.

### Outcome

A working end-to-end pipeline with a FastAPI server, ~255 unit and integration tests, and a landlord UI at `GET /`. The scoring engine handles multi-person households, partial information, and non-applicant messages correctly. The main known limitation is that there is no production-grade database (in-memory only), and the eval harness for measuring LLM extraction quality against golden labels has not been built yet.

---

## 2. Prompts and Development Process

### Initial Problem Framing

The first decision was how to frame the extraction task for the LLM. The naive approach — "extract tenant information from this message" — is insufficiently precise because it leaves several critical ambiguities unspecified:

- What does it mean when a field is absent? (Guess? Default? Leave blank?)
- How should `has_pets` be handled — two-valued or three-valued?
- What counts as a budget? (A tenant writing in implies acceptance of the posted price; only explicit statements like "can do up to 6200" should populate `budget_nis`.)
- How should roommate-reported-per-person budgets be handled?
- How should a message from multiple people be handled?

My initial prompt extracted a flat single-person profile. This worked for solo applicants but produced structurally wrong output for messages like *"Me and 2 friends, can do max 3500 per person"*.

### Prompt Iterations

**Iteration 1 — Per-person budget multiplication**: The first test of the multi-person case returned `budget_nis: 3500` — the per-person figure, not the total. I added:

> "If per-person amounts are given, multiply: '3500 per person' × 3 people = 10500."

**Iteration 2 — Move-in date anchoring**: `move_in_date` required the model to know today's date to resolve relative expressions like "immediately" or "within a month." I injected `Today's date for your calculations is {date}` dynamically. Without this, the model fell back to a training-data date.

**Iteration 3 — Hebrew gender inference**: Hebrew encodes gender grammatically — *"בן 28"* strongly implies male; *"2 שותפות"* implies a female group. I added:

> "Hebrew gendered language is a strong signal: 'בן 25' → male, 'בחורה' → female, '2 שותפות' → female."

But I also had to add a counter-rule immediately after testing:

> "In plural statements, the grammatical gender may not reflect all actual genders ('אנחנו 3 סטודנטים' means at least one male student and we don't know the gender of the rest)."

**Iteration 4 — Provenance at two levels**: I added provenance tracking — for every non-null field, the model must return the exact substring that justified it. The initial design tried to use a `{field: source_span}` dict, but OpenAI's `strict: True` mode forbids `additionalProperties: true` anywhere in the schema. The workaround was to inline all `_prov` keys as flat siblings (`budget_nis_prov`, `move_in_date_prov`, etc.) and reshape them in Python before passing to Pydantic. With the `TenantGroup` model, this had to be done at both levels: group-level fields and per-person fields inside each `applicants` entry.

**Iteration 5 — The `TenantGroup` refactor (biggest change)**: The flat single-profile model broke down on multi-person messages in ways that couldn't be fixed by prompt engineering alone. The fundamental issue was: there's no clean way to output two people's employment statuses and ages in a flat JSON object. The solution was to refactor the output schema to have a top-level household object and a `applicants` array. This required:

- A new `TenantGroup` Pydantic model with `applicants: list[TenantProfile]`
- A new JSON schema (`_TENANT_GROUP_SCHEMA`) with a nested `_PERSON_SCHEMA` for the `applicants` array items
- Updated extraction engine reshaping logic (now has to reshape prov keys at both levels)
- Updated scoring engine and vector functions (per-person dimensions now average over `group.applicants`)
- Updated all tests that depended on `result.profile` to use `result.group` and `result.group.applicants[0]`

The prompt was also rewritten from scratch to explain the two-tier structure with four worked examples (solo, couple with split employment, 3-person group, non-applicant).

**Iteration 6 — Non-application filtering**: Messages like "How much is the rent?" should return `applicants: []` and all-null household fields, not a guessed profile. The rule:

> "If the message is not a rental application, return all household fields as null and `applicants` as an empty array `[]`."

The scoring engine handles these by applying an `_EMPTY_GROUP_PENALTY` (×0.1) — the empty profile path is detected by checking that all scoreable fields on both the group and all applicants are `None`.

### Key Design Decisions

**Why `TenantGroup` instead of a flat profile**: The flat model produced ambiguous output for multi-person applications. Which person's employment do you put in `employment_status`? Both? An average? The `TenantGroup` model makes the structure explicit in the type system: group fields belong to the household, person fields belong to individuals. The scoring engine then averages per-person dimensions across applicants, which is both semantically correct (the landlord cares about the *household*) and computable without guessing.

**Why dot-product ratio instead of cosine similarity**: The scoring algorithm uses `dot(group_vec, criteria_vec) / dot(criteria_vec, criteria_vec)` — the fraction of maximum achievable points earned. Cosine similarity measures angle, not magnitude; two households with very different weight profiles but identical angular orientation would receive the same score. The ratio gives a true [0, 1] range where every unanswered or mismatched dimension costs proportionally to its weight.

**Why `_UNKNOWN = 0.05` (not 0 or 0.5)**: An unknown field means the tenant didn't state something — that should count mildly against them (the landlord has less information), but not as severely as an explicit mismatch. Setting `_UNKNOWN` to 0.05 means an unstated field scores about 5% of maximum — it penalizes silence without treating it as a hard miss. The previous design used `0.5`, which was too generous.

**Why students score `0.5` on employment (not `0.0`)**: `EmploymentStatus.STUDENT` is a softer signal than `UNEMPLOYED`. A student is likely to have some income (loans, parents, part-time work) and is a common tenant type in Tel Aviv. Scoring them at 0.5 puts them in the "uncertain" band rather than "hard fail," which matches real landlord behavior better.

**Why `lowest_price_nis` lives on `ScoringCriteria`, not `Listing`**: The public asking price (`rent_nis`) lives on `Listing` because it's published externally. The private floor (`lowest_price_nis`) lives on `ScoringCriteria` because it's a landlord preference that should not appear in the public listing. The scoring engine takes both as separate parameters.

**Why input is truncated at 1000 characters**: The longest realistic tenant message is well under 800 characters. The truncation guard (`MAX_INPUT_CHARS = 1000`) prevents adversarial or accidental large inputs from causing runaway token costs. Anything beyond 1000 characters is truncated and logged before the API call.

---

## 3. Critical Reflection on AI Output

### What the AI Got Right

The LLM's extraction quality on clear, unambiguous messages was strong from the start. Structured outputs with a strict schema meant the model never produced malformed JSON. Null semantics (don't guess) were respected for most fields in most cases. The per-person provenance was surprisingly accurate — the model correctly attributed age claims to the right person in multi-applicant messages.

### Where the AI Required Active Correction

**The per-person budget hallucination**: `budget_nis: 3500` instead of `10500` for a 3-person group where each pays 3500. This is a domain knowledge gap — Israeli rental convention requires the total, not the per-person figure. The model had no way to know this without an explicit rule.

**Over-eager gender inference in Hebrew plurals**: After adding the Hebrew gender rule, the model began returning `male` for grammatically masculine plural forms (`סטודנטים`, `עובדים`) even when the actual genders of the group members were unknown. Hebrew uses masculine plural as the grammatical default, not as a gender claim. I had to add the counter-rule: *"In plural statements, the grammatical gender may not reflect all actual genders."*

**Inconsistent `applicants` count vs. `household_size`**: Early versions of the multi-person prompt sometimes returned 2 applicants when `household_size = 3`, or 3 applicants for a solo message. The engine now logs a warning on mismatch (`household_size != len(applicants)`), and the prompt has a hard rule: *"Always emit exactly one applicant object per person."* This is still the weakest point of the extraction — the model occasionally produces the wrong count for large groups.

**Provenance hallucination**: The model sometimes returns a paraphrase as the `_prov` span rather than the exact substring. The engine validates every span against the original offer text and logs a warning on mismatches. This surfaces the hallucination for review but doesn't reject the profile — a warning during extraction is better than a pipeline failure, and the extracted value may still be correct even if the cited span is slightly off.

**Stale date in the prompt**: The system prompt contains `Today's date for your calculations is 2026-06-02`. This is a hardcoded string in `prompts.py`. It works correctly today, but will silently produce wrong `move_in_date` values for "immediately" and "within a month" after this date passes. There is no test for this, and fixing it requires templating the prompt at call time rather than at module load time.

### The Limits of Prompt Engineering

The most important lesson: prompt engineering is not a substitute for evaluation. The prompts are tuned against 24 fixtures and a handful of adversarial cases found by inspection. They have failure modes not yet discovered. This is why Station 4 (the eval harness) was designed from the start — the only way to know whether a prompt change is an improvement is to run it against a labeled golden dataset and measure precision/recall per field. Without that harness, every prompt change is a bet.

The current state: the prompts work well on cases observed, but no precision/recall number can be claimed.

### Engineering Judgment vs. AI Trust

Several outputs were structurally valid but semantically wrong in ways the schema couldn't catch:

- The model sometimes inferred `gender` for one applicant in a multi-person message from a context that applied to a different person. The per-person provenance structure helps detect this — if the cited span doesn't mention the right person, it's a misattribution.

- `move_in_date` for "immediately" depends on the injected date. During development, running without `.env` set correctly caused the model to return a date from 2025. This was caught by inspection during a demo run and fixed by making the date injection a prompt invariant, verified in the prompt content tests.

- A message saying *"my friend has a dog"* should not set `has_pets = true` — it's the friend's pet, not the tenant's. The prompt rule says *"return true only if it clearly belongs to the tenant(s) applying"*, but this is an ambiguous case that the model doesn't always resolve correctly. The integration test suite covers the non-tenant-pet case explicitly.

**The pattern throughout**: I never trusted a new rule or field addition until at least one manually checked test case confirmed the expected behavior. Automated tests verify that the *engine* correctly passes data to the model and reshapes the response — they cannot verify that the *model* produces the right data for a given real-world message.

---

## 4. Testing Approach

### Strategy Overview

The test suite is organized around one principle: **isolate the non-deterministic boundary**. The LLM is the only part of the system that can behave differently on identical inputs. Everything downstream — engine response parsing, scoring math, API routing — must be deterministic and fully testable without network calls.

| Layer | Kind | What it covers | File(s) |
|---|---|---|---|
| Data models | Unit | Pydantic validation, enum coercion, null semantics | `test_models.py` |
| Scoring vectors | Unit | Per-dimension compat functions, group averaging, dealbreaker detection | `test_vectors.py` |
| Scoring engine | Unit | Score aggregation, qualification thresholds, dealbreaker penalty, explainability | `test_scoring_engine.py` |
| Extraction engine | Unit (stub) | Prompt content, response reshaping, provenance, null semantics, error handling | `test_extraction.py` |
| In-memory store | Unit | Thread-safety, listing lifecycle, offer CRUD | `test_store.py` |
| FastAPI webhook | Integration | Endpoint routing, gating logic, channel validation, HTTP status codes | `test_webhook.py` |
| Extraction (live) | Integration | Real OpenAI calls on curated fixtures; skipped without API key | `integration/test_extraction_live.py` |
| Cost analysis | Integration (live) | Token usage, cost-per-offer, unit-economics table | `integration/test_cost_analysis.py` |
| Stress | Manual/script | 500 concurrent requests; memory unbounded growth | `scripts/stress_test.py` |

### Unit Tests — Scoring

The scoring engine is a pure function: same inputs, same output, no I/O. The most important test is `test_perfect_profile_scores_100` — it provides a `TenantGroup` with one applicant who satisfies every criterion exactly and asserts `score == 100.0`. This test would fail immediately if any vector math regression were introduced.

The group-aware scoring required new test cases compared to the original flat-profile design:

- A group of two where one has a pet triggers the dealbreaker (the veto is group-level — one member's violation vetoes the household).
- The per-person averaging tests verify that a group where one person is employed and one is a student scores between the individual extremes, not equal to either.
- `is_empty_group` tests both paths: group-level fields all null and applicants all null/empty.

### Unit Tests — Extraction Engine with Stub

The `ExtractionClient` is replaced in every unit test with a `MagicMock` whose `extract_raw()` returns a pre-built dict. This lets the suite cover:

- **Group-level prov reshaping**: flat `budget_nis_prov` keys assembled into `TenantGroup.provenance`.
- **Per-person prov reshaping**: `employment_status_prov` inside each applicant dict assembled into `TenantProfile.provenance`.
- **Null preservation**: `has_pets: False` must survive as `False`, not collapse to `None` (`assert group.has_pets is False`).
- **Provenance hallucination warning**: if a returned span is not a substring of the original text, the engine must log a warning (tested with `caplog`).
- **Error propagation**: a `RuntimeError` from the client surfaces as `ExtractionError`, not a Pydantic traceback.
- **`applicants` count mismatch**: engine logs a warning when `household_size != len(applicants)`.

### Integration Tests — Live Extraction

The live test suite (`test_extraction_live.py`) hits the real OpenAI API with curated fixtures and asserts exact field values. These tests are marked `@pytest.mark.live` and excluded from the default `pytest` run. They cover:

- **Unambiguous English and Hebrew**: full field extraction with exact value assertions.
- **Multi-applicant scenarios**: the couple with split employment, the 3-student group, the married couple in Hebrew — each asserting `household_size`, `len(applicants)`, and per-person field distribution.
- **Partial fields**: stated fields are correct; absent fields are `None` (not guessed).
- **Non-applicant**: price-inquiry message produces `applicants = []` and all-null group fields.
- **Provenance validity**: every extracted `source_span` must be a real substring of the original message.

The most complex assertions test invariants like *"exactly one applicant has `age = 27`, the other two have `age = None`"* — these cannot be expressed as simple equality checks and require reasoning about the structure of the group.

### Cost Analysis Test

`tests/integration/test_cost_analysis.py` runs all 24 fixtures through the real API, captures `response.usage` (prompt tokens, completion tokens, cached tokens) from each call, and produces:

1. **Sanity assertions**: p95 prompt tokens < 3000, p95 completion tokens < 600, p95 cost-per-offer < $0.01.
2. **Unit-economics table**: projected monthly cost at Hobby/SMB/Scale volume tiers, with and without prompt caching, for `gpt-4.1-mini`, `gpt-4.1-nano`, and `gpt-4.1`.
3. **Persistent report**: writes `cost_analysis_results.jsonl` to the project root so numbers are reproducible.

Token usage is captured via a `TokenUsage` dataclass stored on `ExtractionClient.last_usage` after each successful call. The pricing constants are dated and must be re-verified against `platform.openai.com` before any production cost estimate.

### Stress Tests

`scripts/stress_test.py` runs two adversarial tests:

1. **Concurrency storm**: 500 simultaneous requests via `asyncio` / `httpx`. The threading lock in `InMemoryOfferStore` is what prevents data corruption; the test confirms empirically that zero 5xx responses occur under peak load.
2. **Memory flood**: 2000 offers with ~50 KB payloads each. The expected outcome is process death from RAM exhaustion. The test is an *observability* tool — it measures exactly how many MB can be stored before the OS kills the process, informing production deployment decisions.

### What Was Not Tested, and Why

**LLM output quality (precision/recall)**: The extraction engine unit tests verify the engine handles whatever the LLM returns. They do not verify the LLM returns the right thing. This requires Station 4 (the eval harness), not yet built. This is the largest unverified risk: a prompt regression passes all unit tests and is only caught by manual inspection or an eval run.

**`applicants` count correctness at scale**: The integration tests assert correct counts for the curated fixture set. They do not systematically test how often the model produces the wrong count across the full population of possible messages. For a 5-person household described in vague terms, the count can vary between runs.

**Concurrent extraction via `POST /pipeline/run`**: Multiple simultaneous pipeline runs could interfere via the shared `_pipeline_results` dict. Python's GIL makes dict writes atomic for single-key updates, but there is no lock. Accepted as a known limitation for a single-server prototype.

**The landlord UI** (`ui.html`): No automated tests. Manually verified during development but any UI regression would not be caught.

**Multi-process / multi-worker safety**: `InMemoryOfferStore` uses `threading.Lock`, which only protects within one process. Running `uvicorn --workers 4` would partition state across processes. Accepted known limitation — documented, not tested.

**The stale date in the system prompt**: `Today's date for your calculations is 2026-06-02` is hardcoded in `prompts.py`. No test asserts that this date matches the current date. It will silently produce wrong relative date calculations after today.

### A Bug That Tests Caught

During development, I changed the dealbreaker penalty from `×0.15` to `×0.01`. The score-threshold tests still passed because they only asserted `score < 20.0`. A separate test checking the `rule_hits` breakdown caught the real issue: after a scoring engine refactor that reordered the `_build_hits` conditions, dealbreaker profiles were being tagged as `empty_profile` instead of `dealbreaker`. The test:

```python
assert any("DEALBREAKER" in h.reason for h in result.rule_hits)
```

…failed on the refactored version, revealing that the penalty was being applied via the wrong path. The score was low either way, so the score-only assertion would not have caught it.

### A Bug That Slipped Past Tests

The `TenantGroup` refactor introduced a subtle scoring regression: the original `_employment_compat` returned `1.0` when `employment_required = False`, regardless of the applicant list. After the refactor, the function averages over `group.applicants` — but when `applicants = []` (non-application message), `_avg_over_applicants` returns `no_applicants_default = 1.0`, which happens to be the same value. So a non-applicant message still scored 1.0 on employment with no criteria set, and the `is_empty_group` penalty path handled the overall low score. The bug was masked: the *function* was returning the right number for the wrong reason. It was caught during a code review of the refactor, not by any test.

---

## 5. Demo

The server is run locally:

```bash
source .venv/bin/activate
uvicorn rentflow.ingestion.app:app --reload --port 8000
```

The full end-to-end flow can be demonstrated via:

- **`GET /`** — Landlord UI: post a listing, simulate offers from any channel, trigger extraction + scoring, view the ranked household list with per-dimension breakdowns and provenance.
- **`GET /docs`** — FastAPI Swagger UI: interactive API documentation.
- **`python scripts/send_offers.py`** — Sends all 24 fixture messages from `data/fixtures.jsonl` to the running server.
- **`pytest tests/integration/test_cost_analysis.py -m "live and slow" -s -v`** — Runs the cost analysis against the real API and prints the unit-economics table.
- **`python scripts/stress_test.py`** — Concurrency and memory stress tests.

As this is a local prototype with no public hosting, a screen recording of the landlord UI showing a full round-trip (post listing → receive offers → run pipeline → view ranked results) is available on request.
