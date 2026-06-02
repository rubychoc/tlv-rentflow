# TLV-RentFlow — Testing Roadmap

A plan, not code. It lays out **how to exercise each component in isolation**, **how to
verify the components compose**, and two harder tracks the assignment cares about most:
**complex multi-offer ranking scenarios** and **LLM hallucination / consistency probing**.

The pipeline is four stations joined by Pydantic contracts:

```
Listing (landlord)  →  RawOffer  →  TenantProfile  →  ScoreResult  →  ranked list
   POST /listing       webhook       extraction         scoring        pipeline/results
   (Station 0)        (Station 1)    (Station 2)        (Station 3)     (aggregation)
```

Stations 0, 1, and 3 are **deterministic** — they get exhaustive, exact-value tests.
Station 2 is the **only non-deterministic** part — it gets statistical/consistency tests
instead of exact-equality assertions.

---

## Testing layers (vocabulary used below)

| Layer | What it touches | Network? | Speed | When run |
|-------|-----------------|----------|-------|----------|
| **Unit** | one function / class, dependencies stubbed | no | ms | every change (`pytest`) |
| **Component** | one station end-to-end via its public API | no (Station 2 mocked) | ms | every change |
| **Integration** | two+ stations wired together | mostly mocked; one `live` variant hits OpenAI | seconds | pre-commit / CI gate |
| **Scenario** | full pipeline, curated offer sets, ranking asserted | mixed | seconds–min | before submission |
| **LLM-probe** | Station 2 only, real API, repeated calls | yes | min | on prompt change |

Markers to introduce: `@pytest.mark.live` (real OpenAI), `@pytest.mark.slow`,
`@pytest.mark.scenario`. Default `pytest` run excludes `live`/`slow`.

---

## Station 0 — Listing (the landlord)

**Public surface:** `POST /listing`, `GET /listing`, `ScoringCriteria` / `Listing` models.

**Why it matters mechanically:** the listing is the gate. No listing ⇒ webhook returns 409.
Posting a new listing **wipes prior offers and pipeline results** (`set_listing` clears,
`create_listing` clears `_pipeline_results`).

### Mechanical — basic cases
- POST a minimal listing (address + rent_nis, default criteria) → 201, returns id/address/rent.
- POST with an explicit `listing_id` → that id is echoed and stored.
- POST without `listing_id` → server generates `listing_<hex>`.
- `GET /listing` after a POST returns the full listing incl. criteria.
- `GET /listing` with none live → 404.

### Mechanical — edge cases
- POST a second listing → old listing replaced; `GET /listing` shows the new one.
- POST a second listing → previously stored offers are gone (`GET /offers` empty).
- POST a second listing → `_pipeline_results` cleared (`GET /pipeline/results` empty).
- Invalid criteria weights (`budget_weight=101`, negative) → 422 (Pydantic `ge/le` bounds).
- **Landlord floor must be ≤ asking price** (`lowest_price_nis ≤ rent_nis`) — by definition
  the private floor `y` sits below the public asking price `x`. The model does **not** enforce
  this today. Test that an inverted config (`lowest_price_nis > rent_nis`) is **rejected**, and
  add the validation if missing. (Distinct from the *tenant* side: a tenant whose stated budget
  is ≥ asking is perfectly fine — that's the `b >= rent_nis → 1.0` branch and is **not** capped.)
- Threshold inversion (`rejected_threshold > approved_threshold`) — no validation today;
  document expected behavior (everything lands in REVIEW or flips) before writing the test.

---

## Station 1 — Sending & ingesting offers

Two halves: the **sender** (`scripts/send_offers.py`, the simulated platform) and the
**receiver** (`POST /webhook/{channel}` + `InMemoryOfferStore`). Test them separately.

### 1a. The store (`InMemoryOfferStore`) — pure unit
- `save_offer` before any listing → raises `RuntimeError`.
- `set_listing` then `save_offer` → offer retrievable via `all_offers` / `get_offer`.
- `get_offer` unknown id → `None`.
- `all_offers` returns a **copy** (mutating the returned list doesn't affect the store).
- `set_listing` resets the offer list to empty.

**Concurrency / race-condition testing (the focus you asked for).**
The store is the only mutable shared state, and its sole protection is a single
`threading.Lock` around `save_offer`/`all_offers`/`get_offer`. That lock is what these tests
are actually probing. Layer the testing — each layer catches a different class of bug:

- **Store-level race (the real test):** spin up a `ThreadPoolExecutor` with many workers
  (e.g. 64) and have them call `save_offer` thousands of times against one shared store.
  Assert `len(all_offers()) == total_calls` — **exactly**, no lost or double-counted
  appends. This is the test that would catch a dropped lock, because Python's `list.append`
  is *not* guaranteed atomic under all interpreters and the lock is the only thing making it
  safe. Run it with a high enough count that an unlocked version would reliably fail.
- **Mixed read/write race:** while writers hammer `save_offer`, readers loop on `all_offers`
  / `get_offer` → assert no exception (no "list changed size during iteration"), and every
  ID a reader sees is one a writer actually wrote (no torn/partial reads).
- **`set_listing` during writes:** a thread posting a new listing (which **resets** the offer
  list) concurrently with writers → assert the store ends in a consistent state (offer count
  matches whatever survived the reset, never a half-cleared list). Documents the
  reset-vs-append interleaving.
- **Negative control:** temporarily remove the lock (in a fork/spike, not committed) and show
  the count assertion *fails* — proves the test has teeth and the lock is load-bearing, not
  decorative.

These live under `@pytest.mark.slow`. They test the store **directly, in-process** — no HTTP,
no event loop — which isolates "is the data structure thread-safe?" from "does the web layer
hold up?" (the latter is 1b + the existing `stress_test.py`).

**Protocol conformance — what this test is and why it exists.**
`OfferStore` is a `typing.Protocol` (structural interface): any class with the right methods
(`set_listing`, `get_listing`, `save_offer`, …) *is* an `OfferStore`, no inheritance needed.
`app.py` declares `_store: OfferStore = InMemoryOfferStore()` — it depends on the **interface**,
not the concrete class. The test: write a tiny fake (e.g. a `dict`-backed `FakeStore` with the
same methods), assign it to `app._store`, drive the API, and assert everything still works.

Two things it proves:
1. **Decoupling is real, not aspirational.** If someone later makes `app.py` reach into
   `InMemoryOfferStore`-specific internals (a private attr, a method not on the Protocol), the
   fake won't have it and the test breaks — flagging the leak. It guards the "must not break
   the `OfferStore` boundary" invariant from CLAUDE.md.
2. **It's the seam every other test uses.** Because you can swap the store, integration tests
   can inject a pre-seeded store (offers already present) or a store that raises on demand
   (to test error paths) without standing up the real one. "The seam used to mock the store
   elsewhere" just means: this swap-ability is the mechanism the rest of the suite relies on.

(If swapping in a `FakeStore` turns out to be awkward because `_store` is a module global,
that itself is a finding — it suggests the store should be injected via FastAPI dependency
override rather than reassigned, which would make the conformance test cleaner.)

### 1b. The webhook (`POST /webhook/{channel}`) — component, FastAPI TestClient
Basic:
- No listing live → 409.
- Listing live, valid body → 202, `AcceptedResponse` with channel/listing_id/received_at.
- Channel comes from the **URL path**, not the body — POST same body to `/webhook/whatsapp`
  vs `/webhook/facebook` and assert the stored offer's `channel` differs accordingly.
- Omitted `offer_id` → server generates `<channel>_<hex>`.
- Omitted `timestamp` → server stamps `now(utc)`.

Edge:
- Unknown channel (`/webhook/telegram`) → 422 (enum coercion fails at path level).
- Missing required body field (`text`, `sender`) → 422.
- Empty `text` string — currently allowed (no min length). Decide whether an empty message
  *should* be rejected here or allowed through to extraction (where it becomes an empty
  profile). Document, then test the chosen contract.
- Very long `text` (e.g. 50k chars) → accepted; note it flows to extraction untrimmed.
- Duplicate `offer_id` submitted twice — current store **appends both**. Is dedup desired?
  Flag as open question.

### 1b-stress. Webhook under load (HTTP-level concurrency)
A script — `scripts/stress_test.py` — already does this: `--test concurrency` fires N parallel
requests (default 500, up to 100 in-flight) and counts 202 / 4xx / 5xx + latency; `--test
memory` floods large payloads to expose the unbounded store. Use and extend it rather than
rebuild:
- **Scale it up** as you asked: run with `--requests 5000 --concurrency 500` (and higher) →
  the pass condition is **zero 5xx and zero connection failures**; every request is either
  accepted (202) or cleanly rejected (4xx). A 5xx means the server faulted under load.
- **End-state integrity:** after a concurrency storm of K requests against a live listing,
  `GET /healthz` must report `offers_received == K` (minus any cleanly-rejected 4xx). This is
  the HTTP-level analogue of the store count assertion — it proves no offer was silently
  dropped under concurrent ingestion through the real uvicorn worker(s).
- **Note on what HTTP concurrency does *not* prove:** uvicorn here runs the app on a single
  event loop, so async handlers don't truly run in parallel — the in-process `ThreadPoolExecutor`
  store test (1a) is what actually races the lock. Run with multiple workers
  (`uvicorn --workers N`) only if you want to test that path, but note the in-memory store is
  **per-process**, so multi-worker counts won't reconcile — itself worth documenting as a
  scaling limitation.
- **Sustained throughput / latency:** record p99 latency at increasing concurrency to find
  where it degrades. Not a pass/fail gate — a characterization for the writeup.

These are `slow` + require a running server (or a TestClient-driven variant); keep them out of
the default `pytest` run.

### 1c. The sender (`scripts/send_offers.py`) — component against TestClient or live server
- Reads `data/fixtures.jsonl`, posts each line to the matching channel webhook.
- After a run: `GET /healthz` shows `offers_received == len(fixtures)`.
- Malformed fixture line → script reports it without aborting the whole batch (verify the
  failure mode; harden if it currently crashes).
- Run sender with **no listing posted** → every post 409s; script surfaces this clearly.

---

## Station 2 — Extraction (the LLM firewall)

This is where the bulk of effort goes. Split into **deterministic plumbing** (mock the
client — most tests) and **non-deterministic behavior** (real API — the LLM-probe track).

### 2a. Deterministic plumbing — unit, stubbed `ExtractionClient`
The engine's contract is independent of the model. Inject a fake client returning canned dicts.

Happy path:
- Well-formed raw dict → valid `TenantProfile`; all fields land as typed.
- `_prov` keys reshaped into nested `provenance: {field: Provenance}`; null `_prov` dropped.
- A null field with a null `_prov` → field absent from provenance, stays `None`.

Firewall / error mapping:
- Client raises `RuntimeError` → engine raises `ExtractionError` (wrapped, offer_id in msg).
- Client returns a dict that fails `TenantProfile.model_validate` (bad enum, wrong type) →
  `ExtractionError` with the raw response echoed.
- Client returns a dict missing a `_prov` key → reshaping tolerates it (`.pop(..., None)`).

**Where `RuntimeError` actually comes from (your question).** The engine's `try/except` only
catches `RuntimeError` from the client. Reading `ExtractionClient.extract_raw`, it raises
`RuntimeError` in exactly these cases — each gets its own test by stubbing the OpenAI client:
1. **Empty response body** — `response.choices[0].message.content` is falsy →
   `"OpenAI returned an empty response body."`
2. **Invalid JSON** — `json.loads` raises `JSONDecodeError` → wrapped as
   `"OpenAI returned invalid JSON: …"` (note: this is *not* retried — it fails fast).
3. **Non-retryable API error** — an `APIError` with `status_code < 500` (e.g. 400/401/403) →
   `"Non-retryable OpenAI error: …"`, raised immediately without retry.
4. **Retries exhausted** — after `max_retries` transient failures (`RateLimitError`,
   `APITimeoutError`, or `APIError` with `status_code >= 500`) →
   `"OpenAI extraction failed after N attempts. Last error: …"`.

All four must surface to the engine as `ExtractionError` with the offer_id attached. Also test
the **boundary**: a malformed-schema dict that *passes* JSON parsing but *fails*
`TenantProfile.model_validate` is caught by the engine's second `try/except` (the `Exception`
catch around `model_validate`), **not** the client — so it produces an `ExtractionError` via a
different path. Worth a test each so a refactor can't collapse the two.

Provenance auditing (the anti-hallucination guardrails already in code):
- Extracted `provenance` span that is **not a substring** of `offer.text` → a `WARNING` is
  logged listing the bad span. Assert on `caplog`. (This is the hallucinated-citation alarm.)
- A non-null screening field with **no** provenance entry → `WARNING` about uncited field.
- Clean extraction → no warnings.

### 2b. Schema / client wiring — unit

**Server-side API error matrix (stub `self._client.chat.completions.create`).**
You wanted broad coverage of API failure modes — enumerate them, since the retry/raise
behavior differs by class and status code:

| Stubbed failure | Expected behavior |
|-----------------|-------------------|
| `RateLimitError` ×2 then success | retries with backoff, returns on 3rd attempt |
| `APITimeoutError` ×2 then success | same — retried |
| `APIError` status 500 / 502 / 503 / 504 | retried up to `max_retries`, then `RuntimeError` |
| `APIError` status 400 (bad request) | **immediate** `RuntimeError`, no retry |
| `APIError` status 401 (bad key) / 403 | **immediate** `RuntimeError`, no retry |
| `APIError` status 404 (bad model) / 422 | **immediate** `RuntimeError`, no retry |
| `RateLimitError` every attempt | retries exhausted → `RuntimeError` naming last error |
| `APITimeoutError` every attempt | retries exhausted → `RuntimeError` |

Assertions for the matrix:
- **Retry count is honored** — patch `time.sleep` (so tests are instant) and assert
  `create` was called exactly `max_retries` times on the exhausted-retry rows, and exactly
  the expected (failures+1) on the recover-then-succeed rows.
- **Backoff schedule** — assert `time.sleep` was called with the expected growing delays
  (1.0 → 2.0 → … capped at 30.0), so a regression that drops exponential backoff is caught.
- **Status-code boundary** — explicitly test 499 vs 500 to pin the `< 500` cutoff.

**Response-content / value-type failures (the "invalid values" coverage you asked for).**
Distinct from transport errors — here `create` *succeeds* but the body is wrong. Stub the
returned `message.content` string:
- Empty / whitespace-only body → `RuntimeError` (empty-body branch).
- Non-JSON garbage (`"not json {"`) → `RuntimeError` (invalid-JSON branch), not retried.
- Valid JSON but **wrong types**: `budget_nis: "6500"` (string not int), `age: 28.5` (float),
  `has_pets: "yes"` (string not bool), `num_roommates: -1`, `move_in_date: "31/08/2026"`
  (non-ISO) → each must surface as `ExtractionError` at the engine's `model_validate` stage,
  never a silently-coerced or half-built `TenantProfile`. (Pydantic is the firewall; assert it
  actually rejects rather than coerces — e.g. confirm `"6500"`→int coercion is *not* silently
  accepted if strictness matters to you; if it *is* accepted, document that as intended.)
- Valid JSON, valid types, but **out-of-enum** value: `employment_status: "freelancer"`,
  `gender: "M"`, `preferred_language: "fr"` → `ExtractionError` (enum validation).
- **Extra unexpected keys** in the dict → reshaping ignores unknown `_prov` keys via `.pop`,
  but an unexpected top-level field flows into `model_validate`; assert the chosen behavior
  (Pydantic ignores extras by default unless the model forbids them — verify which).
- **Missing required keys** in the returned dict → confirm the engine still produces a clean
  `ExtractionError` rather than a `KeyError` leaking out.

**Schema-drift guard (unchanged):**
- The JSON schema sent to OpenAI has `additionalProperties: false` and lists every field +
  every `_prov` key in `required` (strict mode demands it). Assert `_PROVENANCE_FIELDS` and the
  schema's `properties` stay in sync with `TenantProfile`'s fields — a model field added
  without a schema entry should fail this test.

### 2c. LLM-probe track — real API (see dedicated section below)

---

## Station 3 — Scoring / ranking via vectors

Pure functions, zero I/O. This is the **provably-correct** part, so tests assert **exact
numbers**, not ranges. Two files: per-dimension (`vectors.py`) and the engine (`engine.py`).

### 3a. Per-dimension compatibility functions — unit, exhaustive
For **each** of the 7 dimensions, cover the full truth table. The shape repeats:
match → `1.0`, mismatch → `0.0`, unknown → `_UNKNOWN (0.05)`, no-criterion → `1.0`.

- **budget** (`_budget_compat`): `b ≥ rent` →1.0; `b` in `[floor, rent)` → exact linear
  interpolation value (compute by hand); `b < floor` →0.0; `budget=None` →1.0;
  `floor=None` →1.0 regardless of b. Boundary points `b==floor`, `b==rent`.
- **pets**: `pets_allowed=True` →1.0 always; forbidden + `has_pets=True` →0.0;
  forbidden + `False` →1.0; forbidden + `None` →0.05.
- **move_in**: deadline unset →1.0; date unset →1.0; on-deadline →1.0; assert the decay
  values at 2/7/30/60 days (the docstring table — pin them so the constant can't drift
  silently). Note: this function is symmetric in |days|, but `is_dealbreaker` only vetoes
  **late** move-in — test both directions and document the asymmetry.
- **employment**: not required →1.0; required + employed/self_employed →1.0;
  required + student/unemployed →0.0; required + None →0.05.
- **occupants**: no limit →1.0; `1+roommates ≤ max` →1.0; over →0.0; None →0.05.
  Edge: `num_roommates=0` (alone) counts as 1 occupant.
- **age**: no pref →1.0; inside `[min,max]` →1.0; outside → decay from nearest boundary
  (compute expected with the half-range formula); one-sided ranges (`min` only / `max` only);
  `age=None` →0.05; degenerate `min==max` (half_range clamped to ≥1).
- **gender**: no pref →1.0; match →1.0; mismatch →0.0; None →0.05.

### 3b. Vector builders, weights & normalization — unit

> **Why is `cosine_similarity` still here if we use dot-product?** Good catch — it's a
> **leftover**. The engine (`engine.py:64`) computes `dot(p,c) / dot(c,c)`, the dot-product
> ratio; it never calls `cosine_similarity`. The function is now referenced **only** by the
> old manual `scripts/test_vector.py` and the existing `tests/test_vectors.py`. So in the
> production scoring path it is **dead code**. Action items, not just a test:
> - Decide whether to **delete** `cosine_similarity` (and the stale script) or keep it as a
>   documented alternative. If kept, mark it clearly as not-in-the-scoring-path so a future
>   reader doesn't assume the engine uses it.
> - Either way, the engine's math is the dot-product ratio, and **3c tests assert that**, not
>   cosine. Don't write new engine tests against `cosine_similarity`.
> - If you keep it, its own unit tests stay (identical→1.0, orthogonal→0.0, zero-vector→0.0),
>   but labelled as testing a utility, not the scoring formula.

**Weights add up correctly per listing type (your request).** `_weights` zeroes any dimension
the landlord has no preference on, so the *active* weight set depends on the listing's criteria
shape. Build representative **listing archetypes** and assert the resulting weight vector:
- *Minimal listing* (defaults: pets allowed, no deadline, no employment req, no occupant cap,
  no age/gender pref) → only `budget_weight` is non-zero; all six other slots are 0.
- *Strict screener* (every constraint set) → all 7 slots non-zero, equal to the configured
  weights.
- *Pets-forbidden listing* → pets slot switches on (`pets_weight`); others follow their flags.
- *Age-only listing* (just `min_age`/`max_age`) → age slot on, rest off.
- For each archetype assert: (a) which slots are non-zero matches the criteria flags exactly,
  (b) the non-zero values equal the configured weights, (c) `criteria_to_vector` =
  `sqrt(weight)` on active dims and `0` on inactive. This is the "weights add up per listing
  type" check — it proves a landlord's stated priorities, and *only* those, drive the vector.
- Also assert the **denominator** the engine divides by (`dot(c,c) = Σ active weights`) equals
  the sum of active weights — so the 0–100 scale is normalized to the active dimensions only,
  not diluted by switched-off ones.

**Normalization across the 7 features, incl. ranges & gradients (your request).** Verify each
dimension contributes correctly *after* `sqrt(weight)` scaling, especially the non-binary ones:
- **Binary-ish dims** (pets, employment, occupants, gender): contribution is exactly
  `compat × sqrt(weight)` for compat ∈ {0.0, 0.05, 1.0}. Assert the scaled value directly.
- **Gradient dims** (budget interpolation, move-in decay, age decay): feed a sweep of inputs
  and assert the scaled contribution moves **monotonically** and matches the hand-computed
  curve — e.g. budget stepping floor→rent yields a linear rise; move-in at 2/7/30/60 days
  yields the decaying values; age stepping away from the band decays per the half-range
  formula. This proves the gradients normalize into [0,1]×sqrt(w) consistently and that one
  loud dimension can't exceed its weighted ceiling.
- **Cross-dimension balance:** two listings with the same total weight but split differently
  across dims → a profile that's perfect on the heavy dim and poor on the light dim should
  outscore the reverse. Confirms normalization respects relative weight, not just presence.

- `_scale`: applies `sqrt(weight)`. Verify a known weight vector scales as expected.
- `criteria_to_vector`: all dims 1.0 × sqrt(weight); zeroed dims are 0.
- `is_empty_profile`: all seven scoreable fields None →True; any one set →False.

### 3c. ScoringEngine — unit, exact scores
The engine uses **dot-product ratio** `dot(p,c)/dot(c,c)`, not raw cosine. Test the real math.
- `ideal_profile` vs `base_criteria` (from `conftest`) → **100.0**, qualification APPROVED.
- A profile missing one dimension → score drops by exactly that dim's weight share (hand-calc).
- **Dealbreaker multiplier (×0.01):** each of the four hard constraints, in isolation, drops
  an otherwise-perfect profile to ~1.0 and adds a `dealbreaker` RuleHit:
  - pets when forbidden,
  - budget below private floor,
  - `move_in_strict` + late date,
  - occupants over limit.
- **Empty-profile multiplier (×0.1):** an all-None profile → score = base ×0.1, with an
  `empty_profile` RuleHit. (Distinct from a dealbreaker — test they don't both fire.)
- **Threshold mapping:** construct profiles that land just above `approved_threshold`, between
  the two thresholds (REVIEW), and below `rejected_threshold` (REJECTED). Test boundary
  equality (`score == approved_threshold` → APPROVED, since `>=`).
- **RuleHit breakdown:** `passed` is True at compat 1.0, False at 0.0, None for partial/unknown;
  `points_earned ≤ points_possible`; reason strings present for every dim.
- **Determinism:** same profile scored twice → byte-identical `ScoreResult`.

---

## Cross-component / integration tests

Verify the **contracts between stations** hold when wired together.

- **Listing → webhook gate:** fresh app, no listing → webhook 409; POST listing → webhook 202.
  (Already partly covered; assert the transition explicitly.)
- **Webhook → store → retrieval:** POST 3 offers across 3 channels → `GET /offers` returns
  all 3 with correct per-offer channel; `GET /offers/{id}` round-trips each.
- **Store → extraction → scoring (`POST /pipeline/run`)** with Station 2 **mocked**:
  - known offer + canned profile → result cached with profile, score, and the `vectors`
    block (`landlord` + `tenant` arrays, `dims` labels). Assert vectors match what
    `profile_to_vector`/`criteria_to_vector` produce for that profile+criteria.
  - extraction raises `ExtractionError` → result cached with `profile=None`, `error` set,
    response `status="error"` (pipeline must not 500 on a bad offer).
  - `pipeline/run` for unknown offer_id → 404; with no listing → 409.
- **Ranking aggregation (`GET /pipeline/results`):** run several offers → results sorted by
  `score` desc; **errored offers sort last** (their `score is None` → key `-(-1)`); ties broken
  by earlier `received_at`. Build a deliberate tie to prove the timestamp tiebreak.
- **Lifecycle reset:** run pipeline, then POST a new listing → `GET /pipeline/results` empty
  and `GET /offers` empty (the wipe propagates).
- **Live extraction integration** (`@pytest.mark.live`): real OpenAI on a couple of
  unambiguous fixtures → profile fields match the obvious ground truth; provenance spans are
  real substrings. Keep small (cost); the heavy LLM work is the probe track below.

---

## Complex ranking scenarios (the interesting part)

Goal: feed **curated sets** of tenant profiles against one listing and assert the **resulting
order and qualification bands**, not just individual scores. These are deterministic (build
`TenantProfile`s directly, skip the LLM) so they can assert exact rankings.

Design each scenario as: **one `ScoringCriteria` + a list of labelled profiles + the expected
ranked order + expected qualification per profile.** Suggested scenarios:

1. **Clean separation:** an ideal, a mediocre, and a dealbroken profile → order is
   ideal > mediocre > dealbroken; bands APPROVED / REVIEW / REJECTED respectively.
2. **Dealbreaker outranked by completeness:** a near-perfect tenant who has a pet (forbidden)
   must rank **below** a so-so tenant with no violations — proves the ×0.01 penalty dominates
   raw similarity. This is the headline correctness property.
3. **Unknowns vs. explicit misses:** tenant A leaves employment unstated (0.05), tenant B
   states "unemployed" (0.0) under `employment_required` → A ranks above B. Proves
   "unknown is penalized but less than an explicit fail."
4. **Budget band gradient:** five tenants stating budgets stepping from below-floor → at-floor
   → mid-band → at-asking → above-asking, all else equal → strictly monotonic ranking;
   below-floor one is a dealbreaker (bottom).
5. **Weight sensitivity:** same profile set scored under two criteria that differ only in
   weights (e.g. pets-heavy vs. age-heavy landlord) → the ranking **reorders**. Proves weights
   actually steer the outcome and aren't cosmetic.
6. **Zeroed-criterion neutrality:** a landlord who sets no age/gender/pet preference → those
   dims drop out; two tenants differing only on those fields tie. Proves "no preference =
   no free points and no penalty."
7. **Empty / non-applicant in the mix:** a price-question message (all-None profile) ranks at
   the bottom via the ×0.1 empty penalty, above hard dealbreakers but below any real applicant.
8. **All-dealbreaker pool:** every tenant violates something → all REJECTED; order still
   stable and deterministic (no crash, no NaN).
9. **Tie-break realism:** two identical profiles, different `received_at` → earlier wins
   (mirrors the aggregation rule, tested here at the data level).

For each scenario, record the **expected score vector** alongside the order so a regression
shows *which* tenant moved, not just "order changed." A small table (profile → score → band)
checked into the test as the oracle.

---

## LLM hallucination & consistency probing (Station 2, real API)

The LLM is the only place the system can lie. We can't assert exact output, so we assert
**properties** and **stability across repeats**. Run under `@pytest.mark.live` + `slow`,
gated on `OPENAI_API_KEY`, kept out of the default `pytest` run.

### Method
- Build a curated **probe set** of messages, each tagged with what *should* be extractable
  and what *should stay null*. Distinct from `fixtures.jsonl` (those are realistic; probes are
  adversarial / deliberately obscure).
- For consistency: call `extract` **N times** (e.g. 5–10) on the same message and aggregate.
  Note `temperature=0` makes the API near-deterministic but **not guaranteed** identical —
  measure the actual variance rather than assuming zero.

### Property checks (single call, any message)
These should hold for *every* extraction regardless of content — they're invariants:
- Every non-null `provenance.source_span` is a **real substring** of the input (the engine
  already warns; here we assert it's empty for clean inputs and present for planted bad ones).
- No field is invented when its information is absent (see null-discipline probes below).
- Output always validates against `TenantProfile` (the firewall never leaks a bad object).

### Null-discipline probes (the core hallucination test)
Messages engineered so a field is **genuinely unstated**, then assert the field is `None`:
- "Looking for the apartment, is it still available?" → all screening fields null
  (non-applicant; prompt rule 9).
- A message stating only move-in date → budget/age/gender/pets/employment all null.
- Mentions a dog "my friend has a dog" (not the tenant's) → `has_pets` should be null/false,
  **not** true. (Tests attribution, a classic hallucination trap.)
- "I'm flexible on everything" → no concrete fields populated.

### Ambiguity & obscurity probes (consistency under N repeats)
The user's request: feed **unclear/obscure features** and measure how stable the profile is.
For each, run N times and report the **distribution** of each field; flag any field whose
value flips across runs as a stability risk.
- **Implied gender via Hebrew grammar:** "בן 25" (implies male), "2 שותפות" (implies female),
  vs. a deliberately ungendered Hebrew sentence → does gender stay consistent? Does it
  correctly stay `null` when truly ambiguous?
- **Per-person vs. total budget:** "3500 per person, 3 of us" → budget should resolve to 10500
  every time (prompt rule 2 + example). Measure how often it gets this right.
- **Relative dates:** "next month", "after the holidays", "beginning of summer" → does
  `move_in_date` land on a stable ISO date, or wander? (Prompt pins "today = 2026-06-02" —
  test that relative anchoring is consistent.)
- **Slang / code-switching:** mixed Hebrew-English with slang ("יאללה", "ASAP בלי בעלי חיים")
  → fields stable across runs?
- **Contradictory message:** "no pets, well I have a cat" → does it pick one answer
  consistently, and is the choice defensible? (Document expected behavior; this is a
  judgement call, not a hard pass/fail.)
- **Numeric noise:** an age that could be read as a budget or a date ("I'm 30, looking at the
  3000 area, moving the 1st") → does the LLM keep age=30, budget=3000 separated each run?

### Consistency metrics to report (per probe, over N runs)
- **Field stability rate:** fraction of runs where each field equals the modal value.
- **Hallucination count:** runs where a should-be-null field came back non-null.
- **Provenance validity rate:** fraction of non-null fields whose span is a real substring.
- **Schema-failure rate:** runs that raised `ExtractionError` (should be ~0).

These are **reported and thresholded loosely** (e.g. "≥ 90% stability on unambiguous fields")
rather than asserted exactly — the point is to surface where the prompt is fragile and feed
prompt iteration, not to make CI flaky.

---

## Business / cost analysis of the operation

The dominant running cost is **OpenAI API calls** — exactly one extraction call per offer
(Station 2; everything else is local compute). So cost scales with **# offers**, and each
call's price is set by **tokens in + tokens out**. This section sizes that cost and compares
candidate models, so the model choice (`gpt-4.1-mini` today) is a justified decision, not a default.

### Cost driver model

```
total_cost ≈ Σ_offers ( input_tokens × price_in  +  output_tokens × price_out )
```

- **Listings** are effectively free (local store, no API call) — they set the *denominator*:
  offers-per-listing is the funnel that turns a listing into API spend.
- **Offers** = the call count. One offer → one extraction call (plus retries on transient
  errors, which are rare but should be counted, not assumed zero).
- **Tokens per call** = system prompt (large, fixed — the prompt + 4 few-shot examples) +
  the offer text (small, variable) for **input**, and the fixed JSON schema (~20 keys) for
  **output**. The fixed system prompt dominates input tokens, so **prompt caching** is the
  single biggest lever and should be measured.

### What to measure (instrument, don't estimate)
Capture real numbers from `response.usage` on every extraction call:
- `prompt_tokens`, `completion_tokens`, and (if available) `cached_tokens` per offer.
- Mean / p50 / p95 tokens per call across the fixture + probe sets (offer text varies;
  Hebrew tends to cost more tokens per character than English — worth reporting the split).
- Retry rate (extra calls per offer) observed during stress/integration runs.
- Derive **cost per offer** and **cost per listing** = (cost/offer) × (avg offers/listing).

Persist these to a small report (reuse the existing `extraction_results.jsonl` pattern) so the
numbers are reproducible rather than back-of-envelope.

### Unit-economics table to produce
A table parameterized by volume, e.g.:

| Scale | Listings/mo | Offers/listing | Offers/mo | Tokens/offer (in/out) | $/offer | $/mo |
|-------|-------------|----------------|-----------|-----------------------|---------|------|
| Hobby | 10 | 20 | 200 | measured | … | … |
| SMB | 500 | 30 | 15,000 | measured | … | … |
| Scale | 10,000 | 40 | 400,000 | measured | … | … |

Fill `Tokens/offer` and `$/offer` from instrumentation, then project the two right columns.
Show the same table **with and without prompt caching** to quantify that lever.

### Model comparison
Run the **same probe + fixture set** through several models and compare on three axes
together — cost alone is meaningless without the accuracy it buys:

| Model | $/1M in | $/1M out | $/offer (measured) | Extraction accuracy* | Latency p50 | Notes |
|-------|---------|----------|--------------------|----------------------|-------------|-------|
| `gpt-4.1-mini` (current) | … | … | … | … | … | strict JSON schema OK |
| `gpt-4.1-nano` | … | … | … | … | … | cheapest; does accuracy hold? |
| `gpt-4.1` | … | … | … | … | … | ceiling for accuracy |
| (one non-OpenAI option) | … | … | … | … | … | only if it supports strict structured output |

\* **Accuracy** = the field-level stability + hallucination metrics from the LLM-probe track
above (field stability rate, hallucination count, provenance validity). Reuse that harness —
the cost analysis and the consistency analysis share the same runs.

**Deliverable:** a one-paragraph recommendation — *"`gpt-4.1-mini` costs $X/offer at Y%
field accuracy; nano is Z% cheaper but loses N points on Hebrew gender inference, so mini is
the right tradeoff until volume exceeds …"* — backed by the two tables. Keep prices in a
single dated constants block (they change), and note the date, since pricing is point-in-time.

---

## Suggested order of work

1. Station 3 unit tests (3a→3b→3c) — pure, fast, highest certainty; lock the math first.
2. Station 0 + Station 1 component tests — the deterministic HTTP/store contracts.
3. Station 2 deterministic plumbing (2a/2b) with a stubbed client — the firewall behavior.
4. Cross-component integration with Station 2 mocked — prove the wiring + ranking aggregation.
5. Complex ranking scenarios — the correctness showcase.
6. Live extraction integration (small).
7. LLM hallucination/consistency probes — last, since they inform prompt tuning, not CI gating.
