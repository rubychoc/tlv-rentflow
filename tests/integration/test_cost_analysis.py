"""
Cost analysis for the extraction pipeline — real OpenAI API calls required.

Measures actual token usage across all fixture messages, then projects cost at
three volume tiers (Hobby / SMB / Scale) with and without prompt caching.
Also extrapolates what the same token counts would cost on cheaper/pricier models,
so the current gpt-4.1-mini choice is a backed decision rather than a default.

Run explicitly (excluded from the default pytest run):

    pytest tests/integration/test_cost_analysis.py -m "live and slow" -s -v

Requirements:
    - OPENAI_API_KEY set in .env
    - venv active with pip install -e .

Writes cost_analysis_results.jsonl to the project root on every run so
numbers are reproducible rather than back-of-envelope.
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest
from dotenv import load_dotenv

from rentflow.extraction.client import TokenUsage
from rentflow.extraction.engine import ExtractionEngine, ExtractionError
from rentflow.offer.models import Channel, RawOffer

load_dotenv(Path(__file__).parents[2] / ".env", override=True)

pytestmark = [pytest.mark.live, pytest.mark.slow]

_FIXTURES_PATH = Path(__file__).parents[2] / "data" / "fixtures.jsonl"
_REPORT_PATH = Path(__file__).parent / "cost_analysis_results.json"

# ---------------------------------------------------------------------------
# Pricing constants — verify against platform.openai.com before publishing.
# Last verified: 2026-06-02.
# ---------------------------------------------------------------------------
_PRICING: dict[str, dict[str, float]] = {
    "gpt-4.1-mini": {
        "input_per_1m":        0.40,
        "cached_input_per_1m": 0.10,
        "output_per_1m":       1.60,
    },
    "gpt-4.1-nano": {
        "input_per_1m":        0.10,
        "cached_input_per_1m": 0.025,
        "output_per_1m":       0.40,
    },
    "gpt-4.1": {
        "input_per_1m":        2.00,
        "cached_input_per_1m": 0.50,
        "output_per_1m":       8.00,
    },
}

# Volume scenarios: (label, listings/mo, avg offers/listing)
_VOLUME_TIERS = [
    ("Hobby", 10,    20),
    ("SMB",   500,   30),
    ("Scale", 10_000, 40),
]

# Models to measure. Override with EXTRACTION_MODEL env var to run a single model.
_ALL_MODELS = list(_PRICING.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class CallRecord:
    offer_id: str
    text_len_chars: int
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    model: str
    error: str | None

    @property
    def uncached_prompt_tokens(self) -> int:
        return self.prompt_tokens - self.cached_tokens


def _load_fixtures() -> list[dict]:
    fixtures = []
    with _FIXTURES_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                fixtures.append(json.loads(line))
    return fixtures


def _make_offer(raw: dict, idx: int) -> RawOffer:
    from datetime import datetime, timezone
    return RawOffer(
        offer_id=f"cost_{idx:03d}",
        channel=Channel(raw["channel"]),
        sender=raw["sender"],
        timestamp=datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc),
        text=raw["text"],
    )


def _cost_per_offer(usage: TokenUsage, model: str, with_cache: bool) -> float:
    p = _PRICING[model]
    if with_cache:
        input_cost = (
            usage.cached_tokens * p["cached_input_per_1m"]
            + usage.uncached_prompt_tokens * p["input_per_1m"]
        ) / 1_000_000
    else:
        input_cost = usage.prompt_tokens * p["input_per_1m"] / 1_000_000
    output_cost = usage.completion_tokens * p["output_per_1m"] / 1_000_000
    return input_cost + output_cost


# ---------------------------------------------------------------------------
# Module-scoped fixture: run all offers, collect records
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def records() -> dict[str, list[CallRecord]]:
    """Returns {model_name: [CallRecord, ...]} for every model in _ALL_MODELS."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set — skipping cost analysis")

    # If EXTRACTION_MODEL is set, measure only that model (used by run_model_comparison.sh)
    single_model = os.environ.get("EXTRACTION_MODEL")
    models_to_run = [single_model] if single_model else _ALL_MODELS

    fixtures = _load_fixtures()
    all_records: dict[str, list[CallRecord]] = {}

    for model in models_to_run:
        print(f"\n  Measuring model: {model} ({len(fixtures)} fixtures)...")
        engine = ExtractionEngine(
            client=__import__("rentflow.extraction.client", fromlist=["ExtractionClient"])
            .ExtractionClient(api_key=api_key, model=model)
        )
        results: list[CallRecord] = []

        for idx, raw in enumerate(fixtures):
            offer = _make_offer(raw, idx)
            error_msg: str | None = None
            usage: TokenUsage | None = None

            try:
                engine.extract(offer)
                usage = engine._client.last_usage
            except ExtractionError as exc:
                error_msg = str(exc)

            if usage is not None:
                results.append(CallRecord(
                    offer_id=offer.offer_id,
                    text_len_chars=len(offer.text),
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    cached_tokens=usage.cached_tokens,
                    model=model,
                    error=error_msg,
                ))
            else:
                results.append(CallRecord(
                    offer_id=offer.offer_id,
                    text_len_chars=len(offer.text),
                    prompt_tokens=0, completion_tokens=0, cached_tokens=0,
                    model=model,
                    error=error_msg or "no usage returned",
                ))

        all_records[model] = results

    return all_records


# ---------------------------------------------------------------------------
# Sanity assertions — these are the pass/fail gates
# ---------------------------------------------------------------------------

class TestTokenSanity:
    def test_all_fixtures_processed(self, records):
        """Every fixture produced a record for every model — none silently dropped."""
        n_fixtures = len(_load_fixtures())
        for model, recs in records.items():
            assert len(recs) == n_fixtures, (
                f"Model {model}: expected {n_fixtures} records, got {len(recs)}"
            )

    def test_no_extraction_errors(self, records):
        """All fixtures must extract successfully on every model."""
        all_failed = []
        for model, recs in records.items():
            failed = [r for r in recs if r.error]
            for r in failed:
                all_failed.append(f"  [{model}] {r.offer_id}: {r.error}")
        assert not all_failed, (
            f"{len(all_failed)} extraction failures:\n" + "\n".join(all_failed)
        )

    def test_prompt_tokens_p95_under_limit(self, records):
        """p95 prompt tokens must stay below 3000 on every model."""
        for model, recs in records.items():
            successful = [r for r in recs if not r.error]
            if not successful:
                continue
            p95 = sorted(r.prompt_tokens for r in successful)[int(len(successful) * 0.95)]
            assert p95 < 3000, (
                f"[{model}] p95 prompt_tokens={p95} exceeds 3000"
            )

    def test_completion_tokens_p95_under_limit(self, records):
        """p95 completion tokens must stay below 600 on every model."""
        for model, recs in records.items():
            successful = [r for r in recs if not r.error]
            if not successful:
                continue
            p95 = sorted(r.completion_tokens for r in successful)[int(len(successful) * 0.95)]
            assert p95 < 600, (
                f"[{model}] p95 completion_tokens={p95} exceeds 600"
            )

    def test_cost_per_offer_under_cent(self, records):
        """p95 cost per offer must be under $0.01 for gpt-4.1-mini (cheapest baseline)."""
        mini_records = records.get("gpt-4.1-mini", [])
        successful = [r for r in mini_records if not r.error]
        if not successful:
            pytest.skip("No gpt-4.1-mini records to check")
        costs = [_cost_per_offer(
            TokenUsage(r.prompt_tokens, r.completion_tokens, r.cached_tokens, r.model),
            r.model, with_cache=False,
        ) for r in successful]
        p95_cost = sorted(costs)[int(len(costs) * 0.95)]
        assert p95_cost < 0.01, f"p95 cost/offer on mini = ${p95_cost:.6f}, exceeds $0.01"


# ---------------------------------------------------------------------------
# Report generation — prints and persists the unit-economics table
# ---------------------------------------------------------------------------

class TestCostReport:
    def test_generate_and_print_report(self, records):
        """
        Prints the full unit-economics table for every model measured and saves
        cost_analysis_results.json to the integration folder.
        """
        if not records:
            pytest.skip("No records to report on")

        print("\n")
        print("=" * 72)
        print("TLV-RentFlow — Extraction Cost Analysis")
        print(f"Models measured  : {', '.join(records.keys())}")
        print(f"Run at           : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        print("=" * 72)

        per_model_stats: dict = {}

        for model, recs in records.items():
            successful = [r for r in recs if not r.error]
            if not successful:
                continue

            pt = [r.prompt_tokens for r in successful]
            ct = [r.completion_tokens for r in successful]
            cached = [r.cached_tokens for r in successful]
            text_lens = [r.text_len_chars for r in successful]

            mean_pt     = statistics.mean(pt)
            p50_pt      = statistics.median(pt)
            p95_pt      = sorted(pt)[int(len(pt) * 0.95)]
            mean_ct     = statistics.mean(ct)
            mean_cached = statistics.mean(cached)
            mean_text   = statistics.mean(text_lens)

            avg = TokenUsage(int(mean_pt), int(mean_ct), int(mean_cached), model)

            print(f"\n── {model} — per-call token stats ({'measured' if model in _PRICING else 'extrapolated'}) ──")
            print(f"  Fixtures measured     : {len(successful)} / {len(recs)}")
            print(f"  Avg input text length : {mean_text:.0f} chars")
            print(f"  Prompt tokens  mean   : {mean_pt:.0f}")
            print(f"  Prompt tokens  p50    : {p50_pt:.0f}")
            print(f"  Prompt tokens  p95    : {p95_pt:.0f}")
            print(f"  Completion tkns mean  : {mean_ct:.0f}")
            print(f"  Cached tokens  mean   : {mean_cached:.0f}  "
                  f"(cache hit rate: {mean_cached/mean_pt*100:.1f}%)")

            print(f"\n  Cost per offer ({model}):")
            print(f"    {'Priced as':<20}  {'$/offer (no cache)':>20}  {'$/offer (cached)':>18}")
            print(f"    {'-'*20}  {'-'*20}  {'-'*18}")
            for priced_as in _PRICING:
                u = TokenUsage(int(mean_pt), int(mean_ct), int(mean_cached), priced_as)
                nc = _cost_per_offer(u, priced_as, with_cache=False)
                c  = _cost_per_offer(u, priced_as, with_cache=True)
                marker = " ◀" if priced_as == model else ""
                print(f"    {priced_as:<20}  ${nc:>19.6f}  ${c:>17.6f}{marker}")

            print(f"\n  Unit economics — {model} — projected monthly cost:")
            cost_nc = _cost_per_offer(avg, model, with_cache=False)
            cost_c  = _cost_per_offer(avg, model, with_cache=True)
            print(f"  {'Tier':<8}  {'Listings':>9}  {'Off/lst':>7}  {'Offers/mo':>10}  "
                  f"{'$/mo (no cache)':>16}  {'$/mo (cached)':>14}")
            print(f"  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*10}  {'-'*16}  {'-'*14}")
            for label, listings, opl in _VOLUME_TIERS:
                offers_mo = listings * opl
                print(f"  {label:<8}  {listings:>9,}  {opl:>7}  {offers_mo:>10,}  "
                      f"${offers_mo*cost_nc:>15,.2f}  ${offers_mo*cost_c:>13,.2f}")
            saving_pct = (cost_nc - cost_c) / cost_nc * 100 if cost_nc else 0
            print(f"\n  Prompt caching saves {saving_pct:.1f}% per call.")

            per_model_stats[model] = {
                "fixture_count": len(recs),
                "successful_count": len(successful),
                "stats": {
                    "prompt_tokens": {"mean": mean_pt, "p50": p50_pt, "p95": p95_pt},
                    "completion_tokens": {"mean": mean_ct},
                    "cached_tokens": {"mean": mean_cached, "hit_rate_pct": mean_cached / mean_pt * 100},
                    "text_len_chars": {"mean": mean_text},
                },
                "cost_per_offer": {
                    priced_as: {
                        "no_cache": _cost_per_offer(
                            TokenUsage(int(mean_pt), int(mean_ct), int(mean_cached), priced_as),
                            priced_as, with_cache=False),
                        "with_cache": _cost_per_offer(
                            TokenUsage(int(mean_pt), int(mean_ct), int(mean_cached), priced_as),
                            priced_as, with_cache=True),
                    }
                    for priced_as in _PRICING
                },
                "per_call_records": [asdict(r) for r in recs],
            }

        print("\n── Pricing constants used (verify at platform.openai.com) ──────────")
        for model, p in _PRICING.items():
            print(f"  {model:<20}  in=${p['input_per_1m']:.2f}/1M  "
                  f"cached=${p['cached_input_per_1m']:.3f}/1M  out=${p['output_per_1m']:.2f}/1M")
        print("=" * 72)

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "models_measured": list(records.keys()),
            "pricing_date": "2026-06-02",
            "per_model": per_model_stats,
        }
        _REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  Report written to: {_REPORT_PATH}")

        assert len(per_model_stats) > 0
        # Verify at least one model has a positive cost
        for model_stats in per_model_stats.values():
            assert model_stats["cost_per_offer"]["gpt-4.1-mini"]["no_cache"] > 0
            break
