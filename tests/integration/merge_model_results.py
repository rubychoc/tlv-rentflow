"""
Merges per-model results from run_model_comparison.sh into one comparison file.

Reads from:  tests/integration/model_results/<model>/{cost.json, extraction.json, probes.json}
Writes to:   tests/integration/model_comparison.json

Run automatically by run_model_comparison.sh, or manually:
    python tests/integration/merge_model_results.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "model_results"
OUTPUT_FILE = Path(__file__).parent / "model_comparison.json"


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"error": str(e)}


def _summarise_cost(cost: dict | None, model: str) -> dict:
    if not cost:
        return {"available": False}
    # New format: cost is nested under per_model[model]
    per_model = cost.get("per_model", {})
    model_data = per_model.get(model, cost)  # fall back to flat format for old files
    stats = model_data.get("stats", {})
    return {
        "available": True,
        "fixtures_measured": model_data.get("successful_count"),
        "prompt_tokens": stats.get("prompt_tokens", {}),
        "completion_tokens_mean": stats.get("completion_tokens", {}).get("mean"),
        "cached_tokens_mean": stats.get("cached_tokens", {}).get("mean"),
        "cache_hit_rate_pct": stats.get("cached_tokens", {}).get("hit_rate_pct"),
        "cost_per_offer": model_data.get("cost_per_offer", {}),
    }


def _summarise_extraction(report: dict | None) -> dict:
    """Parse pytest-json-report output for extraction live tests."""
    if not report:
        return {"available": False}
    tests = report.get("tests", [])
    passed = sum(1 for t in tests if t.get("outcome") == "passed")
    failed = sum(1 for t in tests if t.get("outcome") == "failed")
    total = len(tests)
    failures = [
        {
            "test": t.get("nodeid", "").split("::")[-1],
            "message": t.get("call", {}).get("longrepr", "")[:300],
        }
        for t in tests if t.get("outcome") == "failed"
    ]
    return {
        "available": True,
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": f"{passed}/{total}",
        "failures": failures,
    }


def _summarise_probes(probes: list | None) -> dict:
    if not probes:
        return {"available": False}

    total = len(probes)
    passed = sum(1 for p in probes if p.get("passed"))
    failed = total - passed

    # Group by kind
    by_kind: dict[str, dict] = {}
    for p in probes:
        kind = p.get("kind", "unknown")
        if kind not in by_kind:
            by_kind[kind] = {"total": 0, "passed": 0, "failed": 0, "failures": []}
        by_kind[kind]["total"] += 1
        if p.get("passed"):
            by_kind[kind]["passed"] += 1
        else:
            by_kind[kind]["failed"] += 1
            by_kind[kind]["failures"].append({
                "test": p.get("test"),
                "field": p.get("field"),
                "expected": p.get("expected"),
                "actual": p.get("actual") or p.get("run_results"),
                "distribution": p.get("distribution"),
            })

    return {
        "available": True,
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": f"{passed}/{total}",
        "by_kind": by_kind,
        "all_results": [
            {
                "test": p.get("test"),
                "message": p.get("message"),
                "field": p.get("field"),
                "kind": p.get("kind"),
                "expected": p.get("expected"),
                "actual": p.get("actual"),
                "n_runs": p.get("n_runs"),
                "correct_runs": p.get("correct_runs"),
                "distribution": p.get("distribution"),
                "passed": p.get("passed"),
            }
            for p in probes
        ],
    }


def merge() -> dict:
    if not RESULTS_DIR.exists():
        print(f"No model_results directory found at {RESULTS_DIR}")
        return {}

    models = sorted(d.name for d in RESULTS_DIR.iterdir() if d.is_dir())
    if not models:
        print("No model result directories found.")
        return {}

    comparison: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models_compared": models,
        "results": {},
    }

    for model in models:
        model_dir = RESULTS_DIR / model
        cost = _load_json(model_dir / "cost.json")
        extraction = _load_json(model_dir / "extraction.json")
        probes = _load_json(model_dir / "probes.json")

        comparison["results"][model] = {
            "cost": _summarise_cost(cost, model),
            "extraction_accuracy": _summarise_extraction(extraction),
            "probe_accuracy": _summarise_probes(probes),
        }

    # Side-by-side summary table (printed, not stored)
    print("\n" + "=" * 72)
    print("Model Comparison Summary")
    print("=" * 72)
    fmt = f"  {'Model':<22}  {'$/offer (no cache)':>20}  {'Extraction':>12}  {'Probes':>10}"
    print(fmt)
    print(f"  {'-'*22}  {'-'*20}  {'-'*12}  {'-'*10}")
    for model, r in comparison["results"].items():
        cost_str = "n/a"
        if r["cost"].get("available"):
            mini_cost = r["cost"]["cost_per_offer"].get(model, {}).get("no_cache")
            if mini_cost:
                cost_str = f"${mini_cost:.6f}"

        extr = r["extraction_accuracy"]
        extr_str = extr.get("pass_rate", "n/a") if extr.get("available") else "n/a"

        probe = r["probe_accuracy"]
        probe_str = probe.get("pass_rate", "n/a") if probe.get("available") else "n/a"

        print(f"  {model:<22}  {cost_str:>20}  {extr_str:>12}  {probe_str:>10}")
    print("=" * 72)

    return comparison


if __name__ == "__main__":
    result = merge()
    if result:
        OUTPUT_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nFull comparison written to: {OUTPUT_FILE}")
