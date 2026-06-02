"""
Quick demo: runs the extraction engine on a few fixture messages and prints
the resulting TenantProfiles.

How to run:
  export OPENAI_API_KEY=sk-...
  python scripts/extract_demo.py

Optional: pick specific fixture lines by index (0-based):
  python scripts/extract_demo.py --lines 0 4 12
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for editable installs that haven't been reinstalled
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rentflow.extraction.engine import ExtractionEngine, ExtractionError
from rentflow.offer.models import Channel, RawOffer

FIXTURES = Path(__file__).parent.parent / "data" / "fixtures.jsonl"


def load_fixtures(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_offer(raw: dict, idx: int) -> RawOffer:
    channel = raw["channel"]
    return RawOffer(
        offer_id=f"demo_{idx:02d}",
        channel=Channel(channel),
        sender=raw["sender"],
        timestamp="2026-06-01T10:00:00",
        text=raw["text"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lines", type=int, nargs="+",
        help="Which fixture lines to process (0-based). Default: first 3.",
    )
    args = parser.parse_args()

    fixtures = load_fixtures(FIXTURES)
    indices = args.lines if args.lines else [0, 1, 2]

    try:
        engine = ExtractionEngine.from_env()
    except EnvironmentError as exc:
        print(exc)
        sys.exit(1)

    for idx in indices:
        if idx >= len(fixtures):
            print(f"[SKIP] No fixture at index {idx} (only {len(fixtures)} loaded).")
            continue

        raw = fixtures[idx]
        offer = make_offer(raw, idx)

        print(f"\n{'=' * 60}")
        print(f"Fixture [{idx}] — channel: {offer.channel.value}")
        print(f"Text: {offer.text}")
        print(f"{'=' * 60}")

        try:
            result = engine.extract(offer)
            profile_json = result.profile.model_dump(mode="json")
            print(json.dumps(profile_json, ensure_ascii=False, indent=2))
        except ExtractionError as exc:
            print(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()
