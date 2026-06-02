"""
Quick manual test for the extraction pipeline.

Usage (venv active):
    python scripts/test_extraction.py
"""

import json
import sys
from datetime import datetime

sys.path.insert(0, "src")

from rentflow.extraction.client import ExtractionClient
from rentflow.extraction.engine import ExtractionEngine
from rentflow.extraction.prompts import SYSTEM_PROMPT
from rentflow.offer.models import Channel, RawOffer


def main() -> None:
    print("Enter the tenant message (blank line to finish):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    text = "\n".join(lines).strip()
    if not text:
        print("No input. Exiting.")
        return

    import os
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env", override=True)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        return

    client = ExtractionClient(api_key=api_key)

    offer = RawOffer(
        offer_id="test-001",
        channel=Channel.WHATSAPP,
        sender="tester",
        timestamp=datetime.now(),
        text=text,
    )

    print("\n--- RAW LLM OUTPUT ---")
    raw = client.extract_raw(system_prompt=SYSTEM_PROMPT, user_text=text)
    print(json.dumps(raw, indent=2, ensure_ascii=False))

    print("\n--- PARSED TenantProfile ---")
    engine = ExtractionEngine(client=client)
    result = engine.extract(offer)
    print(json.dumps(result.profile.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
