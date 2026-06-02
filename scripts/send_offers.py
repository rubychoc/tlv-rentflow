"""
Sender script — simulates the "flood" of incoming tenant messages.

What this does:
  Reads every message from data/fixtures.jsonl and POSTs each one to the
  running webhook server, exactly as WhatsApp / Facebook / Yad2 would if they
  were configured to deliver messages to this server.

  This is "Level 2" of the simulation: the webhook receives and processes the
  message identically whether this script rang the doorbell or Meta did.

How to run (from the project root, with the server already running):
  PYTHONPATH=src .venv/bin/python scripts/send_offers.py

Optional flags:
  --delay 0.5     Pause 0.5 seconds between messages (simulates a trickle).
  --delay 0       Fire all messages as fast as possible (simulates the flood).
  --host          Base URL of the server (default: http://localhost:8000).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

FIXTURES_PATH = Path(__file__).parent.parent / "data" / "fixtures.jsonl"


def load_fixtures(path: Path) -> list[dict]:
    messages = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] Line {lineno}: bad JSON — {exc}", file=sys.stderr)
    return messages


def send_all(host: str, delay: float) -> None:
    fixtures = load_fixtures(FIXTURES_PATH)
    print(f"Loaded {len(fixtures)} messages from {FIXTURES_PATH.name}")
    print(f"Sending to {host}  (delay={delay}s between messages)\n")

    ok = 0
    failed = 0

    with httpx.Client(base_url=host, timeout=10.0) as client:
        # Confirm the server is up before sending anything.
        try:
            health = client.get("/healthz")
            health.raise_for_status()
            print(f"Server is healthy: {health.json()}\n")
        except Exception as exc:
            print(f"[ERROR] Cannot reach {host}/healthz — is the server running?\n  {exc}")
            sys.exit(1)

        for i, msg in enumerate(fixtures, start=1):
            channel = msg.get("channel", "unknown")
            sender  = msg.get("sender", "")
            text    = msg.get("text", "")
            preview = text[:60] + ("…" if len(text) > 60 else "")

            payload = {"sender": sender, "text": text}

            try:
                resp = client.post(f"/webhook/{channel}", json=payload)
                if resp.status_code == 202:
                    data = resp.json()
                    print(f"[{i:02d}] OK  {channel:<10}  {data['offer_id']}  '{preview}'")
                    ok += 1
                else:
                    print(f"[{i:02d}] FAIL  {channel:<10}  HTTP {resp.status_code}  '{preview}'")
                    print(f"       detail: {resp.text[:200]}")
                    failed += 1
            except Exception as exc:
                print(f"[{i:02d}] ✗  {channel:<10}  error: {exc}")
                failed += 1

            if delay > 0 and i < len(fixtures):
                time.sleep(delay)

    print(f"\nDone. {ok} accepted, {failed} failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Send fixture offers to the ingestion webhook.")
    parser.add_argument("--host", default="http://localhost:8000", help="Webhook base URL.")
    parser.add_argument("--delay", type=float, default=0.3, help="Seconds between messages.")
    args = parser.parse_args()
    send_all(host=args.host, delay=args.delay)


if __name__ == "__main__":
    main()
