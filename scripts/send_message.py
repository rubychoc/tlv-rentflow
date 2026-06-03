"""
Send a single tenant message to the running RentFlow server.

Usage (with the server already running):
  python scripts/send_message.py "היי, מעוניין בדירה. תקציב 7000, עובד בהייטק."
  python scripts/send_message.py "Hi, budget 8500, 2 people, no pets." --channel facebook
  python scripts/send_message.py "Looking for a place" --sender "+972501234567" --channel whatsapp

Options:
  --channel   whatsapp / facebook / yad2  (default: whatsapp)
  --sender    Sender identifier            (default: auto-generated)
  --host      Server base URL              (default: http://localhost:8000)
  --no-run    Ingest only — skip the extraction+scoring pipeline step
"""

import argparse
import sys
import time

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a single message to the RentFlow webhook.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("message", help="The raw tenant message text.")
    parser.add_argument(
        "--channel", default="whatsapp",
        choices=["whatsapp", "facebook", "yad2"],
        help="Which channel the message arrives on (default: whatsapp).",
    )
    parser.add_argument(
        "--sender", default=None,
        help="Sender identifier, e.g. +972541234567 or fb_user_001 (auto-generated if omitted).",
    )
    parser.add_argument(
        "--host", default="http://localhost:8000",
        help="Server base URL (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--no-run", action="store_true",
        help="Skip the extraction+scoring pipeline step after ingestion.",
    )
    args = parser.parse_args()

    sender = args.sender or f"cli_{int(time.time())}"
    channel = args.channel
    text = args.message

    with httpx.Client(base_url=args.host, timeout=30.0) as client:
        # Health check
        try:
            health = client.get("/healthz").raise_for_status().json()
        except Exception as exc:
            print(f"ERROR  Cannot reach {args.host}/healthz — is the server running?\n  {exc}", file=sys.stderr)
            sys.exit(1)

        if not health.get("listing_live"):
            print("ERROR  No active listing on server. Create one via the UI first.", file=sys.stderr)
            sys.exit(1)

        # Ingest
        print(f"Sending to /{channel} as {sender!r} …")
        try:
            resp = client.post(
                f"/webhook/{channel}",
                json={"sender": sender, "text": text},
            ).raise_for_status().json()
        except httpx.HTTPStatusError as exc:
            print(f"ERROR  Ingestion failed — HTTP {exc.response.status_code}", file=sys.stderr)
            try:
                print(f"       {exc.response.json()}", file=sys.stderr)
            except Exception:
                pass
            sys.exit(1)

        offer_id = resp["offer_id"]
        print(f"OK     Ingested as {offer_id}")

        if args.no_run:
            print("Skipped pipeline (--no-run).")
            return

        # Run pipeline
        print("Running extraction + scoring pipeline …")
        try:
            run_resp = client.post(
                "/pipeline/run",
                json={"offer_id": offer_id},
            ).raise_for_status().json()
        except httpx.HTTPStatusError as exc:
            print(f"ERROR  Pipeline call failed — HTTP {exc.response.status_code}", file=sys.stderr)
            sys.exit(1)

        if run_resp.get("status") == "error":
            print(f"ERROR  Pipeline returned error: {run_resp.get('error')}", file=sys.stderr)
            sys.exit(1)

        # Fetch and print the score
        try:
            results = client.get("/pipeline/results").raise_for_status().json()
        except Exception:
            results = []

        hit = next((r for r in results if r.get("offer", {}).get("offer_id") == offer_id), None)
        if hit and hit.get("score"):
            sc = hit["score"]
            print(f"\nResult: {sc['qualification']}  —  score {sc['score']:.1f} / 100")
            if hit.get("group", {}).get("applicants"):
                appl = hit["group"]["applicants"]
                name = appl[0].get("name") or "(name unknown)"
                size = hit["group"].get("household_size") or len(appl)
                print(f"        {name}  ·  household {size}")
        else:
            print("Pipeline finished (no score data returned).")


if __name__ == "__main__":
    main()
