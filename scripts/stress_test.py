"""
Stress test for the ingestion webhook.

Two separate tests, run in sequence:

TEST 1 — CONCURRENCY STORM
  Fires a large number of requests simultaneously (true parallel HTTP calls
  using async I/O). Goal: find race conditions, connection pool exhaustion,
  or server crashes under instantaneous load. The server should accept all
  of them or reject them cleanly — it must never crash or return a 5xx.

TEST 2 — MEMORY FLOOD
  Sends offers one after another, each with a large text payload (~50 KB),
  until the server's in-memory store runs out of RAM and either crashes or
  the OS kills the process. Goal: prove that InMemoryOfferStore has no size
  cap and will grow without bound. This is the expected failure — the point
  of the test is to *observe* it and understand where the limit is.

How to run (server must be running first):

  Terminal 1:
    source .venv/bin/activate
    uvicorn rentflow.ingestion.app:app --port 8000

  Terminal 2:
    source .venv/bin/activate
    python scripts/stress_test.py                  # both tests
    python scripts/stress_test.py --test concurrency
    python scripts/stress_test.py --test memory
    python scripts/stress_test.py --test memory --payload-kb 200 --max-offers 500
"""

import argparse
import asyncio
import random
import string
import sys
import time
from dataclasses import dataclass, field

import httpx

BASE_URL = "http://localhost:8000"
CHANNELS = ["whatsapp", "facebook", "yad2"]
SAMPLE_TEXTS = [
    "היי, מעוניין בדירה. תקציב 6500, נכנס מיידי. עובד בהייטק, יש לי כלב.",
    "Hello! Interested in the apartment. Budget 7k NIS. Can move in August 1st.",
    "שלום, אני ועוד שותף אחד, תקציב 9000 ביחד. כניסה ספטמבר.",
    "Hey is this still available? Me + 2 roommates, combined budget ~10500.",
    "מעוניינת! סטודנטית, תקציב 5500, כניסה ב-15 לאוגוסט, ללא חיות.",
]


# ---------------------------------------------------------------------------
# Shared result tracker
# ---------------------------------------------------------------------------

@dataclass
class Results:
    accepted: int = 0
    rejected: int = 0       # 4xx — bad request, validation error
    errors: int = 0         # 5xx or connection failure
    latencies_ms: list[float] = field(default_factory=list)

    def record(self, status: int, latency_ms: float) -> None:
        self.latencies_ms.append(latency_ms)
        if status == 202:
            self.accepted += 1
        elif 400 <= status < 500:
            self.rejected += 1
        else:
            self.errors += 1

    def summary(self) -> str:
        total = self.accepted + self.rejected + self.errors
        if not self.latencies_ms:
            return "No requests completed."
        avg = sum(self.latencies_ms) / len(self.latencies_ms)
        p99 = sorted(self.latencies_ms)[int(len(self.latencies_ms) * 0.99)]
        return (
            f"  Total sent : {total}\n"
            f"  Accepted   : {self.accepted} (HTTP 202)\n"
            f"  Rejected   : {self.rejected} (HTTP 4xx — validation failures)\n"
            f"  Errors     : {self.errors}   (HTTP 5xx or connection failure)\n"
            f"  Avg latency: {avg:.1f} ms\n"
            f"  P99 latency: {p99:.1f} ms"
        )


# ---------------------------------------------------------------------------
# TEST 1 — Concurrency storm
# ---------------------------------------------------------------------------

async def _send_one(client: httpx.AsyncClient, results: Results) -> None:
    channel = random.choice(CHANNELS)
    payload = {
        "sender": f"+972{random.randint(500000000, 599999999)}",
        "text": random.choice(SAMPLE_TEXTS),
    }
    t0 = time.perf_counter()
    try:
        resp = await client.post(f"/webhook/{channel}", json=payload)
        latency = (time.perf_counter() - t0) * 1000
        results.record(resp.status_code, latency)
        if resp.status_code >= 500:
            print(f"  [5xx] {resp.status_code}: {resp.text[:120]}", file=sys.stderr)
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        results.record(0, latency)
        print(f"  [ERR] {exc}", file=sys.stderr)


async def run_concurrency_test(n_requests: int, concurrency: int) -> Results:
    results = Results()
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_send(client: httpx.AsyncClient) -> None:
        async with semaphore:
            await _send_one(client, results)

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0, limits=limits) as client:
        tasks = [asyncio.create_task(bounded_send(client)) for _ in range(n_requests)]
        await asyncio.gather(*tasks)

    return results


# ---------------------------------------------------------------------------
# TEST 2 — Memory flood
# ---------------------------------------------------------------------------

def _large_text(kb: int) -> str:
    # Realistic-looking padding: repeating the kind of long text a tenant might
    # paste (property description, lease terms, prior messages, etc.)
    base = "".join(random.choices(string.ascii_letters + " אבגדהוזחטיכלמנסעפצקרשת\n", k=1024))
    return base * kb


def run_memory_test(payload_kb: int, max_offers: int, report_every: int) -> None:
    print(f"  Each payload: ~{payload_kb} KB of text")
    print(f"  Will send up to {max_offers} offers (Ctrl+C to stop early)\n")

    sent = 0
    t_start = time.perf_counter()

    with httpx.Client(base_url=BASE_URL, timeout=15.0) as client:
        for i in range(1, max_offers + 1):
            channel = random.choice(CHANNELS)
            payload = {
                "sender": f"+972{random.randint(500000000, 599999999)}",
                "text": _large_text(payload_kb),
            }
            t0 = time.perf_counter()
            try:
                resp = client.post(f"/webhook/{channel}", json=payload)
                latency_ms = (time.perf_counter() - t0) * 1000

                if resp.status_code == 202:
                    sent += 1
                else:
                    print(f"\n  [!] Unexpected HTTP {resp.status_code} at offer #{i}: {resp.text[:200]}")

                if i % report_every == 0:
                    elapsed = time.perf_counter() - t_start
                    approx_mb = (sent * payload_kb) / 1024
                    print(
                        f"  [{i:>5}] sent={sent}  "
                        f"~{approx_mb:.1f} MB stored  "
                        f"last latency={latency_ms:.0f}ms  "
                        f"elapsed={elapsed:.1f}s"
                    )

            except httpx.ConnectError:
                print(f"\n  [DEAD] Server stopped responding at offer #{i}.")
                print(f"  Sent {sent} offers (~{sent * payload_kb / 1024:.1f} MB) before crash.")
                return
            except KeyboardInterrupt:
                print(f"\n  Stopped manually at offer #{i}.")
                return
            except Exception as exc:
                print(f"\n  [ERR] #{i}: {exc}")

    print(f"\n  Finished {max_offers} offers. Server survived.")
    print(f"  Total data pushed: ~{sent * payload_kb / 1024:.1f} MB")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def check_health() -> bool:
    try:
        resp = httpx.get(f"{BASE_URL}/healthz", timeout=3.0)
        data = resp.json()
        print(f"  Server is up. Offers already stored: {data.get('offers_received', '?')}")
        return True
    except Exception as exc:
        print(f"  Cannot reach {BASE_URL}/healthz — is the server running?\n  {exc}")
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stress test the ingestion webhook.")
    parser.add_argument(
        "--test",
        choices=["concurrency", "memory", "both"],
        default="both",
        help="Which test to run (default: both).",
    )
    # Concurrency test options
    parser.add_argument("--requests", type=int, default=500,
                        help="Total requests for the concurrency test (default: 500).")
    parser.add_argument("--concurrency", type=int, default=100,
                        help="Max simultaneous in-flight requests (default: 100).")
    # Memory test options
    parser.add_argument("--payload-kb", type=int, default=50,
                        help="Size of each offer text payload in KB (default: 50).")
    parser.add_argument("--max-offers", type=int, default=2000,
                        help="Max offers to send in the memory test (default: 2000).")
    parser.add_argument("--report-every", type=int, default=50,
                        help="Print a memory report every N offers (default: 50).")
    args = parser.parse_args()

    print("=" * 60)
    print("TLV-RentFlow — Ingestion Stress Test")
    print("=" * 60)
    print("\nChecking server health...")
    if not check_health():
        sys.exit(1)

    run_concurrency = args.test in ("concurrency", "both")
    run_memory = args.test in ("memory", "both")

    # ---- TEST 1 ----
    if run_concurrency:
        print(f"\n{'=' * 60}")
        print(f"TEST 1: Concurrency Storm")
        print(f"  {args.requests} requests, up to {args.concurrency} in-flight at once")
        print(f"{'=' * 60}")
        t0 = time.perf_counter()
        results = asyncio.run(run_concurrency_test(args.requests, args.concurrency))
        elapsed = time.perf_counter() - t0
        print(results.summary())
        print(f"  Wall time  : {elapsed:.2f}s")
        rps = args.requests / elapsed if elapsed > 0 else 0
        print(f"  Throughput : {rps:.0f} req/s")

        if results.errors > 0:
            print(f"\n  WARNING: {results.errors} requests returned 5xx or failed to connect.")
            print("  The server may have crashed or be in a bad state.")
            print("  Check the server terminal for tracebacks.")
        else:
            print("\n  PASS — server handled all requests without a 5xx error.")

    # ---- TEST 2 ----
    if run_memory:
        print(f"\n{'=' * 60}")
        print("TEST 2: Memory Flood")
        print(f"{'=' * 60}")
        try:
            run_memory_test(args.payload_kb, args.max_offers, args.report_every)
        except KeyboardInterrupt:
            print("\n  Stopped.")


if __name__ == "__main__":
    main()
