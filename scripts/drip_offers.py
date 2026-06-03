"""
Drip 200 tenant offers to the running server immediately after the listing
goes live. Gaps between batches: 0.01-0.3 seconds.

Listing context:
  Asking rent : 6,500 NIS  |  Floor : 6,300 NIS
  No pets (dealbreaker)    |  Max 3 occupants (dealbreaker)
  Employment preferred     |  Age 25-40 (soft)
  Move-in by 2026-07-01    (soft)

Mix: ~40% strong, ~25% borderline, ~35% weak/dealbreaker

Usage:
  python scripts/drip_offers.py
  python scripts/drip_offers.py --host http://localhost:8000
"""

import argparse
import random
import sys
import time

import httpx

# fmt: off
OFFERS = [
    # non-applications
    {"channel": "whatsapp", "sender": "+972547044044", "text": "כמה חדרים יש בדירה?"},
    {"channel": "facebook", "sender": "fb_3332",      "text": "האם יש חניה?"},
    {"channel": "whatsapp", "sender": "+972548045045", "text": "עד מתי הדירה פנויה?"},
    {"channel": "facebook", "sender": "fb_3333",      "text": "אפשר לראות תמונות נוספות?"},
    {"channel": "whatsapp", "sender": "+972549046046", "text": "מה הקומה?"},

    {"channel": "whatsapp", "sender": "+972545060060", "text": "היי מה קורה? יש מצב להכניסה באמצע יולי?"},
    {"channel": "whatsapp", "sender": "+972545060060", "text": "מה המצב אחי תראה אני שמעון אני עובד בנגריה, אני מאוד מעוניין בדירה אבל יש לי קצת בעיה עם כניבה ביולי, אפשר כניסה באוגוסט?"},

    {"channel": "whatsapp", "sender": "+972543456960", "text": " היי מה קורה אנחנו 3 בנות שממש מתעניינות בדירה, נשמח לבוא לראות! כולנו עובדות במשרד החינוך"},
    {"channel": "whatsapp", "sender": "+97254890360", "text": " אנחנו 2 שותפים בני 27 ו30 שרוצים לעבור לגור ביחד. שנינו עובדים בהייטק. אין לנו בעלי חיים"},
    {"channel": "facebook", "sender": "fb_3347",      "text": "מה קורה? מחפש עבור אמא שלי דירה, היא בת 56 וממה אהבה את הדירה."},

    {"channel": "whatsapp", "sender": "+972545012360", "text": " היי. אנחנו זוג בני 25, אני מוראה והיא עובדת במלצרות. אין לנו חיות"},
    {"channel": "facebook", "sender": "fb_3347",      "text": " היי מה העניינים? אני נתעניים בדירה. אני בן 29, כרגע סטודנט להנדסת בניין. אין בע״ח"},
    {"channel": "whatsapp", "sender": "+972545048000", "text": " אני בת 27 כרגע בין עבודות אבל מחפשת לעבוד. אין לי בעילי חיים ואני גרה לבד :)"},
    {"channel": "facebook", "sender": "fb_3347",      "text": " היי אני מחפשת עבורי ובעלי, אני בת 29 והוא בן 40. אני כרגע לא עובדת והוא עובד בסטארטאפ. יש לנו כלב אחד קטן וחמוד "},
    {"channel": "whatsapp", "sender": "+972545060530", "text": "  היי מה העניינים? הדירה מדהימה אבל קצת יקרה לנו... אפשר אולי קצת לרדת במחיר ל6300?"},
    {"channel": "facebook", "sender": "fb_3347",      "text": " הדירה פצצה. אנחנו 2 שותפים בני 29 ועובדים בהייטק. היא טיפה יקרה יש מצב שאתה תרד במחיר ל6000? "},



    {"channel": "whatsapp", "sender": "+972545690060", "text": "היי אני גל, בת 25 מהצפון במקור.אני מחפשת לעבור לבד. אני כרגע סטודנטית להנדסת חשמל ומגיעה בלי בעלי חיים"},
    {"channel": "whatsapp", "sender": "+972224442060", "text": " אשמח לבוא לראות את הדירה? אנחנו נאור ודפנה, בני 30 ו-29, עובדים בהייטק. אין לנו חיות"},
    {"channel": "whatsapp", "sender": "+97254505670", "text": "היי אנחנו עובדים בהייטק, בני 27, אבל נשמח להכנס באמצע יולי, יש מצב כזה? "},

    {"channel": "whatsapp", "sender": "+972545345760", "text": "hi I'm Maria, staying in TLV for the next year. I'm 28 and currently on an exchange program for my studies. I'm looking with one more roomate who is 27 and in the same programn with me. We dont have any pets :)"},

       {"channel": "whatsapp", "sender": "+972544566060", "text": "היי מה קורה אנחנו במכינה קדם צבאית, בני 19 ורוצים לשכור דירה בתל אביב. אנחנו בלי בעלי חיים"},

    {"channel": "whatsapp", "sender": "+972544566060", "text": "היי אנחנו 4 שותפים שרוצים להיכנס לדירה, כולנו עובדים במלצרות. "},

]
# fmt: on



def build_batches(offers: list[dict]) -> list[list[dict]]:
    """Group into batches -- some simultaneous, most solo."""
    random.shuffle(offers)
    batches = []
    i = 0
    while i < len(offers):
        if random.random() < 0.25 and i + 1 < len(offers):
            size = random.randint(2, min(4, len(offers) - i))
            batches.append(offers[i:i + size])
            i += size
        else:
            batches.append([offers[i]])
            i += 1
    return batches


def send_offer(client: httpx.Client, offer: dict) -> str | None:
    channel = offer["channel"]
    payload = {"sender": offer["sender"], "text": offer["text"]}
    try:
        resp = client.post(f"/webhook/{channel}", json=payload, timeout=10.0)
        if resp.status_code == 202:
            offer_id = resp.json()["offer_id"]
            client.post("/pipeline/run", json={"offer_id": offer_id}, timeout=30.0)
            return offer_id
        else:
            print(f"  WARN  HTTP {resp.status_code} for {offer['sender']}: {resp.text[:80]}")
            return None
    except Exception as exc:
        print(f"  ERR   {offer['sender']}: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Drip 200 offers to the RentFlow server.")
    parser.add_argument("--host", default="http://localhost:8000")
    args = parser.parse_args()

    with httpx.Client(base_url=args.host) as client:
        try:
            health = client.get("/healthz", timeout=5.0).raise_for_status().json()
        except Exception as exc:
            print(f"ERROR  Cannot reach {args.host}/healthz\n  {exc}")
            sys.exit(1)

        if not health.get("listing_live"):
            print("ERROR  No active listing. Create one via the UI first.")
            sys.exit(1)

        print(f"Listing live: {health.get('listing_id')} -- starting drip now.\n")

        batches = build_batches(list(OFFERS))
        total = len(OFFERS)
        sent = 0

        for batch in batches:
            [send_offer(client, o) for o in batch]
            sent += len(batch)
            print(f"  [{sent:>3}/{total}]  {', '.join(o['sender'] for o in batch)[:80]}")
            if sent < total:
                time.sleep(random.uniform(0.01, 0.3))

        print(f"\nDone. {sent} offers sent.")


if __name__ == "__main__":
    main()
