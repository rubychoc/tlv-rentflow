"""
Live integration tests for Station 2 — hits the real OpenAI API.

Two kinds of tests:

  Fully-specified messages — every field is stated; assert exact values.
  Partially-specified messages — only some fields are stated; assert the
    stated ones are correct AND the absent ones are null (not guessed).

The partial-field tests are the more important ones: they verify the model
obeys the "null = not stated, never guessed" invariant from the prompt.
A hallucination here (filling in a missing field) is a correctness bug.

Marked @pytest.mark.live so they are excluded from the default pytest run.
Run them explicitly when the prompt changes:

    pytest tests/integration/test_extraction_live.py -m live -v

Requirements:
  - OPENAI_API_KEY set in .env or the environment
  - venv active with pip install -e .
"""

from pathlib import Path

import pytest
from datetime import datetime, timezone
from dotenv import load_dotenv

from rentflow.extraction.engine import ExtractionEngine
from rentflow.offer.models import Channel, EmploymentStatus, Gender, RawOffer

load_dotenv(Path(__file__).parents[2] / ".env", override=True)

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def engine():
    """One shared engine for all live tests — avoids re-loading .env repeatedly."""
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — skipping live tests")
    return ExtractionEngine.from_env()


def _offer(text: str, offer_id: str) -> RawOffer:
    return RawOffer(
        offer_id=offer_id,
        channel=Channel.WHATSAPP,
        sender="+972541234567",
        timestamp=datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
        text=text,
    )


# ---------------------------------------------------------------------------
# Unambiguous English fixture — clear values for all screening fields
# ---------------------------------------------------------------------------

class TestLiveClearEnglish:
    """
    Message: employed woman, 25 years old, no pets, alone, budget 7000, moves Aug 1.
    Every screening field is explicit and unambiguous.
    """

    TEXT = (
        "Hi! I'm a 25-year-old woman, employed full-time. "
        "No pets, looking to live alone. "
        "Budget up to 7000 NIS. Can move in August 1st."
    )

    def test_employment_status_is_employed(self, engine):
        profile = engine.extract(_offer(self.TEXT, "live_en_1")).profile
        assert profile.employment_status == EmploymentStatus.EMPLOYED

    def test_age_is_25(self, engine):
        profile = engine.extract(_offer(self.TEXT, "live_en_2")).profile
        assert profile.age == 25

    def test_gender_is_female(self, engine):
        profile = engine.extract(_offer(self.TEXT, "live_en_3")).profile
        assert profile.gender == Gender.FEMALE

    def test_has_pets_is_false(self, engine):
        profile = engine.extract(_offer(self.TEXT, "live_en_4")).profile
        assert profile.has_pets is False

    def test_num_roommates_is_zero(self, engine):
        profile = engine.extract(_offer(self.TEXT, "live_en_5")).profile
        assert profile.num_roommates == 0

    def test_budget_is_7000(self, engine):
        profile = engine.extract(_offer(self.TEXT, "live_en_6")).profile
        assert profile.budget_nis == 7000

    def test_move_in_date_is_august_1(self, engine):
        from datetime import date
        profile = engine.extract(_offer(self.TEXT, "live_en_7")).profile
        assert profile.move_in_date == date(2026, 8, 1)


# ---------------------------------------------------------------------------
# Unambiguous Hebrew fixture — clear values, gendered language
# ---------------------------------------------------------------------------

class TestLiveClearHebrew:
    """
    Message: employed man, 30 years old, immediate move-in, no pets, alone.
    'עובד' (employed, masculine), 'בן 30' (age 30, masculine).
    """

    TEXT = "שלום, אני בן 30, עובד בהייטק. מחפש לגור לבד, ללא חיות. נכנס מיידי."

    def test_employment_status_is_employed(self, engine):
        profile = engine.extract(_offer(self.TEXT, "live_he_1")).profile
        assert profile.employment_status == EmploymentStatus.EMPLOYED

    def test_age_is_30(self, engine):
        profile = engine.extract(_offer(self.TEXT, "live_he_2")).profile
        assert profile.age == 30

    def test_has_pets_is_false(self, engine):
        profile = engine.extract(_offer(self.TEXT, "live_he_3")).profile
        assert profile.has_pets is False

    def test_num_roommates_is_zero(self, engine):
        profile = engine.extract(_offer(self.TEXT, "live_he_4")).profile
        assert profile.num_roommates == 0

    def test_move_in_date_is_today(self, engine):
        # "מיידי" = immediately → today's date as pinned in the prompt (2026-06-02)
        from datetime import date
        profile = engine.extract(_offer(self.TEXT, "live_he_5")).profile
        assert profile.move_in_date == date(2026, 6, 2)


# ---------------------------------------------------------------------------
# Non-applicant — all screening fields must be null
# ---------------------------------------------------------------------------

class TestLiveNonApplicant:
    """
    A price-inquiry message, not a rental application.
    Prompt rule 9: return all screening fields as null.
    """

    TEXT = "כמה עולה הדירה בחודש? אפשר לראות תמונות?"

    def test_all_screening_fields_are_null(self, engine):
        profile = engine.extract(_offer(self.TEXT, "live_noapp_1")).profile
        for field in ("budget_nis", "move_in_date", "employment_status",
                      "has_pets", "num_roommates", "age", "gender"):
            assert getattr(profile, field) is None, f"Expected {field} to be None"


# ---------------------------------------------------------------------------
# Provenance validity — all spans must be real substrings of the input
# ---------------------------------------------------------------------------

class TestLiveProvenanceValidity:
    """
    For any extraction, every non-null provenance span must be a real substring
    of the original message. A span not found in the text is a hallucination.
    """

    TEXT = (
        "Hi! I'm a 25-year-old woman, employed full-time. "
        "No pets, looking to live alone. "
        "Budget up to 7000 NIS. Can move in August 1st."
    )

    def test_all_provenance_spans_are_real_substrings(self, engine):
        offer = _offer(self.TEXT, "live_prov_1")
        result = engine.extract(offer)
        bad = [
            f"{field}={prov.source_span!r}"
            for field, prov in result.profile.provenance.items()
            if prov.source_span not in offer.text
        ]
        assert bad == [], f"Hallucinated provenance spans: {bad}"


# ---------------------------------------------------------------------------
# Partial fields — stated fields correct, absent fields null (not guessed)
# ---------------------------------------------------------------------------

class TestLiveNoAgeNoBudget:
    """
    "שלום, ראיתי ביד2. אני בחורה בת 25, עובדת בבנק. מחפשת לבד ומיידי. ללא חיות."
    States: gender (female via 'בחורה'), age (25), employment (employed),
            num_roommates (0 via 'לבד'), move_in (immediate), has_pets (false).
    Does NOT state: budget, phone, name.
    """
    TEXT = "שלום, ראיתי ביד2. אני בחורה בת 25, עובדת בבנק. מחפשת לבד ומיידי. ללא חיות."

    def test_stated_gender_is_female(self, engine):
        # 'בחורה' explicitly marks female gender
        profile = engine.extract(_offer(self.TEXT, "partial_1a")).profile
        assert profile.gender == Gender.FEMALE

    def test_stated_age_is_25(self, engine):
        profile = engine.extract(_offer(self.TEXT, "partial_1b")).profile
        assert profile.age == 25

    def test_stated_employment_is_employed(self, engine):
        profile = engine.extract(_offer(self.TEXT, "partial_1c")).profile
        assert profile.employment_status == EmploymentStatus.EMPLOYED

    def test_stated_roommates_is_zero(self, engine):
        profile = engine.extract(_offer(self.TEXT, "partial_1d")).profile
        assert profile.num_roommates == 0

    def test_stated_pets_is_false(self, engine):
        profile = engine.extract(_offer(self.TEXT, "partial_1e")).profile
        assert profile.has_pets is False

    def test_unstated_budget_is_null(self, engine):
        # No budget mentioned — must be null, not guessed from any context
        profile = engine.extract(_offer(self.TEXT, "partial_1f")).profile
        assert profile.budget_nis is None


class TestLiveNoPetsNoMoveIn:
    """
    "Interested. Single occupant, employed as a software engineer.
     No pets. Can provide references and pay 2 months deposit.
     Available from July. I'm a 32-year-old man."
    States: employment (employed), num_roommates (0), has_pets (false),
            move_in (July → 2026-07-01), age (32), gender (male via 'man').
    Does NOT state: budget, name, phone.
    """
    TEXT = (
        "Interested. Single occupant, employed as a software engineer. "
        "No pets. Can provide references and pay 2 months deposit. "
        "Available from July. I'm a 32-year-old man."
    )

    def test_stated_employment_is_employed(self, engine):
        profile = engine.extract(_offer(self.TEXT, "partial_2a")).profile
        assert profile.employment_status == EmploymentStatus.EMPLOYED

    def test_stated_roommates_is_zero(self, engine):
        profile = engine.extract(_offer(self.TEXT, "partial_2b")).profile
        assert profile.num_roommates == 0

    def test_stated_pets_is_false(self, engine):
        profile = engine.extract(_offer(self.TEXT, "partial_2c")).profile
        assert profile.has_pets is False

    def test_stated_age_is_32(self, engine):
        profile = engine.extract(_offer(self.TEXT, "partial_2d")).profile
        assert profile.age == 32

    def test_stated_gender_is_male(self, engine):
        profile = engine.extract(_offer(self.TEXT, "partial_2e")).profile
        assert profile.gender == Gender.MALE

    def test_unstated_budget_is_null(self, engine):
        # "pay 2 months deposit" is about deposit, not budget — must not populate budget_nis
        profile = engine.extract(_offer(self.TEXT, "partial_2f")).profile
        assert profile.budget_nis is None


class TestLiveExplicitBudgetNoAgeNoGender:
    """
    "Hey! Me and my girlfriend are looking. Both employed, mid-20s, no pets.
     We can move in from August. Is the price negotiable at all? We'd do 6800."
    States: employment (employed), has_pets (false), move_in (August → 2026-08-01),
            num_roommates (1 via 'me and my girlfriend'), budget (6800).
    Does NOT state: age (only 'mid-20s', not a specific integer), gender.
    """
    TEXT = (
        "Hey! Me and my girlfriend are looking. Both employed, mid-20s, no pets. "
        "We can move in from August. Is the price negotiable at all? We'd do 6800."
    )

    def test_unstated_gender_is_null(self, engine):
        # Sender's gender is ambiguous from context — must not infer
        profile = engine.extract(_offer(self.TEXT, "partial_3f")).profile
        assert profile.gender is None

    def test_unstated_age_is_null(self, engine):
        # "mid-20s" is a range, not a specific integer — must not guess an age
        profile = engine.extract(_offer(self.TEXT, "partial_3e")).profile
        assert profile.age is None

    def test_stated_budget_is_6800(self, engine):
        # "We'd do 6800" is an explicit budget offer
        profile = engine.extract(_offer(self.TEXT, "partial_3a")).profile
        assert profile.budget_nis == 6800

    def test_stated_roommates_is_one(self, engine):
        # "Me and my girlfriend" = 1 other occupant
        profile = engine.extract(_offer(self.TEXT, "partial_3b")).profile
        assert profile.num_roommates == 1

    def test_stated_employment_is_employed(self, engine):
        profile = engine.extract(_offer(self.TEXT, "partial_3c")).profile
        assert profile.employment_status == EmploymentStatus.EMPLOYED

    def test_stated_pets_is_false(self, engine):
        profile = engine.extract(_offer(self.TEXT, "partial_3d")).profile
        assert profile.has_pets is False

    

    


class TestLiveRoommatesNoEmploymentNoAge:
    """
    "Hi, is this place still available? We are 3 students looking for a shared
     apartment. Move-in flexible, ideally October. All non-smokers, no pets."
    States: num_roommates (2 others + sender = 3 total → num_roommates=2),
            has_pets (false), employment (student).
    Does NOT state: budget, age, gender, specific move_in date ('flexible').
    """
    TEXT = (
        "Hi, is this place still available? We are 3 students looking for a "
        "shared apartment. Move-in flexible, ideally October. All non-smokers, no pets."
    )

    def test_stated_roommates_is_two(self, engine):
        # "We are 3" → sender + 2 others → num_roommates=2
        profile = engine.extract(_offer(self.TEXT, "partial_4a")).profile
        assert profile.num_roommates == 2

    def test_stated_pets_is_false(self, engine):
        profile = engine.extract(_offer(self.TEXT, "partial_4b")).profile
        assert profile.has_pets is False

    def test_stated_employment_is_student(self, engine):
        profile = engine.extract(_offer(self.TEXT, "partial_4c")).profile
        assert profile.employment_status == EmploymentStatus.STUDENT

    def test_unstated_budget_is_null(self, engine):
        profile = engine.extract(_offer(self.TEXT, "partial_4d")).profile
        assert profile.budget_nis is None

    def test_unstated_age_is_null(self, engine):
        profile = engine.extract(_offer(self.TEXT, "partial_4e")).profile
        assert profile.age is None

    def test_flexible_move_in_is_null(self, engine):
        # "flexible" with no concrete date → null, not a guessed date
        profile = engine.extract(_offer(self.TEXT, "partial_4f")).profile
        assert profile.move_in_date is None


class TestLiveHasPetsNoRoommates:
    """
    "Looking for something pet-friendly. I have a golden retriever 🐕.
     Just me, no roommates. Self-employed consultant. Flexible on dates. 38 years old, male."
    States: has_pets (true), num_roommates (0), employment (self_employed),
            age (38), gender (male).
    Does NOT state: budget, move_in date (flexible → null).
    """
    TEXT = (
        "Looking for something pet-friendly. I have a golden retriever 🐕. "
        "Just me, no roommates. Self-employed consultant. Flexible on dates. 38 years old, male."
    )

    def test_stated_pets_is_true(self, engine):
        profile = engine.extract(_offer(self.TEXT, "partial_5a")).profile
        assert profile.has_pets is True

    def test_stated_roommates_is_zero(self, engine):
        profile = engine.extract(_offer(self.TEXT, "partial_5b")).profile
        assert profile.num_roommates == 0

    def test_stated_employment_is_self_employed(self, engine):
        profile = engine.extract(_offer(self.TEXT, "partial_5c")).profile
        assert profile.employment_status == EmploymentStatus.SELF_EMPLOYED

    def test_stated_age_is_38(self, engine):
        profile = engine.extract(_offer(self.TEXT, "partial_5d")).profile
        assert profile.age == 38

    def test_stated_gender_is_male(self, engine):
        profile = engine.extract(_offer(self.TEXT, "partial_5e")).profile
        assert profile.gender == Gender.MALE

    def test_unstated_budget_is_null(self, engine):
        profile = engine.extract(_offer(self.TEXT, "partial_5f")).profile
        assert profile.budget_nis is None

    def test_flexible_move_in_is_null(self, engine):
        profile = engine.extract(_offer(self.TEXT, "partial_5g")).profile
        assert profile.move_in_date is None
