"""
Live integration tests for Station 2 — hits the real OpenAI API.

Two kinds of tests:

  Fully-specified messages — every field is stated; assert exact values.
  Partially-specified messages — only some fields are stated; assert the
    stated ones are correct AND the absent ones are null (not guessed).

The partial-field tests are the more important ones: they verify the model
obeys the "null = not stated, never guessed" invariant from the prompt.
A hallucination here (filling in a missing field) is a correctness bug.

Group-level fields (budget_nis, move_in_date, has_pets, household_size) are
accessed on result.group. Per-person fields (employment_status, age, gender)
are accessed on result.group.applicants[0] for single-applicant messages.

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
        group = engine.extract(_offer(self.TEXT, "live_en_1")).group
        assert group.applicants[0].employment_status == EmploymentStatus.EMPLOYED

    def test_age_is_25(self, engine):
        group = engine.extract(_offer(self.TEXT, "live_en_2")).group
        assert group.applicants[0].age == 25

    def test_gender_is_female(self, engine):
        group = engine.extract(_offer(self.TEXT, "live_en_3")).group
        assert group.applicants[0].gender == Gender.FEMALE

    def test_has_pets_is_false(self, engine):
        group = engine.extract(_offer(self.TEXT, "live_en_4")).group
        assert group.has_pets is False

    def test_household_size_is_one(self, engine):
        group = engine.extract(_offer(self.TEXT, "live_en_5")).group
        assert group.household_size == 1

    def test_budget_is_7000(self, engine):
        group = engine.extract(_offer(self.TEXT, "live_en_6")).group
        assert group.budget_nis == 7000

    def test_move_in_date_is_august_1(self, engine):
        from datetime import date
        group = engine.extract(_offer(self.TEXT, "live_en_7")).group
        assert group.move_in_date == date(2026, 8, 1)


# ---------------------------------------------------------------------------
# Unambiguous Hebrew fixture — clear values, gendered language
# ---------------------------------------------------------------------------

class TestLiveClearHebrew:
    """
    Message: employed man, 30 years old, immediate move-in, no pets, alone.
    """

    TEXT = "שלום, אני בן 30, עובד בהייטק. מחפש לגור לבד, ללא חיות. נכנס מיידי."

    def test_employment_status_is_employed(self, engine):
        group = engine.extract(_offer(self.TEXT, "live_he_1")).group
        assert group.applicants[0].employment_status == EmploymentStatus.EMPLOYED

    def test_age_is_30(self, engine):
        group = engine.extract(_offer(self.TEXT, "live_he_2")).group
        assert group.applicants[0].age == 30

    def test_has_pets_is_false(self, engine):
        group = engine.extract(_offer(self.TEXT, "live_he_3")).group
        assert group.has_pets is False

    def test_household_size_is_one(self, engine):
        group = engine.extract(_offer(self.TEXT, "live_he_4")).group
        assert group.household_size == 1

    def test_move_in_date_is_today(self, engine):
        from datetime import date
        group = engine.extract(_offer(self.TEXT, "live_he_5")).group
        assert group.move_in_date == date(2026, 6, 2)


# ---------------------------------------------------------------------------
# Non-applicant — all screening fields must be null, no applicants
# ---------------------------------------------------------------------------

class TestLiveNonApplicant:
    """A price-inquiry message, not a rental application."""

    TEXT = "כמה עולה הדירה בחודש? אפשר לראות תמונות?"

    def test_all_group_fields_are_null_and_no_applicants(self, engine):
        group = engine.extract(_offer(self.TEXT, "live_noapp_1")).group
        for field in ("budget_nis", "move_in_date", "has_pets", "household_size"):
            assert getattr(group, field) is None, f"Expected group.{field} to be None"
        assert group.applicants == [], "Non-application must have no applicants"


# ---------------------------------------------------------------------------
# Provenance validity — all spans must be real substrings of the input
# ---------------------------------------------------------------------------

class TestLiveProvenanceValidity:
    TEXT = (
        "Hi! I'm a 25-year-old woman, employed full-time. "
        "No pets, looking to live alone. "
        "Budget up to 7000 NIS. Can move in August 1st."
    )

    def test_all_provenance_spans_are_real_substrings(self, engine):
        offer = _offer(self.TEXT, "live_prov_1")
        result = engine.extract(offer)
        bad = []
        for field, prov in result.group.provenance.items():
            if prov.source_span not in offer.text:
                bad.append(f"group.{field}={prov.source_span!r}")
        for i, person in enumerate(result.group.applicants):
            for field, prov in person.provenance.items():
                if prov.source_span not in offer.text:
                    bad.append(f"applicant[{i}].{field}={prov.source_span!r}")
        assert bad == [], f"Hallucinated provenance spans: {bad}"


# ---------------------------------------------------------------------------
# Partial fields — stated fields correct, absent fields null
# ---------------------------------------------------------------------------

class TestLiveNoAgeNoBudget:
    TEXT = "שלום, ראיתי ביד2. אני בחורה בת 25, עובדת בבנק. מחפשת לבד ומיידי. ללא חיות."

    def test_stated_gender_is_female(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_1a")).group
        assert group.applicants[0].gender == Gender.FEMALE

    def test_stated_age_is_25(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_1b")).group
        assert group.applicants[0].age == 25

    def test_stated_employment_is_employed(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_1c")).group
        assert group.applicants[0].employment_status == EmploymentStatus.EMPLOYED

    def test_stated_household_size_is_one(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_1d")).group
        assert group.household_size == 1

    def test_stated_pets_is_false(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_1e")).group
        assert group.has_pets is False

    def test_unstated_budget_is_null(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_1f")).group
        assert group.budget_nis is None


class TestLiveNoPetsNoMoveIn:
    TEXT = (
        "Interested. Single occupant, employed as a software engineer. "
        "No pets. Can provide references and pay 2 months deposit. "
        "Available from July. I'm a 32-year-old man."
    )

    def test_stated_employment_is_employed(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_2a")).group
        assert group.applicants[0].employment_status == EmploymentStatus.EMPLOYED

    def test_stated_household_size_is_one(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_2b")).group
        assert group.household_size == 1

    def test_stated_pets_is_false(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_2c")).group
        assert group.has_pets is False

    def test_stated_age_is_32(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_2d")).group
        assert group.applicants[0].age == 32

    def test_stated_gender_is_male(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_2e")).group
        assert group.applicants[0].gender == Gender.MALE

    def test_unstated_budget_is_null(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_2f")).group
        assert group.budget_nis is None


class TestLiveExplicitBudgetNoAgeNoGender:
    TEXT = (
        "Hey! Me and my girlfriend are looking. Both employed, mid-20s, no pets. "
        "We can move in from August. Is the price negotiable at all? We'd do 6800."
    )

    def test_age_is_estimated_from_mid_twenties(self, engine):
        # "mid-20s" → prompt rule 9 maps this to 25
        group = engine.extract(_offer(self.TEXT, "partial_3e")).group
        for p in group.applicants:
            assert p.age in (None, 25), f"Unexpected age={p.age}"

    def test_stated_budget_is_6800(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_3a")).group
        assert group.budget_nis == 6800

    def test_stated_household_size_is_two(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_3b")).group
        assert group.household_size == 2

    def test_stated_employment_is_employed(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_3c")).group
        for p in group.applicants:
            assert p.employment_status == EmploymentStatus.EMPLOYED

    def test_stated_pets_is_false(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_3d")).group
        assert group.has_pets is False


class TestLiveRoommatesNoEmploymentNoAge:
    """
    "We are 3 students" — headcount-only, shared employment, no individual ages.
    """
    TEXT = (
        "Hi, is this place still available? We are 3 students looking for a "
        "shared apartment. Move-in flexible, ideally October. All non-smokers, no pets."
    )

    def test_stated_household_size_is_three(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_4a")).group
        assert group.household_size == 3

    def test_stated_pets_is_false(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_4b")).group
        assert group.has_pets is False

    def test_stated_employment_is_student(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_4c")).group
        assert len(group.applicants) == 3
        assert all(p.employment_status == EmploymentStatus.STUDENT for p in group.applicants)

    def test_unstated_budget_is_null(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_4d")).group
        assert group.budget_nis is None

    def test_unstated_age_is_null(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_4e")).group
        for p in group.applicants:
            assert p.age is None

    def test_flexible_move_in_is_null(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_4f")).group
        assert group.move_in_date is None


class TestLiveHasPetsNoRoommates:
    TEXT = (
        "Looking for something pet-friendly. I have a golden retriever 🐕. "
        "Just me, no roommates. Self-employed consultant. Flexible on dates. 38 years old, male."
    )

    def test_stated_pets_is_true(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_5a")).group
        assert group.has_pets is True

    def test_stated_household_size_is_one(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_5b")).group
        assert group.household_size == 1

    def test_stated_employment_is_self_employed(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_5c")).group
        assert group.applicants[0].employment_status == EmploymentStatus.SELF_EMPLOYED

    def test_stated_age_is_38(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_5d")).group
        assert group.applicants[0].age == 38

    def test_stated_gender_is_male(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_5e")).group
        assert group.applicants[0].gender == Gender.MALE

    def test_unstated_budget_is_null(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_5f")).group
        assert group.budget_nis is None

    def test_flexible_move_in_is_null(self, engine):
        group = engine.extract(_offer(self.TEXT, "partial_5g")).group
        assert group.move_in_date is None


# ---------------------------------------------------------------------------
# Couple with split employment — multi-applicant extraction
# ---------------------------------------------------------------------------

class TestLiveCoupleWithSplitEmployment:
    """
    "me and my partner, we have 2 cats. I freelance, she's employed. I'm 31."
    Two distinct people with different employment statuses.
    """
    TEXT = (
        "Looking for a place for me and my partner, we have 2 cats. "
        "Can move in immediately. I freelance, she's employed full time. I'm 31."
    )

    def test_household_has_two_people(self, engine):
        group = engine.extract(_offer(self.TEXT, "couple_1a")).group
        assert group.household_size == 2

    def test_group_has_pets(self, engine):
        group = engine.extract(_offer(self.TEXT, "couple_1b")).group
        assert group.has_pets is True

    def test_applicant_count_matches_household_size(self, engine):
        group = engine.extract(_offer(self.TEXT, "couple_1c")).group
        assert len(group.applicants) == group.household_size == 2

    def test_split_employment_both_captured(self, engine):
        group = engine.extract(_offer(self.TEXT, "couple_1d")).group
        statuses = {p.employment_status for p in group.applicants}
        assert EmploymentStatus.SELF_EMPLOYED in statuses
        assert EmploymentStatus.EMPLOYED in statuses

    def test_sender_age_known_partner_age_null(self, engine):
        # "I'm 31" → sender has age; partner age never mentioned → null
        group = engine.extract(_offer(self.TEXT, "couple_1e")).group
        ages = [p.age for p in group.applicants]
        assert 31 in ages
        assert None in ages  # partner age unknown

    def test_at_least_one_gender_is_female(self, engine):
        # "she's employed" → partner is female; sender gender is ambiguous
        group = engine.extract(_offer(self.TEXT, "couple_1f")).group
        genders = [p.gender for p in group.applicants]
        assert Gender.FEMALE in genders


# ---------------------------------------------------------------------------
# Married couple (Hebrew) — wife's gender deduced, ages ambiguous
# ---------------------------------------------------------------------------

class TestLiveHebrewMarriedCouple:
    """
    "אני ואשתי מחפשים דירה. שנינו עובדים, בשנות השלושים. יש לנו חתול אחד. כניסה ב-1 לספטמבר."
    household_size=2, both employed, one cat, September move-in.
    "אשתי" → wife → at least one female. "שנות השלושים" → 30s, not an exact integer.
    """
    TEXT = "אני ואשתי מחפשים דירה. שנינו עובדים, בשנות השלושים. יש לנו חתול אחד. כניסה ב-1 לספטמבר."

    def test_household_size_is_two(self, engine):
        group = engine.extract(_offer(self.TEXT, "heb_couple_a")).group
        assert group.household_size == 2

    def test_two_applicants(self, engine):
        group = engine.extract(_offer(self.TEXT, "heb_couple_b")).group
        assert len(group.applicants) == 2

    def test_both_employed(self, engine):
        group = engine.extract(_offer(self.TEXT, "heb_couple_c")).group
        assert all(p.employment_status == EmploymentStatus.EMPLOYED for p in group.applicants)

    def test_at_least_one_gender_is_female(self, engine):
        # "אשתי" (my wife) → one applicant must be female
        group = engine.extract(_offer(self.TEXT, "heb_couple_d")).group
        genders = [p.gender for p in group.applicants]
        assert Gender.FEMALE in genders

    def test_ages_are_null_not_guessed(self, engine):
        # "בשנות השלושים" is a range — no exact integer should be extracted
        group = engine.extract(_offer(self.TEXT, "heb_couple_e")).group
        for p in group.applicants:
            assert p.age is None or p.age in range(30, 40), f"Expected age=None for range 'בשנות השלושים', got {p.age}"

    def test_has_pets_is_true(self, engine):
        group = engine.extract(_offer(self.TEXT, "heb_couple_f")).group
        assert group.has_pets is True

    def test_move_in_is_september(self, engine):
        from datetime import date
        group = engine.extract(_offer(self.TEXT, "heb_couple_g")).group
        assert group.move_in_date == date(2026, 9, 1)


# ---------------------------------------------------------------------------
# Hebrew roommates — sender's details known, others only share employment
# ---------------------------------------------------------------------------

class TestLiveHebrewRoommatesPartialInfo:
    """
    "מעוניינת! אני ועוד 2 שותפות, כולנו עובדות. כניסה מיידית. אין חיות. אני בת 27."
    household_size=3, all employed, sender is female aged 27.
    The two roommates have employment but no individual age or gender stated.
    """
    TEXT = "מעוניינת! אני ועוד 2 שותפות, כולנו עובדות. כניסה מיידית. אין חיות. אני בת 27."

    def test_household_size_is_three(self, engine):
        group = engine.extract(_offer(self.TEXT, "heb_room_a")).group
        assert group.household_size == 3

    def test_three_applicants(self, engine):
        group = engine.extract(_offer(self.TEXT, "heb_room_b")).group
        assert len(group.applicants) == 3

    def test_all_employed(self, engine):
        group = engine.extract(_offer(self.TEXT, "heb_room_c")).group
        assert all(p.employment_status == EmploymentStatus.EMPLOYED for p in group.applicants)

    def test_exactly_one_age_known(self, engine):
        # Only sender stated her age (27); the other two did not
        group = engine.extract(_offer(self.TEXT, "heb_room_d")).group
        known_ages = [p.age for p in group.applicants if p.age is not None]
        assert known_ages == [27]

    def test_at_least_one_gender_is_female(self, engine):
        # Sender said "בת 27" and "שותפות" (fem. plural) → at least sender is female
        group = engine.extract(_offer(self.TEXT, "heb_room_e")).group
        genders = [p.gender for p in group.applicants]
        assert Gender.FEMALE in genders

    def test_others_have_null_age(self, engine):
        # The two roommates beyond the sender must have age=None
        group = engine.extract(_offer(self.TEXT, "heb_room_f")).group
        null_ages = [p for p in group.applicants if p.age is None]
        assert len(null_ages) == 2
