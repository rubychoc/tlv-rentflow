"""
LLM hallucination and consistency probing — Station 2, real OpenAI API.

These tests are *adversarial*: they use deliberately ambiguous, obscure, or
contradictory messages to find where the model hallucinates or behaves
inconsistently across repeated calls.

Two kinds of assertions:

  Hard assertions  — things that must always hold regardless of content.
                     A single failure here is a real bug.

  Consistency runs — the same message is extracted N times; we measure the
                     distribution of each field and flag instability. These
                     are reported, not strict pass/fail, because temperature=0
                     is near-deterministic but not guaranteed identical.
                     The threshold is ≥ STABILITY_THRESHOLD of runs agreeing
                     on the modal value. Failing this surfaces prompt fragility
                     worth fixing before shipping.

Marked @pytest.mark.live — excluded from the default pytest run.
Run on prompt changes:

    pytest tests/integration/test_llm_probes.py -m live -v -s

Requirements: OPENAI_API_KEY in .env
"""

import json
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone

import pytest
from dotenv import load_dotenv

from rentflow.extraction.engine import ExtractionEngine
from rentflow.offer.models import Channel, EmploymentStatus, Gender, RawOffer

load_dotenv(Path(__file__).parents[2] / ".env", override=True)

pytestmark = pytest.mark.live

# How many times to call the model per consistency probe.
N_RUNS = 5

# Fraction of runs that must agree on the modal value for a field to be "stable".
STABILITY_THRESHOLD = 0.8

# Report file written alongside this file.
_REPORT_PATH = Path(__file__).parent / "probe_report.json"
_report_entries: list[dict] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    return ExtractionEngine.from_env()


@pytest.fixture(scope="session", autouse=True)
def write_report():
    """Write all probe results to probe_report.json after the session ends."""
    yield
    if _report_entries:
        with _REPORT_PATH.open("w", encoding="utf-8") as f:
            json.dump(_report_entries, f, ensure_ascii=False, default=str, indent=2)


def _offer(text: str, offer_id: str) -> RawOffer:
    return RawOffer(
        offer_id=offer_id,
        channel=Channel.WHATSAPP,
        sender="+972541234567",
        timestamp=datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
        text=text,
    )


def _extract_n(engine, text: str, base_id: str, n: int = N_RUNS):
    """Run extraction N times and return list of TenantGroup results."""
    return [
        engine.extract(_offer(text, f"{base_id}_{i}")).group
        for i in range(n)
    ]


def _stability(values: list) -> tuple[float, object]:
    """Return (modal_fraction, modal_value) for a list of values."""
    if not values:
        return 0.0, None
    counts = Counter(str(v) for v in values)
    modal_str, modal_count = counts.most_common(1)[0]
    modal_val = next(v for v in values if str(v) == modal_str)
    return modal_count / len(values), modal_val


def _record(
    test: str,
    message: str,
    field: str,
    expected,
    values: list,
    kind: str,
    passed: bool,
):
    """Append one result entry to the session report."""
    counts = Counter(str(v) for v in values)
    n = len(values)
    correct = sum(1 for v in values if v == expected) if expected is not None else None
    _report_entries.append({
        "test": test,
        "message": message,
        "field": field,
        "kind": kind,
        "expected": str(expected) if expected is not None else "(no single correct answer)",
        "actual": str(values[0]) if n == 1 else None,
        "n_runs": n,
        "passed": passed,
        "correct_runs": f"{correct}/{n}" if correct is not None else None,
        "run_results": [str(v) for v in values],
        "distribution": dict(counts),
    })


def _assert_stable(values: list, field: str, message: str, test: str,
                   threshold: float = STABILITY_THRESHOLD):
    """Assert a field is consistent across N runs (used for ambiguous inputs only)."""
    rate, modal = _stability(values)
    passed = rate >= threshold
    _record(test=test, message=message, field=field, expected=None,
            values=values, kind="stability", passed=passed)
    assert passed, (
        f"\n  Test     : {test}"
        f"\n  Field    : {field}"
        f"\n  Kind     : stability — ambiguous input, no single correct answer"
        f"\n  Threshold: {threshold:.0%}"
        f"\n  Stability: {rate:.0%}  ← FAIL"
        f"\n  Modal    : {modal!r}"
        f"\n  All runs : {values}"
    )


def _assert_all_correct(values: list, expected, field: str, message: str, test: str):
    """Assert every run returned the expected value (correctness + consistency in one)."""
    wrong = [(i, v) for i, v in enumerate(values) if v != expected]
    passed = not wrong
    _record(test=test, message=message, field=field, expected=expected,
            values=values, kind="correctness", passed=passed)
    assert passed, (
        f"\n  Test     : {test}"
        f"\n  Field    : {field}"
        f"\n  Expected : {expected!r}"
        f"\n  Correct  : {len(values) - len(wrong)}/{len(values)}"
        f"\n  Wrong    : " + ", ".join(f"run {i} → {v!r}" for i, v in wrong) +
        f"\n  All runs : {values}"
    )


def _record_single(test: str, message: str, field: str, expected, actual, passed: bool):
    """Record a single-call test result."""
    _record(test=test, message=message, field=field, expected=expected,
            values=[actual], kind="single", passed=passed)


# ---------------------------------------------------------------------------
# Hard invariants — single call, always true
# ---------------------------------------------------------------------------

class TestHardInvariants:
    """
    Properties that must hold for every extraction regardless of content.
    One failure = a real bug, not a statistical issue.
    """

    def test_all_provenance_spans_are_real_substrings(self, engine):
        text = (
            "Hi! Me and my partner are looking. Both employed, no pets. "
            "Can move in August. Budget 7000. I'm 29, she's 31."
        )
        offer = _offer(text, "inv_prov")
        result = engine.extract(offer)
        g = result.group
        bad = []
        for field, prov in g.provenance.items():
            if prov.source_span not in offer.text:
                bad.append(f"group.{field}={prov.source_span!r}")
        for i, person in enumerate(g.applicants):
            for field, prov in person.provenance.items():
                if prov.source_span not in offer.text:
                    bad.append(f"applicants[{i}].{field}={prov.source_span!r}")
        passed = bad == []
        _record_single("test_all_provenance_spans_are_real_substrings", text,
                       "provenance spans", expected="all substrings of input",
                       actual="OK" if passed else str(bad), passed=passed)
        assert passed, f"Hallucinated provenance spans: {bad}"

    def test_non_applicant_returns_empty_applicants(self, engine):
        msg = "מה המחיר? אפשר לראות תמונות?"
        g = engine.extract(_offer(msg, "inv_noapp")).group
        passed = g.applicants == [] and all(getattr(g, f) is None
                 for f in ("budget_nis", "move_in_date", "has_pets", "household_size"))
        _record_single("test_non_applicant_returns_empty_applicants", msg,
                       "applicants + shared fields", expected="[] and all null",
                       actual=f"applicants={g.applicants}, budget={g.budget_nis}", passed=passed)
        assert g.applicants == []
        for field in ("budget_nis", "move_in_date", "has_pets", "household_size"):
            assert getattr(g, field) is None, f"Expected group.{field} to be None"

    def test_result_always_validates_as_tenant_group(self, engine):
        from rentflow.offer.models import TenantGroup
        msg = "maybe interested, not sure yet"
        g = engine.extract(_offer(msg, "inv_schema")).group
        passed = isinstance(g, TenantGroup)
        _record_single("test_result_always_validates_as_tenant_group", msg,
                       "schema", expected="valid TenantGroup", actual=type(g).__name__, passed=passed)
        assert passed

    def test_friend_pet_not_attributed_to_applicant(self, engine):
        msg = "Hi, interested in the apartment. My friend has a dog but I don't. No pets with me."
        g = engine.extract(_offer(msg, "inv_pet_attr")).group
        _record_single("test_friend_pet_not_attributed_to_applicant", msg,
                       "has_pets", expected=False, actual=g.has_pets, passed=g.has_pets is False)
        assert g.has_pets is False

    def test_flexible_move_in_does_not_produce_date(self, engine):
        msg = "Interested! Flexible on move-in dates, whenever works for you."
        g = engine.extract(_offer(msg, "inv_flexible")).group
        _record_single("test_flexible_move_in_does_not_produce_date", msg,
                       "move_in_date", expected=None, actual=g.move_in_date,
                       passed=g.move_in_date is None)
        assert g.move_in_date is None

    def test_applicant_count_matches_household_size_when_stated(self, engine):
        msg = "We are 2 students looking for a place. Both employed part-time. No pets."
        g = engine.extract(_offer(msg, "inv_count")).group
        actual = len(g.applicants)
        passed = g.household_size is None or actual == g.household_size
        _record_single("test_applicant_count_matches_household_size_when_stated", msg,
                       "len(applicants)", expected=g.household_size, actual=actual, passed=passed)
        if g.household_size is not None:
            assert actual == g.household_size


# ---------------------------------------------------------------------------
# Null-discipline probes — fields that must stay null
# ---------------------------------------------------------------------------

class TestNullDiscipline:
    """
    Messages engineered so specific fields are genuinely absent.
    The model must return null for those fields, not guess.
    """

    def test_no_budget_stated_stays_null(self, engine):
        msg = "Hello, very interested in the apartment. Employed, no pets, can move in September."
        g = engine.extract(_offer(msg, "null_budget")).group
        _record_single("test_no_budget_stated_stays_null", msg,
                       "budget_nis", expected=None, actual=g.budget_nis, passed=g.budget_nis is None)
        assert g.budget_nis is None

    def test_deposit_mention_does_not_populate_budget(self, engine):
        msg = "Interested. Single occupant, employed. No pets. Can pay 2 months deposit upfront."
        g = engine.extract(_offer(msg, "null_deposit")).group
        _record_single("test_deposit_mention_does_not_populate_budget", msg,
                       "budget_nis", expected=None, actual=g.budget_nis, passed=g.budget_nis is None)
        assert g.budget_nis is None

    def test_age_range_produces_null_or_midpoint(self, engine):
        msg = "Looking for a place, just me. Employed, no pets. I'm in my mid-20s."
        persons = engine.extract(_offer(msg, "null_age_range")).group.applicants
        actual = persons[0].age if persons else None
        passed = actual in (None, 25)
        _record_single("test_age_range_produces_null_or_midpoint", msg,
                       "age", expected="None or 25", actual=actual, passed=passed)
        assert passed, f"Unexpected age: {actual}"

    def test_i_am_flexible_produces_no_fields(self, engine):
        msg = "Hi! I'm flexible on everything — dates, budget, whatever works for you."
        g = engine.extract(_offer(msg, "null_flexible_all")).group
        passed = g.budget_nis is None and g.move_in_date is None
        _record_single("test_i_am_flexible_produces_no_fields", msg,
                       "budget_nis + move_in_date", expected="both None",
                       actual=f"budget={g.budget_nis}, move_in={g.move_in_date}", passed=passed)
        assert g.budget_nis is None
        assert g.move_in_date is None

    def test_no_age_mentioned_stays_null(self, engine):
        msg = "Employed software engineer, no pets, looking alone, can move August."
        persons = engine.extract(_offer(msg, "null_age_none")).group.applicants
        actual = persons[0].age if persons else None
        _record_single("test_no_age_mentioned_stays_null", msg,
                       "age", expected=None, actual=actual, passed=actual is None)
        if persons:
            assert persons[0].age is None


# ---------------------------------------------------------------------------
# Consistency probes — N repeated calls, stability measured
# ---------------------------------------------------------------------------

class TestConsistencyHebrew:
    """
    Hebrew messages with grammatical gender signals.
    'בן 25' is a strong male signal; '2 שותפות' is a strong female signal.
    These should be highly stable at temperature=0.
    """

    def test_ben_25_gender_is_consistently_male(self, engine):
        msg = "מעוניין, בן 25, עובד. גר לבד, ללא חיות."
        groups = _extract_n(engine, msg, "heb_male")
        genders = [g.applicants[0].gender if g.applicants else None for g in groups]
        _assert_all_correct(genders, Gender.MALE, "gender",
                            message=msg, test="test_ben_25_gender_is_consistently_male")

    def test_shuftot_gender_is_consistently_female(self, engine):
        msg = "אנחנו 2 שותפות, עובדות. מחפשות דירה מספטמבר. ללא חיות."
        groups = _extract_n(engine, msg, "heb_female")
        all_genders = [p.gender for g in groups for p in g.applicants]
        _assert_all_correct(all_genders, Gender.FEMALE, "gender",
                            message=msg, test="test_shuftot_gender_is_consistently_female")

    def test_immediate_move_in_date_is_correct(self, engine):
        from datetime import date
        msg = "מעוניינת, נכנסת מיידי. עובדת, ללא חיות. בת 27."
        groups = _extract_n(engine, msg, "heb_immediate")
        dates = [g.move_in_date for g in groups]
        _assert_all_correct(dates, date(2026, 6, 2), "move_in_date",
                            message=msg, test="test_immediate_move_in_date_is_correct")


class TestConsistencyBudget:
    """
    Per-person budget multiplication.
    "3500 per person, 3 of us" → budget_nis = 10500. This requires arithmetic
    and is a known failure mode for LLMs.
    """

    def test_per_person_budget_multiplied_correctly(self, engine):
        msg = "Hi! Me and 2 friends looking for a place. All employed, no pets. Can do max 3500 per person."
        groups = _extract_n(engine, msg, "budget_mult")
        budgets = [g.budget_nis for g in groups]
        _assert_all_correct(budgets, 10500, "budget_nis",
                            message=msg, test="test_per_person_budget_multiplied_correctly")

    def test_explicit_total_budget_correct(self, engine):
        msg = "Looking for an apartment, just me. Budget up to 7000 NIS. Employed, no pets."
        groups = _extract_n(engine, msg, "budget_total")
        budgets = [g.budget_nis for g in groups]
        _assert_all_correct(budgets, 7000, "budget_nis",
                            message=msg, test="test_explicit_total_budget_correct")


class TestConsistencyRelativeDates:
    """
    Relative date anchoring. The prompt pins today = 2026-06-02.
    "next month" and "within a month" should resolve to 2026-07-02.
    """

    def test_within_a_month_resolves_correctly(self, engine):
        from datetime import date
        msg = "Interested, just me, employed, no pets. Can move in within a month."
        groups = _extract_n(engine, msg, "date_within_month")
        dates = [g.move_in_date for g in groups]
        _assert_all_correct(dates, date(2026, 7, 2), "move_in_date",
                            message=msg, test="test_within_a_month_resolves_correctly")

    def test_specific_month_name_resolves_correctly(self, engine):
        from datetime import date
        msg = "Hi, me and my partner looking. Both employed, no pets. Can move in from August."
        groups = _extract_n(engine, msg, "date_august")
        dates = [g.move_in_date for g in groups]
        _assert_all_correct(dates, date(2026, 8, 1), "move_in_date",
                            message=msg, test="test_specific_month_name_resolves_correctly")


class TestConsistencySlang:
    """
    Mixed Hebrew-English with slang. Stable extraction matters here because
    slang varies in LLM training data.
    """

    def test_asap_move_in_is_correct(self, engine):
        from datetime import date
        msg = "יאללה, מעוניין! ASAP להיכנס. עובד, בלי בעלי חיים. בן 26."
        groups = _extract_n(engine, msg, "slang_asap")
        dates = [g.move_in_date for g in groups]
        _assert_all_correct(dates, date(2026, 6, 2), "move_in_date",
                            message=msg, test="test_asap_move_in_is_correct")

    def test_slang_no_pets_is_correct(self, engine):
        msg = "Hey!! super interested 🙏 moving asap. no pets obv. work in marketing. no roommates. 26 btw"
        groups = _extract_n(engine, msg, "slang_pets")
        pets = [g.has_pets for g in groups]
        _assert_all_correct(pets, False, "has_pets",
                            message=msg, test="test_slang_no_pets_is_correct")


class TestConsistencyContradictory:
    """
    Contradictory messages. We can't assert a specific value but we can assert
    the model picks one answer consistently rather than alternating.
    """

    def test_contradictory_pets_is_consistent(self, engine):
        msg = "No pets, well I actually have a small cat. She's very quiet though."
        groups = _extract_n(engine, msg, "contradict_pets")
        pets = [g.has_pets for g in groups]
        _assert_stable(pets, "has_pets", message=msg, test="test_contradictory_pets_is_consistent")

    def test_contradictory_employment_is_consistent(self, engine):
        msg = "I work as a freelance designer, well between projects right now to be honest."
        groups = _extract_n(engine, msg, "contradict_employment")
        statuses = [p.employment_status for g in groups for p in g.applicants]
        _assert_stable(statuses, "employment_status",
                       message=msg, test="test_contradictory_employment_is_consistent")


class TestConsistencyNumericNoise:
    """
    Messages where numbers could be misread as the wrong field type.
    Classic trap: age vs budget vs date vs apartment number.
    """

    def test_age_and_budget_kept_separate(self, engine):
        msg = "I'm 30 years old, my budget is 3000 NIS, can move in on the 1st of August."
        groups = _extract_n(engine, msg, "numeric_noise")
        _assert_all_correct([g.budget_nis for g in groups], 3000, "budget_nis",
                            message=msg, test="test_age_and_budget_kept_separate")
        ages = [g.applicants[0].age for g in groups if g.applicants]
        _assert_all_correct(ages, 30, "age",
                            message=msg, test="test_age_and_budget_kept_separate")

    def test_apartment_number_not_extracted_as_budget(self, engine):
        msg = "Interested in apartment 4B. Employed, no pets. Rent around 7500 is fine."
        g = engine.extract(_offer(msg, "numeric_apt")).group
        actual_budget = g.budget_nis
        actual_age = g.applicants[0].age if g.applicants else None
        passed = actual_budget in (None, 7500) and actual_age != 4
        _record_single("test_apartment_number_not_extracted_as_budget", msg,
                       "budget_nis + age", expected="budget in (None,7500), age ≠ 4",
                       actual=f"budget={actual_budget}, age={actual_age}", passed=passed)
        assert actual_budget in (None, 7500), f"Unexpected budget: {actual_budget}"
        if g.applicants:
            assert actual_age != 4, "Apartment number extracted as age"


# ---------------------------------------------------------------------------
# Multi-person consistency
# ---------------------------------------------------------------------------

class TestConsistencyMultiPerson:
    """
    Couples and groups where per-person fields should be distributed correctly.
    """

    def test_couple_split_employment_stable(self, engine):
        msg = "Looking for a place for me and my partner. I freelance, she's employed full time. No pets. Can move in immediately. I'm 31."
        groups = _extract_n(engine, msg, "multi_employment")
        all_status_sets = [{p.employment_status for p in g.applicants} for g in groups]
        passed = all(
            EmploymentStatus.SELF_EMPLOYED in s and EmploymentStatus.EMPLOYED in s
            for s in all_status_sets
        )
        flat = [str(s) for g in groups for p in g.applicants for s in [p.employment_status]]
        _record(test="test_couple_split_employment_stable", message=msg,
                field="employment_status (each applicant)",
                expected="{self_employed, employed} per run",
                values=flat, kind="correctness", passed=passed)
        for statuses in all_status_sets:
            assert EmploymentStatus.SELF_EMPLOYED in statuses, f"Expected self_employed, got {statuses}"
            assert EmploymentStatus.EMPLOYED in statuses, f"Expected employed, got {statuses}"

    def test_shared_employment_copied_to_all_members(self, engine):
        msg = "We are 3 friends looking for an apartment. All employed, no pets. Can move in September."
        groups = _extract_n(engine, msg, "multi_shared_emp")
        all_statuses = [p.employment_status for g in groups for p in g.applicants]
        _assert_all_correct(all_statuses, EmploymentStatus.EMPLOYED, "employment_status",
                            message=msg, test="test_shared_employment_copied_to_all_members")

    def test_known_age_on_sender_unknown_on_others(self, engine):
        msg = "Hi! Me and my two roommates are looking. I'm 28, we're all employed. No pets. Can move in August."
        groups = _extract_n(engine, msg, "multi_age_split")
        for g in groups:
            ages = [p.age for p in g.applicants]
            known = [a for a in ages if a is not None]
            assert len(known) <= 1, f"Only sender's age should be known, got {ages}"
            if known:
                assert known[0] == 28, f"Sender age should be 28, got {known[0]}"
        sender_ages = [g.applicants[0].age for g in groups if g.applicants]
        _assert_all_correct(sender_ages, 28, "age (sender only)",
                            message=msg, test="test_known_age_on_sender_unknown_on_others")


# ---------------------------------------------------------------------------
# Obscure attribution probes
# ---------------------------------------------------------------------------

class TestObscureAttribution:
    """
    Messages where the key information belongs to someone other than the applicant,
    or is stated in a way that requires careful attribution to extract correctly.

    These are the hardest hallucination traps: the information *is* in the text,
    just not about the applicant.
    """

    def test_sister_pet_not_attributed_to_applicant(self, engine):
        msg = "Hi, interested in the apartment. My sister has a dog but I don't keep any pets."
        g = engine.extract(_offer(msg, "obscure_sister_pet")).group
        _record_single("test_sister_pet_not_attributed_to_applicant", msg,
                       "has_pets", expected=False, actual=g.has_pets, passed=g.has_pets is False)
        assert g.has_pets is False, f"Sister's dog must not populate has_pets, got {g.has_pets}"

    def test_past_pet_ownership_does_not_set_has_pets(self, engine):
        msg = "I used to have a dog but he passed away last year. Currently no pets."
        g = engine.extract(_offer(msg, "obscure_past_pet")).group
        _record_single("test_past_pet_ownership_does_not_set_has_pets", msg,
                       "has_pets", expected=False, actual=g.has_pets, passed=g.has_pets is False)
        assert g.has_pets is False, f"Past pet must not populate has_pets as True, got {g.has_pets}"

    def test_considering_getting_a_pet_stays_null_or_unknown(self, engine):
        msg = "No pets right now, though I'm thinking about getting a cat in the future."
        g = engine.extract(_offer(msg, "obscure_future_pet")).group
        passed = g.has_pets is not True
        _record_single("test_considering_getting_a_pet_stays_null_or_unknown", msg,
                       "has_pets", expected="False or None", actual=g.has_pets, passed=passed)
        assert passed, f"Hypothetical future pet must not set has_pets=True, got {g.has_pets}"

    def test_friend_budget_not_attributed_to_applicant(self, engine):
        msg = "My friend looked at a similar apartment and paid 6500. I'm looking for a place, employed, no pets, can move in August."
        g = engine.extract(_offer(msg, "obscure_friend_budget")).group
        _record_single("test_friend_budget_not_attributed_to_applicant", msg,
                       "budget_nis", expected=None, actual=g.budget_nis, passed=g.budget_nis is None)
        assert g.budget_nis is None, f"Friend's rent must not populate budget_nis, got {g.budget_nis}"

    def test_previous_tenant_age_not_attributed(self, engine):
        msg = "I heard the previous tenant was 45 and lived there for years. I'm looking for a place, just me, employed, no pets."
        g = engine.extract(_offer(msg, "obscure_prev_tenant_age")).group
        actual = g.applicants[0].age if g.applicants else None
        passed = actual != 45
        _record_single("test_previous_tenant_age_not_attributed", msg,
                       "age", expected="not 45", actual=actual, passed=passed)
        if g.applicants:
            assert actual != 45, f"Previous tenant's age must not populate applicant age"

    def test_landlord_asking_price_not_extracted_as_tenant_budget(self, engine):
        msg = "I saw the listing, the rent is 7500 NIS. I'm interested, employed, no pets."
        g = engine.extract(_offer(msg, "obscure_asking_price")).group
        _record_single("test_landlord_asking_price_not_extracted_as_tenant_budget", msg,
                       "budget_nis", expected=None, actual=g.budget_nis, passed=g.budget_nis is None)
        assert g.budget_nis is None, f"Landlord's asking price must not populate budget_nis, got {g.budget_nis}"


# ---------------------------------------------------------------------------
# Obscure date probes
# ---------------------------------------------------------------------------

class TestObscureDates:
    """
    Move-in dates expressed vaguely, seasonally, or relative to events.
    The model must either resolve them to a concrete ISO date or return null —
    never a hallucinated date with no basis in the text.
    """

    def test_end_of_summer_produces_date_or_null(self, engine):
        msg = "Interested! Employed, no pets, just me. Want to move in before the end of summer."
        groups = _extract_n(engine, msg, "date_end_summer")
        dates = [g.move_in_date for g in groups]
        _record(test="test_end_of_summer_produces_date_or_null", message=msg,
                field="move_in_date", expected=None, values=dates, kind="observe", passed=True)
        for d in dates:
            if d is not None:
                assert d.month <= 9, f"'End of summer' resolved to {d}, expected ≤ September"

    def test_after_the_holidays_produces_date_or_null(self, engine):
        msg = "מעוניין, עובד, ללא חיות. כניסה אחרי החגים."
        groups = _extract_n(engine, msg, "date_after_holidays")
        dates = [g.move_in_date for g in groups]
        _record(test="test_after_the_holidays_produces_date_or_null", message=msg,
                field="move_in_date", expected=None, values=dates, kind="observe", passed=True)
        for d in dates:
            if d is not None:
                assert d.year == 2026, f"Date year must be 2026, got {d}"

    def test_beginning_of_next_month_resolves_correctly(self, engine):
        from datetime import date
        msg = "Hi, just me, employed, no pets. Can move in at the beginning of next month."
        groups = _extract_n(engine, msg, "date_next_month_start")
        dates = [g.move_in_date for g in groups]
        _assert_all_correct(dates, date(2026, 7, 1), "move_in_date",
                            message=msg, test="test_beginning_of_next_month_resolves_correctly")


# ---------------------------------------------------------------------------
# Obscure budget probes
# ---------------------------------------------------------------------------

class TestObscureBudget:
    """
    Budget signals that are indirect, hedged, or expressed relative to the asking price.
    The model must not populate budget_nis unless the tenant explicitly states
    what they are willing or able to pay.
    """

    def test_rent_is_a_little_tight_does_not_populate_budget(self, engine):
        msg = "The rent is a little tight for me but I think I can manage. Employed, no pets, looking alone."
        g = engine.extract(_offer(msg, "obscure_budget_tight")).group
        _record_single("test_rent_is_a_little_tight_does_not_populate_budget", msg,
                       "budget_nis", expected=None, actual=g.budget_nis, passed=g.budget_nis is None)
        assert g.budget_nis is None, f"Vague 'a little tight' must not produce a budget, got {g.budget_nis}"

    def test_can_we_negotiate_does_not_populate_budget(self, engine):
        msg = "Very interested! Is there any flexibility on the price? Employed full time, no pets, can move immediately."
        g = engine.extract(_offer(msg, "obscure_budget_negotiate")).group
        _record_single("test_can_we_negotiate_does_not_populate_budget", msg,
                       "budget_nis", expected=None, actual=g.budget_nis, passed=g.budget_nis is None)
        assert g.budget_nis is None, f"Negotiation without a number must not produce a budget, got {g.budget_nis}"

    def test_explicit_counteroffer_does_populate_budget(self, engine):
        msg = "Interested! Would it be possible to do 6800 instead of 7000? Employed, no pets, moving alone."
        groups = _extract_n(engine, msg, "obscure_budget_counteroffer")
        _assert_all_correct([g.budget_nis for g in groups], 6800, "budget_nis",
                            message=msg, test="test_explicit_counteroffer_does_populate_budget")

    def test_price_per_person_with_unclear_group_size(self, engine):
        msg = "Hey, a few of us are looking. We can do around 3000 each. All employed, no pets, flexible move-in."
        groups = _extract_n(engine, msg, "obscure_budget_perperson_vague")
        budgets = [g.budget_nis for g in groups]
        sizes = [g.household_size for g in groups]
        _record(test="test_price_per_person_with_unclear_group_size", message=msg,
                field="budget_nis", expected=None, values=budgets, kind="observe", passed=True)
        _record(test="test_price_per_person_with_unclear_group_size", message=msg,
                field="household_size", expected=None, values=sizes, kind="observe", passed=True)
        for b in budgets:
            if b is not None:
                assert b > 3000, f"Per-person budget must be multiplied, got {b}"


# ---------------------------------------------------------------------------
# Obscure employment probes
# ---------------------------------------------------------------------------

class TestObscureEmployment:
    """
    Employment signals that are indirect, contextual, or temporally qualified.
    """

    def test_on_parental_leave_is_not_unemployed(self, engine):
        msg = "I'm currently on maternity leave but going back to work in two months. Looking for a place alone, no pets, move in August."
        groups = _extract_n(engine, msg, "obscure_emp_parental")
        statuses = [p.employment_status for g in groups for p in g.applicants]
        passed = all(s != EmploymentStatus.UNEMPLOYED for s in statuses)
        _record(test="test_on_parental_leave_is_not_unemployed", message=msg,
                field="employment_status", expected="not unemployed", values=statuses,
                kind="correctness", passed=passed)
        for s in statuses:
            assert s != EmploymentStatus.UNEMPLOYED, f"Maternity leave must not be classified as unemployed, got {s}"

    def test_between_jobs_is_consistently_classified(self, engine):
        msg = "I'm between jobs right now but have strong savings. Looking for a place alone, no pets."
        groups = _extract_n(engine, msg, "obscure_emp_between")
        statuses = [p.employment_status for g in groups for p in g.applicants]
        _assert_stable(statuses, "employment_status",
                       message=msg, test="test_between_jobs_is_consistently_classified")
