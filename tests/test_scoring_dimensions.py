"""
Tests for the per-dimension compatibility functions in scoring/vectors.py.

Shared dims (budget, pets, move_in, occupants) accept a TenantGroup.
Per-person dims (employment, age, gender) are tested at two levels:
  - _*_compat_one: single-person function (TenantProfile)
  - _*_compat: group-level averaged function (TenantGroup)

Dealbreaker and is_empty_group are also tested here.
"""

import math
import pytest
from datetime import date

from rentflow.offer.models import EmploymentStatus, Gender, ScoringCriteria, TenantGroup, TenantProfile
from rentflow.scoring.vectors import (
    _UNKNOWN,
    _STUDENT_COMPAT,
    _age_compat,
    _age_compat_one,
    _budget_compat,
    _employment_compat,
    _employment_compat_one,
    _gender_compat,
    _gender_compat_one,
    _move_in_compat,
    _occupants_compat,
    _pets_compat,
    is_dealbreaker,
    is_empty_group,
    _MOVE_IN_DECAY,
)

RENT = 7000
FLOOR = 6000


def _criteria(**kwargs) -> ScoringCriteria:
    defaults = dict(
        lowest_price_nis=FLOOR,
        pets_allowed=False,
        move_in_by=date(2026, 8, 31),
        move_in_strict=True,
        employment_required=True,
        max_occupants=2,
        min_age=23, max_age=33,
        preferred_gender=Gender.FEMALE,
    )
    defaults.update(kwargs)
    return ScoringCriteria(**defaults)


def _group(**kwargs) -> TenantGroup:
    """Minimal group with one null applicant. Override any field."""
    person_fields = {"employment_status", "age", "gender", "name", "phone"}
    person_kwargs = {k: v for k, v in kwargs.items() if k in person_fields}
    group_kwargs = {k: v for k, v in kwargs.items() if k not in person_fields}
    applicant = TenantProfile(**person_kwargs) if person_kwargs else TenantProfile()
    return TenantGroup(applicants=[applicant], **group_kwargs)


def _group_no_applicants(**kwargs) -> TenantGroup:
    return TenantGroup(applicants=[], **kwargs)


# ---------------------------------------------------------------------------
# Budget (group-level)
# ---------------------------------------------------------------------------

class TestBudgetDimension:
    def test_budget_above_asking_scores_full(self):
        assert _budget_compat(_group(budget_nis=8000), _criteria(), RENT) == 1.0

    def test_budget_exactly_at_asking_scores_full(self):
        assert _budget_compat(_group(budget_nis=RENT), _criteria(), RENT) == 1.0

    def test_budget_midpoint_in_band_scores_near_half(self):
        expected = (6500 - FLOOR + 1) / (RENT - FLOOR + 1)
        assert _budget_compat(_group(budget_nis=6500), _criteria(), RENT) == pytest.approx(expected)

    def test_budget_at_floor_scores_small_positive_not_zero(self):
        expected = 1 / (RENT - FLOOR + 1)
        result = _budget_compat(_group(budget_nis=FLOOR), _criteria(), RENT)
        assert result == pytest.approx(expected)
        assert result > 0.0

    def test_budget_below_floor_scores_zero(self):
        assert _budget_compat(_group(budget_nis=5000), _criteria(), RENT) == 0.0

    def test_budget_interpolation_is_linear(self):
        expected = (6250 - FLOOR + 1) / (RENT - FLOOR + 1)
        assert _budget_compat(_group(budget_nis=6250), _criteria(), RENT) == pytest.approx(expected)

    def test_unknown_budget_scores_full_regardless_of_floor(self):
        assert _budget_compat(_group(budget_nis=None), _criteria(), RENT) == 1.0

    def test_unknown_budget_scores_full_when_no_floor_set(self):
        assert _budget_compat(_group(budget_nis=None), _criteria(lowest_price_nis=None), RENT) == 1.0

    def test_no_floor_set_any_stated_budget_scores_full(self):
        assert _budget_compat(_group(budget_nis=3000), _criteria(lowest_price_nis=None), RENT) == 1.0


# ---------------------------------------------------------------------------
# Pets (group-level)
# ---------------------------------------------------------------------------

class TestPetsDimension:
    def test_no_pets_when_forbidden_scores_full(self):
        assert _pets_compat(_group(has_pets=False), _criteria(pets_allowed=False)) == 1.0

    def test_has_pets_when_forbidden_scores_zero(self):
        assert _pets_compat(_group(has_pets=True), _criteria(pets_allowed=False)) == 0.0

    def test_unknown_pets_when_forbidden_scores_unknown(self):
        assert _pets_compat(_group(has_pets=None), _criteria(pets_allowed=False)) == _UNKNOWN

    def test_pets_allowed_with_pets_scores_full(self):
        assert _pets_compat(_group(has_pets=True), _criteria(pets_allowed=True)) == 1.0

    def test_pets_allowed_without_pets_scores_full(self):
        assert _pets_compat(_group(has_pets=False), _criteria(pets_allowed=True)) == 1.0

    def test_pets_allowed_unknown_scores_full(self):
        assert _pets_compat(_group(has_pets=None), _criteria(pets_allowed=True)) == 1.0


# ---------------------------------------------------------------------------
# Move-in (group-level)
# ---------------------------------------------------------------------------

class TestMoveInDimension:
    def test_no_deadline_set_scores_full(self):
        assert _move_in_compat(_group(move_in_date=date(2026, 12, 1)), _criteria(move_in_by=None)) == 1.0

    def test_no_stated_date_scores_full(self):
        assert _move_in_compat(_group(move_in_date=None), _criteria()) == 1.0

    def test_on_deadline_scores_full(self):
        assert _move_in_compat(_group(move_in_date=date(2026, 8, 31)), _criteria()) == 1.0

    def test_2_days_late_matches_decay_formula(self):
        c = _criteria(move_in_by=date(2026, 8, 31))
        g = _group(move_in_date=date(2026, 9, 2))
        expected = 1.0 / (1.0 + _MOVE_IN_DECAY * 2)
        assert _move_in_compat(g, c) == pytest.approx(expected)

    def test_7_days_late_matches_decay_formula(self):
        c = _criteria(move_in_by=date(2026, 8, 31))
        g = _group(move_in_date=date(2026, 9, 7))
        expected = 1.0 / (1.0 + _MOVE_IN_DECAY * 7)
        assert _move_in_compat(g, c) == pytest.approx(expected)

    def test_further_from_deadline_scores_lower(self):
        c = _criteria(move_in_by=date(2026, 8, 31))
        s7 = _move_in_compat(_group(move_in_date=date(2026, 9, 7)), c)
        s30 = _move_in_compat(_group(move_in_date=date(2026, 9, 30)), c)
        assert s7 > s30


# ---------------------------------------------------------------------------
# Employment (per-person, averaged at group level)
# ---------------------------------------------------------------------------

class TestEmploymentDimension:
    def test_employed_when_required_scores_full(self):
        assert _employment_compat_one(TenantProfile(employment_status=EmploymentStatus.EMPLOYED), _criteria()) == 1.0

    def test_self_employed_when_required_scores_full(self):
        assert _employment_compat_one(TenantProfile(employment_status=EmploymentStatus.SELF_EMPLOYED), _criteria()) == 1.0

    def test_student_when_required_scores_partial(self):
        assert _employment_compat_one(TenantProfile(employment_status=EmploymentStatus.STUDENT), _criteria()) == _STUDENT_COMPAT

    def test_unemployed_when_required_scores_zero(self):
        assert _employment_compat_one(TenantProfile(employment_status=EmploymentStatus.UNEMPLOYED), _criteria()) == 0.0

    def test_unknown_employment_when_required_scores_unknown(self):
        assert _employment_compat_one(TenantProfile(employment_status=None), _criteria()) == _UNKNOWN

    def test_employment_not_required_always_scores_full(self):
        c = _criteria(employment_required=False)
        for status in (EmploymentStatus.STUDENT, EmploymentStatus.UNEMPLOYED, None):
            assert _employment_compat_one(TenantProfile(employment_status=status), c) == 1.0

    def test_group_employment_averages_two_applicants(self):
        # employed (1.0) + student (0.5) → avg = 0.75
        c = _criteria()
        g = TenantGroup(
            applicants=[
                TenantProfile(employment_status=EmploymentStatus.EMPLOYED),
                TenantProfile(employment_status=EmploymentStatus.STUDENT),
            ]
        )
        assert _employment_compat(g, c) == pytest.approx(0.75)

    def test_group_no_applicants_defaults_to_one(self):
        # No applicants → no-op listing constraint → full score
        c = _criteria(employment_required=False)
        assert _employment_compat(_group_no_applicants(), c) == 1.0


# ---------------------------------------------------------------------------
# Occupants (group-level, uses household_size)
# ---------------------------------------------------------------------------

class TestOccupantsDimension:
    def test_alone_within_limit_scores_full(self):
        assert _occupants_compat(_group(household_size=1), _criteria(max_occupants=2)) == 1.0

    def test_at_limit_scores_full(self):
        assert _occupants_compat(_group(household_size=2), _criteria(max_occupants=2)) == 1.0

    def test_one_over_limit_scores_zero(self):
        assert _occupants_compat(_group(household_size=3), _criteria(max_occupants=2)) == 0.0

    def test_unknown_household_falls_back_to_applicant_count(self):
        # household_size=None, 1 applicant → total=1 → within limit 2 → 1.0
        g = TenantGroup(household_size=None, applicants=[TenantProfile()])
        assert _occupants_compat(g, _criteria(max_occupants=2)) == 1.0

    def test_no_occupant_limit_always_scores_full(self):
        assert _occupants_compat(_group(household_size=10), _criteria(max_occupants=None)) == 1.0


# ---------------------------------------------------------------------------
# Age (per-person, averaged at group level)
# ---------------------------------------------------------------------------

class TestAgeDimension:
    def test_age_at_lower_bound_scores_full(self):
        assert _age_compat_one(TenantProfile(age=23), _criteria(min_age=23, max_age=33)) == 1.0

    def test_age_at_upper_bound_scores_full(self):
        assert _age_compat_one(TenantProfile(age=33), _criteria(min_age=23, max_age=33)) == 1.0

    def test_age_inside_range_scores_full(self):
        assert _age_compat_one(TenantProfile(age=28), _criteria(min_age=23, max_age=33)) == 1.0

    def test_age_one_half_range_above_max_scores_half(self):
        assert _age_compat_one(TenantProfile(age=38), _criteria(min_age=23, max_age=33)) == pytest.approx(0.5)

    def test_age_one_half_range_below_min_scores_half(self):
        assert _age_compat_one(TenantProfile(age=18), _criteria(min_age=23, max_age=33)) == pytest.approx(0.5)

    def test_further_outside_range_scores_lower(self):
        c = _criteria(min_age=23, max_age=33)
        assert _age_compat_one(TenantProfile(age=48), c) < _age_compat_one(TenantProfile(age=38), c)

    def test_no_age_preference_always_scores_full(self):
        assert _age_compat_one(TenantProfile(age=70), _criteria(min_age=None, max_age=None)) == 1.0

    def test_unknown_age_scores_unknown(self):
        assert _age_compat_one(TenantProfile(age=None), _criteria()) == _UNKNOWN

    def test_group_age_averages_two_applicants(self):
        # age=28 (1.0, in-range) + age=38 (0.5, half-range above max) → avg = 0.75
        c = _criteria(min_age=23, max_age=33)
        g = TenantGroup(applicants=[TenantProfile(age=28), TenantProfile(age=38)])
        assert _age_compat(g, c) == pytest.approx(0.75)

    def test_degenerate_range_min_equals_max_does_not_crash(self):
        c = _criteria(min_age=28, max_age=28)
        result = _age_compat_one(TenantProfile(age=30), c)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Gender (per-person, averaged at group level)
# ---------------------------------------------------------------------------

class TestGenderDimension:
    def test_matching_gender_scores_full(self):
        assert _gender_compat_one(TenantProfile(gender=Gender.FEMALE), _criteria(preferred_gender=Gender.FEMALE)) == 1.0

    def test_mismatched_gender_scores_zero(self):
        assert _gender_compat_one(TenantProfile(gender=Gender.MALE), _criteria(preferred_gender=Gender.FEMALE)) == 0.0

    def test_unknown_gender_scores_unknown(self):
        assert _gender_compat_one(TenantProfile(gender=None), _criteria(preferred_gender=Gender.FEMALE)) == _UNKNOWN

    def test_no_gender_preference_always_scores_full(self):
        assert _gender_compat_one(TenantProfile(gender=Gender.MALE), _criteria(preferred_gender=None)) == 1.0

    def test_group_gender_averages_mixed_couple(self):
        # female (1.0) + male (0.0) → avg = 0.5
        c = _criteria(preferred_gender=Gender.FEMALE)
        g = TenantGroup(applicants=[TenantProfile(gender=Gender.FEMALE), TenantProfile(gender=Gender.MALE)])
        assert _gender_compat(g, c) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Dealbreaker detection (group-level)
# ---------------------------------------------------------------------------

class TestDealbreaker:
    def test_group_with_pets_when_forbidden_is_dealbreaker(self):
        violated, reason = is_dealbreaker(_group(has_pets=True), _criteria(pets_allowed=False), RENT)
        assert violated is True
        assert "pets" in reason.lower()

    def test_group_without_pets_when_forbidden_is_not_dealbreaker(self):
        violated, _ = is_dealbreaker(_group(has_pets=False), _criteria(pets_allowed=False), RENT)
        assert violated is False

    def test_unknown_pets_when_forbidden_is_not_dealbreaker(self):
        violated, _ = is_dealbreaker(_group(has_pets=None), _criteria(pets_allowed=False), RENT)
        assert violated is False

    def test_budget_below_floor_is_dealbreaker(self):
        violated, reason = is_dealbreaker(_group(budget_nis=5000), _criteria(lowest_price_nis=6000), RENT)
        assert violated is True
        assert "floor" in reason.lower()

    def test_budget_in_band_is_not_dealbreaker(self):
        violated, _ = is_dealbreaker(_group(budget_nis=6500), _criteria(lowest_price_nis=6000), RENT)
        assert violated is False

    def test_unknown_budget_is_not_dealbreaker(self):
        violated, _ = is_dealbreaker(_group(budget_nis=None), _criteria(lowest_price_nis=6000), RENT)
        assert violated is False

    def test_strict_deadline_missed_is_dealbreaker(self):
        c = _criteria(move_in_by=date(2026, 8, 31), move_in_strict=True)
        violated, reason = is_dealbreaker(_group(move_in_date=date(2026, 10, 1)), c, RENT)
        assert violated is True

    def test_strict_deadline_met_is_not_dealbreaker(self):
        c = _criteria(move_in_by=date(2026, 8, 31), move_in_strict=True)
        violated, _ = is_dealbreaker(_group(move_in_date=date(2026, 6, 1)), c, RENT)
        assert violated is False

    def test_non_strict_deadline_missed_is_not_dealbreaker(self):
        c = _criteria(move_in_by=date(2026, 8, 31), move_in_strict=False)
        violated, _ = is_dealbreaker(_group(move_in_date=date(2026, 12, 1)), c, RENT)
        assert violated is False

    def test_household_size_over_limit_is_dealbreaker(self):
        # household_size=3, limit=2 → veto
        g = TenantGroup(household_size=3, applicants=[TenantProfile()])
        violated, reason = is_dealbreaker(g, _criteria(max_occupants=2), RENT)
        assert violated is True
        assert "3" in reason and "2" in reason

    def test_household_size_at_limit_is_not_dealbreaker(self):
        g = TenantGroup(household_size=2, applicants=[TenantProfile()])
        violated, _ = is_dealbreaker(g, _criteria(max_occupants=2), RENT)
        assert violated is False

    def test_pets_veto_fires_even_when_employment_is_fine(self):
        # Dealbreaker fires regardless of other dims being fine
        g = TenantGroup(
            has_pets=True,
            applicants=[TenantProfile(employment_status=EmploymentStatus.EMPLOYED)]
        )
        violated, _ = is_dealbreaker(g, _criteria(pets_allowed=False), RENT)
        assert violated is True


# ---------------------------------------------------------------------------
# Empty group detection
# ---------------------------------------------------------------------------

class TestEmptyGroup:
    def test_all_none_with_no_applicants_is_empty(self):
        assert is_empty_group(TenantGroup(applicants=[])) is True

    def test_all_none_with_one_null_applicant_is_empty(self):
        assert is_empty_group(TenantGroup(applicants=[TenantProfile()])) is True

    def test_group_with_budget_is_not_empty(self):
        assert is_empty_group(TenantGroup(budget_nis=5000, applicants=[TenantProfile()])) is False

    def test_group_with_has_pets_false_is_not_empty(self):
        assert is_empty_group(TenantGroup(has_pets=False, applicants=[TenantProfile()])) is False

    def test_group_with_applicant_employment_is_not_empty(self):
        g = TenantGroup(applicants=[TenantProfile(employment_status=EmploymentStatus.EMPLOYED)])
        assert is_empty_group(g) is False

    def test_non_screening_fields_do_not_count(self):
        # name/phone on an applicant are not scoring fields; still empty
        g = TenantGroup(applicants=[TenantProfile(name="Dan", phone="050")])
        assert is_empty_group(g) is True
