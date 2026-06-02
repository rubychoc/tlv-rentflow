"""
Tests for the 7 per-dimension compatibility functions in scoring/vectors.py.

Each function takes a TenantProfile + ScoringCriteria (+ rent_nis for budget)
and returns a float in [0.0, 1.0]. The contract for every dimension is:
  - 1.0  = perfect match
  - 0.0  = worst possible (explicit fail or hard boundary)
  - 0.05 = unknown (tenant did not state this field, _UNKNOWN constant)
  - 1.0  = no criterion set by landlord (dimension is inactive)

Budget is the only gradient dimension (linear interpolation between floor and asking).
Move-in and age use decay functions, not binary jumps.
"""

import math
import pytest
from datetime import date

from rentflow.offer.models import EmploymentStatus, Gender, ScoringCriteria, TenantProfile
from rentflow.scoring.vectors import (
    _UNKNOWN,
    _STUDENT_COMPAT,
    _age_compat,
    _budget_compat,
    _employment_compat,
    _gender_compat,
    _move_in_compat,
    _occupants_compat,
    _pets_compat,
    is_dealbreaker,
    is_empty_profile,
    _MOVE_IN_DECAY,
)

RENT = 7000
FLOOR = 6000


def _criteria(**kwargs) -> ScoringCriteria:
    """Base criteria with all constraints active; override specific fields per test."""
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


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

class TestBudgetDimension:
    def test_budget_above_asking_scores_full(self):
        assert _budget_compat(TenantProfile(budget_nis=8000), _criteria(), RENT) == 1.0

    def test_budget_exactly_at_asking_scores_full(self):
        assert _budget_compat(TenantProfile(budget_nis=RENT), _criteria(), RENT) == 1.0

    def test_budget_midpoint_in_band_scores_near_half(self):
        # Interpolation with +1 shift: (6500 - 6000 + 1) / (7000 - 6000 + 1) = 501/1001
        expected = (6500 - FLOOR + 1) / (RENT - FLOOR + 1)
        assert _budget_compat(TenantProfile(budget_nis=6500), _criteria(), RENT) == pytest.approx(expected)

    def test_budget_at_floor_scores_small_positive_not_zero(self):
        # Exactly at floor → minimum band value: 1 / (rent - floor + 1), not zero
        expected = 1 / (RENT - FLOOR + 1)
        result = _budget_compat(TenantProfile(budget_nis=FLOOR), _criteria(), RENT)
        assert result == pytest.approx(expected)
        assert result > 0.0

    def test_budget_below_floor_scores_zero(self):
        assert _budget_compat(TenantProfile(budget_nis=5000), _criteria(), RENT) == 0.0

    def test_budget_interpolation_is_linear(self):
        # At 25% of the band with +1 shift: (6250 - 6000 + 1) / (7000 - 6000 + 1) = 251/1001
        expected = (6250 - FLOOR + 1) / (RENT - FLOOR + 1)
        assert _budget_compat(TenantProfile(budget_nis=6250), _criteria(), RENT) == pytest.approx(expected)

    def test_unknown_budget_scores_full_regardless_of_floor(self):
        # No stated budget = implicit acceptance of asking price
        assert _budget_compat(TenantProfile(budget_nis=None), _criteria(), RENT) == 1.0

    def test_unknown_budget_scores_full_when_no_floor_set(self):
        assert _budget_compat(TenantProfile(budget_nis=None), _criteria(lowest_price_nis=None), RENT) == 1.0

    def test_no_floor_set_any_stated_budget_scores_full(self):
        # When landlord sets no floor, any tenant budget is acceptable
        assert _budget_compat(TenantProfile(budget_nis=3000), _criteria(lowest_price_nis=None), RENT) == 1.0


# ---------------------------------------------------------------------------
# Pets
# ---------------------------------------------------------------------------

class TestPetsDimension:
    def test_no_pets_when_forbidden_scores_full(self):
        assert _pets_compat(TenantProfile(has_pets=False), _criteria(pets_allowed=False)) == 1.0

    def test_has_pets_when_forbidden_scores_zero(self):
        assert _pets_compat(TenantProfile(has_pets=True), _criteria(pets_allowed=False)) == 0.0

    def test_unknown_pets_when_forbidden_scores_unknown(self):
        assert _pets_compat(TenantProfile(has_pets=None), _criteria(pets_allowed=False)) == _UNKNOWN

    def test_pets_allowed_with_pets_scores_full(self):
        # When pets are welcome, tenant having pets is a perfect match
        assert _pets_compat(TenantProfile(has_pets=True), _criteria(pets_allowed=True)) == 1.0

    def test_pets_allowed_without_pets_scores_full(self):
        # When pets are welcome, no constraint exists — any answer is fine
        assert _pets_compat(TenantProfile(has_pets=False), _criteria(pets_allowed=True)) == 1.0

    def test_pets_allowed_unknown_scores_full(self):
        assert _pets_compat(TenantProfile(has_pets=None), _criteria(pets_allowed=True)) == 1.0


# ---------------------------------------------------------------------------
# Move-in
# ---------------------------------------------------------------------------

class TestMoveInDimension:
    def test_no_deadline_set_scores_full(self):
        assert _move_in_compat(TenantProfile(move_in_date=date(2026, 12, 1)), _criteria(move_in_by=None)) == 1.0

    def test_no_stated_date_scores_full(self):
        # No stated date = implicit acceptance of deadline
        assert _move_in_compat(TenantProfile(move_in_date=None), _criteria()) == 1.0

    def test_on_deadline_scores_full(self):
        assert _move_in_compat(TenantProfile(move_in_date=date(2026, 8, 31)), _criteria()) == 1.0

    def test_2_days_late_matches_decay_formula(self):
        # 1 / (1 + 0.6 * 2) = 1/2.2 ≈ 0.4545
        c = _criteria(move_in_by=date(2026, 8, 31))
        p = TenantProfile(move_in_date=date(2026, 9, 2))
        expected = 1.0 / (1.0 + _MOVE_IN_DECAY * 2)
        assert _move_in_compat(p, c) == pytest.approx(expected)

    def test_7_days_late_matches_decay_formula(self):
        c = _criteria(move_in_by=date(2026, 8, 31))
        p = TenantProfile(move_in_date=date(2026, 9, 7))
        expected = 1.0 / (1.0 + _MOVE_IN_DECAY * 7)
        assert _move_in_compat(p, c) == pytest.approx(expected)

    def test_30_days_late_matches_decay_formula(self):
        # Sept 30 is 30 days after Aug 31
        c = _criteria(move_in_by=date(2026, 8, 31))
        p = TenantProfile(move_in_date=date(2026, 9, 30))
        expected = 1.0 / (1.0 + _MOVE_IN_DECAY * 30)
        assert _move_in_compat(p, c) == pytest.approx(expected)

    def test_further_from_deadline_scores_lower(self):
        # Score decays monotonically as distance from deadline grows
        c = _criteria(move_in_by=date(2026, 8, 31))
        s7 = _move_in_compat(TenantProfile(move_in_date=date(2026, 9, 7)), c)
        s30 = _move_in_compat(TenantProfile(move_in_date=date(2026, 9, 30)), c)
        assert s7 > s30

    def test_early_move_in_also_uses_distance(self):
        # Decay is symmetric in |days|: arriving early is also distance from deadline
        c = _criteria(move_in_by=date(2026, 8, 31))
        early = TenantProfile(move_in_date=date(2026, 8, 24))  # 7 days early
        late = TenantProfile(move_in_date=date(2026, 9, 7))    # 7 days late
        assert _move_in_compat(early, c) == pytest.approx(_move_in_compat(late, c))


# ---------------------------------------------------------------------------
# Employment
# ---------------------------------------------------------------------------

class TestEmploymentDimension:
    def test_employed_when_required_scores_full(self):
        assert _employment_compat(TenantProfile(employment_status=EmploymentStatus.EMPLOYED), _criteria()) == 1.0

    def test_self_employed_when_required_scores_full(self):
        assert _employment_compat(TenantProfile(employment_status=EmploymentStatus.SELF_EMPLOYED), _criteria()) == 1.0

    def test_student_when_required_scores_partial(self):
        # Student gets partial credit: has some income capacity but not stable employment
        assert _employment_compat(TenantProfile(employment_status=EmploymentStatus.STUDENT), _criteria()) == _STUDENT_COMPAT

    def test_student_scores_less_than_employed(self):
        # Student must score strictly below employed/self-employed
        assert _employment_compat(TenantProfile(employment_status=EmploymentStatus.STUDENT), _criteria()) < 1.0

    def test_unemployed_when_required_scores_zero(self):
        assert _employment_compat(TenantProfile(employment_status=EmploymentStatus.UNEMPLOYED), _criteria()) == 0.0

    def test_student_scores_more_than_unemployed(self):
        # Student ranks above unemployed but below employed
        c = _criteria()
        assert (
            _employment_compat(TenantProfile(employment_status=EmploymentStatus.UNEMPLOYED), c)
            < _employment_compat(TenantProfile(employment_status=EmploymentStatus.STUDENT), c)
            < _employment_compat(TenantProfile(employment_status=EmploymentStatus.EMPLOYED), c)
        )

    def test_unknown_employment_when_required_scores_unknown(self):
        assert _employment_compat(TenantProfile(employment_status=None), _criteria()) == _UNKNOWN

    def test_employment_not_required_always_scores_full(self):
        # Any status is fine when landlord has no employment requirement
        c = _criteria(employment_required=False)
        for status in (EmploymentStatus.STUDENT, EmploymentStatus.UNEMPLOYED, None):
            assert _employment_compat(TenantProfile(employment_status=status), c) == 1.0


# ---------------------------------------------------------------------------
# Occupants
# ---------------------------------------------------------------------------

class TestOccupantsDimension:
    def test_alone_within_limit_scores_full(self):
        # num_roommates=0 means 1 total occupant; limit is 2
        assert _occupants_compat(TenantProfile(num_roommates=0), _criteria(max_occupants=2)) == 1.0

    def test_at_limit_scores_full(self):
        # num_roommates=1 means 2 total occupants; limit is 2
        assert _occupants_compat(TenantProfile(num_roommates=1), _criteria(max_occupants=2)) == 1.0

    def test_one_over_limit_scores_zero(self):
        # num_roommates=2 means 3 total occupants; limit is 2
        assert _occupants_compat(TenantProfile(num_roommates=2), _criteria(max_occupants=2)) == 0.0

    def test_unknown_roommates_scores_unknown(self):
        assert _occupants_compat(TenantProfile(num_roommates=None), _criteria(max_occupants=2)) == _UNKNOWN

    def test_no_occupant_limit_always_scores_full(self):
        assert _occupants_compat(TenantProfile(num_roommates=10), _criteria(max_occupants=None)) == 1.0


# ---------------------------------------------------------------------------
# Age
# ---------------------------------------------------------------------------

class TestAgeDimension:
    def test_age_at_lower_bound_scores_full(self):
        assert _age_compat(TenantProfile(age=23), _criteria(min_age=23, max_age=33)) == 1.0

    def test_age_at_upper_bound_scores_full(self):
        assert _age_compat(TenantProfile(age=33), _criteria(min_age=23, max_age=33)) == 1.0

    def test_age_inside_range_scores_full(self):
        assert _age_compat(TenantProfile(age=28), _criteria(min_age=23, max_age=33)) == 1.0

    def test_age_one_half_range_above_max_scores_half(self):
        # range 23-33: half_range=5; age=38 → dist=5 → 1/(1+5/5) = 0.5
        assert _age_compat(TenantProfile(age=38), _criteria(min_age=23, max_age=33)) == pytest.approx(0.5)

    def test_age_one_half_range_below_min_scores_half(self):
        # age=18 → dist=5 below min=23 → same decay as above
        assert _age_compat(TenantProfile(age=18), _criteria(min_age=23, max_age=33)) == pytest.approx(0.5)

    def test_further_outside_range_scores_lower(self):
        # Decay is monotonic: farther from boundary → lower score
        c = _criteria(min_age=23, max_age=33)
        assert _age_compat(TenantProfile(age=48), c) < _age_compat(TenantProfile(age=38), c)

    def test_min_only_range_below_scores_decay(self):
        # Only min_age set; half_range falls back to min_age itself
        c = _criteria(min_age=25, max_age=None)
        assert _age_compat(TenantProfile(age=20), c) < 1.0

    def test_max_only_range_above_scores_decay(self):
        c = _criteria(min_age=None, max_age=35)
        assert _age_compat(TenantProfile(age=40), c) < 1.0

    def test_no_age_preference_always_scores_full(self):
        assert _age_compat(TenantProfile(age=70), _criteria(min_age=None, max_age=None)) == 1.0

    def test_unknown_age_scores_unknown(self):
        assert _age_compat(TenantProfile(age=None), _criteria()) == _UNKNOWN

    def test_degenerate_range_min_equals_max_does_not_crash(self):
        # half_range clamped to ≥1 to avoid division by zero
        c = _criteria(min_age=28, max_age=28)
        result = _age_compat(TenantProfile(age=30), c)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Gender
# ---------------------------------------------------------------------------

class TestGenderDimension:
    def test_matching_gender_scores_full(self):
        assert _gender_compat(TenantProfile(gender=Gender.FEMALE), _criteria(preferred_gender=Gender.FEMALE)) == 1.0

    def test_mismatched_gender_scores_zero(self):
        assert _gender_compat(TenantProfile(gender=Gender.MALE), _criteria(preferred_gender=Gender.FEMALE)) == 0.0

    def test_unknown_gender_scores_unknown(self):
        assert _gender_compat(TenantProfile(gender=None), _criteria(preferred_gender=Gender.FEMALE)) == _UNKNOWN

    def test_no_gender_preference_always_scores_full(self):
        assert _gender_compat(TenantProfile(gender=Gender.MALE), _criteria(preferred_gender=None)) == 1.0


# ---------------------------------------------------------------------------
# Dealbreaker detection
# ---------------------------------------------------------------------------

class TestDealbreaker:
    def test_tenant_with_pets_when_forbidden_is_dealbreaker(self):
        violated, reason = is_dealbreaker(TenantProfile(has_pets=True), _criteria(pets_allowed=False), RENT)
        assert violated is True
        assert "pets" in reason.lower()

    def test_tenant_without_pets_when_forbidden_is_not_dealbreaker(self):
        violated, _ = is_dealbreaker(TenantProfile(has_pets=False), _criteria(pets_allowed=False), RENT)
        assert violated is False

    def test_unknown_pets_when_forbidden_is_not_dealbreaker(self):
        # Unknown pets is penalized in the vector (0.05) but not a hard veto
        violated, _ = is_dealbreaker(TenantProfile(has_pets=None), _criteria(pets_allowed=False), RENT)
        assert violated is False

    def test_pets_allowed_with_pets_is_not_dealbreaker(self):
        violated, _ = is_dealbreaker(TenantProfile(has_pets=True), _criteria(pets_allowed=True), RENT)
        assert violated is False

    def test_budget_below_floor_is_dealbreaker(self):
        violated, reason = is_dealbreaker(TenantProfile(budget_nis=5000), _criteria(lowest_price_nis=6000), RENT)
        assert violated is True
        assert "floor" in reason.lower()

    def test_budget_in_band_is_not_dealbreaker(self):
        violated, _ = is_dealbreaker(TenantProfile(budget_nis=6500), _criteria(lowest_price_nis=6000), RENT)
        assert violated is False

    def test_budget_above_asking_is_not_dealbreaker(self):
        violated, _ = is_dealbreaker(TenantProfile(budget_nis=8000), _criteria(lowest_price_nis=6000), RENT)
        assert violated is False

    def test_unknown_budget_is_not_dealbreaker(self):
        # No stated budget = implicit acceptance; floor not checked
        violated, _ = is_dealbreaker(TenantProfile(budget_nis=None), _criteria(lowest_price_nis=6000), RENT)
        assert violated is False

    def test_no_floor_set_any_budget_is_not_dealbreaker(self):
        violated, _ = is_dealbreaker(TenantProfile(budget_nis=100), _criteria(lowest_price_nis=None), RENT)
        assert violated is False

    def test_strict_deadline_missed_is_dealbreaker(self):
        c = _criteria(move_in_by=date(2026, 8, 31), move_in_strict=True)
        violated, reason = is_dealbreaker(TenantProfile(move_in_date=date(2026, 10, 1)), c, RENT)
        assert violated is True
        assert "deadline" in reason.lower() or "strict" in reason.lower() or "misses" in reason.lower()

    def test_strict_deadline_met_is_not_dealbreaker(self):
        c = _criteria(move_in_by=date(2026, 8, 31), move_in_strict=True)
        violated, _ = is_dealbreaker(TenantProfile(move_in_date=date(2026, 6, 1)), c, RENT)
        assert violated is False

    def test_strict_deadline_no_stated_date_is_not_dealbreaker(self):
        # No stated date = implicit acceptance; deadline not violated
        c = _criteria(move_in_by=date(2026, 8, 31), move_in_strict=True)
        violated, _ = is_dealbreaker(TenantProfile(move_in_date=None), c, RENT)
        assert violated is False

    def test_non_strict_deadline_missed_is_not_dealbreaker(self):
        c = _criteria(move_in_by=date(2026, 8, 31), move_in_strict=False)
        violated, _ = is_dealbreaker(TenantProfile(move_in_date=date(2026, 12, 1)), c, RENT)
        assert violated is False

    def test_occupants_over_limit_is_dealbreaker(self):
        # 1 + 2 roommates = 3 total, limit is 2
        violated, reason = is_dealbreaker(TenantProfile(num_roommates=2), _criteria(max_occupants=2), RENT)
        assert violated is True
        assert "3" in reason and "2" in reason

    def test_occupants_at_limit_is_not_dealbreaker(self):
        # 1 + 1 roommate = 2 total, limit is 2
        violated, _ = is_dealbreaker(TenantProfile(num_roommates=1), _criteria(max_occupants=2), RENT)
        assert violated is False

    def test_move_in_asymmetry_only_late_is_dealbreaker(self):
        # is_dealbreaker only vetoes LATE move-in; arriving EARLY with strict deadline is not a veto
        c = _criteria(move_in_by=date(2026, 8, 31), move_in_strict=True)
        early = TenantProfile(move_in_date=date(2026, 7, 1))  # 61 days early
        late = TenantProfile(move_in_date=date(2026, 10, 1))  # 31 days late
        violated_early, _ = is_dealbreaker(early, c, RENT)
        violated_late, _ = is_dealbreaker(late, c, RENT)
        assert violated_early is False  # early is NOT a hard veto
        assert violated_late is True    # late IS a hard veto


# ---------------------------------------------------------------------------
# Empty profile detection
# ---------------------------------------------------------------------------

class TestEmptyProfile:
    def test_all_none_fields_is_empty(self):
        assert is_empty_profile(TenantProfile()) is True

    def test_any_single_field_set_is_not_empty(self):
        # Setting any one scoreable field makes the profile non-empty
        assert is_empty_profile(TenantProfile(budget_nis=5000)) is False
        assert is_empty_profile(TenantProfile(has_pets=False)) is False
        assert is_empty_profile(TenantProfile(employment_status=EmploymentStatus.EMPLOYED)) is False
        assert is_empty_profile(TenantProfile(num_roommates=0)) is False
        assert is_empty_profile(TenantProfile(age=25)) is False
        assert is_empty_profile(TenantProfile(gender=Gender.FEMALE)) is False

    def test_non_screening_fields_do_not_count(self):
        # name/phone/language are not screening fields; setting them alone → still empty
        p = TenantProfile(name="Dan", phone="050-1234567")
        assert is_empty_profile(p) is True
