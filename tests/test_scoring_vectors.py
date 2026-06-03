"""
Tests for vector construction, weight activation, and normalization in scoring/vectors.py.

The scoring engine represents both the landlord's ideal and the household's group
as 7-dimensional vectors. Shared dims (budget, pets, move_in, occupants) are scored
at the group level. Per-person dims (employment, age, gender) are averaged across applicants.
"""

import math
import pytest
from datetime import date

from rentflow.offer.models import EmploymentStatus, Gender, ScoringCriteria, TenantGroup, TenantProfile
from rentflow.scoring.vectors import (
    _scale,
    _weights,
    cosine_similarity,
    criteria_to_vector,
    group_to_vector,
    is_empty_group,
)

RENT = 7000
FLOOR = 6000

DIM_BUDGET = 0
DIM_PETS = 1
DIM_MOVE_IN = 2
DIM_EMPLOYMENT = 3
DIM_OCCUPANTS = 4
DIM_AGE = 5
DIM_GENDER = 6


def _group(**kwargs) -> TenantGroup:
    """Minimal group with one applicant. Split kwargs between group and person."""
    person_fields = {"employment_status", "age", "gender"}
    person_kwargs = {k: v for k, v in kwargs.items() if k in person_fields}
    group_kwargs = {k: v for k, v in kwargs.items() if k not in person_fields}
    applicant = TenantProfile(**person_kwargs) if person_kwargs else TenantProfile()
    return TenantGroup(applicants=[applicant], **group_kwargs)


# ---------------------------------------------------------------------------
# Weight activation per listing archetype
# ---------------------------------------------------------------------------

class TestWeightActivation:
    def test_minimal_listing_only_budget_slot_active(self):
        c = ScoringCriteria(budget_weight=10)
        w = _weights(c)
        assert w[DIM_BUDGET] == 10
        assert w[DIM_PETS] == 0
        assert w[DIM_MOVE_IN] == 0
        assert w[DIM_EMPLOYMENT] == 0
        assert w[DIM_OCCUPANTS] == 0
        assert w[DIM_AGE] == 0
        assert w[DIM_GENDER] == 0

    def test_strict_screener_all_seven_slots_active(self):
        c = ScoringCriteria(
            budget_weight=10,
            pets_allowed=False, pets_weight=25,
            move_in_by=date(2026, 8, 31), move_in_weight=20,
            employment_required=True, employment_weight=15,
            max_occupants=2, occupants_weight=10,
            min_age=23, max_age=33, age_weight=10,
            preferred_gender=Gender.FEMALE, gender_weight=10,
        )
        w = _weights(c)
        assert all(wi > 0 for wi in w)

    def test_pets_forbidden_activates_pets_slot(self):
        c = ScoringCriteria(pets_allowed=False, pets_weight=25)
        assert _weights(c)[DIM_PETS] == 25

    def test_pets_allowed_deactivates_pets_slot(self):
        c = ScoringCriteria(pets_allowed=True, pets_weight=25)
        assert _weights(c)[DIM_PETS] == 0

    def test_age_range_activates_age_slot(self):
        c = ScoringCriteria(min_age=25, max_age=35, age_weight=10)
        assert _weights(c)[DIM_AGE] == 10

    def test_no_age_range_deactivates_age_slot(self):
        c = ScoringCriteria(min_age=None, max_age=None, age_weight=10)
        assert _weights(c)[DIM_AGE] == 0

    def test_gender_preference_activates_gender_slot(self):
        c = ScoringCriteria(preferred_gender=Gender.MALE, gender_weight=10)
        assert _weights(c)[DIM_GENDER] == 10

    def test_no_gender_preference_deactivates_gender_slot(self):
        c = ScoringCriteria(preferred_gender=None, gender_weight=10)
        assert _weights(c)[DIM_GENDER] == 0


# ---------------------------------------------------------------------------
# sqrt(weight) scaling
# ---------------------------------------------------------------------------

class TestScaling:
    def test_scale_applies_sqrt_of_weight(self):
        dims = [1.0, 0.5, 0.0]
        weights = [4.0, 9.0, 16.0]
        result = _scale(dims, weights)
        assert result[0] == pytest.approx(1.0 * math.sqrt(4.0))
        assert result[1] == pytest.approx(0.5 * math.sqrt(9.0))
        assert result[2] == pytest.approx(0.0 * math.sqrt(16.0))

    def test_scale_zero_weight_yields_zero_regardless_of_dim(self):
        assert _scale([1.0], [0.0])[0] == 0.0

    def test_scale_weight_one_leaves_dim_unchanged(self):
        assert _scale([0.7], [1.0])[0] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# criteria_to_vector
# ---------------------------------------------------------------------------

class TestCriteriaVector:
    def test_criteria_vector_length_is_seven(self):
        assert len(criteria_to_vector(ScoringCriteria())) == 7

    def test_active_dims_equal_sqrt_of_weight(self):
        c = ScoringCriteria(budget_weight=10, pets_allowed=False, pets_weight=25)
        cv = criteria_to_vector(c)
        assert cv[DIM_BUDGET] == pytest.approx(math.sqrt(10))
        assert cv[DIM_PETS] == pytest.approx(math.sqrt(25))

    def test_inactive_dims_are_zero(self):
        c = ScoringCriteria(budget_weight=10)
        cv = criteria_to_vector(c)
        assert cv[DIM_PETS] == 0.0
        assert cv[DIM_MOVE_IN] == 0.0
        assert cv[DIM_AGE] == 0.0
        assert cv[DIM_GENDER] == 0.0

    def test_denominator_equals_sum_of_active_weights(self):
        c = ScoringCriteria(
            budget_weight=10,
            pets_allowed=False, pets_weight=25,
            employment_required=True, employment_weight=15,
        )
        cv = criteria_to_vector(c)
        denominator = sum(x * x for x in cv)
        assert denominator == pytest.approx(10 + 25 + 15)


# ---------------------------------------------------------------------------
# group_to_vector
# ---------------------------------------------------------------------------

class TestGroupVector:
    def test_group_vector_length_is_seven(self):
        c = ScoringCriteria()
        assert len(group_to_vector(TenantGroup(applicants=[]), c, RENT)) == 7

    def test_ideal_group_vector_matches_criteria_vector(self):
        c = ScoringCriteria(
            lowest_price_nis=FLOOR,
            pets_allowed=False, pets_weight=25,
            move_in_by=date(2026, 8, 31), move_in_weight=20,
            employment_required=True, employment_weight=15,
            max_occupants=2, occupants_weight=10,
            min_age=23, max_age=33, age_weight=10,
            preferred_gender=Gender.FEMALE, gender_weight=10,
            budget_weight=10,
        )
        g = TenantGroup(
            budget_nis=RENT,
            has_pets=False,
            move_in_date=date(2026, 8, 31),
            household_size=2,
            applicants=[
                TenantProfile(
                    employment_status=EmploymentStatus.EMPLOYED,
                    age=28,
                    gender=Gender.FEMALE,
                )
            ],
        )
        cv = criteria_to_vector(c)
        gv = group_to_vector(g, c, RENT)
        for i, (ci, gi) in enumerate(zip(cv, gv)):
            assert ci == pytest.approx(gi), f"Dim {i} mismatch: criteria={ci}, group={gi}"

    def test_inactive_dim_is_zero_in_group_vector(self):
        c = ScoringCriteria(preferred_gender=None, gender_weight=10)
        g = _group(gender=Gender.MALE)
        gv = group_to_vector(g, c, RENT)
        assert gv[DIM_GENDER] == 0.0


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_budget_gradient_is_monotonically_increasing(self):
        c = ScoringCriteria(lowest_price_nis=FLOOR, budget_weight=10)
        budgets = [FLOOR, 6250, 6500, 6750, RENT]
        scores = [group_to_vector(_group(budget_nis=b), c, RENT)[DIM_BUDGET] for b in budgets]
        assert scores == sorted(scores)

    def test_age_decay_is_monotonically_decreasing_outside_range(self):
        c = ScoringCriteria(min_age=23, max_age=33, age_weight=10)
        ages_outside = [34, 38, 43, 53]
        scores = [group_to_vector(_group(age=a), c, RENT)[DIM_AGE] for a in ages_outside]
        assert scores == sorted(scores, reverse=True)

    def test_move_in_decay_is_monotonically_decreasing_with_distance(self):
        c = ScoringCriteria(move_in_by=date(2026, 8, 31), move_in_weight=20)
        dates = [date(2026, 9, 2), date(2026, 9, 9), date(2026, 9, 30), date(2026, 11, 1)]
        scores = [group_to_vector(_group(move_in_date=d), c, RENT)[DIM_MOVE_IN] for d in dates]
        assert scores == sorted(scores, reverse=True)

    def test_no_dimension_exceeds_its_weighted_ceiling(self):
        c = ScoringCriteria(
            lowest_price_nis=FLOOR, budget_weight=10,
            pets_allowed=False, pets_weight=25,
            move_in_by=date(2026, 8, 31), move_in_weight=20,
            employment_required=True, employment_weight=15,
            max_occupants=2, occupants_weight=10,
            min_age=23, max_age=33, age_weight=10,
            preferred_gender=Gender.FEMALE, gender_weight=10,
        )
        g = TenantGroup(
            budget_nis=RENT + 1000, has_pets=False, move_in_date=date(2026, 8, 31),
            household_size=1,
            applicants=[TenantProfile(
                employment_status=EmploymentStatus.EMPLOYED, age=28, gender=Gender.FEMALE
            )],
        )
        cv = criteria_to_vector(c)
        gv = group_to_vector(g, c, RENT)
        for i, (ci, gi) in enumerate(zip(cv, gv)):
            assert gi <= ci + 1e-9, f"Dim {i}: group value {gi} exceeds criteria ceiling {ci}"

    def test_inactive_dims_do_not_dilute_score(self):
        c = ScoringCriteria(
            budget_weight=10,
            pets_allowed=False, pets_weight=25,
            employment_required=True, employment_weight=15,
        )
        g = TenantGroup(
            budget_nis=RENT, has_pets=False,
            applicants=[TenantProfile(employment_status=EmploymentStatus.EMPLOYED)],
        )
        cv = criteria_to_vector(c)
        gv = group_to_vector(g, c, RENT)
        denom = sum(x * x for x in cv)
        ratio = sum(a * b for a, b in zip(gv, cv)) / denom
        assert ratio == pytest.approx(1.0)

    def test_heavy_dim_outweighs_light_dim(self):
        c_pets_heavy = ScoringCriteria(
            pets_allowed=False, pets_weight=80,
            preferred_gender=Gender.FEMALE, gender_weight=20,
        )
        c_gender_heavy = ScoringCriteria(
            pets_allowed=False, pets_weight=20,
            preferred_gender=Gender.FEMALE, gender_weight=80,
        )
        group_a = TenantGroup(has_pets=False, applicants=[TenantProfile(gender=Gender.MALE)])
        group_b = TenantGroup(has_pets=True, applicants=[TenantProfile(gender=Gender.FEMALE)])

        def dot_ratio(g, c):
            gv = group_to_vector(g, c, RENT)
            cv = criteria_to_vector(c)
            denom = sum(x * x for x in cv)
            return sum(a * b for a, b in zip(gv, cv)) / denom if denom else 0.0

        assert dot_ratio(group_a, c_pets_heavy) > dot_ratio(group_b, c_pets_heavy)
        assert dot_ratio(group_b, c_gender_heavy) > dot_ratio(group_a, c_gender_heavy)


# ---------------------------------------------------------------------------
# cosine_similarity utility
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_give_one(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_give_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_returns_zero_not_error(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_partial_match_hand_computed(self):
        a = [1.0, 1.0, 0.0]
        b = [1.0, 1.0, 1.0]
        expected = 2.0 / (math.sqrt(2) * math.sqrt(3))
        assert cosine_similarity(a, b) == pytest.approx(expected)
