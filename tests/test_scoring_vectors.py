"""
Tests for vector construction, weight activation, normalization, and cosine utility
in scoring/vectors.py.

The scoring engine represents both the landlord's ideal and the tenant's profile as
7-dimensional vectors. Weights control which dimensions matter for a given listing:
a dimension with no criterion set gets weight 0 and drops out entirely.
Dimensions are scaled by sqrt(weight) so cosine similarity properly reflects priorities.

The engine itself uses dot-product ratio (dot(p,c)/dot(c,c)), not raw cosine.
cosine_similarity is a utility present in vectors.py but not called by the engine —
it is tested here as a standalone function, not as part of the scoring formula.
"""

import math
import pytest
from datetime import date

from rentflow.offer.models import EmploymentStatus, Gender, ScoringCriteria, TenantProfile
from rentflow.scoring.vectors import (
    _scale,
    _weights,
    cosine_similarity,
    criteria_to_vector,
    is_empty_profile,
    profile_to_vector,
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


# ---------------------------------------------------------------------------
# Weight activation per listing archetype
# ---------------------------------------------------------------------------

class TestWeightActivation:
    def test_minimal_listing_only_budget_slot_active(self):
        # All soft constraints off: pets welcome, no deadline, no employment req, no cap, no age/gender pref
        c = ScoringCriteria(budget_weight=10)
        w = _weights(c)
        assert w[DIM_BUDGET] == 10
        assert w[DIM_PETS] == 0      # pets_allowed=True → no constraint
        assert w[DIM_MOVE_IN] == 0   # move_in_by=None
        assert w[DIM_EMPLOYMENT] == 0
        assert w[DIM_OCCUPANTS] == 0
        assert w[DIM_AGE] == 0
        assert w[DIM_GENDER] == 0

    def test_strict_screener_all_seven_slots_active(self):
        # Every constraint set → all 7 weights are non-zero
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
        # pets_allowed=False turns on the pets dimension
        c = ScoringCriteria(pets_allowed=False, pets_weight=25)
        w = _weights(c)
        assert w[DIM_PETS] == 25

    def test_pets_allowed_deactivates_pets_slot(self):
        # pets_allowed=True (default) → no constraint → weight zeroed
        c = ScoringCriteria(pets_allowed=True, pets_weight=25)
        w = _weights(c)
        assert w[DIM_PETS] == 0

    def test_age_range_activates_age_slot(self):
        c = ScoringCriteria(min_age=25, max_age=35, age_weight=10)
        w = _weights(c)
        assert w[DIM_AGE] == 10

    def test_no_age_range_deactivates_age_slot(self):
        c = ScoringCriteria(min_age=None, max_age=None, age_weight=10)
        w = _weights(c)
        assert w[DIM_AGE] == 0

    def test_gender_preference_activates_gender_slot(self):
        c = ScoringCriteria(preferred_gender=Gender.MALE, gender_weight=10)
        w = _weights(c)
        assert w[DIM_GENDER] == 10

    def test_no_gender_preference_deactivates_gender_slot(self):
        c = ScoringCriteria(preferred_gender=None, gender_weight=10)
        w = _weights(c)
        assert w[DIM_GENDER] == 0


# ---------------------------------------------------------------------------
# sqrt(weight) scaling
# ---------------------------------------------------------------------------

class TestScaling:
    def test_scale_applies_sqrt_of_weight(self):
        # Each dimension value is multiplied by sqrt of its weight
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
        c = ScoringCriteria(
            budget_weight=10,
            pets_allowed=False, pets_weight=25,
        )
        cv = criteria_to_vector(c)
        assert cv[DIM_BUDGET] == pytest.approx(math.sqrt(10))
        assert cv[DIM_PETS] == pytest.approx(math.sqrt(25))

    def test_inactive_dims_are_zero(self):
        # Minimal listing: only budget active
        c = ScoringCriteria(budget_weight=10)
        cv = criteria_to_vector(c)
        assert cv[DIM_PETS] == 0.0
        assert cv[DIM_MOVE_IN] == 0.0
        assert cv[DIM_AGE] == 0.0
        assert cv[DIM_GENDER] == 0.0

    def test_denominator_equals_sum_of_active_weights(self):
        # dot(c,c) = Σ weight_i for active dims; this is what the engine divides by
        c = ScoringCriteria(
            budget_weight=10,
            pets_allowed=False, pets_weight=25,
            employment_required=True, employment_weight=15,
        )
        cv = criteria_to_vector(c)
        denominator = sum(x * x for x in cv)
        assert denominator == pytest.approx(10 + 25 + 15)


# ---------------------------------------------------------------------------
# profile_to_vector
# ---------------------------------------------------------------------------

class TestProfileVector:
    def test_profile_vector_length_is_seven(self):
        c = ScoringCriteria()
        assert len(profile_to_vector(TenantProfile(), c, RENT)) == 7

    def test_ideal_profile_vector_matches_criteria_vector(self):
        # A perfect tenant's vector equals the criteria vector element-by-element
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
        p = TenantProfile(
            budget_nis=RENT,
            has_pets=False,
            move_in_date=date(2026, 8, 31),
            employment_status=EmploymentStatus.EMPLOYED,
            num_roommates=1,  # 1+1=2 = max_occupants
            age=28,
            gender=Gender.FEMALE,
        )
        cv = criteria_to_vector(c)
        pv = profile_to_vector(p, c, RENT)
        for i, (ci, pi) in enumerate(zip(cv, pv)):
            assert ci == pytest.approx(pi), f"Dim {i} mismatch: criteria={ci}, profile={pi}"

    def test_inactive_dim_is_zero_in_profile_vector(self):
        # A zero-weight dim is zero regardless of the tenant's value on that field
        c = ScoringCriteria(preferred_gender=None, gender_weight=10)  # gender inactive
        p = TenantProfile(gender=Gender.MALE)
        pv = profile_to_vector(p, c, RENT)
        assert pv[DIM_GENDER] == 0.0


# ---------------------------------------------------------------------------
# Normalization: gradient and binary dimensions scale within their weight ceiling
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_budget_gradient_is_monotonically_increasing(self):
        # As tenant budget rises from floor to asking, the scaled budget dim rises
        c = ScoringCriteria(lowest_price_nis=FLOOR, budget_weight=10)
        budgets = [FLOOR, 6250, 6500, 6750, RENT]
        scores = [profile_to_vector(TenantProfile(budget_nis=b), c, RENT)[DIM_BUDGET] for b in budgets]
        assert scores == sorted(scores), "Budget dimension must increase monotonically"

    def test_age_decay_is_monotonically_decreasing_outside_range(self):
        # As tenant age moves further from the preferred range, the scaled age dim falls
        c = ScoringCriteria(min_age=23, max_age=33, age_weight=10)
        ages_outside = [34, 38, 43, 53]
        scores = [profile_to_vector(TenantProfile(age=a), c, RENT)[DIM_AGE] for a in ages_outside]
        assert scores == sorted(scores, reverse=True), "Age dimension must decay monotonically outside range"

    def test_move_in_decay_is_monotonically_decreasing_with_distance(self):
        # As move-in date drifts further from the deadline, the score falls
        c = ScoringCriteria(move_in_by=date(2026, 8, 31), move_in_weight=20)
        dates = [date(2026, 9, 2), date(2026, 9, 9), date(2026, 9, 30), date(2026, 11, 1)]
        scores = [profile_to_vector(TenantProfile(move_in_date=d), c, RENT)[DIM_MOVE_IN] for d in dates]
        assert scores == sorted(scores, reverse=True), "Move-in dimension must decay monotonically"

    def test_no_dimension_exceeds_its_weighted_ceiling(self):
        # Every profile vector dimension must be ≤ the corresponding criteria vector dimension
        c = ScoringCriteria(
            lowest_price_nis=FLOOR, budget_weight=10,
            pets_allowed=False, pets_weight=25,
            move_in_by=date(2026, 8, 31), move_in_weight=20,
            employment_required=True, employment_weight=15,
            max_occupants=2, occupants_weight=10,
            min_age=23, max_age=33, age_weight=10,
            preferred_gender=Gender.FEMALE, gender_weight=10,
        )
        p = TenantProfile(budget_nis=RENT + 1000, has_pets=False, move_in_date=date(2026, 8, 31),
                          employment_status=EmploymentStatus.EMPLOYED, num_roommates=0,
                          age=28, gender=Gender.FEMALE)
        cv = criteria_to_vector(c)
        pv = profile_to_vector(p, c, RENT)
        for i, (ci, pi) in enumerate(zip(cv, pv)):
            assert pi <= ci + 1e-9, f"Dim {i}: profile value {pi} exceeds criteria ceiling {ci}"

    def test_heavy_dim_outweighs_light_dim(self):
        # Two listings with same total weight, different distribution:
        # a tenant perfect on the heavy dim but poor on the light should beat the reverse
        c_pets_heavy = ScoringCriteria(
            pets_allowed=False, pets_weight=80,
            preferred_gender=Gender.FEMALE, gender_weight=20,
        )
        c_gender_heavy = ScoringCriteria(
            pets_allowed=False, pets_weight=20,
            preferred_gender=Gender.FEMALE, gender_weight=80,
        )
        # Tenant A: passes pets (heavy in c_pets_heavy), fails gender
        tenant_a = TenantProfile(has_pets=False, gender=Gender.MALE)
        # Tenant B: fails pets, passes gender (heavy in c_gender_heavy)
        tenant_b = TenantProfile(has_pets=True, gender=Gender.FEMALE)

        def dot_ratio(p, c):
            pv = profile_to_vector(p, c, RENT)
            cv = criteria_to_vector(c)
            denom = sum(x * x for x in cv)
            return sum(a * b for a, b in zip(pv, cv)) / denom if denom else 0.0

        assert dot_ratio(tenant_a, c_pets_heavy) > dot_ratio(tenant_b, c_pets_heavy)
        assert dot_ratio(tenant_b, c_gender_heavy) > dot_ratio(tenant_a, c_gender_heavy)

    def test_inactive_dims_do_not_dilute_score(self):
        # A listing with 3 active dims should still allow a perfect score of 1.0
        # even though 4 dims are zero — they don't pull the denominator down
        c = ScoringCriteria(
            budget_weight=10,
            pets_allowed=False, pets_weight=25,
            employment_required=True, employment_weight=15,
        )
        p = TenantProfile(
            budget_nis=RENT,
            has_pets=False,
            employment_status=EmploymentStatus.EMPLOYED,
        )
        cv = criteria_to_vector(c)
        pv = profile_to_vector(p, c, RENT)
        denom = sum(x * x for x in cv)
        ratio = sum(a * b for a, b in zip(pv, cv)) / denom
        assert ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# cosine_similarity utility (not used by the engine, tested as a standalone)
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_give_one(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_give_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_returns_zero_not_error(self):
        # Guard against division by zero when one vector is all zeros
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_partial_match_hand_computed(self):
        a = [1.0, 1.0, 0.0]
        b = [1.0, 1.0, 1.0]
        expected = 2.0 / (math.sqrt(2) * math.sqrt(3))
        assert cosine_similarity(a, b) == pytest.approx(expected)
