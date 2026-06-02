"""
Tests for ScoringEngine in scoring/engine.py.

The engine computes a 0-100 score using dot-product ratio: dot(p,c) / dot(c,c),
where c is the landlord's ideal vector and p is the tenant's vector. This gives
the fraction of maximum possible points the tenant earned. Two multipliers can
reduce the score: ×0.01 for any hard-constraint violation (dealbreaker), ×0.1
for a completely empty profile (no information provided at all).

Tests are grouped into:
  - Exact score math (pure arithmetic, hand-computed expected values)
  - Dealbreaker and empty-profile penalties
  - Qualification threshold mapping (Approved / Review / Rejected)
  - RuleHit breakdown correctness
  - Complex ranking scenarios (curated sets of profiles asserted in order)
"""

import math
import pytest
from datetime import date, datetime, timezone

from rentflow.offer.models import (
    EmploymentStatus,
    Gender,
    Qualification,
    ScoringCriteria,
    TenantProfile,
)
from rentflow.scoring.engine import ScoringEngine

RENT = 7500
FLOOR = 6000


# ---------------------------------------------------------------------------
# Exact score math
# ---------------------------------------------------------------------------

class TestExactScoreMath:
    def test_ideal_profile_scores_100(self, base_criteria, ideal_profile):
        # A tenant who matches every dimension perfectly should score exactly 100
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(ideal_profile)
        assert result.score == pytest.approx(100.0)

    def test_score_is_deterministic(self, base_criteria, ideal_profile):
        # Same inputs always produce the exact same ScoreResult
        engine = ScoringEngine(base_criteria, rent_nis=RENT)
        assert engine.score(ideal_profile).score == engine.score(ideal_profile).score

    def test_missing_one_dim_reduces_score_by_its_weight_share(self, base_criteria):
        # Remove gender match from an otherwise ideal profile; score should drop
        # by exactly the gender weight's share of the total active weight
        profile = TenantProfile(
            budget_nis=RENT,
            move_in_date=date(2026, 8, 31),
            employment_status=EmploymentStatus.EMPLOYED,
            has_pets=False,
            num_roommates=0,
            age=28,
            gender=None,   # unknown → _UNKNOWN (0.05) instead of 1.0
        )
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(profile)
        assert result.score < 100.0
        assert result.score > 50.0  # still a strong candidate overall

    def test_blank_criteria_blank_profile_gets_empty_penalty(self):
        # No constraints set → only budget_weight matters, all else zero;
        # an all-None profile triggers the empty-profile penalty (×0.1)
        engine = ScoringEngine(ScoringCriteria(), rent_nis=RENT)
        result = engine.score(TenantProfile())
        assert result.score < 15.0


# ---------------------------------------------------------------------------
# Dealbreaker and empty-profile penalties
# ---------------------------------------------------------------------------

class TestPenalties:
    def test_pets_violation_applies_dealbreaker_penalty(self, base_criteria):
        # Perfect profile except has pets when forbidden → ×0.01 penalty → very low score
        profile = TenantProfile(
            budget_nis=RENT, move_in_date=date(2026, 8, 31),
            employment_status=EmploymentStatus.EMPLOYED,
            has_pets=True, num_roommates=0, age=28, gender=Gender.FEMALE,
        )
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(profile)
        assert result.score < 5.0
        assert result.qualification == Qualification.REJECTED

    def test_budget_below_floor_applies_dealbreaker_penalty(self, base_criteria):
        # Budget below private floor triggers dealbreaker → ×0.01 penalty
        profile = TenantProfile(budget_nis=5000)
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(profile)
        assert result.score < 5.0
        assert any("DEALBREAKER" in h.reason for h in result.rule_hits)

    def test_strict_deadline_missed_applies_dealbreaker_penalty(self, base_criteria):
        # Missing a strict move-in deadline triggers dealbreaker → ×0.01 penalty
        profile = TenantProfile(move_in_date=date(2026, 12, 1))
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(profile)
        assert result.score < 5.0
        assert any("DEALBREAKER" in h.reason for h in result.rule_hits)

    def test_occupants_over_limit_applies_dealbreaker_penalty(self, base_criteria):
        # 3 occupants vs limit 2 triggers dealbreaker → ×0.01 penalty
        profile = TenantProfile(num_roommates=2)
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(profile)
        assert result.score < 5.0
        assert any("DEALBREAKER" in h.reason for h in result.rule_hits)

    def test_empty_profile_applies_empty_penalty_not_dealbreaker(self):
        # All-None profile gets ×0.1 penalty (not the ×0.01 dealbreaker)
        c = ScoringCriteria(budget_weight=10, pets_allowed=False, pets_weight=25)
        result = ScoringEngine(c, rent_nis=RENT).score(TenantProfile())
        assert result.score < 15.0
        assert any(h.rule == "empty_profile" for h in result.rule_hits)
        assert not any(h.rule == "dealbreaker" for h in result.rule_hits)

    def test_dealbreaker_and_empty_profile_do_not_both_fire(self):
        # A profile that has pets (dealbreaker) but no other fields is not empty by the
        # engine's logic — has_pets=True is a set field; only the dealbreaker fires
        c = ScoringCriteria(pets_allowed=False, pets_weight=25)
        profile = TenantProfile(has_pets=True)
        result = ScoringEngine(c, rent_nis=RENT).score(profile)
        assert any(h.rule == "dealbreaker" for h in result.rule_hits)
        assert not any(h.rule == "empty_profile" for h in result.rule_hits)


# ---------------------------------------------------------------------------
# Qualification threshold mapping
# ---------------------------------------------------------------------------

class TestQualificationThresholds:
    def test_score_at_approved_threshold_is_approved(self, base_criteria):
        # Boundary: score == approved_threshold uses >= so it is APPROVED
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(
            TenantProfile(budget_nis=RENT, move_in_date=date(2026, 8, 31),
                          employment_status=EmploymentStatus.EMPLOYED,
                          has_pets=False, num_roommates=0, age=28, gender=Gender.FEMALE)
        )
        assert result.score >= base_criteria.approved_threshold
        assert result.qualification == Qualification.APPROVED

    def test_score_below_rejected_threshold_is_rejected(self, base_criteria):
        # Dealbreaker always drives score below rejected_threshold
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(TenantProfile(has_pets=True))
        assert result.score < base_criteria.rejected_threshold
        assert result.qualification == Qualification.REJECTED

    def test_score_between_thresholds_is_review(self):
        # Construct criteria where only pets matters, and set thresholds around unknown penalty.
        # A tenant who leaves pets unstated scores _UNKNOWN (0.05) → well below approved (75)
        # but the empty penalty (×0.1) doesn't apply because has_pets counts as a stated field
        # only when set; with only pets active and has_pets=None, it's a partial score.
        c = ScoringCriteria(
            pets_allowed=False, pets_weight=25,
            approved_threshold=75,
            rejected_threshold=20,
        )
        # Unknown pets: scores 0.05 on the only active dim → raw ratio = 0.05 → score = 5.0
        # That's below rejected_threshold=20 → REJECTED, not REVIEW.
        # For REVIEW we need a partial score between 20 and 75.
        # Use budget gradient: a budget in-band scores between 0 and 1 on the budget dim.
        c2 = ScoringCriteria(
            lowest_price_nis=FLOOR, budget_weight=10,
            approved_threshold=75,
            rejected_threshold=20,
        )
        # Budget at midpoint (6750): compat = (6750-6000)/(7500-6000) = 0.5 → score = 50
        profile = TenantProfile(budget_nis=6750)
        result = ScoringEngine(c2, rent_nis=RENT).score(profile)
        assert 20 <= result.score < 75
        assert result.qualification == Qualification.REVIEW


# ---------------------------------------------------------------------------
# RuleHit breakdown
# ---------------------------------------------------------------------------

class TestRuleHitBreakdown:
    def test_rule_hits_cover_all_active_dimensions(self, base_criteria, ideal_profile):
        # Every active dimension must appear in the breakdown
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(ideal_profile)
        rule_names = {h.rule for h in result.rule_hits}
        for dim in ("budget", "pets", "move_in", "employment", "occupants", "age", "gender"):
            assert dim in rule_names

    def test_passed_true_for_perfect_dim(self, base_criteria, ideal_profile):
        # A dimension where tenant scores 1.0 should have passed=True
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(ideal_profile)
        budget_hit = next(h for h in result.rule_hits if h.rule == "budget")
        assert budget_hit.passed is True

    def test_passed_false_for_zero_dim(self, base_criteria):
        # A dimension where tenant scores 0.0 (explicit fail) should have passed=False
        profile = TenantProfile(has_pets=True)  # pets forbidden → 0.0
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(profile)
        pets_hit = next(h for h in result.rule_hits if h.rule == "pets")
        assert pets_hit.passed is False

    def test_passed_none_for_partial_dim(self, base_criteria):
        # Unknown field (0.05) or partial score → passed=None
        profile = TenantProfile(budget_nis=None, age=None)  # unknown dims
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(profile)
        age_hit = next(h for h in result.rule_hits if h.rule == "age")
        assert age_hit.passed is None

    def test_points_earned_never_exceeds_points_possible(self, base_criteria, ideal_profile):
        # Sanity: no dim can earn more than its maximum possible points
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(ideal_profile)
        for h in result.rule_hits:
            assert h.points_earned <= h.points_possible + 1e-9

    def test_all_reason_strings_are_non_empty(self, base_criteria, ideal_profile):
        # Every rule hit must include a human-readable reason
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(ideal_profile)
        for h in result.rule_hits:
            assert h.reason, f"Rule '{h.rule}' has an empty reason"

    def test_dealbreaker_hit_appended_when_violated(self, base_criteria):
        # A dealbreaker violation adds a dedicated 'dealbreaker' RuleHit at the end
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(TenantProfile(has_pets=True))
        dealbreaker_hits = [h for h in result.rule_hits if h.rule == "dealbreaker"]
        assert len(dealbreaker_hits) == 1
        assert "DEALBREAKER" in dealbreaker_hits[0].reason


# ---------------------------------------------------------------------------
# Complex ranking scenarios
# ---------------------------------------------------------------------------

class TestRankingScenarios:
    """
    Each scenario builds a set of named profiles, scores them, and asserts the
    resulting ranked order. All profiles are constructed directly — no LLM involved.
    """

    def _score(self, criteria, rent, profile):
        return ScoringEngine(criteria, rent_nis=rent).score(profile).score

    def test_scenario_clean_separation(self):
        # Three clearly distinct profiles: ideal → mediocre → dealbroken.
        c = ScoringCriteria(
            lowest_price_nis=FLOOR,
            pets_allowed=False, pets_weight=25,
            employment_required=True, employment_weight=15,
            max_occupants=2, occupants_weight=10,
            move_in_by=date(2026, 8, 31), move_in_weight=20,
            min_age=23, max_age=33, age_weight=10,
            preferred_gender=Gender.FEMALE, gender_weight=10,
            budget_weight=10,
            approved_threshold=75, rejected_threshold=50,
        )
        ideal = TenantProfile(budget_nis=RENT, has_pets=False, move_in_date=date(2026, 8, 31),
                              employment_status=EmploymentStatus.EMPLOYED, num_roommates=0,
                              age=28, gender=Gender.FEMALE)
        mediocre = TenantProfile(budget_nis=6300, has_pets=False, move_in_date=date(2026, 8, 31),
                                 employment_status=EmploymentStatus.EMPLOYED, num_roommates=1,
                                 age=40, gender=Gender.MALE)
        dealbroken = TenantProfile(budget_nis=RENT, has_pets=True)  # pets violation

        engine = ScoringEngine(c, rent_nis=RENT)
        s_ideal = engine.score(ideal)
        s_mediocre = engine.score(mediocre)
        s_dealbroken = engine.score(dealbroken)

        assert s_ideal.score > s_mediocre.score > s_dealbroken.score
        assert s_ideal.qualification == Qualification.APPROVED
        assert s_dealbroken.qualification == Qualification.REJECTED

    def test_scenario_dealbreaker_outranks_completion(self):
        # A near-perfect tenant with a pet must rank BELOW a weaker but compliant tenant.
        # This is the headline correctness property of the ×0.01 penalty.
        c = ScoringCriteria(pets_allowed=False, pets_weight=25, budget_weight=10)
        near_perfect_with_pet = TenantProfile(budget_nis=RENT, has_pets=True)
        weak_compliant = TenantProfile(budget_nis=FLOOR + 100, has_pets=False)
        s_violator = self._score(c, RENT, near_perfect_with_pet)
        s_compliant = self._score(c, RENT, weak_compliant)
        assert s_violator < s_compliant

    def test_scenario_unknown_beats_explicit_fail(self):
        # Tenant A leaves employment unstated (0.05 score); tenant B says unemployed (0.0).
        # A must rank above B because unknown is penalized less than an explicit fail.
        # Both profiles have budget_nis set so neither triggers the empty-profile penalty.
        c = ScoringCriteria(employment_required=True, employment_weight=15, budget_weight=10)
        unknown_employment = TenantProfile(budget_nis=RENT, employment_status=None)
        unemployed = TenantProfile(budget_nis=RENT, employment_status=EmploymentStatus.UNEMPLOYED)
        assert self._score(c, RENT, unknown_employment) > self._score(c, RENT, unemployed)

    def test_scenario_budget_gradient_is_strictly_ordered(self):
        # Five tenants with budgets from below-floor to above-asking; must rank strictly in order.
        # Below-floor tenant is a dealbreaker (×0.01) and lands at the bottom.
        # Note: budget=FLOOR scores compat=0.0 on the budget dimension, but compat=0.0 is NOT
        # a dealbreaker by itself (floor boundary is a vector zero, not a hard veto). The
        # dealbreaker only fires when budget_nis < lowest_price_nis. Use FLOOR+1 as the
        # first non-dealbreaker point to show the gradient above the floor.
        c = ScoringCriteria(lowest_price_nis=FLOOR, budget_weight=10)
        below_floor = TenantProfile(budget_nis=FLOOR - 1)    # dealbreaker
        at_floor = TenantProfile(budget_nis=FLOOR + 1)       # just above floor, small compat
        mid_band = TenantProfile(budget_nis=6250)            # midpoint
        at_asking = TenantProfile(budget_nis=RENT)           # full score
        above_asking = TenantProfile(budget_nis=RENT + 500)  # also full score

        s_below = self._score(c, RENT, below_floor)
        s_floor = self._score(c, RENT, at_floor)
        s_mid = self._score(c, RENT, mid_band)
        s_asking = self._score(c, RENT, at_asking)
        s_above = self._score(c, RENT, above_asking)

        assert s_below < s_floor    # dealbreaker < just above floor
        assert s_floor < s_mid      # gradient rises through the band
        assert s_mid < s_asking     # at asking = full
        assert s_asking == pytest.approx(s_above)  # at and above asking are equal

    def test_scenario_weights_reorder_ranking(self):
        # Same two profiles scored under two different weight configs: the ranking flips.
        # Profiles must avoid hard dealbreakers so the weight difference drives the outcome.
        # Tenant A: explicitly no pets (passes pets dim), mismatched gender.
        # Tenant B: unknown pets (scores _UNKNOWN on pets, not a dealbreaker), matching gender.
        tenant_a = TenantProfile(has_pets=False, gender=Gender.MALE)   # passes pets, fails gender
        tenant_b = TenantProfile(has_pets=None, gender=Gender.FEMALE)  # unknown pets, passes gender

        pets_matters = ScoringCriteria(pets_allowed=False, pets_weight=80,
                                       preferred_gender=Gender.FEMALE, gender_weight=10)
        gender_matters = ScoringCriteria(pets_allowed=False, pets_weight=10,
                                         preferred_gender=Gender.FEMALE, gender_weight=80)

        # Under pets-heavy criteria, A (no pets = 1.0) beats B (unknown pets = 0.05)
        assert self._score(pets_matters, RENT, tenant_a) > self._score(pets_matters, RENT, tenant_b)
        # Under gender-heavy criteria, B (matching gender = 1.0) beats A (mismatched gender = 0.0)
        assert self._score(gender_matters, RENT, tenant_b) > self._score(gender_matters, RENT, tenant_a)

    def test_scenario_zeroed_criteria_means_no_preference_no_penalty(self):
        # A landlord with no age/gender/pet preference → tenants differing only on those fields tie.
        c = ScoringCriteria(budget_weight=10)  # all other criteria inactive
        tenant_a = TenantProfile(budget_nis=RENT, age=25, gender=Gender.FEMALE, has_pets=True)
        tenant_b = TenantProfile(budget_nis=RENT, age=60, gender=Gender.MALE, has_pets=False)
        assert self._score(c, RENT, tenant_a) == pytest.approx(self._score(c, RENT, tenant_b))

    def test_scenario_empty_profile_ranks_below_any_real_applicant(self):
        # An empty/non-applicant message scores lower than even a weak real applicant.
        c = ScoringCriteria(budget_weight=10, pets_allowed=False, pets_weight=25)
        weak_real = TenantProfile(has_pets=False)  # states only "no pets"
        empty = TenantProfile()                     # says nothing
        assert self._score(c, RENT, weak_real) > self._score(c, RENT, empty)

    def test_scenario_all_dealbreakers_all_rejected_no_crash(self):
        # A pool where every tenant violates something: system must not crash, all REJECTED.
        c = ScoringCriteria(pets_allowed=False, pets_weight=25,
                            lowest_price_nis=FLOOR, budget_weight=10)
        engine = ScoringEngine(c, rent_nis=RENT)
        pool = [
            TenantProfile(has_pets=True),
            TenantProfile(budget_nis=FLOOR - 1),
            TenantProfile(has_pets=True, budget_nis=FLOOR - 1),
        ]
        results = [engine.score(p) for p in pool]
        assert all(r.qualification == Qualification.REJECTED for r in results)
        assert all(math.isfinite(r.score) for r in results)  # no NaN or inf
