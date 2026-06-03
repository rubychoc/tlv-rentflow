"""
Tests for ScoringEngine in scoring/engine.py.

The engine computes a 0-100 score using dot-product ratio: dot(g,c) / dot(c,c),
where c is the landlord's ideal vector and g is the household group vector.
Shared dims (budget, pets, move_in, occupants) are scored at the group level.
Per-person dims (employment, age, gender) are averaged across applicants.

Hard constraints (dealbreakers) are strict: one member's violation vetoes the group.
"""

import math
import pytest
from datetime import date

from rentflow.offer.models import (
    EmploymentStatus,
    Gender,
    Qualification,
    ScoringCriteria,
    TenantGroup,
    TenantProfile,
)
from rentflow.scoring.engine import ScoringEngine

RENT = 7500
FLOOR = 6000


def _group(**kwargs) -> TenantGroup:
    """Build a minimal TenantGroup. Person-level kwargs go to the applicant."""
    person_fields = {"employment_status", "age", "gender"}
    person_kwargs = {k: v for k, v in kwargs.items() if k in person_fields}
    group_kwargs = {k: v for k, v in kwargs.items() if k not in person_fields}
    applicant = TenantProfile(**person_kwargs) if person_kwargs else TenantProfile()
    return TenantGroup(applicants=[applicant], **group_kwargs)


def _empty_group() -> TenantGroup:
    return TenantGroup(applicants=[])


# ---------------------------------------------------------------------------
# Exact score math
# ---------------------------------------------------------------------------

class TestExactScoreMath:
    def test_ideal_group_scores_100(self, base_criteria, ideal_group):
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(ideal_group)
        assert result.score == pytest.approx(100.0)

    def test_score_is_deterministic(self, base_criteria, ideal_group):
        engine = ScoringEngine(base_criteria, rent_nis=RENT)
        assert engine.score(ideal_group).score == engine.score(ideal_group).score

    def test_missing_one_dim_reduces_score(self, base_criteria):
        g = TenantGroup(
            budget_nis=RENT,
            move_in_date=date(2026, 8, 31),
            has_pets=False,
            household_size=1,
            applicants=[TenantProfile(
                employment_status=EmploymentStatus.EMPLOYED,
                age=28,
                gender=None,   # unknown → _UNKNOWN (0.05)
            )],
        )
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(g)
        assert result.score < 100.0
        assert result.score > 50.0

    def test_blank_criteria_empty_group_gets_empty_penalty(self):
        engine = ScoringEngine(ScoringCriteria(), rent_nis=RENT)
        result = engine.score(_empty_group())
        assert result.score < 15.0


# ---------------------------------------------------------------------------
# Dealbreaker and empty-group penalties
# ---------------------------------------------------------------------------

class TestPenalties:
    def test_pets_violation_applies_dealbreaker_penalty(self, base_criteria):
        g = TenantGroup(
            budget_nis=RENT, move_in_date=date(2026, 8, 31),
            has_pets=True, household_size=1,
            applicants=[TenantProfile(
                employment_status=EmploymentStatus.EMPLOYED, age=28, gender=Gender.FEMALE
            )],
        )
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(g)
        assert result.score < 5.0
        assert result.qualification == Qualification.REJECTED

    def test_budget_below_floor_applies_dealbreaker_penalty(self, base_criteria):
        g = _group(budget_nis=5000)
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(g)
        assert result.score < 5.0
        assert any("DEALBREAKER" in h.reason for h in result.rule_hits)

    def test_strict_deadline_missed_applies_dealbreaker_penalty(self, base_criteria):
        g = _group(move_in_date=date(2026, 12, 1))
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(g)
        assert result.score < 5.0
        assert any("DEALBREAKER" in h.reason for h in result.rule_hits)

    def test_occupants_over_limit_applies_dealbreaker_penalty(self, base_criteria):
        # household_size=3, limit=2 → dealbreaker
        g = TenantGroup(household_size=3, applicants=[TenantProfile()])
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(g)
        assert result.score < 5.0
        assert any("DEALBREAKER" in h.reason for h in result.rule_hits)

    def test_empty_group_applies_empty_penalty_not_dealbreaker(self):
        c = ScoringCriteria(budget_weight=10, pets_allowed=False, pets_weight=25)
        result = ScoringEngine(c, rent_nis=RENT).score(_empty_group())
        assert result.score < 15.0
        assert any(h.rule == "empty_group" for h in result.rule_hits)
        assert not any(h.rule == "dealbreaker" for h in result.rule_hits)

    def test_dealbreaker_and_empty_group_do_not_both_fire(self):
        # has_pets=True (dealbreaker) — not empty because has_pets is set
        c = ScoringCriteria(pets_allowed=False, pets_weight=25)
        g = TenantGroup(has_pets=True, applicants=[])
        result = ScoringEngine(c, rent_nis=RENT).score(g)
        assert any(h.rule == "dealbreaker" for h in result.rule_hits)
        assert not any(h.rule == "empty_group" for h in result.rule_hits)

    def test_one_applicant_pet_veto_fires_for_whole_group(self, base_criteria):
        # Group-level has_pets=True → veto, even though only one member
        g = TenantGroup(
            budget_nis=RENT, move_in_date=date(2026, 8, 31),
            has_pets=True, household_size=2,
            applicants=[
                TenantProfile(employment_status=EmploymentStatus.EMPLOYED, age=28, gender=Gender.FEMALE),
                TenantProfile(employment_status=EmploymentStatus.EMPLOYED, age=30, gender=Gender.FEMALE),
            ],
        )
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(g)
        assert result.score < 5.0


# ---------------------------------------------------------------------------
# Averaging behaviour for per-person dims
# ---------------------------------------------------------------------------

class TestAveragingBehaviour:
    def test_couple_employment_averages_employed_and_student(self):
        # employed=1.0, student=0.5 → avg=0.75 for a couple
        c = ScoringCriteria(employment_required=True, employment_weight=100)
        couple = TenantGroup(
            applicants=[
                TenantProfile(employment_status=EmploymentStatus.EMPLOYED),
                TenantProfile(employment_status=EmploymentStatus.STUDENT),
            ]
        )
        solo_employed = TenantGroup(applicants=[TenantProfile(employment_status=EmploymentStatus.EMPLOYED)])
        solo_student = TenantGroup(applicants=[TenantProfile(employment_status=EmploymentStatus.STUDENT)])
        e = ScoringEngine(c, rent_nis=RENT)
        s_couple = e.score(couple).score
        s_employed = e.score(solo_employed).score
        s_student = e.score(solo_student).score
        assert s_student < s_couple < s_employed

    def test_couple_gender_averages_one_match_one_mismatch(self):
        # female=1.0, male=0.0 → avg=0.5
        c = ScoringCriteria(preferred_gender=Gender.FEMALE, gender_weight=100)
        couple = TenantGroup(
            applicants=[
                TenantProfile(gender=Gender.FEMALE),
                TenantProfile(gender=Gender.MALE),
            ]
        )
        e = ScoringEngine(c, rent_nis=RENT)
        s = e.score(couple).score
        assert 40 < s < 60  # avg=0.5 → should be midway

    def test_couple_scores_between_two_individuals(self):
        # Solo female employed > couple (employed female + unemployed male) > solo unemployed male
        c = ScoringCriteria(
            employment_required=True, employment_weight=40,
            preferred_gender=Gender.FEMALE, gender_weight=60,
        )
        best = TenantGroup(applicants=[TenantProfile(employment_status=EmploymentStatus.EMPLOYED, gender=Gender.FEMALE)])
        couple = TenantGroup(applicants=[
            TenantProfile(employment_status=EmploymentStatus.EMPLOYED, gender=Gender.FEMALE),
            TenantProfile(employment_status=EmploymentStatus.UNEMPLOYED, gender=Gender.MALE),
        ])
        worst = TenantGroup(applicants=[TenantProfile(employment_status=EmploymentStatus.UNEMPLOYED, gender=Gender.MALE)])
        e = ScoringEngine(c, rent_nis=RENT)
        assert e.score(best).score > e.score(couple).score > e.score(worst).score


# ---------------------------------------------------------------------------
# Qualification threshold mapping
# ---------------------------------------------------------------------------

class TestQualificationThresholds:
    def test_score_at_approved_threshold_is_approved(self, base_criteria, ideal_group):
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(ideal_group)
        assert result.score >= base_criteria.approved_threshold
        assert result.qualification == Qualification.APPROVED

    def test_score_below_rejected_threshold_is_rejected(self, base_criteria):
        g = TenantGroup(has_pets=True, applicants=[])
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(g)
        assert result.score < base_criteria.rejected_threshold
        assert result.qualification == Qualification.REJECTED

    def test_score_between_thresholds_is_review(self):
        c2 = ScoringCriteria(
            lowest_price_nis=FLOOR, budget_weight=10,
            approved_threshold=75, rejected_threshold=20,
        )
        g = _group(budget_nis=6750)
        result = ScoringEngine(c2, rent_nis=RENT).score(g)
        assert 20 <= result.score < 75
        assert result.qualification == Qualification.REVIEW


# ---------------------------------------------------------------------------
# RuleHit breakdown
# ---------------------------------------------------------------------------

class TestRuleHitBreakdown:
    def test_rule_hits_cover_all_active_dimensions(self, base_criteria, ideal_group):
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(ideal_group)
        rule_names = {h.rule for h in result.rule_hits}
        for dim in ("budget", "pets", "move_in", "employment", "occupants", "age", "gender"):
            assert dim in rule_names

    def test_passed_true_for_perfect_dim(self, base_criteria, ideal_group):
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(ideal_group)
        budget_hit = next(h for h in result.rule_hits if h.rule == "budget")
        assert budget_hit.passed is True

    def test_passed_false_for_zero_dim(self, base_criteria):
        g = TenantGroup(has_pets=True, applicants=[])
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(g)
        pets_hit = next(h for h in result.rule_hits if h.rule == "pets")
        assert pets_hit.passed is False

    def test_passed_none_for_partial_dim(self, base_criteria):
        g = TenantGroup(budget_nis=None, applicants=[TenantProfile(age=None)])
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(g)
        age_hit = next(h for h in result.rule_hits if h.rule == "age")
        assert age_hit.passed is None

    def test_points_earned_never_exceeds_points_possible(self, base_criteria, ideal_group):
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(ideal_group)
        for h in result.rule_hits:
            assert h.points_earned <= h.points_possible + 1e-9

    def test_all_reason_strings_are_non_empty(self, base_criteria, ideal_group):
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(ideal_group)
        for h in result.rule_hits:
            assert h.reason, f"Rule '{h.rule}' has an empty reason"

    def test_dealbreaker_hit_appended_when_violated(self, base_criteria):
        g = TenantGroup(has_pets=True, applicants=[])
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(g)
        dealbreaker_hits = [h for h in result.rule_hits if h.rule == "dealbreaker"]
        assert len(dealbreaker_hits) == 1
        assert "DEALBREAKER" in dealbreaker_hits[0].reason

    def test_couple_breakdown_shows_avg_compat(self, base_criteria):
        g = TenantGroup(
            budget_nis=RENT, move_in_date=date(2026, 8, 31),
            has_pets=False, household_size=2,
            applicants=[
                TenantProfile(employment_status=EmploymentStatus.EMPLOYED, age=28, gender=Gender.FEMALE),
                TenantProfile(employment_status=EmploymentStatus.STUDENT, age=30, gender=Gender.MALE),
            ],
        )
        result = ScoringEngine(base_criteria, rent_nis=RENT).score(g)
        employ_hit = next(h for h in result.rule_hits if h.rule == "employment")
        # avg of 1.0 and 0.5 = 0.75 → not full pass (1.0) and not full fail (0.0) → passed=None
        assert employ_hit.passed is None


# ---------------------------------------------------------------------------
# Complex ranking scenarios
# ---------------------------------------------------------------------------

class TestRankingScenarios:
    def _score(self, criteria, rent, group):
        return ScoringEngine(criteria, rent_nis=rent).score(group).score

    def test_scenario_clean_separation(self):
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
        ideal = TenantGroup(
            budget_nis=RENT, has_pets=False, move_in_date=date(2026, 8, 31), household_size=1,
            applicants=[TenantProfile(employment_status=EmploymentStatus.EMPLOYED, age=28, gender=Gender.FEMALE)],
        )
        mediocre = TenantGroup(
            budget_nis=6300, has_pets=False, move_in_date=date(2026, 8, 31), household_size=2,
            applicants=[TenantProfile(employment_status=EmploymentStatus.EMPLOYED, age=40, gender=Gender.MALE)],
        )
        dealbroken = TenantGroup(budget_nis=RENT, has_pets=True, applicants=[])

        engine = ScoringEngine(c, rent_nis=RENT)
        s_ideal = engine.score(ideal)
        s_mediocre = engine.score(mediocre)
        s_dealbroken = engine.score(dealbroken)

        assert s_ideal.score > s_mediocre.score > s_dealbroken.score
        assert s_ideal.qualification == Qualification.APPROVED
        assert s_dealbroken.qualification == Qualification.REJECTED

    def test_scenario_dealbreaker_outranks_completion(self):
        c = ScoringCriteria(pets_allowed=False, pets_weight=25, budget_weight=10)
        near_perfect_with_pet = TenantGroup(budget_nis=RENT, has_pets=True, applicants=[])
        weak_compliant = TenantGroup(budget_nis=FLOOR + 100, has_pets=False, applicants=[])
        s_violator = self._score(c, RENT, near_perfect_with_pet)
        s_compliant = self._score(c, RENT, weak_compliant)
        assert s_violator < s_compliant

    def test_scenario_unknown_beats_explicit_fail(self):
        c = ScoringCriteria(employment_required=True, employment_weight=15, budget_weight=10)
        unknown_employ = TenantGroup(budget_nis=RENT, applicants=[TenantProfile(employment_status=None)])
        unemployed = TenantGroup(budget_nis=RENT, applicants=[TenantProfile(employment_status=EmploymentStatus.UNEMPLOYED)])
        assert self._score(c, RENT, unknown_employ) > self._score(c, RENT, unemployed)

    def test_scenario_budget_gradient_is_strictly_ordered(self):
        c = ScoringCriteria(lowest_price_nis=FLOOR, budget_weight=10)
        below_floor = _group(budget_nis=FLOOR - 1)
        at_floor = _group(budget_nis=FLOOR + 1)
        mid_band = _group(budget_nis=6250)
        at_asking = _group(budget_nis=RENT)
        above_asking = _group(budget_nis=RENT + 500)

        s_below = self._score(c, RENT, below_floor)
        s_floor = self._score(c, RENT, at_floor)
        s_mid = self._score(c, RENT, mid_band)
        s_asking = self._score(c, RENT, at_asking)
        s_above = self._score(c, RENT, above_asking)

        assert s_below < s_floor < s_mid < s_asking
        assert s_asking == pytest.approx(s_above)

    def test_scenario_weights_reorder_ranking(self):
        group_a = TenantGroup(has_pets=False, applicants=[TenantProfile(gender=Gender.MALE)])
        group_b = TenantGroup(has_pets=None, applicants=[TenantProfile(gender=Gender.FEMALE)])

        pets_matters = ScoringCriteria(pets_allowed=False, pets_weight=80,
                                       preferred_gender=Gender.FEMALE, gender_weight=10)
        gender_matters = ScoringCriteria(pets_allowed=False, pets_weight=10,
                                         preferred_gender=Gender.FEMALE, gender_weight=80)

        assert self._score(pets_matters, RENT, group_a) > self._score(pets_matters, RENT, group_b)
        assert self._score(gender_matters, RENT, group_b) > self._score(gender_matters, RENT, group_a)

    def test_scenario_zeroed_criteria_means_no_preference_no_penalty(self):
        c = ScoringCriteria(budget_weight=10)
        group_a = TenantGroup(budget_nis=RENT, applicants=[TenantProfile(age=25, gender=Gender.FEMALE)])
        group_b = TenantGroup(budget_nis=RENT, has_pets=False, applicants=[TenantProfile(age=60, gender=Gender.MALE)])
        assert self._score(c, RENT, group_a) == pytest.approx(self._score(c, RENT, group_b))

    def test_scenario_empty_group_ranks_below_any_real_applicant(self):
        c = ScoringCriteria(budget_weight=10, pets_allowed=False, pets_weight=25)
        weak_real = TenantGroup(has_pets=False, applicants=[])
        empty = _empty_group()
        assert self._score(c, RENT, weak_real) > self._score(c, RENT, empty)

    def test_scenario_all_dealbreakers_all_rejected_no_crash(self):
        c = ScoringCriteria(pets_allowed=False, pets_weight=25,
                            lowest_price_nis=FLOOR, budget_weight=10)
        engine = ScoringEngine(c, rent_nis=RENT)
        pool = [
            TenantGroup(has_pets=True, applicants=[]),
            TenantGroup(budget_nis=FLOOR - 1, applicants=[]),
            TenantGroup(has_pets=True, budget_nis=FLOOR - 1, applicants=[]),
        ]
        results = [engine.score(g) for g in pool]
        assert all(r.qualification == Qualification.REJECTED for r in results)
        assert all(math.isfinite(r.score) for r in results)
