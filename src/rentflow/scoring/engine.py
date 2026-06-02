"""
Scoring engine — Station 3 of the TLV-RentFlow pipeline.

Algorithm:
  1. Build a criteria vector (the landlord's ideal) and a profile vector.
  2. Compute dot-product ratio: dot(p, c) / dot(c, c) -> raw score in [0, 1].
     This measures what fraction of maximum possible points the tenant earned.
  3. Apply a dealbreaker penalty (×0.01) if any hard constraint is violated.
  4. Multiply by 100 to yield a 0-100 score.
  5. Map to Approved / Review / Rejected via configurable thresholds.

This module is pure: no I/O, no randomness. Same inputs -> same output.
"""

from __future__ import annotations

import math

from rentflow.offer.models import (
    Qualification,
    RuleHit,
    ScoringCriteria,
    ScoreResult,
    TenantProfile,
)
from rentflow.scoring.vectors import (
    DIM_LABELS,
    _age_compat,
    _budget_compat,
    _employment_compat,
    _gender_compat,
    _move_in_compat,
    _occupants_compat,
    _pets_compat,
    _weights,
    criteria_to_vector,
    is_dealbreaker,
    is_empty_profile,
    profile_to_vector,
)

_DEALBREAKER_PENALTY = 0.01
_EMPTY_PROFILE_PENALTY = 0.1  # completely uninformative message


class ScoringEngine:
    def __init__(self, criteria: ScoringCriteria, rent_nis: int) -> None:
        self._criteria = criteria
        self._rent_nis = rent_nis

    def score(self, profile: TenantProfile) -> ScoreResult:
        """
        Computes the dot-product ratio score for one TenantProfile.

        score = dot(profile_vec, criteria_vec) / dot(criteria_vec, criteria_vec)

        This measures what fraction of maximum possible points the tenant earned,
        giving a true [0, 1] range where every dimension miss is felt proportionally.
        """
        c_vec = criteria_to_vector(self._criteria)
        p_vec = profile_to_vector(profile, self._criteria, self._rent_nis)

        max_score = sum(x * x for x in c_vec)
        raw_ratio = sum(a * b for a, b in zip(p_vec, c_vec)) / max_score if max_score else 0.0

        violated, veto_reason = is_dealbreaker(profile, self._criteria, self._rent_nis)
        if violated:
            multiplier = _DEALBREAKER_PENALTY
        elif is_empty_profile(profile):
            multiplier = _EMPTY_PROFILE_PENALTY
        else:
            multiplier = 1.0

        score = round(raw_ratio * multiplier * 100, 1)

        if score >= self._criteria.approved_threshold:
            qualification = Qualification.APPROVED
        elif score < self._criteria.rejected_threshold:
            qualification = Qualification.REJECTED
        else:
            qualification = Qualification.REVIEW

        empty = is_empty_profile(profile)
        hits = self._build_hits(profile, violated, veto_reason, empty)
        return ScoreResult(score=score, qualification=qualification, rule_hits=hits)

    def _build_hits(
        self,
        profile: TenantProfile,
        dealbreaker_violated: bool,
        veto_reason: str,
        empty_profile: bool = False,
    ) -> list[RuleHit]:
        """Builds the per-dimension breakdown for explainability."""
        c = self._criteria
        r = self._rent_nis

        compat_fns = [
            lambda p: _budget_compat(p, c, r),
            lambda p: _pets_compat(p, c),
            lambda p: _move_in_compat(p, c),
            lambda p: _employment_compat(p, c),
            lambda p: _occupants_compat(p, c),
            lambda p: _age_compat(p, c),
            lambda p: _gender_compat(p, c),
        ]
        weights = _weights(c)
        hits: list[RuleHit] = []

        for label, fn, w in zip(DIM_LABELS, compat_fns, weights):
            compat = fn(profile)
            sw = math.sqrt(w)
            earned = compat * sw
            possible = sw

            if compat == 1.0:
                passed = True
            elif compat == 0.0:
                passed = False
            else:
                passed = None  # partial / unknown

            reason = self._reason(label, profile, compat)
            hits.append(RuleHit(
                rule=label,
                passed=passed,
                points_earned=round(earned, 3),
                points_possible=round(possible, 3),
                reason=reason,
            ))

        if dealbreaker_violated:
            hits.append(RuleHit(
                rule="dealbreaker",
                passed=False,
                points_earned=0.0,
                points_possible=0.0,
                reason=f"DEALBREAKER (×{_DEALBREAKER_PENALTY}): {veto_reason}",
            ))
        elif empty_profile:
            hits.append(RuleHit(
                rule="empty_profile",
                passed=False,
                points_earned=0.0,
                points_possible=0.0,
                reason=f"EMPTY PROFILE (×{_EMPTY_PROFILE_PENALTY}): message contained no tenant information.",
            ))

        return hits

    def _reason(self, label: str, profile: TenantProfile, compat: float) -> str:
        c = self._criteria
        r = self._rent_nis

        if label == "budget":
            if profile.budget_nis is None:
                return "Tenant did not state a budget (implicitly accepts posted price)."
            if c.lowest_price_nis is None:
                return f"Tenant stated {profile.budget_nis} NIS; no private floor set."
            if profile.budget_nis >= r:
                return f"Tenant budget {profile.budget_nis} NIS >= asking {r} NIS."
            if profile.budget_nis < c.lowest_price_nis:
                return (
                    f"Tenant budget {profile.budget_nis} NIS below private floor "
                    f"{c.lowest_price_nis} NIS (dealbreaker)."
                )
            return (
                f"Tenant budget {profile.budget_nis} NIS in negotiation band "
                f"[{c.lowest_price_nis}, {r}] NIS (compat={compat:.2f})."
            )

        if label == "pets":
            if c.pets_allowed:
                return "Pets welcome — perfect match regardless."
            if profile.has_pets is None:
                return "Tenant did not mention pets (no pets policy — counts against them)."
            return "No pets — OK." if not profile.has_pets else "Tenant has pets; landlord forbids them."

        if label == "move_in":
            if c.move_in_by is None:
                return "No move-in deadline set."
            strict_tag = " [strict]" if c.move_in_strict else " [flexible]"
            if profile.move_in_date is None:
                return f"Tenant did not state a move-in date (implicitly accepts deadline {c.move_in_by}{strict_tag})."
            return f"Move-in date {profile.move_in_date} vs deadline {c.move_in_by}{strict_tag}."

        if label == "employment":
            if not c.employment_required:
                return "No employment requirement set."
            if profile.employment_status is None:
                return "Tenant did not state employment."
            return f"Employment: {profile.employment_status.value}."

        if label == "occupants":
            if c.max_occupants is None:
                return "No occupant limit set."
            if profile.num_roommates is None:
                return "Tenant did not state number of roommates."
            total = 1 + profile.num_roommates
            return f"{total} occupant(s) vs limit {c.max_occupants}."

        if label == "age":
            if c.min_age is None and c.max_age is None:
                return "No age preference set."
            if profile.age is None:
                return "Tenant did not state age."
            public = " [public]" if c.age_pref_public else " [private]"
            range_str = f"{c.min_age or '∞'}–{c.max_age or '∞'}"
            return f"Tenant age {profile.age} vs range [{range_str}]{public} (compat={compat:.2f})."

        if label == "gender":
            if c.preferred_gender is None:
                return "No gender preference set."
            if profile.gender is None:
                return "Tenant did not state gender."
            public = " [public preference]" if c.gender_pref_public else " [private preference]"
            return (
                f"Tenant gender: {profile.gender.value}, preferred: {c.preferred_gender.value}"
                f"{public}."
            )

        return ""
