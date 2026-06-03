"""
Scoring engine — Station 3 of the TLV-RentFlow pipeline.

Algorithm:
  1. Build a criteria vector (the landlord's ideal) and a group vector.
  2. Compute dot-product ratio: dot(g, c) / dot(c, c) -> raw score in [0, 1].
     Shared dims (budget, pets, move_in, occupants) are scored once at the
     group level. Per-person dims (employment, age, gender) are averaged
     across all applicants.
  3. Apply a dealbreaker penalty (×0.01) if any hard constraint is violated.
     Hard constraints are strict: one member's violation vetoes the whole group.
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
    TenantGroup,
)
from rentflow.scoring.vectors import (
    DIM_LABELS,
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
    _weights,
    criteria_to_vector,
    group_to_vector,
    is_dealbreaker,
    is_empty_group,
)

_DEALBREAKER_PENALTY = 0.01
_EMPTY_GROUP_PENALTY = 0.1


class ScoringEngine:
    def __init__(self, criteria: ScoringCriteria, rent_nis: int) -> None:
        self._criteria = criteria
        self._rent_nis = rent_nis

    def score(self, group: TenantGroup) -> ScoreResult:
        """
        Computes the dot-product ratio score for one TenantGroup.

        score = dot(group_vec, criteria_vec) / dot(criteria_vec, criteria_vec)
        """
        c_vec = criteria_to_vector(self._criteria)
        g_vec = group_to_vector(group, self._criteria, self._rent_nis)

        max_score = sum(x * x for x in c_vec)
        raw_ratio = sum(a * b for a, b in zip(g_vec, c_vec)) / max_score if max_score else 0.0

        violated, veto_reason = is_dealbreaker(group, self._criteria, self._rent_nis)
        if violated:
            multiplier = _DEALBREAKER_PENALTY
        elif is_empty_group(group):
            multiplier = _EMPTY_GROUP_PENALTY
        else:
            multiplier = 1.0

        score = round(raw_ratio * multiplier * 100, 1)

        if score >= self._criteria.approved_threshold:
            qualification = Qualification.APPROVED
        elif score < self._criteria.rejected_threshold:
            qualification = Qualification.REJECTED
        else:
            qualification = Qualification.REVIEW

        empty = is_empty_group(group)
        hits = self._build_hits(group, violated, veto_reason, empty)
        return ScoreResult(score=score, qualification=qualification, rule_hits=hits)

    def _build_hits(
        self,
        group: TenantGroup,
        dealbreaker_violated: bool,
        veto_reason: str,
        empty_group: bool = False,
    ) -> list[RuleHit]:
        c = self._criteria
        r = self._rent_nis

        compat_fns = [
            lambda g: _budget_compat(g, c, r),
            lambda g: _pets_compat(g, c),
            lambda g: _move_in_compat(g, c),
            lambda g: _employment_compat(g, c),
            lambda g: _occupants_compat(g, c),
            lambda g: _age_compat(g, c),
            lambda g: _gender_compat(g, c),
        ]
        weights = _weights(c)
        hits: list[RuleHit] = []

        for label, fn, w in zip(DIM_LABELS, compat_fns, weights):
            compat = fn(group)
            sw = math.sqrt(w)
            earned = compat * sw
            possible = sw

            if compat == 1.0:
                passed = True
            elif compat == 0.0:
                passed = False
            else:
                passed = None

            reason = self._reason(label, group, compat)
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
        elif empty_group:
            hits.append(RuleHit(
                rule="empty_group",
                passed=False,
                points_earned=0.0,
                points_possible=0.0,
                reason=f"EMPTY GROUP (×{_EMPTY_GROUP_PENALTY}): message contained no tenant information.",
            ))

        return hits

    def _reason(self, label: str, group: TenantGroup, compat: float) -> str:
        c = self._criteria
        r = self._rent_nis
        n = len(group.applicants)

        if label == "budget":
            if group.budget_nis is None:
                return "Group did not state a budget (implicitly accepts posted price)."
            if c.lowest_price_nis is None:
                return f"Group budget {group.budget_nis} NIS; no private floor set."
            if group.budget_nis >= r:
                return f"Group budget {group.budget_nis} NIS >= asking {r} NIS."
            if group.budget_nis < c.lowest_price_nis:
                return (
                    f"Group budget {group.budget_nis} NIS below private floor "
                    f"{c.lowest_price_nis} NIS (dealbreaker)."
                )
            return (
                f"Group budget {group.budget_nis} NIS in negotiation band "
                f"[{c.lowest_price_nis}, {r}] NIS (compat={compat:.2f})."
            )

        if label == "pets":
            if c.pets_allowed:
                return "Pets welcome — perfect match regardless."
            if group.has_pets is None:
                return "Group did not mention pets (no pets policy — counts against them)."
            return "No pets — OK." if not group.has_pets else "Household has pets; landlord forbids them."

        if label == "move_in":
            if c.move_in_by is None:
                return "No move-in deadline set."
            strict_tag = " [strict]" if c.move_in_strict else " [flexible]"
            if group.move_in_date is None:
                return f"Group did not state move-in date (implicitly accepts deadline {c.move_in_by}{strict_tag})."
            return f"Move-in date {group.move_in_date} vs deadline {c.move_in_by}{strict_tag}."

        if label == "employment":
            if not c.employment_required:
                return "No employment requirement set."
            if not group.applicants:
                return "No applicant details provided."
            statuses = [
                p.employment_status.value if p.employment_status else "unknown"
                for p in group.applicants
            ]
            per_person = ", ".join(statuses)
            if n == 1:
                return f"Employment: {per_person}."
            indiv = [_employment_compat_one(p, c) for p in group.applicants]
            return (
                f"Employment avg over {n} applicants: {per_person} "
                f"→ compat={compat:.2f} (avg of {[round(v,2) for v in indiv]})."
            )

        if label == "occupants":
            if c.max_occupants is None:
                return "No occupant limit set."
            total = group.household_size or (len(group.applicants) if group.applicants else None)
            if total is None:
                return "Group did not state number of occupants."
            return f"{total} occupant(s) vs limit {c.max_occupants}."

        if label == "age":
            if c.min_age is None and c.max_age is None:
                return "No age preference set."
            if not group.applicants:
                return "No applicant details provided."
            ages = [str(p.age) if p.age is not None else "unknown" for p in group.applicants]
            public = " [public]" if c.age_pref_public else " [private]"
            range_str = f"{c.min_age or '∞'}–{c.max_age or '∞'}"
            if n == 1:
                return f"Tenant age {ages[0]} vs range [{range_str}]{public} (compat={compat:.2f})."
            indiv = [_age_compat_one(p, c) for p in group.applicants]
            return (
                f"Ages {ages} vs range [{range_str}]{public} "
                f"→ avg compat={compat:.2f} (individual: {[round(v,2) for v in indiv]})."
            )

        if label == "gender":
            if c.preferred_gender is None:
                return "No gender preference set."
            if not group.applicants:
                return "No applicant details provided."
            genders = [
                p.gender.value if p.gender else "unknown"
                for p in group.applicants
            ]
            public = " [public preference]" if c.gender_pref_public else " [private preference]"
            if n == 1:
                return (
                    f"Tenant gender: {genders[0]}, preferred: {c.preferred_gender.value}{public}."
                )
            indiv = [_gender_compat_one(p, c) for p in group.applicants]
            return (
                f"Genders {genders}, preferred: {c.preferred_gender.value}{public} "
                f"→ avg compat={compat:.2f} (individual: {[round(v,2) for v in indiv]})."
            )

        return ""
