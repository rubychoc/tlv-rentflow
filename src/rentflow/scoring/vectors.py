"""
Vector builder and scoring functions for the scoring engine.

Each household group and the landlord's criteria are mapped to the same
fixed-dimension feature vector. Every dimension is a compatibility score
in [0.0, 1.0]:
  - 1.0  = perfect match on this dimension
  - 0.0  = worst compatible (or hard-constraint boundary)
  - 0.05 = unknown / field not stated

Dimensions (in order):
  0  budget       price-band compatibility          — group-level
  1  pets         policy compatibility              — group-level
  2  move_in      availability vs. deadline         — group-level
  3  employment   status vs. requirement            — avg over applicants
  4  occupants    headcount vs. limit               — group-level (household_size)
  5  age          proximity to preferred age        — avg over applicants
  6  gender       match to preferred gender         — avg over applicants

Shared dimensions (budget, pets, move_in, occupants) are scored once at the
group level. Per-person dimensions (employment, age, gender) are averaged
across all applicants — a group is assessed by the mean of its members.

Dealbreakers are NOT encoded in the vector — they are a separate multiplier
applied by the engine AFTER the ratio is computed.
"""

from __future__ import annotations

import math

from rentflow.offer.models import (
    EmploymentStatus,
    ScoringCriteria,
    TenantGroup,
    TenantProfile,
)

_UNKNOWN = 0.05  # unknown fields count against the tenant
_DIM_COUNT = 7   # must match the ordered list above


# ---------------------------------------------------------------------------
# Individual dimension compatibility functions  (all return float in [0, 1])
# ---------------------------------------------------------------------------

def _budget_compat(group: TenantGroup, criteria: ScoringCriteria, rent_nis: int) -> float:
    if group.budget_nis is None:
        return 1.0  # implicit acceptance of asking price
    b = group.budget_nis
    if b >= rent_nis:
        return 1.0
    floor = criteria.lowest_price_nis
    if floor is None:
        return 1.0
    if b < floor:
        return 0.0
    return (b - floor + 1) / (rent_nis - floor + 1)


def _pets_compat(group: TenantGroup, criteria: ScoringCriteria) -> float:
    if criteria.pets_allowed:
        return 1.0
    if group.has_pets is None:
        return _UNKNOWN
    return 0.0 if group.has_pets else 1.0


_MOVE_IN_DECAY = 0.6


def _move_in_compat(group: TenantGroup, criteria: ScoringCriteria) -> float:
    if criteria.move_in_by is None:
        return 1.0
    if group.move_in_date is None:
        return 1.0
    dist = abs((group.move_in_date - criteria.move_in_by).days)
    return 1.0 / (1.0 + _MOVE_IN_DECAY * dist)


_STUDENT_COMPAT = 0.5


def _employment_compat_one(person: TenantProfile, criteria: ScoringCriteria) -> float:
    if not criteria.employment_required:
        return 1.0
    if person.employment_status is None:
        return _UNKNOWN
    if person.employment_status in (EmploymentStatus.EMPLOYED, EmploymentStatus.SELF_EMPLOYED):
        return 1.0
    if person.employment_status == EmploymentStatus.STUDENT:
        return _STUDENT_COMPAT
    return 0.0  # UNEMPLOYED


def _age_compat_one(person: TenantProfile, criteria: ScoringCriteria) -> float:
    if criteria.min_age is None and criteria.max_age is None:
        return 1.0
    if person.age is None:
        return _UNKNOWN
    age = person.age
    lo = criteria.min_age
    hi = criteria.max_age
    if (lo is None or age >= lo) and (hi is None or age <= hi):
        return 1.0
    if lo is not None and age < lo:
        dist = lo - age
        half_range = (hi - lo) / 2 if hi is not None else lo
    else:
        dist = age - hi
        half_range = (hi - lo) / 2 if lo is not None else hi
    half_range = max(half_range, 1)
    return 1.0 / (1.0 + dist / half_range)


def _gender_compat_one(person: TenantProfile, criteria: ScoringCriteria) -> float:
    if criteria.preferred_gender is None:
        return 1.0
    if person.gender is None:
        return _UNKNOWN
    return 1.0 if person.gender == criteria.preferred_gender else 0.0


def _avg_over_applicants(
    group: TenantGroup,
    per_person_fn,
    criteria: ScoringCriteria,
    no_applicants_default: float = 1.0,
) -> float:
    """Average a per-person compat function over all applicants.

    Returns `no_applicants_default` when the group has no applicants
    (non-application message) — avoids division by zero and keeps the
    empty-profile penalty path clean.
    """
    if not group.applicants:
        return no_applicants_default
    total = sum(per_person_fn(p, criteria) for p in group.applicants)
    return total / len(group.applicants)


def _occupants_compat(group: TenantGroup, criteria: ScoringCriteria) -> float:
    if criteria.max_occupants is None:
        return 1.0
    total = group.household_size
    if total is None:
        # Fall back to applicant count if household_size not stated
        n = len(group.applicants)
        total = n if n > 0 else None
    if total is None:
        return _UNKNOWN
    return 1.0 if total <= criteria.max_occupants else 0.0


# ---------------------------------------------------------------------------
# Aggregate group-level compat functions (used by engine._build_hits)
# ---------------------------------------------------------------------------

def _employment_compat(group: TenantGroup, criteria: ScoringCriteria) -> float:
    return _avg_over_applicants(group, _employment_compat_one, criteria)


def _age_compat(group: TenantGroup, criteria: ScoringCriteria) -> float:
    return _avg_over_applicants(group, _age_compat_one, criteria)


def _gender_compat(group: TenantGroup, criteria: ScoringCriteria) -> float:
    return _avg_over_applicants(group, _gender_compat_one, criteria)


# ---------------------------------------------------------------------------
# Vector builders
# ---------------------------------------------------------------------------

def _weights(criteria: ScoringCriteria) -> list[float]:
    return [
        criteria.budget_weight,
        0 if criteria.pets_allowed else criteria.pets_weight,
        criteria.move_in_weight if criteria.move_in_by else 0,
        criteria.employment_weight if criteria.employment_required else 0,
        criteria.occupants_weight if criteria.max_occupants else 0,
        criteria.age_weight if (criteria.min_age or criteria.max_age) else 0,
        criteria.gender_weight if criteria.preferred_gender else 0,
    ]


def _scale(dims: list[float], weights: list[float]) -> list[float]:
    return [d * math.sqrt(w) for d, w in zip(dims, weights)]


def group_to_vector(
    group: TenantGroup,
    criteria: ScoringCriteria,
    rent_nis: int,
) -> list[float]:
    """Returns the household compatibility vector (sqrt-weight scaled)."""
    dims = [
        _budget_compat(group, criteria, rent_nis),
        _pets_compat(group, criteria),
        _move_in_compat(group, criteria),
        _employment_compat(group, criteria),
        _occupants_compat(group, criteria),
        _age_compat(group, criteria),
        _gender_compat(group, criteria),
    ]
    return _scale(dims, _weights(criteria))


def criteria_to_vector(criteria: ScoringCriteria) -> list[float]:
    """Returns the landlord's 'ideal tenant' vector: all dims = 1.0, sqrt-weight scaled."""
    return _scale([1.0] * _DIM_COUNT, _weights(criteria))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Dealbreaker check  (strict: any single member's violation vetoes the group)
# ---------------------------------------------------------------------------

def is_dealbreaker(
    group: TenantGroup,
    criteria: ScoringCriteria,
    rent_nis: int,
) -> tuple[bool, str]:
    """
    Returns (True, reason) if any hard constraint is violated, else (False, "").

    Hard constraints are strict — one member's violation vetoes the whole group:
      - Household has pets but landlord forbids them.
      - Group budget is below the private price floor.
      - move_in_strict=True and group cannot meet the deadline.
      - Total occupants exceed max_occupants.
    """
    if not criteria.pets_allowed and group.has_pets is True:
        return True, "Household has pets but landlord does not allow pets."

    if (
        criteria.lowest_price_nis is not None
        and group.budget_nis is not None
        and group.budget_nis < criteria.lowest_price_nis
    ):
        return True, (
            f"Group budget {group.budget_nis} NIS is below "
            f"the private floor {criteria.lowest_price_nis} NIS."
        )

    if criteria.move_in_strict and criteria.move_in_by is not None:
        if group.move_in_date is not None and group.move_in_date > criteria.move_in_by:
            return True, (
                f"Group move-in date {group.move_in_date} misses "
                f"the strict deadline {criteria.move_in_by}."
            )

    total = group.household_size or (len(group.applicants) if group.applicants else None)
    if (
        criteria.max_occupants is not None
        and total is not None
        and total > criteria.max_occupants
    ):
        return True, (
            f"Total occupants {total} exceeds the limit of {criteria.max_occupants}."
        )

    return False, ""


def is_empty_group(group: TenantGroup) -> bool:
    """True when the group stated nothing scoreable — every field is null/empty."""
    group_empty = all(f is None for f in [
        group.budget_nis,
        group.move_in_date,
        group.has_pets,
        group.household_size,
    ])
    if not group_empty:
        return False
    if not group.applicants:
        return True
    return all(
        p.employment_status is None and p.age is None and p.gender is None
        for p in group.applicants
    )


# ---------------------------------------------------------------------------
# Dimension labels  (used by engine to build RuleHit breakdown)
# ---------------------------------------------------------------------------

DIM_LABELS = ["budget", "pets", "move_in", "employment", "occupants", "age", "gender"]
