"""
Vector builder and cosine similarity for the scoring engine.

Each tenant profile and the landlord's criteria are mapped to the same
fixed-dimension feature vector. Every dimension is a compatibility score
in [0.0, 1.0]:
  - 1.0  = perfect match on this dimension
  - 0.0  = worst compatible (or hard-constraint boundary)
  - 0.05 = unknown / tenant did not state this field
  - None is never emitted; unknown always becomes 0.05

Dimensions (in order):
  0  budget       price-band compatibility
  1  pets         policy compatibility
  2  move_in      availability vs. deadline
  3  employment   status vs. requirement
  4  occupants    headcount vs. limit
  5  age          proximity to preferred age
  6  gender       match to preferred gender

Weights (from ScoringCriteria) are applied as sqrt(w) scaling so cosine
similarity properly reflects the landlord's priorities.

Dealbreakers are NOT encoded in the vector — they are a separate multiplier
applied by the engine AFTER cosine is computed. This keeps the math clean.
"""

from __future__ import annotations

import math

from rentflow.offer.models import (
    EmploymentStatus,
    ScoringCriteria,
    TenantProfile,
)

_UNKNOWN = 0.05  # unknown fields count against the tenant
_DIM_COUNT = 7  # must match the ordered list above


# ---------------------------------------------------------------------------
# Individual dimension compatibility functions  (all return float in [0,1])
# ---------------------------------------------------------------------------

def _budget_compat(profile: TenantProfile, criteria: ScoringCriteria, rent_nis: int) -> float:
    """
    Maps the tenant's stated budget against the price band [floor, asking].

    rent_nis  = x  (public asking price, from Listing)
    floor_nis = y  (private minimum, from criteria; None means any budget ok)

    >= x  -> 1.0
    [y,x) -> linear interpolation
    < y   -> 0.0 (dealbreaker handled by engine, but dimension also bottoms out)
    None  -> 1.0 always (implicit acceptance of asking price, floor irrelevant)
    """
    if profile.budget_nis is None:
        return 1.0  # no stated budget = implicit acceptance of asking price
    b = profile.budget_nis
    if b >= rent_nis:
        return 1.0
    floor = criteria.lowest_price_nis
    if floor is None:
        # No private floor set — any stated budget is fine.
        return 1.0
    if b < floor:
        return 0.0
    # Linear interpolation in (floor, rent_nis]: +1 shift so b==floor → small positive, not zero.
    return (b - floor + 1) / (rent_nis - floor + 1)


def _pets_compat(profile: TenantProfile, criteria: ScoringCriteria) -> float:
    if criteria.pets_allowed:
        return 1.0  # pets welcome — no penalty regardless of tenant's answer
    # Landlord forbids pets.
    if profile.has_pets is None:
        return _UNKNOWN
    return 0.0 if profile.has_pets else 1.0


_MOVE_IN_DECAY = 0.6  # k in 1/(1 + k*days): 2d→0.71, 7d→0.42, 30d→0.14, 60d→0.08


def _move_in_compat(profile: TenantProfile, criteria: ScoringCriteria) -> float:
    if criteria.move_in_by is None:
        return 1.0
    if profile.move_in_date is None:
        return 1.0  # no stated date = implicit acceptance of the deadline
    dist = abs((profile.move_in_date - criteria.move_in_by).days)
    return 1.0 / (1.0 + _MOVE_IN_DECAY * dist)


_STUDENT_COMPAT = 0.5  # student: partial income, not stable employment

def _employment_compat(profile: TenantProfile, criteria: ScoringCriteria) -> float:
    if not criteria.employment_required:
        return 1.0
    if profile.employment_status is None:
        return _UNKNOWN
    if profile.employment_status in (EmploymentStatus.EMPLOYED, EmploymentStatus.SELF_EMPLOYED):
        return 1.0
    if profile.employment_status == EmploymentStatus.STUDENT:
        return _STUDENT_COMPAT
    return 0.0  # UNEMPLOYED


def _occupants_compat(profile: TenantProfile, criteria: ScoringCriteria) -> float:
    if criteria.max_occupants is None:
        return 1.0
    if profile.num_roommates is None:
        return _UNKNOWN
    total = 1 + profile.num_roommates
    return 1.0 if total <= criteria.max_occupants else 0.0


def _age_compat(profile: TenantProfile, criteria: ScoringCriteria) -> float:
    """
    1.0 within [min_age, max_age], logarithmic decay outside.
    Distance is measured from the nearest boundary; half-range used as decay unit.
    """
    if criteria.min_age is None and criteria.max_age is None:
        return 1.0
    if profile.age is None:
        return _UNKNOWN
    age = profile.age
    lo = criteria.min_age
    hi = criteria.max_age
    if (lo is None or age >= lo) and (hi is None or age <= hi):
        return 1.0
    # Distance past the nearest boundary
    if lo is not None and age < lo:
        dist = lo - age
        half_range = (hi - lo) / 2 if hi is not None else lo
    else:
        dist = age - hi
        half_range = (hi - lo) / 2 if lo is not None else hi
    half_range = max(half_range, 1)  # avoid division by zero
    return 1.0 / (1.0 + dist / half_range)


def _gender_compat(profile: TenantProfile, criteria: ScoringCriteria) -> float:
    if criteria.preferred_gender is None:
        return 1.0
    if profile.gender is None:
        return _UNKNOWN
    return 1.0 if profile.gender == criteria.preferred_gender else 0.0


# ---------------------------------------------------------------------------
# Vector builders
# ---------------------------------------------------------------------------

def _weights(criteria: ScoringCriteria) -> list[float]:
    """
    Zero out weights for dimensions the landlord has no preference on.
    This prevents free points for irrelevant fields.
    """
    return [
        criteria.budget_weight,
        0 if criteria.pets_allowed else criteria.pets_weight,       # allowed = no constraint
        criteria.move_in_weight if criteria.move_in_by else 0,
        criteria.employment_weight if criteria.employment_required else 0,
        criteria.occupants_weight if criteria.max_occupants else 0,
        criteria.age_weight if (criteria.min_age or criteria.max_age) else 0,
        criteria.gender_weight if criteria.preferred_gender else 0,
    ]


def _scale(dims: list[float], weights: list[float]) -> list[float]:
    """Apply sqrt(weight) scaling to each dimension."""
    return [d * math.sqrt(w) for d, w in zip(dims, weights)]


def profile_to_vector(
    profile: TenantProfile,
    criteria: ScoringCriteria,
    rent_nis: int,
) -> list[float]:
    """Returns the tenant compatibility vector (sqrt-weight scaled)."""
    dims = [
        _budget_compat(profile, criteria, rent_nis),
        _pets_compat(profile, criteria),
        _move_in_compat(profile, criteria),
        _employment_compat(profile, criteria),
        _occupants_compat(profile, criteria),
        _age_compat(profile, criteria),
        _gender_compat(profile, criteria),
    ]
    return _scale(dims, _weights(criteria))


def criteria_to_vector(criteria: ScoringCriteria) -> list[float]:
    """
    Returns the landlord's 'ideal tenant' vector: all dimensions = 1.0,
    scaled by sqrt(weight). This is what every profile is compared against.
    """
    return _scale([1.0] * _DIM_COUNT, _weights(criteria))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 if either is zero."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Dealbreaker check  (separate from the vector — penalty is applied by engine)
# ---------------------------------------------------------------------------

def is_dealbreaker(
    profile: TenantProfile,
    criteria: ScoringCriteria,
    rent_nis: int,
) -> tuple[bool, str]:
    """
    Returns (True, reason) if any hard constraint is violated, else (False, "").

    Hard constraints:
      - Tenant has pets but landlord forbids them.
      - Tenant's stated budget is below the private price floor.
      - move_in_strict=True and tenant cannot meet the deadline.
      - Tenant's total occupants exceed max_occupants.
    """
    # Pets veto
    if not criteria.pets_allowed and profile.has_pets is True:
        return True, "Tenant has pets but landlord does not allow pets."

    # Budget below private floor
    if (
        criteria.lowest_price_nis is not None
        and profile.budget_nis is not None
        and profile.budget_nis < criteria.lowest_price_nis
    ):
        return True, (
            f"Tenant budget {profile.budget_nis} NIS is below "
            f"the private floor {criteria.lowest_price_nis} NIS."
        )

    # Strict move-in deadline
    if criteria.move_in_strict and criteria.move_in_by is not None:
        if profile.move_in_date is not None and profile.move_in_date > criteria.move_in_by:
            return True, (
                f"Tenant move-in date {profile.move_in_date} misses "
                f"the strict deadline {criteria.move_in_by}."
            )

    # Occupants over limit
    if (
        criteria.max_occupants is not None
        and profile.num_roommates is not None
        and 1 + profile.num_roommates > criteria.max_occupants
    ):
        return True, (
            f"Total occupants {1 + profile.num_roommates} exceeds "
            f"the limit of {criteria.max_occupants}."
        )

    return False, ""


def is_empty_profile(profile: TenantProfile) -> bool:
    """
    Returns True if the tenant stated nothing at all — every scoreable field is null.
    An empty profile indicates a non-application or a completely uninformative message.
    """
    return all(f is None for f in [
        profile.budget_nis,
        profile.move_in_date,
        profile.employment_status,
        profile.has_pets,
        profile.num_roommates,
        profile.age,
        profile.gender,
    ])


# ---------------------------------------------------------------------------
# Dimension labels  (used by engine to build RuleHit breakdown)
# ---------------------------------------------------------------------------

DIM_LABELS = ["budget", "pets", "move_in", "employment", "occupants", "age", "gender"]
