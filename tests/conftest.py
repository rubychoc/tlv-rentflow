"""
Shared fixtures used across the test suite.

None of these make network calls or hit the real OpenAI API.
"""

import pytest
from datetime import date, datetime, timezone

from rentflow.offer.models import (
    Channel,
    EmploymentStatus,
    Gender,
    Language,
    Listing,
    RawOffer,
    ScoringCriteria,
    TenantProfile,
)


@pytest.fixture
def base_criteria() -> ScoringCriteria:
    """A fully-specified landlord criteria used as the baseline in scoring tests."""
    return ScoringCriteria(
        lowest_price_nis=6000,        budget_weight=10,
        pets_allowed=False,          pets_weight=25,
        move_in_by=date(2026, 8, 31), move_in_strict=True, move_in_weight=20,
        employment_required=True,    employment_weight=15,
        max_occupants=2,             occupants_weight=10,
        min_age=23, max_age=33,      age_weight=10,
        preferred_gender=Gender.FEMALE, gender_weight=10,
        approved_threshold=75,
        rejected_threshold=50,
    )


@pytest.fixture
def ideal_profile() -> TenantProfile:
    """A profile that achieves perfect compatibility against base_criteria."""
    return TenantProfile(
        budget_nis=7500,                           # >= rent_nis (7500) → 1.0
        move_in_date=date(2026, 8, 31),             # exactly the deadline → 1.0
        employment_status=EmploymentStatus.EMPLOYED,
        has_pets=False,
        num_roommates=0,
        age=28,
        gender=Gender.FEMALE,
        preferred_language=Language.ENGLISH,
    )


@pytest.fixture
def raw_offer() -> RawOffer:
    return RawOffer(
        offer_id="test_001",
        channel=Channel.WHATSAPP,
        sender="+972541234567",
        timestamp=datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
        text="מעוניין בדירה, נכנס מיידי, עובד, ללא חיות.",
    )


@pytest.fixture
def active_listing(base_criteria) -> Listing:
    return Listing(
        listing_id="listing_test_01",
        address="Rothschild Blvd 22, Tel Aviv",
        rent_nis=7500,
        created_at=datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc),
        criteria=base_criteria,
    )
