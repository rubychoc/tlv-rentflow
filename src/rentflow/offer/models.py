"""
Data models for TLV-RentFlow.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field


class Channel(str, Enum):
    WHATSAPP = "whatsapp"
    FACEBOOK = "facebook"
    YAD2 = "yad2"


class Language(str, Enum):
    HEBREW = "he"
    ENGLISH = "en"


class EmploymentStatus(str, Enum):
    EMPLOYED = "employed"
    SELF_EMPLOYED = "self_employed"
    STUDENT = "student"
    UNEMPLOYED = "unemployed"


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class RawOffer(BaseModel):
    """A single incoming tenant message in reply to a listing."""

    offer_id: str = Field(..., description="Unique ID for this message.")
    channel: Channel = Field(..., description="Which platform the message came from.")
    sender: str = Field(..., description="Identifier for who sent it.")
    timestamp: datetime = Field(..., description="When the message was sent.")
    text: str = Field(..., description="The raw, unstructured message text.")


class Provenance(BaseModel):
    """The slice of the original text that justified an extracted value."""
    source_span: str = Field(..., description="Exact substring from the offer text.")


class TenantProfile(BaseModel):
    """Per-person data for one applicant extracted from a RawOffer.

    Every extracted field is Optional — None means 'not stated', never guessed.
    Shared facts (budget, move-in, pets, household size) live on TenantGroup.
    """

    # --- Per-person screening fields ---
    employment_status: EmploymentStatus | None = Field(default=None)
    age: int | None = Field(
        default=None,
        description="Tenant's age in years. None if not stated.",
    )
    gender: Gender | None = Field(
        default=None,
        description="Tenant's gender if stated. None if not mentioned.",
    )

    # --- Contact & identity ---
    name: str | None = Field(default=None)
    phone: str | None = Field(default=None)

    # --- Per-person provenance ---
    provenance: dict[str, Provenance] = Field(default_factory=dict)


class TenantGroup(BaseModel):
    """All applicants from one RawOffer, plus the facts shared by the household.

    A single applicant is represented as a group of one (len(applicants) == 1).
    Always emit one TenantProfile per occupant — len(applicants) == household_size
    when household_size is known. Shared facts are copied to every profile;
    per-person unknown fields are left null.
    """

    # --- Shared / household-level fields ---
    budget_nis: int | None = Field(
        default=None,
        description="Total group budget in NIS/month. None if not stated.",
    )
    move_in_date: date | None = Field(
        default=None,
        description=(
            "Earliest date the group can move in. "
            "Immediate → today, within_month → today+30, flexible/unstated → null."
        ),
    )
    has_pets: bool | None = Field(
        default=None,
        description="True=household has pets, False=explicitly none, None=not mentioned.",
    )
    household_size: int | None = Field(
        default=None,
        description=(
            "Total number of occupants (including the sender). "
            "1=alone, 2=couple, etc. None=not stated. "
            "Must equal len(applicants) when known."
        ),
    )
    preferred_language: Language | None = Field(default=None)

    # --- Per-person profiles (>= 1 when message is an application) ---
    applicants: list[TenantProfile] = Field(default_factory=list)

    # --- Group-level provenance ---
    provenance: dict[str, Provenance] = Field(default_factory=dict)


class ScoringCriteria(BaseModel):
    """The landlord's screening criteria.

    Price band: landlord posts rent_nis (on Listing) as the public asking price x.
    lowest_price_nis is the private minimum y the landlord is actually willing to
    accept. Tenants >= x score full marks; tenants in [y, x) score proportionally;
    tenants < y are a hard dealbreaker. Tenants who don't state a budget score 0.5
    (they implicitly accept the posted price by writing in).
    """

    # --- Price band ---
    # Public asking price lives on Listing.rent_nis (x).
    # lowest_price_nis is the private floor (y <= x).
    lowest_price_nis: int | None = Field(
        default=None,
        description="Private minimum price the landlord will accept (NIS/month). "
                    "None means any tenant budget is fine. Must be <= Listing.rent_nis.",
    )
    budget_weight: int = Field(default=10, ge=0, le=100)

    # --- Pets ---
    pets_allowed: bool = Field(default=True)
    pets_weight: int = Field(default=25, ge=0, le=100)

    # --- Move-in deadline ---
    move_in_by: date | None = Field(default=None)
    move_in_strict: bool = Field(
        default=True,
        description="If True and move_in_by is set, missing the deadline is a dealbreaker.",
    )
    move_in_weight: int = Field(default=20, ge=0, le=100)

    # --- Employment ---
    employment_required: bool = Field(default=False)
    employment_weight: int = Field(default=15, ge=0, le=100)

    # --- Max occupants ---
    max_occupants: int | None = Field(default=None)
    occupants_weight: int = Field(default=10, ge=0, le=100)

    # --- Age preference ---
    min_age: int | None = Field(default=None, description="Minimum preferred tenant age. None = no lower bound.")
    max_age: int | None = Field(default=None, description="Maximum preferred tenant age. None = no upper bound.")
    age_weight: int = Field(default=10, ge=0, le=100)
    # Whether to advertise the age preference in the public listing (scoring unaffected).
    age_pref_public: bool = Field(default=False)

    # --- Gender preference ---
    preferred_gender: Gender | None = Field(
        default=None,
        description="Landlord's preferred tenant gender. None = no preference.",
    )
    gender_weight: int = Field(default=10, ge=0, le=100)
    # Whether to advertise the gender preference in the public listing (scoring unaffected).
    gender_pref_public: bool = Field(default=False)

    # --- Qualification thresholds (applied to the 0-100 cosine score) ---
    approved_threshold: int = Field(default=75, ge=0, le=100)
    rejected_threshold: int = Field(default=50, ge=0, le=100)


class Qualification(str, Enum):
    APPROVED = "Approved"
    REVIEW = "Review"
    REJECTED = "Rejected"


class RuleHit(BaseModel):
    """Result of evaluating one scoring dimension."""
    rule: str
    passed: bool | None   # None = field unknown
    points_earned: float
    points_possible: float
    reason: str


class ScoreResult(BaseModel):
    """Output of the scoring engine for one TenantProfile."""
    score: float = Field(description="Cosine-based compatibility score, 0-100.")
    qualification: Qualification
    rule_hits: list[RuleHit] = Field(description="Per-dimension breakdown.")


class Listing(BaseModel):
    """An apartment listing posted by the landlord."""

    listing_id: str = Field(..., description="Unique listing ID.")
    address: str = Field(..., description="Full address.")
    description: str | None = Field(default=None)
    rent_nis: int = Field(..., description="Public asking rent in NIS/month.")
    created_at: datetime = Field(..., description="When the listing was posted.")
    criteria: ScoringCriteria = Field(
        default_factory=ScoringCriteria,
        description="Landlord's screening preferences.",
    )
