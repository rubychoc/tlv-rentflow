"""
Extraction engine — Station 2 of the TLV-RentFlow pipeline.

Takes a RawOffer, calls the LLM via ExtractionClient, and returns a
validated TenantGroup (with >= 1 TenantProfile in applicants for real
applications, and an empty applicants list for non-application messages).

This is the only place in the pipeline where non-deterministic behaviour
(the LLM) is allowed. Everything downstream (scoring, evaluation) is
deterministic.

The engine's job is to be the firewall: if the LLM returns something that
doesn't match the TenantGroup schema, the ValidationError is caught here,
logged, and re-raised as a clean ExtractionError rather than leaking
Pydantic internals up the stack.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from rentflow.extraction.client import (
    ExtractionClient,
    _GROUP_PROV_FIELDS,
    _PERSON_PROV_FIELDS,
)
from rentflow.extraction.prompts import SYSTEM_PROMPT
from rentflow.offer.models import Provenance, RawOffer, TenantGroup

_ENV_PATH = Path(__file__).parents[3] / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Raised when extraction fails after all retries, or the response is invalid."""


@dataclass
class ExtractionResult:
    """Pairs the input offer with the extracted group for traceability."""
    offer: RawOffer
    group: TenantGroup


class ExtractionEngine:
    """
    Orchestrates the full extraction flow for one RawOffer.

    Usage:
        engine = ExtractionEngine.from_env()
        result = engine.extract(offer)
        print(result.group.budget_nis)
        for person in result.group.applicants:
            print(person.age, person.employment_status)
    """

    def __init__(self, client: ExtractionClient) -> None:
        self._client = client

    @classmethod
    def from_env(cls) -> "ExtractionEngine":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Export it in your shell before running:\n"
                "  export OPENAI_API_KEY=sk-..."
            )
        model = os.environ.get("EXTRACTION_MODEL", "gpt-4.1-mini")
        return cls(client=ExtractionClient(api_key=api_key, model=model))

    # Longest realistic tenant message (bilingual, very detailed) is well under 800 chars.
    # 1000 chars is the hard ceiling — anything beyond is truncated before the API call
    # to guard against prompt injection and runaway token costs.
    MAX_INPUT_CHARS = 1000

    def extract(self, offer: RawOffer) -> ExtractionResult:
        """
        Extracts a TenantGroup from one RawOffer.

        Raises:
            ExtractionError: on LLM failure or schema validation failure.
        """
        logger.info("Extracting offer %s (channel=%s)", offer.offer_id, offer.channel.value)

        user_text = offer.text
        if len(user_text) > self.MAX_INPUT_CHARS:
            logger.warning(
                "Offer %s: input truncated from %d to %d chars.",
                offer.offer_id, len(user_text), self.MAX_INPUT_CHARS,
            )
            user_text = user_text[:self.MAX_INPUT_CHARS]

        try:
            raw_dict = self._client.extract_raw(
                system_prompt=SYSTEM_PROMPT,
                user_text=user_text,
            )
        except RuntimeError as exc:
            raise ExtractionError(f"LLM call failed for offer {offer.offer_id}: {exc}") from exc

        # --- Reshape group-level provenance ---
        group_prov: dict[str, Provenance] = {}
        for field in _GROUP_PROV_FIELDS:
            prov_key = f"{field}_prov"
            span = raw_dict.pop(prov_key, None)
            if span is not None:
                group_prov[field] = Provenance(source_span=span)
        raw_dict["provenance"] = group_prov

        # --- Reshape per-person provenance within each applicant ---
        reshaped_applicants = []
        for person_dict in raw_dict.get("applicants", []):
            person_prov: dict[str, Provenance] = {}
            for field in _PERSON_PROV_FIELDS:
                prov_key = f"{field}_prov"
                span = person_dict.pop(prov_key, None)
                if span is not None:
                    person_prov[field] = Provenance(source_span=span)
            person_dict["provenance"] = person_prov
            reshaped_applicants.append(person_dict)
        raw_dict["applicants"] = reshaped_applicants

        try:
            group = TenantGroup.model_validate(raw_dict)
        except Exception as exc:
            raise ExtractionError(
                f"LLM response for offer {offer.offer_id} failed schema validation: {exc}\n"
                f"Raw response: {raw_dict}"
            ) from exc

        # --- Validate provenance spans are real substrings ---
        bad_spans: list[str] = []
        for field, prov in group.provenance.items():
            if prov.source_span not in offer.text:
                bad_spans.append(f"group.{field}={prov.source_span!r}")
        for i, person in enumerate(group.applicants):
            for field, prov in person.provenance.items():
                if prov.source_span not in offer.text:
                    bad_spans.append(f"applicant[{i}].{field}={prov.source_span!r}")
        if bad_spans:
            logger.warning(
                "Offer %s: provenance spans not found in source text: %s",
                offer.offer_id,
                ", ".join(bad_spans),
            )

        # --- Warn on non-null group fields missing provenance ---
        GROUP_SCREENING = {"budget_nis", "move_in_date", "has_pets", "household_size"}
        uncited_group = [
            f for f in GROUP_SCREENING
            if getattr(group, f) is not None and f not in group.provenance
        ]
        if uncited_group:
            logger.warning(
                "Offer %s: non-null group fields missing provenance: %s",
                offer.offer_id, ", ".join(uncited_group),
            )

        # --- Warn on per-person fields missing provenance ---
        PERSON_SCREENING = {"employment_status", "age", "gender"}
        for i, person in enumerate(group.applicants):
            uncited = [
                f for f in PERSON_SCREENING
                if getattr(person, f) is not None and f not in person.provenance
            ]
            if uncited:
                logger.warning(
                    "Offer %s applicant[%d]: non-null fields missing provenance: %s",
                    offer.offer_id, i, ", ".join(uncited),
                )

        # Warn if the LLM didn't produce one profile per occupant.
        n = len(group.applicants)
        if group.household_size is not None and n != group.household_size:
            logger.warning(
                "Offer %s: household_size=%d but got %d applicant(s) — mismatch.",
                offer.offer_id, group.household_size, n,
            )

        logger.info(
            "Extracted offer %s — budget=%s, move_in=%s, pets=%s, "
            "household_size=%s, applicants=%d",
            offer.offer_id,
            group.budget_nis,
            group.move_in_date,
            group.has_pets,
            group.household_size,
            n,
        )
        if n > 0:
            for i, p in enumerate(group.applicants):
                logger.info(
                    "  applicant[%d]: employ=%s, age=%s, gender=%s",
                    i, p.employment_status, p.age, p.gender,
                )

        return ExtractionResult(offer=offer, group=group)
