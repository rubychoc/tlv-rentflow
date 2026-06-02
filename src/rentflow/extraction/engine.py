"""
Extraction engine — Station 2 of the TLV-RentFlow pipeline.

Takes a RawOffer, calls the LLM via ExtractionClient, and returns a
validated TenantProfile. This is the only place in the pipeline where
non-deterministic behaviour (the LLM) is allowed. Everything downstream
(scoring, evaluation) is deterministic.

The engine's job is to be the firewall: if the LLM returns something that
doesn't match the TenantProfile schema, the ValidationError is caught here,
logged, and re-raised as a clean ExtractionError rather than leaking
Pydantic internals up the stack.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from rentflow.extraction.client import ExtractionClient, _PROVENANCE_FIELDS
from rentflow.extraction.prompts import SYSTEM_PROMPT
from rentflow.offer.models import Provenance, RawOffer, TenantProfile

# Load .env from the project root, overriding any existing shell variables.
# override=True means .env always wins, so running `export OPENAI_API_KEY=x`
# in your shell won't silently shadow a key you set in the file.
_ENV_PATH = Path(__file__).parents[3] / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Raised when extraction fails after all retries, or the response is invalid."""


@dataclass
class ExtractionResult:
    """Pairs the input offer with the extracted profile for traceability."""
    offer: RawOffer
    profile: TenantProfile


class ExtractionEngine:
    """
    Orchestrates the full extraction flow for one RawOffer.

    Usage:
        engine = ExtractionEngine.from_env()
        result = engine.extract(offer)
        print(result.profile.budget_nis)
    """

    def __init__(self, client: ExtractionClient) -> None:
        self._client = client

    @classmethod
    def from_env(cls) -> "ExtractionEngine":
        """
        Constructs an ExtractionEngine using OPENAI_API_KEY from the environment.
        This is the normal way to instantiate in production and in scripts.
        For tests, construct directly with a stub client.
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Export it in your shell before running:\n"
                "  export OPENAI_API_KEY=sk-..."
            )
        return cls(client=ExtractionClient(api_key=api_key))

    def extract(self, offer: RawOffer) -> ExtractionResult:
        """
        Extracts a TenantProfile from one RawOffer.

        The offer text is sent to the LLM with the system prompt. The response
        is validated against TenantProfile's schema by Pydantic. Any field the
        LLM left as null stays null — we never fill in defaults here.

        Raises:
            ExtractionError: on LLM failure or schema validation failure.
        """
        logger.info("Extracting offer %s (channel=%s)", offer.offer_id, offer.channel.value)

        try:
            raw_dict = self._client.extract_raw(
                system_prompt=SYSTEM_PROMPT,
                user_text=offer.text,
            )
        except RuntimeError as exc:
            raise ExtractionError(f"LLM call failed for offer {offer.offer_id}: {exc}") from exc

        # Reshape the flat per-field provenance keys (e.g. "budget_nis_prov")
        # back into the nested dict[str, Provenance] that TenantProfile expects.
        # Null prov keys (field was null) are dropped; the dict stays sparse.
        provenance: dict[str, Provenance] = {}
        for field in _PROVENANCE_FIELDS:
            prov_key = f"{field}_prov"
            span = raw_dict.pop(prov_key, None)
            if span is not None:
                provenance[field] = Provenance(source_span=span)
        raw_dict["provenance"] = provenance

        try:
            profile = TenantProfile.model_validate(raw_dict)
        except Exception as exc:
            raise ExtractionError(
                f"LLM response for offer {offer.offer_id} failed schema validation: {exc}\n"
                f"Raw response: {raw_dict}"
            ) from exc

        # Verify every provenance source_span is actually a substring of the
        # original offer text.  A span that doesn't appear in the text is a
        # hallucination — flag it loudly rather than silently accepting it.
        bad_spans = [
            f"{field}={prov.source_span!r}"
            for field, prov in profile.provenance.items()
            if prov.source_span not in offer.text
        ]
        if bad_spans:
            logger.warning(
                "Offer %s: provenance spans not found in source text: %s",
                offer.offer_id,
                ", ".join(bad_spans),
            )

        # Warn when a non-null screening field has no provenance citation.
        SCREENING_FIELDS = {
            "budget_nis", "move_in_date",
            "employment_status", "has_pets", "num_roommates",
            "age", "gender",
        }
        uncited = [
            f for f in SCREENING_FIELDS
            if getattr(profile, f) is not None and f not in profile.provenance
        ]
        if uncited:
            logger.warning(
                "Offer %s: non-null fields missing provenance citation: %s",
                offer.offer_id,
                ", ".join(uncited),
            )

        logger.info(
            "Extracted offer %s — budget=%s, move_in=%s, pets=%s, roommates=%s, "
            "age=%s, gender=%s, provenance_fields=%s",
            offer.offer_id,
            profile.budget_nis,
            profile.move_in_date,
            profile.has_pets,
            profile.num_roommates,
            profile.age,
            profile.gender,
            list(profile.provenance.keys()),
        )

        return ExtractionResult(offer=offer, profile=profile)
