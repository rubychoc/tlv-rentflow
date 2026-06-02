"""
In-memory store for the active Listing and received RawOffers.

The store enforces the pipeline's starting condition: a Listing must exist
before any RawOffer can be saved. Callers check is_listing_live() and call
save_offer() only after that returns True; the store also raises if they don't.

The OfferStore Protocol defines the interface both the real store and any
test stub must satisfy, keeping app.py decoupled from the implementation.
"""

import threading
from typing import Protocol, runtime_checkable

from rentflow.offer.models import Listing, RawOffer


@runtime_checkable
class OfferStore(Protocol):
    def set_listing(self, listing: Listing) -> None: ...
    def get_listing(self) -> Listing | None: ...
    def is_listing_live(self) -> bool: ...
    def save_offer(self, offer: RawOffer) -> None: ...
    def all_offers(self) -> list[RawOffer]: ...
    def get_offer(self, offer_id: str) -> RawOffer | None: ...


class InMemoryOfferStore:
    """Holds one active Listing and all RawOffers received against it."""

    def __init__(self) -> None:
        self._listing: Listing | None = None
        self._offers: list[RawOffer] = []
        self._lock = threading.Lock()

    def set_listing(self, listing: Listing) -> None:
        """Publish a new listing. Replaces any previous listing and clears offers."""
        with self._lock:
            self._listing = listing
            self._offers = []

    def get_listing(self) -> Listing | None:
        with self._lock:
            return self._listing

    def is_listing_live(self) -> bool:
        with self._lock:
            return self._listing is not None

    def save_offer(self, offer: RawOffer) -> None:
        """Save an offer. Raises RuntimeError if no listing is live."""
        with self._lock:
            if self._listing is None:
                raise RuntimeError("Cannot save offer: no active listing.")
            self._offers.append(offer)

    def all_offers(self) -> list[RawOffer]:
        with self._lock:
            return list(self._offers)

    def get_offer(self, offer_id: str) -> RawOffer | None:
        with self._lock:
            for offer in self._offers:
                if offer.offer_id == offer_id:
                    return offer
        return None
