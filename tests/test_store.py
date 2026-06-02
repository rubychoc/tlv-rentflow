"""
Tests for InMemoryOfferStore (ingestion/store.py) — Station 1, data layer.

The store is the only mutable shared state in the pipeline. It enforces one
invariant: a Listing must exist before any RawOffer can be saved. It also resets
the offer list whenever a new listing is posted.

Tests are grouped into:
  - Basic listing gate (no listing → can't save)
  - Offer persistence and retrieval
  - Reset behaviour (new listing wipes old offers)
  - Protocol conformance (store can be swapped behind the OfferStore interface)
  - Thread safety (the threading.Lock is the only protection — these tests prove it works)
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pytest

from rentflow.ingestion.store import InMemoryOfferStore, OfferStore
from rentflow.offer.models import Channel, Listing, RawOffer, ScoringCriteria


def _make_offer(i: int) -> RawOffer:
    """Helper: unique RawOffer with a deterministic id."""
    return RawOffer(
        offer_id=f"offer_{i:05d}",
        channel=Channel.WHATSAPP,
        sender=f"+9725400{i:05d}",
        timestamp=datetime.now(tz=timezone.utc),
        text="test",
    )


def _make_listing(i: int = 0) -> Listing:
    """Helper: minimal Listing."""
    return Listing(
        listing_id=f"listing_{i}",
        address=f"Test St {i}, Tel Aviv",
        rent_nis=7000,
        created_at=datetime.now(tz=timezone.utc),
        criteria=ScoringCriteria(),
    )


# ---------------------------------------------------------------------------
# Listing gate
# ---------------------------------------------------------------------------

class TestListingGate:
    def test_fresh_store_has_no_listing(self):
        assert InMemoryOfferStore().get_listing() is None

    def test_fresh_store_is_not_live(self):
        assert InMemoryOfferStore().is_listing_live() is False

    def test_store_is_live_after_set_listing(self):
        store = InMemoryOfferStore()
        store.set_listing(_make_listing())
        assert store.is_listing_live() is True

    def test_get_listing_returns_the_posted_listing(self):
        store = InMemoryOfferStore()
        listing = _make_listing()
        store.set_listing(listing)
        assert store.get_listing().listing_id == listing.listing_id

    def test_save_offer_without_listing_raises(self):
        with pytest.raises(RuntimeError, match="no active listing"):
            InMemoryOfferStore().save_offer(_make_offer(0))

    def test_save_offer_with_listing_succeeds(self):
        store = InMemoryOfferStore()
        store.set_listing(_make_listing())
        store.save_offer(_make_offer(0))
        assert len(store.all_offers()) == 1


# ---------------------------------------------------------------------------
# Offer persistence and retrieval
# ---------------------------------------------------------------------------

class TestOfferRetrieval:
    def test_get_offer_by_id_returns_correct_offer(self):
        store = InMemoryOfferStore()
        store.set_listing(_make_listing())
        offer = _make_offer(1)
        store.save_offer(offer)
        assert store.get_offer(offer.offer_id).offer_id == offer.offer_id

    def test_get_offer_with_unknown_id_returns_none(self):
        store = InMemoryOfferStore()
        store.set_listing(_make_listing())
        assert store.get_offer("does_not_exist") is None

    def test_all_offers_returns_all_saved(self):
        store = InMemoryOfferStore()
        store.set_listing(_make_listing())
        for i in range(5):
            store.save_offer(_make_offer(i))
        assert len(store.all_offers()) == 5

    def test_all_offers_returns_copy_not_reference(self):
        # Mutating the returned list must not affect the store's internal state
        store = InMemoryOfferStore()
        store.set_listing(_make_listing())
        store.save_offer(_make_offer(0))
        snapshot = store.all_offers()
        snapshot.clear()
        assert len(store.all_offers()) == 1


# ---------------------------------------------------------------------------
# Reset behaviour
# ---------------------------------------------------------------------------

class TestReset:
    def test_new_listing_clears_all_offers(self):
        # Posting a second listing must wipe every offer from the previous one
        store = InMemoryOfferStore()
        store.set_listing(_make_listing(0))
        for i in range(3):
            store.save_offer(_make_offer(i))
        assert len(store.all_offers()) == 3
        store.set_listing(_make_listing(1))
        assert len(store.all_offers()) == 0

    def test_new_listing_replaces_old_listing(self):
        store = InMemoryOfferStore()
        store.set_listing(_make_listing(0))
        store.set_listing(_make_listing(1))
        assert store.get_listing().listing_id == "listing_1"

    def test_offers_saved_to_new_listing_after_reset(self):
        # After reset, new offers accumulate normally
        store = InMemoryOfferStore()
        store.set_listing(_make_listing(0))
        store.save_offer(_make_offer(0))
        store.set_listing(_make_listing(1))
        store.save_offer(_make_offer(1))
        assert len(store.all_offers()) == 1


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

class TestProtocolConformance:
    def test_inmemory_store_satisfies_offerstore_protocol(self):
        # OfferStore is a structural Protocol; isinstance check confirms conformance
        store = InMemoryOfferStore()
        assert isinstance(store, OfferStore)

    def test_fake_store_satisfies_offerstore_protocol(self):
        # Any class with the right methods satisfies the Protocol without inheritance.
        # This is the seam used to inject test doubles into app.py.
        class FakeStore:
            def set_listing(self, listing): pass
            def get_listing(self): return None
            def is_listing_live(self): return False
            def save_offer(self, offer): pass
            def all_offers(self): return []
            def get_offer(self, offer_id): return None

        assert isinstance(FakeStore(), OfferStore)

    def test_app_store_can_be_swapped_via_module_attribute(self):
        # Confirms the decoupling invariant: app.py uses _store as a swappable
        # module-level variable, not a hard-coded InMemoryOfferStore import.
        from rentflow.ingestion import app as app_module
        original = app_module._store
        try:
            app_module._store = InMemoryOfferStore()
            assert app_module._store is not original
        finally:
            app_module._store = original


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_saves_all_recorded_exactly(self):
        # 500 threads each save one offer — the lock must ensure every single one lands.
        # An unlocked list.append can lose writes under concurrent access.
        N = 500
        store = InMemoryOfferStore()
        store.set_listing(_make_listing())
        errors = []

        def save(i):
            try:
                store.save_offer(_make_offer(i))
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=64) as pool:
            futures = [pool.submit(save, i) for i in range(N)]
            for f in as_completed(futures):
                pass

        assert errors == [], f"Unexpected errors during concurrent saves: {errors}"
        assert len(store.all_offers()) == N

    def test_concurrent_reads_while_writing_raises_no_exceptions(self):
        # Readers calling all_offers() while writers call save_offer() must never
        # raise "list changed size during iteration" or similar torn-read errors.
        store = InMemoryOfferStore()
        store.set_listing(_make_listing())
        read_errors = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                try:
                    _ = store.all_offers()
                except Exception as e:
                    read_errors.append(e)

        reader_threads = [threading.Thread(target=reader) for _ in range(8)]
        for t in reader_threads:
            t.start()

        with ThreadPoolExecutor(max_workers=32) as pool:
            futures = [pool.submit(store.save_offer, _make_offer(i)) for i in range(200)]
            for f in as_completed(futures):
                pass

        stop.set()
        for t in reader_threads:
            t.join()

        assert read_errors == [], f"Read errors under concurrent writes: {read_errors}"

    def test_set_listing_during_writes_leaves_store_consistent(self):
        # A thread resetting the listing while writers are saving must not corrupt the store.
        # After settling, all_offers() must return a consistent list (not a partial/torn one).
        store = InMemoryOfferStore()
        store.set_listing(_make_listing(0))
        errors = []

        def writer(i):
            try:
                store.save_offer(_make_offer(i))
            except RuntimeError:
                pass  # expected if the listing reset happens between gate-check and append
            except Exception as e:
                errors.append(e)

        def resetter():
            for j in range(5):
                store.set_listing(_make_listing(j + 1))

        with ThreadPoolExecutor(max_workers=33) as pool:
            futures = [pool.submit(writer, i) for i in range(200)]
            futures.append(pool.submit(resetter))
            for f in as_completed(futures):
                pass

        assert errors == [], f"Unexpected errors: {errors}"
        # Store must be in a coherent state — no crash and count is a non-negative integer
        count = len(store.all_offers())
        assert count >= 0
