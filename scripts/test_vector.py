"""
Quick manual test for vector creation and cosine scoring.

Usage (venv active):
    python scripts/test_vector.py

You'll be prompted to paste a TenantProfile JSON (output from test_extraction.py).
Default ScoringCriteria and rent_nis are used — edit the defaults below to change them.
"""

import json
import sys

sys.path.insert(0, "src")

from rentflow.offer.models import ScoringCriteria, TenantProfile
from rentflow.scoring.vectors import (
    DIM_LABELS,
    cosine_similarity,
    criteria_to_vector,
    is_dealbreaker,
    profile_to_vector,
)

# --- Edit these defaults to match the listing under test ---
DEFAULT_RENT_NIS = 7000
DEFAULT_CRITERIA = ScoringCriteria(
    lowest_price_nis=6500,
    pets_allowed=False,
    employment_required=True,
    max_occupants=2,
)


def main() -> None:
    print("Paste TenantProfile JSON (blank line to finish):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    raw = "\n".join(lines).strip()
    if not raw:
        print("No input. Exiting.")
        return

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON: {exc}")
        return

    profile = TenantProfile.model_validate(data)
    criteria = DEFAULT_CRITERIA
    rent_nis = DEFAULT_RENT_NIS

    pv = profile_to_vector(profile, criteria, rent_nis)
    cv = criteria_to_vector(criteria)
    score = cosine_similarity(pv, cv) * 100
    dealbreaker, reason = is_dealbreaker(profile, criteria, rent_nis)

    print("\n--- VECTORS ---")
    print(f"{'Dimension':<14} {'Profile':>8}  {'Ideal':>8}")
    print("-" * 36)
    for label, p, c in zip(DIM_LABELS, pv, cv):
        print(f"{label:<14} {p:>8.4f}  {c:>8.4f}")

    print(f"\n--- SCORE ---")
    print(f"Cosine similarity : {score:.1f} / 100")
    if dealbreaker:
        final = score * 0.15
        print(f"Dealbreaker       : YES — {reason}")
        print(f"Final score       : {final:.1f} / 100  (×0.15 penalty)")
    else:
        print(f"Dealbreaker       : no")
        print(f"Final score       : {score:.1f} / 100")

    approved = criteria.approved_threshold
    rejected = criteria.rejected_threshold
    final_score = score * (0.15 if dealbreaker else 1.0)
    if final_score >= approved:
        qual = "Approved"
    elif final_score >= rejected:
        qual = "Review"
    else:
        qual = "Rejected"
    print(f"Qualification     : {qual}  (approved>={approved}, review>={rejected})")


if __name__ == "__main__":
    main()
