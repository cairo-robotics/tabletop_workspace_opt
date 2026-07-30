"""Self-consistency vote aggregation over K parsed VLM responses."""
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def aggregate(votes: List[Dict[str, Any]], k: int, eta: int,
              valid_ids: Optional[List[str]] = None
              ) -> Tuple[Optional[str], float, Dict[str, int]]:
    """Returns (winner_id or None, confidence, tally).

    winner is None unless: the modal candidate_id is a valid non-UNCLEAR
    candidate, has at least eta votes, and is not tied at the top.
    confidence = modal_count / k (0.0 when there is no winner).
    """
    ids = []
    for vote in votes:
        cid = vote.get("candidate_id")
        if not isinstance(cid, str):
            continue
        cid = cid.strip().upper()
        if valid_ids is not None and cid != "UNCLEAR" and cid not in valid_ids:
            continue
        ids.append(cid)
    tally = dict(Counter(ids))
    if not tally:
        return None, 0.0, tally

    ranked = sorted(tally.items(), key=lambda kv: -kv[1])
    top_id, top_count = ranked[0]
    tied = len(ranked) > 1 and ranked[1][1] == top_count
    if top_id == "UNCLEAR" or tied or top_count < eta:
        return None, 0.0, tally
    return top_id, top_count / float(k), tally
