"""Group training source questions for the recorded annotation process.

Questions are grouped by dialogue and target movie with their available context
cutoffs. build_full.py uses the resulting keys to attach the fixed annotations;
these packets do not include the reference answer labels.
"""

import sys, json, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from convert_v1 import load_sources

ROOT = Path(__file__).resolve().parents[1]


def title(r):
    """Extract the target movie title by removing the fixed RecToM question prefix
    and question mark. This connects belief and desire questions about the same
    movie during annotation preparation.
    """
    q = r["question"]
    return (
        q.removeprefix(
            "How does the recommender believe the seeker's attitude about the "
        )
        .removeprefix("Is the seeker likely to watch the ")
        .removesuffix("?")
    )


def norm(x):
    """Collapse repeated whitespace and case-fold text for matching. This groups
    equivalent title spellings without changing the original title shown in
    source records.
    """
    return " ".join(x.split()).casefold()


def groups():
    """Group training questions by dialogue and movie, recording each question
    cutoff and the longest available dialogue prefix. Return source packets
    without answer labels so the recorded annotations can be attached to the
    appropriate dialogue/movie target.
    """
    rows, s = load_sources(ROOT)
    by = {}
    for r in rows:
        if str(r["dialogue_id"]) in s["train"]:
            by.setdefault(str(r["dialogue_id"]), []).append(r)
    out = []
    for did, rr in sorted(by.items(), key=lambda p: int(p[0])):
        longest = max(rr, key=lambda r: len(r["utterance_context"]))[
            "utterance_context"
        ].splitlines()
        assert all(
            (
                longest[: len(r["utterance_context"].splitlines())]
                == r["utterance_context"].splitlines()
                for r in rr
            )
        )
        targets = list(dict.fromkeys((norm(title(r)) for r in rr)))
        out.append(
            {
                "dialogue_id": did,
                "lines": longest,
                "targets": [
                    {
                        "key": f"{did}/{i + 1}",
                        "title": next((title(r) for r in rr if norm(title(r)) == name)),
                        "rows": [r["record_id"] for r in rr if norm(title(r)) == name],
                        "cutoffs": sorted(
                            set(
                                (
                                    len(r["utterance_context"].splitlines())
                                    for r in rr
                                    if norm(title(r)) == name
                                )
                            )
                        ),
                    }
                    for i, name in enumerate(targets)
                ],
            }
        )
    return out


if __name__ == "__main__":
    gs = groups()
    (ROOT / "reviews" / "source_packets.json").write_text(json.dumps(gs, indent=2))
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    b = int(sys.argv[2]) if len(sys.argv) > 2 else a + 15
    for g in gs[a:b]:
        print("\nD", g["dialogue_id"])
        for i, l in enumerate(g["lines"], 1):
            print(
                f"{i} "
                + l.replace("RECOMMENDER says:", "R:").replace("SEEKER says:", "S:")
            )
        print(
            "TARGETS "
            + "; ".join(
                (f"{t['key']} {t['title']} @ {t['cutoffs']}" for t in g["targets"])
            )
        )
    print("TOTAL", len(gs), "TARGETS", sum((len(g["targets"]) for g in gs)))
