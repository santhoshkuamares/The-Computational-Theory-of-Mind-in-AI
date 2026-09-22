"""Inspect the citation-control examples in revision-4 supervision.

The audit checks cases where a missing citation should be recovered and cases
where a real citation does not establish the selected claim under the recorded
annotation. It writes construction receipts and counts, without claiming
independent semantic validation or shortcut-free learning.
"""

import json, hashlib
from collections import Counter, defaultdict
from pathlib import Path
from dataset_io import load_training

ROOT = Path(__file__).resolve().parents[1]


def previous_state(example):
    """Read the current-state JSON embedded in a review-training prompt. The audit
    needs this actual input state to compare citation presence with the
    supervised correction.
    """
    return json.loads(
        example["messages"][0]["content"]
        .split("Current perspective state (may contain mistakes): ")[1]
        .split("\nSelected gap:")[0]
    )


def run():
    """Inspect review examples where a known value lacks citations or an
    unsupported value carries a misleading citation. Check their targets and
    write counts and receipts showing that citation presence alone does not
    determine whether a field should remain known; this is not proof that no
    shortcut was learned.
    """
    ledger = {
        r["record_id"]: r
        for r in map(
            json.loads,
            (ROOT / "data/review/all_training_records.jsonl").read_text().splitlines(),
        )
    }
    rows = load_training(ROOT / "data", "subjectesis_full_train.jsonl")
    counts = Counter()
    by_field = defaultdict(Counter)
    selections = Counter()
    receipts = []
    for row in rows:
        if row["kind"] != "review_state":
            continue
        response = json.loads(row["messages"][1]["content"])
        field = response["field"]
        before = previous_state(row)[field]
        record = ledger[row["record_id"]]
        correct = record["grounded_target"]["state"][field]
        assert response["updated_claim"] == correct
        if before["value"] == "unknown":
            continue
        pattern = (
            (
                "asserted_with_citation"
                if before["evidence_turns"]
                else "asserted_without_citation"
            )
            + " -> "
            + ("unknown" if correct["value"] == "unknown" else "known")
        )
        counts[pattern] += 1
        by_field[row["task"] + "/" + field][pattern] += 1
        case = row["construction"]["case"]
        if case not in ("recover_citation", "withdraw_misleading_citation"):
            continue
        lines = record["input"]["utterance_context"].splitlines()
        if case == "recover_citation":
            assert (
                before["value"] == correct["value"] != "unknown"
                and before["evidence_turns"] == []
            )
            assert correct["evidence_turns"] and response["decision"] == "preserve"
            assert response["evidence"] == [
                {"turn": n, "quote": lines[n - 1]} for n in correct["evidence_turns"]
            ]
        else:
            assert (
                correct["value"] == "unknown" and correct["basis"] == "not_established"
            )
            assert (
                response["decision"] == "revise"
                and (not response["evidence"])
                and (not correct["evidence_turns"])
            )
            assert len(before["evidence_turns"]) == 1
            turn = before["evidence_turns"][0]
            assert 1 <= turn <= len(lines)
            d = row["construction"]["distractor"]
            assert d["turn"] == turn and d["quote"] == lines[turn - 1]
            selections[d["selection"]] += 1
        receipts.append(
            {
                "example_id": row["example_id"],
                "record_id": row["record_id"],
                "field": field,
                "case": case,
                "input_claim": before,
                "output_claim": correct,
                "input_citations": [
                    {"turn": n, "quote": lines[n - 1]} for n in before["evidence_turns"]
                ],
                "output_evidence": response["evidence"],
                "relationship_basis": "unchanged_assistant_reviewed_prefix_state",
                "source_record_sha256": record["source_record_sha256"],
            }
        )
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    (out / "Citation_Control_Receipts.jsonl").write_text(
        "".join((json.dumps(r, ensure_ascii=False) + "\n" for r in receipts))
    )
    summary = {
        "preparation_revision": 4,
        "counts_from_actual_prompts_and_targets": dict(counts),
        "by_field": {k: dict(v) for k, v in by_field.items()},
        "new_control_examples": len(receipts),
        "misleading_citation_selection_counts": dict(selections),
        "data_sha256": hashlib.sha256(
            (ROOT / "data/sft/subjectesis_full_train.jsonl").read_bytes()
        ).hexdigest(),
        "semantic_limit": "Non-entailment follows the existing reviewed whole-prefix unknown annotation, not citation bounds. This is not independent human validation or proof of shortcut-free learning.",
    }
    for k in (
        "asserted_with_citation -> unknown",
        "asserted_without_citation -> known",
    ):
        assert counts[k] > 0
    (out / "Citation_Control_Audit.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
