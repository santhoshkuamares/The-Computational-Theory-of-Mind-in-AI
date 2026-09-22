"""Revision-4 data preparation: convert v1."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import re
from collections import Counter

REVISION = "317b91d3db47f4e278b8bb3eefcbb733fb7d639b"
SOURCES = {
    "belief": (
        "8_belief_rec_2_com.json",
        "f26e9519af66b82160fc03e900588877be0d1682a1e75b2c5086d5a8205abf97",
    ),
    "desire": (
        "7_desire_seeker_com.json",
        "2f5bd1a426bee1533f8f1931b448da13107f8c56b92e634bc358aa51b599f8f4",
    ),
}
PUBLIC = ("task", "utterance_context", "question", "choices")
DOMAINS = {
    "belief": {
        "proposer": ["recommender", "seeker", "unknown"],
        "seen": ["yes", "no", "unknown"],
        "recommendation_response": ["accepted", "declined", "unknown"],
        "appraisal": ["likes", "dislikes", "unknown"],
    },
    "desire": {"intends_to_watch": ["yes", "no", "unknown"]},
}
FIELDS = {
    "proposer": "who introduced the target movie into the discussion",
    "seen": "whether the seeker has watched the target movie",
    "recommendation_response": "whether the seeker accepts or declines this recommendation",
    "appraisal": "the seeker's positive or negative attitude toward the target movie",
    "intends_to_watch": "the seeker's likely willingness to watch, a prediction rather than a guaranteed future action",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write_jsonl(path, rows):
    """Write the records in order, one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join((json.dumps(r, ensure_ascii=False) + "\n" for r in rows)),
        encoding="utf-8",
    )


def write_json(path, value):
    """Write a JSON result using the original serialization convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_sources(root):
    """Read the fixed RecToM task sources and dialogue-level split manifest."""
    rows = []
    for task, (name, expected) in SOURCES.items():
        raw = (root / "original" / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Original source checksum mismatch: " + name)
        for index, record in enumerate(json.loads(raw)):
            if (
                len(record["answer"]) != 1
                or record["answer"][0] not in record["choices"]
            ):
                raise ValueError("Invalid source answer")
            rows.append(dict(record, task=task, record_id=f"{task}_{index}"))
    ids = sorted({str(r["dialogue_id"]) for r in rows})
    if len(rows) != 3210 or len(ids) != 336:
        raise ValueError("Unexpected source counts")
    random.Random(8).shuffle(ids)
    splits = {
        "train": set(ids[:235]),
        "validation": set(ids[235:269]),
        "test": set(ids[269:]),
    }
    original = json.loads((root / "original" / "split_manifest.json").read_text())
    if original["source_revision"] != REVISION or original["seed"] != 8:
        raise ValueError("Split provenance changed")
    for split, group in splits.items():
        if group != set(original["dialogue_ids"][split]):
            raise ValueError("Previously agreed dialogue split changed")
    return (rows, splits)


def public_row(row):
    """Remove answer-bearing fields before constructing the model input."""
    return {
        k: copy.deepcopy(row[k])
        for k in ("record_id", "dialogue_id", "utterance_pos") + PUBLIC
    }


def input_text(row):
    """Render the original dialogue, question and forced-choice options."""
    return (
        "Task: "
        + row["task"]
        + "\nDialogue:\n"
        + "\n".join(
            (
                f"{i}: {s}"
                for i, s in enumerate(row["utterance_context"].splitlines(), 1)
            )
        )
        + "\nQuestion: "
        + row["question"]
        + "\nOptions:\n"
        + "\n".join((f"{key}: {value}" for key, value in row["choices"].items()))
    )


def benchmark_state(row):
    """Decode the benchmark answer into its recorded task-state interpretation."""
    text = " ".join(row["choices"][row["answer"][0]].lower().split())
    if row["task"] == "desire":
        if text not in ("yes", "no"):
            raise ValueError("Unexpected desire answer wording")
        return {"intends_to_watch": text}
    state = {}
    match = re.search("proposed by the (reommender|recommender|seeker)\\b", text)
    if match:
        state["proposer"] = "seeker" if match.group(1) == "seeker" else "recommender"
    if "seeker has not seen" in text:
        state["seen"] = "no"
    elif "seeker has already seen" in text:
        state["seen"] = "yes"
    if "did not accept" in text:
        state["recommendation_response"] = "declined"
    elif "accepted the suggestion" in text:
        state["recommendation_response"] = "accepted"
    if "does not seem to like" in text:
        state["appraisal"] = "dislikes"
    elif "appears to like" in text or "seems to like" in text:
        state["appraisal"] = "likes"
    if len(state) != 3:
        raise ValueError("Unknown belief option semantics: " + text)
    return state


def training_example(row, kind, prompt, response, suffix=""):
    """Build one prompt/completion example with its record and split metadata."""
    return {
        "example_id": row["record_id"] + "/" + kind + suffix,
        "record_id": row["record_id"],
        "dialogue_id": str(row["dialogue_id"]),
        "split": "train",
        "task": row["task"],
        "kind": kind,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
    }


def answer_example(row, suffix=""):
    """Build an ordinary answer-only completion for the matched control."""
    return training_example(
        row,
        "answer_only",
        input_text(row) + "\nReturn only the selected option letter.",
        row["answer"][0],
        suffix,
    )


def state_prompt(row):
    """Build the perspective-state prompt used in the structured supervision."""
    return input_text(row) + (
        "\nBuild an evidence-grounded perspective record and choose an answer. Separate observed facts from qualified inferences. Retain unknown for facts the supplied context does not establish. Movie liking and recommendation acceptance are different fields. For a forced-choice answer, list any option claims not established by the evidence. Return JSON with perspective, evidence, state, missing_information, answer and answer_claims_not_established. Each state field has value, evidence_turns, basis and reason. Fields: "
        + json.dumps(FIELDS_FOR_TASK(row))
    )


def review_prompt(row, current_state, name):
    """Ask for evidence-sensitive review of the selected state field."""
    if set(current_state) != set(DOMAINS[row["task"]]) or name not in current_state:
        raise ValueError(
            "Review input must contain only the task state fields, not a reference answer"
        )
    state_view = {
        field: {"value": claim["value"], "evidence_turns": claim["evidence_turns"]}
        for field, claim in current_state.items()
    }
    return (
        input_text(row)
        + "\nCurrent perspective state (may contain mistakes): "
        + json.dumps(state_view, ensure_ascii=False)
        + "\nSelected gap: Check "
        + FIELDS[name]
        + ".\nReview the supplied evidence. Return JSON with field, decision, updated_claim, evidence and stop_reason. Revise supported mistakes. If neither known value is supported, withdraw the claim to unknown; otherwise preserve a supported claim or retain unknown. Do not add facts from later turns."
    )


def validate_review(row, review):
    """Check the review schema and selected field, not whether the evidence entails the claim."""
    if review["source_record_sha256"] != digest(row):
        raise ValueError(
            "Review refers to a changed source record: " + row["record_id"]
        )
    if review["review_status"] != "assistant_semantically_reviewed":
        raise ValueError("Unreviewed item cannot enter grounded SFT")
    if not review.get("review_notes"):
        raise ValueError("Missing semantic review notes")
    slots = review["slots"]
    if set(slots) != set(DOMAINS[row["task"]]):
        raise ValueError("Review has wrong fields")
    lines = row["utterance_context"].splitlines()
    for name, claim in slots.items():
        if claim["value"] not in DOMAINS[row["task"]][name]:
            raise ValueError("Invalid reviewed value")
        if not claim.get("reason") or claim["basis"] not in (
            "explicit",
            "inferred",
            "not_established",
        ):
            raise ValueError("Missing evidence strength or rationale")
        refs = claim["evidence_turns"]
        if (
            not isinstance(refs, list)
            or len(refs) != len(set(refs))
            or any((type(n) is not int or not 1 <= n <= len(lines) for n in refs))
        ):
            raise ValueError("Invalid reviewed citation")
        if claim["value"] != "unknown" and (
            not refs or claim["basis"] == "not_established"
        ):
            raise ValueError("Known claim has no supporting evidence")
        if claim["value"] == "unknown" and claim["basis"] != "not_established":
            raise ValueError(
                "Unknown must be distinguished from a known inferred value"
            )
    target = benchmark_state(row)
    conflicts = [
        name
        for name, value in target.items()
        if slots[name]["value"] not in ("unknown", value)
    ]
    if conflicts:
        raise ValueError(
            "Reviewed evidence conflicts with official target; quarantine before export: "
            + str(conflicts)
        )


def grounded_target(row, review):
    slots = copy.deepcopy(review["slots"])
    lines = row["utterance_context"].splitlines()
    refs = sorted({n for c in slots.values() for n in c["evidence_turns"]})
    return {
        "perspective": (
            "recommender about seeker"
            if row["task"] == "belief"
            else "seeker likely willingness"
        ),
        "evidence": [{"turn": n, "quote": lines[n - 1]} for n in refs],
        "state": slots,
        "missing_information": [
            FIELDS[n] for n, c in slots.items() if c["value"] == "unknown"
        ],
        "answer": row["answer"][0],
        "answer_claims_not_established": [
            n for n in benchmark_state(row) if slots[n]["value"] == "unknown"
        ],
    }


def grounded_examples(row, review):
    target = grounded_target(row, review)
    examples = [
        training_example(
            row,
            "build_state",
            state_prompt(row),
            json.dumps(target, ensure_ascii=False),
        )
    ]
    for name, correct in target["state"].items():
        cases = (
            ["retain_unknown"]
            if correct["value"] == "unknown"
            else ["masked", "wrong", "preserve"]
        )
        for case in cases:
            previous = copy.deepcopy(target["state"])
            if case == "masked":
                previous[name] = {
                    "value": "unknown",
                    "evidence_turns": [],
                    "basis": "not_established",
                    "reason": "This field has not yet been checked.",
                }
            elif case == "wrong":
                alternatives = [
                    v
                    for v in DOMAINS[row["task"]][name]
                    if v not in ("unknown", correct["value"])
                ]
                previous[name] = {
                    "value": alternatives[0],
                    "evidence_turns": correct["evidence_turns"],
                    "basis": "inferred",
                    "reason": "Unverified preliminary interpretation.",
                }
            prompt = review_prompt(row, previous, name)
            decision = (
                "retain_unknown"
                if case == "retain_unknown"
                else "preserve" if case == "preserve" else "revise"
            )
            response = {
                "field": name,
                "decision": decision,
                "updated_claim": correct,
                "evidence": [
                    e
                    for e in target["evidence"]
                    if e["turn"] in correct["evidence_turns"]
                ],
                "stop_reason": (
                    "evidence_not_sufficient"
                    if decision == "retain_unknown"
                    else "selected_field_reviewed"
                ),
            }
            sample = training_example(
                row,
                "review_state",
                prompt,
                json.dumps(response, ensure_ascii=False),
                "/" + name + "/" + case,
            )
            sample["construction"] = {
                "synthetic_state": True,
                "case": case,
                "selected_field": name,
                "derived_from_review_id": review["review_id"],
            }
            examples.append(sample)
    return examples


def FIELDS_FOR_TASK(row):
    return {name: FIELDS[name] for name in DOMAINS[row["task"]]}


def convert(root, output):
    rows, splits = load_sources(root)
    by_id = {r["record_id"]: r for r in rows}
    partitions = {
        name: [r for r in rows if str(r["dialogue_id"]) in group]
        for name, group in splits.items()
    }
    assert [len(partitions[k]) for k in ("train", "validation", "test")] == [
        2270,
        319,
        621,
    ]
    train = partitions["train"]
    reviews = [
        json.loads(line)
        for line in (root / "reviews" / "assistant_reviews.jsonl")
        .read_text()
        .splitlines()
    ]
    if len({r["record_id"] for r in reviews}) != len(reviews):
        raise ValueError("Duplicate review")
    grounded, matched, matched_steps, ledger = ([], [], [], [])
    for review in reviews:
        row = by_id[review["record_id"]]
        if str(row["dialogue_id"]) not in splits["train"]:
            raise ValueError("Validation/test evidence cannot enter training")
        validate_review(row, review)
        examples = grounded_examples(row, review)
        grounded.extend(examples)
        matched.append(answer_example(row))
        matched_steps.extend(
            (answer_example(row, f"/repeat_{i}") for i in range(len(examples)))
        )
        ledger.append(
            dict(
                review,
                benchmark_target_state=benchmark_state(row),
                grounded_target=grounded_target(row, review),
            )
        )
    all_answer = [answer_example(row) for row in train]
    label_state = [
        training_example(
            row,
            "label_state_control",
            input_text(row)
            + "\nReturn JSON with benchmark_target_state and answer. These states follow the benchmark answer semantics.",
            json.dumps(
                {
                    "benchmark_target_state": benchmark_state(row),
                    "answer": row["answer"][0],
                }
            ),
        )
        for row in train
    ]
    outputs = {
        "answer_only_train": all_answer,
        "label_state_control_train": label_state,
        "subjectesis_seed_train": grounded,
        "answer_only_matched_seed": matched,
        "answer_only_matched_steps": matched_steps,
        "subjectesis_mixed_train": all_answer + grounded,
        "answer_only_matched_mixed_steps": all_answer
        + [
            dict(example, example_id=example["example_id"] + "/auxiliary_control")
            for example in matched_steps
        ],
    }
    for name, records in outputs.items():
        write_jsonl(output / "sft" / f"{name}.jsonl", records)
    reviewed_ids = {r["record_id"] for r in reviews}
    pending = [
        dict(
            public_row(row),
            official_training_answer=row["answer"][0],
            benchmark_target_state=benchmark_state(row),
            evidence_review_status="not_reviewed",
            grounded_state=None,
        )
        for row in train
        if row["record_id"] not in reviewed_ids
    ]
    write_jsonl(output / "review" / "pending_training_records.jsonl", pending)
    write_jsonl(output / "review" / "reviewed_training_records.jsonl", ledger)
    for split in ("validation", "test"):
        write_jsonl(
            output / "evaluation" / f"{split}_inputs.jsonl",
            [
                dict(
                    public_row(r),
                    prompt=input_text(r),
                    answer_only_prompt=input_text(r)
                    + "\nReturn only the selected option letter.",
                    subjectesis_prompt=state_prompt(r),
                )
                for r in partitions[split]
            ],
        )
        write_jsonl(
            output / "scoring_only" / f"{split}_references.jsonl",
            [
                {
                    "record_id": r["record_id"],
                    "dialogue_id": str(r["dialogue_id"]),
                    "task": r["task"],
                    "answer": r["answer"][0],
                }
                for r in partitions[split]
            ],
        )
    manifest = {
        "source_revision": REVISION,
        "split_seed": 8,
        "split_unit": "dialogue",
        "source_sha256": {v[0]: v[1] for v in SOURCES.values()},
        "dialogue_ids": {k: sorted(v) for k, v in splits.items()},
        "counts": {
            k: {
                "dialogues": len(splits[k]),
                "questions": len(v),
                "task_labels": dict(
                    sorted(
                        Counter((r["task"] + ":" + r["answer"][0] for r in v)).items()
                    )
                ),
            }
            for k, v in partitions.items()
        },
        "sft_rows": {k: len(v) for k, v in outputs.items()},
        "semantically_reviewed_training_records": len(reviews),
        "pending_evidence_reviews": len(pending),
        "review_provenance": "Assistant semantic inspection, label-aware; no independent human validation",
        "seed_selection": "Prior balanced 18-record pilot plus short training records across labels; includes all three training E records. Convenience seed, not representative.",
        "training_run": False,
        "all_training_answers_exported": True,
        "all_evidence_reviewed": False,
        "recommended_recipe": "mixed partial evidence supervision: original answer targets for all training records plus checked state/revision auxiliary targets from the reviewed seed",
        "notes": [
            "Label-state export is a control, not a verified Subjectesis loop.",
            "Compare seed conditions on matched underlying records; do not compare full-data answer-only against seed-only Subjectesis and attribute differences to method.",
            "Matched-step repetition matches example count, not token count or compute.",
            "No external benchmark was downloaded, used for annotation, training or tuning.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    file_hashes = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "checksums.json"
    }
    write_json(output / "checksums.json", file_hashes)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = convert(args.root, args.output or args.root / "data")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "counts",
                    "sft_rows",
                    "semantically_reviewed_training_records",
                    "pending_evidence_reviews",
                )
            },
            indent=2,
        )
    )
