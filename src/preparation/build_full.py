"""Revision-4 data preparation: build full."""

import argparse, copy, hashlib, json, re
from collections import Counter
from pathlib import Path
from convert_v1 import (
    load_sources,
    digest,
    DOMAINS,
    FIELDS,
    public_row,
    input_text,
    state_prompt,
    review_prompt,
    training_example,
    answer_example,
    benchmark_state,
    write_jsonl,
    write_json,
)
from packets import groups, norm, title

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = {
    "pR": (
        "proposer",
        "recommender",
        "explicit",
        "The recommender introduces this movie, using an explicit name or an unambiguous local reference before the full title appears.",
    ),
    "pS": (
        "proposer",
        "seeker",
        "explicit",
        "The seeker introduces this movie, using an explicit name or an unambiguous local reference before the full title appears.",
    ),
    "sY": (
        "seen",
        "yes",
        "explicit",
        "The seeker reports having watched the target, possibly only in part; this does not guarantee completing it.",
    ),
    "sN": (
        "seen",
        "no",
        "explicit",
        "The seeker reports not having watched the target in the supplied context.",
    ),
    "sIy": (
        "seen",
        "yes",
        "inferred",
        "The seeker's recollection or description suggests prior viewing; this is a qualified inference, not an explicit certainty.",
    ),
    "sIn": (
        "seen",
        "no",
        "inferred",
        "The seeker's lack of familiarity or unrealized first-viewing plan suggests not yet watched; this is a qualified inference.",
    ),
    "aL": (
        "appraisal",
        "likes",
        "explicit",
        "The seeker expresses a favorable attitude to the target.",
    ),
    "aD": (
        "appraisal",
        "dislikes",
        "explicit",
        "The seeker expresses an unfavorable attitude to the target.",
    ),
    "aIl": (
        "appraisal",
        "likes",
        "inferred",
        "The seeker's preference example, comparison, or positive interest suggests a favorable attitude; this need not be a post-viewing appraisal.",
    ),
    "aId": (
        "appraisal",
        "dislikes",
        "inferred",
        "The seeker's qualified negative reaction suggests an unfavorable attitude; this is an inference.",
    ),
    "rA": (
        "recommendation_response",
        "accepted",
        "explicit",
        "The seeker accepts, chooses, or states an intention to try this suggestion.",
    ),
    "rD": (
        "recommendation_response",
        "declined",
        "explicit",
        "The seeker declines this suggestion for the current request; this need not mean disliking the film generally.",
    ),
    "rIa": (
        "recommendation_response",
        "accepted",
        "inferred",
        "The seeker's positive uptake suggests tentative acceptance; it is not a guaranteed viewing commitment.",
    ),
    "rId": (
        "recommendation_response",
        "declined",
        "inferred",
        "The seeker's objection or mismatch with the current request suggests declining this suggestion; this is a qualified inference.",
    ),
    "wY": (
        "intends_to_watch",
        "yes",
        "explicit",
        "The seeker states willingness or an intention to watch the target; a stated intention does not guarantee the future action.",
    ),
    "wN": (
        "intends_to_watch",
        "no",
        "explicit",
        "The seeker explicitly states unwillingness to watch the target.",
    ),
}


def annotations(root):
    """Load the fixed source-inspected annotation decisions."""
    path = root / "reviews" / "annotations.txt"
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, codes, note = line.split("|", 2)
        if key in result:
            raise ValueError("Duplicate semantic annotation: " + key)
        claims = {}
        for token in [] if codes == "-" else codes.split():
            code, refs = token.split(":")
            field, value, basis, reason = CLAIMS[code]
            if field in claims:
                raise ValueError("Multiple claims for a field: " + key)
            citations = [int(n) for n in refs.split(",")]
            if len(set(citations)) != len(citations):
                raise ValueError("Duplicate citation: " + key)
            claims[field] = {
                "value": value,
                "evidence_turns": sorted(citations),
                "basis": basis,
                "reason": reason,
            }
        result[key] = {"claims": claims, "audit_note": note}
    return result


def unknown(field):
    """Represent an unsupported field using the explicit unknown state."""
    return {
        "value": "unknown",
        "evidence_turns": [],
        "basis": "not_established",
        "reason": "The supplied dialogue prefix does not unambiguously establish "
        + FIELDS[field]
        + " for the target movie.",
    }


def source_state(row, item):
    """Compile annotated perspective claims with their source-turn references."""
    lines = row["utterance_context"].splitlines()
    claims = copy.deepcopy(item["claims"])
    for field, c in claims.items():
        if any(
            (
                type(n) is not int or not 1 <= n <= len(lines)
                for n in c["evidence_turns"]
            )
        ):
            raise ValueError(
                "Future/out-of-bounds evidence: " + row["record_id"] + "/" + field
            )
        if field != "proposer" and (
            not any(
                (lines[n - 1].startswith("SEEKER says:") for n in c["evidence_turns"])
            )
        ):
            raise ValueError(
                "Only other-speaker evidence: " + row["record_id"] + "/" + field
            )
    if row["task"] == "belief":
        slots = {
            field: claims.get(field, unknown(field)) for field in DOMAINS["belief"]
        }
        mentions = [
            (i, line)
            for i, line in enumerate(lines, 1)
            if norm(title(row)) in norm(line)
        ]
        if mentions and "proposer" not in claims:
            i, line = mentions[0]
            if line.startswith("SEEKER says:"):
                speaker = "seeker"
            elif line.startswith("RECOMMENDER says:"):
                speaker = "recommender"
            else:
                raise ValueError("Unknown speaker")
            slots["proposer"] = {
                "value": speaker,
                "evidence_turns": [i],
                "basis": "explicit",
                "reason": "This speaker makes the first explicit mention of the target movie in the supplied prefix, including preference examples.",
            }
        return slots
    if "intends_to_watch" in claims:
        return {"intends_to_watch": claims["intends_to_watch"]}
    if "recommendation_response" in claims:
        c = claims["recommendation_response"]
        positive = c["value"] == "accepted"
        return {
            "intends_to_watch": {
                "value": "yes" if positive else "no",
                "evidence_turns": c["evidence_turns"],
                "basis": "inferred",
                "reason": "The seeker's "
                + ("positive" if positive else "negative")
                + " response to this recommendation supports a "
                + ("positive" if positive else "negative")
                + " willingness prediction; it does not guarantee a future action.",
            }
        }
    if "appraisal" in claims:
        c = claims["appraisal"]
        positive = c["value"] == "likes"
        return {
            "intends_to_watch": {
                "value": "yes" if positive else "no",
                "evidence_turns": c["evidence_turns"],
                "basis": "inferred",
                "reason": "The seeker's "
                + ("favorable" if positive else "unfavorable")
                + " attitude supports a "
                + ("positive" if positive else "negative")
                + " willingness prediction, including possible rewatching. No explicit viewing commitment is established.",
            }
        }
    return {"intends_to_watch": unknown("intends_to_watch")}


def check_state(row, slots):
    """Check allowed state values and the structural evidence contract."""
    if set(slots) != set(DOMAINS[row["task"]]):
        raise ValueError("Incorrect state fields")
    lines = row["utterance_context"].splitlines()
    for name, c in slots.items():
        if c["value"] not in DOMAINS[row["task"]][name]:
            raise ValueError("Invalid domain")
        if c["basis"] not in ("explicit", "inferred", "not_established"):
            raise ValueError("Invalid basis")
        if not c["reason"]:
            raise ValueError("Missing reason")
        if c["value"] == "unknown":
            if c["basis"] != "not_established" or c["evidence_turns"]:
                raise ValueError("Unknown claim not clean")
        elif not c["evidence_turns"] or c["basis"] == "not_established":
            raise ValueError("Unsupported known claim")
        for n in c["evidence_turns"]:
            if type(n) is not int or not 1 <= n <= len(lines):
                raise ValueError("Future evidence")
    refs = sorted({n for c in slots.values() for n in c["evidence_turns"]})
    return [{"turn": n, "quote": lines[n - 1]} for n in refs]


def target_for(row, slots):
    """Construct the recorded structured completion target for one question."""
    state = benchmark_state(row)
    conflicts = [
        name
        for name, value in state.items()
        if slots[name]["value"] not in ("unknown", value)
    ]
    return {
        "perspective": (
            "recommender about seeker"
            if row["task"] == "belief"
            else "seeker likely willingness"
        ),
        "evidence": check_state(row, slots),
        "state": slots,
        "missing_information": [
            FIELDS[n] for n, c in slots.items() if c["value"] == "unknown"
        ],
        "answer": row["answer"][0],
        "answer_claims_not_established": [
            n for n in state if slots[n]["value"] == "unknown"
        ],
        "answer_claims_conflicting": conflicts,
    }


def misleading_citation(row, target, field):
    """Select a legal distractor under the unchanged reviewed unknown state.

    Prefer a turn already used as evidence for a DIFFERENT state field.
    Otherwise use a title-bearing turn, then any prefix turn. No gold access.
    The non-entailment judgment comes from the reviewed whole-prefix unknown
    annotation, not from citation bounds or the selection heuristic."""
    claim = target["state"][field]
    if claim["value"] != "unknown" or claim["evidence_turns"]:
        raise ValueError("Distractor construction requires a reviewed unknown")
    lines = row["utterance_context"].splitlines()
    other = sorted(
        {
            n
            for f, c in target["state"].items()
            if f != field
            for n in c["evidence_turns"]
        }
    )
    mentions = [i for i, line in enumerate(lines, 1) if norm(title(row)) in norm(line)]
    pool = other or mentions or list(range(1, len(lines) + 1))
    if not pool:
        raise ValueError("No legal citation in an empty prefix")
    offset = int(
        hashlib.sha256((row["record_id"] + "/" + field).encode()).hexdigest(), 16
    ) % len(pool)
    turn = pool[offset]
    if not 1 <= turn <= len(lines):
        raise ValueError("Illegal distractor citation")
    return {
        "turn": turn,
        "quote": lines[turn - 1],
        "selection": (
            "other_field_evidence"
            if other
            else "title_mention" if mentions else "prefix_turn"
        ),
        "relationship": "does_not_establish_selected_field_under_reviewed_unknown_annotation",
    }


def examples(row, target):
    """Expand a training question into the recorded answer, state and review examples."""
    if target["answer_claims_conflicting"]:
        raise ValueError("Conflict must be quarantined")
    target = copy.deepcopy(target)
    target.pop("answer_claims_conflicting")
    out = [
        training_example(
            row,
            "build_state",
            state_prompt(row),
            json.dumps(target, ensure_ascii=False),
        )
    ]
    for field, correct in target["state"].items():
        cases = (
            ["retain_unknown"]
            + [
                kind + ":" + v
                for kind in ["withdraw_unsupported", "withdraw_misleading_citation"]
                for v in DOMAINS[row["task"]][field]
                if v != "unknown"
            ]
            if correct["value"] == "unknown"
            else ["masked", "wrong", "preserve", "recover_citation"]
        )
        for case in cases:
            previous = copy.deepcopy(target["state"])
            construction_case = case.split(":")[0]
            if construction_case in (
                "withdraw_unsupported",
                "withdraw_misleading_citation",
            ):
                previous[field]["value"] = case.split(":", 1)[1]
            if construction_case == "withdraw_misleading_citation":
                distractor = misleading_citation(row, target, field)
                previous[field]["evidence_turns"] = [distractor["turn"]]
            elif case == "recover_citation":
                previous[field]["evidence_turns"] = []
            if case == "masked":
                previous[field] = unknown(field)
            elif case == "wrong":
                previous[field]["value"] = next(
                    (
                        v
                        for v in DOMAINS[row["task"]][field]
                        if v not in ("unknown", correct["value"])
                    )
                )
            decision = (
                "retain_unknown"
                if case == "retain_unknown"
                else (
                    "preserve" if case in ("preserve", "recover_citation") else "revise"
                )
            )
            response = {
                "field": field,
                "decision": decision,
                "updated_claim": correct,
                "evidence": [
                    e
                    for e in target["evidence"]
                    if e["turn"] in correct["evidence_turns"]
                ],
                "stop_reason": (
                    "evidence_not_sufficient"
                    if correct["value"] == "unknown"
                    else "selected_field_reviewed"
                ),
            }
            example = training_example(
                row,
                "review_state",
                review_prompt(row, previous, field),
                json.dumps(response, ensure_ascii=False),
                "/" + field + "/" + case,
            )
            example["construction"] = {
                "synthetic_state": True,
                "case": construction_case,
                "selected_field": field,
            }
            if construction_case in (
                "withdraw_unsupported",
                "withdraw_misleading_citation",
            ):
                example["construction"]["unsupported_value"] = previous[field]["value"]
            if construction_case == "withdraw_misleading_citation":
                example["construction"]["distractor"] = distractor
            out.append(example)
    return out


def run(root=ROOT, output=None):
    output = output or root / "data"
    rows, splits = load_sources(root)
    train = [r for r in rows if str(r["dialogue_id"]) in splits["train"]]
    ann = annotations(root)
    packets = groups()
    expected = {t["key"] for g in packets for t in g["targets"]}
    if set(ann) != expected:
        raise ValueError("Incomplete target coverage")
    row_keys = {
        rid: t["key"] for g in packets for t in g["targets"] for rid in t["rows"]
    }
    if set(row_keys) != {r["record_id"] for r in train}:
        raise ValueError("Nontraining or missing rows")
    ledger = []
    quarantine = []
    clean = []
    reasons = Counter()
    all_sources = {r["record_id"]: r for r in train}
    for row in train:
        key = row_keys[row["record_id"]]
        a = ann[key]
        slots = source_state(row, a)
        target = target_for(row, slots)
        flags = []
        if target["answer_claims_conflicting"]:
            flags.append("evidence_label_conflict")
        if not any(
            (
                norm(title(row)) in norm(line)
                for line in row["utterance_context"].splitlines()
            )
        ):
            flags.append("target_not_explicitly_named")
        item = {
            "record_id": row["record_id"],
            "dialogue_id": str(row["dialogue_id"]),
            "task": row["task"],
            "source_record_sha256": digest(row),
            "review_status": "assistant_source_inspected",
            "independent_human_review": False,
            "annotation_key": key,
            "review_protocol": "full_source_inspection_v2_with_targeted_audit_v3",
            "audit_note": a["audit_note"],
            "quality_flags": flags,
            "official_answer": row["answer"][0],
            "official_answer_state": benchmark_state(row),
            "grounded_target": target,
            "input": public_row(row),
            "training_eligible": not flags,
        }
        ledger.append(item)
        if flags:
            quarantine.append(item)
            reasons.update(flags)
        else:
            clean.append(item)
    sft = []
    matched = []
    state_only = []
    plain = []
    factual = []
    for item in clean:
        row = all_sources[item["record_id"]]
        ex = examples(row, item["grounded_target"])
        sft.extend([answer_example(row)] + ex)
        state_only.append(ex[0])
        factual.extend(ex)
        plain.append(answer_example(row))
        matched.extend(
            (answer_example(row, "/repeat_" + str(i)) for i in range(1 + len(ex)))
        )
    write_jsonl(output / "review" / "all_training_records.jsonl", ledger)
    write_jsonl(output / "review" / "quarantined_training_records.jsonl", quarantine)
    exports = {
        "subjectesis_full_train": sft,
        "answer_only_matched_train": matched,
        "subjectesis_state_only_train": state_only,
        "subjectesis_auxiliary_train": factual,
        "answer_only_clean_train": plain,
        "answer_only_original_train": [answer_example(r) for r in train],
    }
    for name, items in exports.items():
        write_jsonl(output / "sft" / f"{name}.jsonl", items)
    for split in ("validation", "test"):
        rr = [r for r in rows if str(r["dialogue_id"]) in splits[split]]
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
                for r in rr
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
                for r in rr
            ],
        )
    manifest = {
        "version": 4,
        "source_revision": "317b91d3db47f4e278b8bb3eefcbb733fb7d639b",
        "split_unit": "dialogue",
        "seed": 8,
        "dialogue_ids": {k: sorted(v) for k, v in splits.items()},
        "split_counts": {
            k: {
                "dialogues": len(v),
                "questions": sum((str(r["dialogue_id"]) in v for r in rows)),
            }
            for k, v in splits.items()
        },
        "reviewed_training_questions": len(ledger),
        "reviewed_dialogue_movie_targets": len(ann),
        "pending_annotation_questions": 0,
        "training_eligible_questions": len(clean),
        "quarantined_questions": len(quarantine),
        "quarantine_reasons": dict(reasons),
        "eligible_training_dialogues": len({r["dialogue_id"] for r in clean}),
        "training_labels_original": dict(
            Counter((r["task"] + ":" + r["answer"][0] for r in train))
        ),
        "training_labels_eligible": dict(
            Counter((r["task"] + ":" + r["official_answer"] for r in clean))
        ),
        "claim_basis_all": dict(
            Counter(
                (
                    c["basis"]
                    for r in ledger
                    for c in r["grounded_target"]["state"].values()
                )
            )
        ),
        "questions_with_unsupported_answer_claims": sum(
            (
                bool(r["grounded_target"]["answer_claims_not_established"])
                for r in ledger
            )
        ),
        "sft_examples": {k: len(v) for k, v in exports.items()},
        "provenance": "Assistant source inspection of every training dialogue and target. No independent human validation. Audit notes are not model targets. Desire from appraisal/response is qualified rule-assisted inference.",
        "strictly_blinded_annotation": False,
        "annotation_packet_includes_reference_labels": False,
        "annotation_notes": "The assistant had prior context about 51 seed questions. Source packets themselves omit answer labels. All structured states are created before gold-label comparisons. Exclusion rules apply equally to both training arms.",
        "revision_cases": dict(
            Counter((r["construction"]["case"] for r in sft if "construction" in r))
        ),
        "eligible_questions_with_unsupported_answer_claims": sum(
            (bool(r["grounded_target"]["answer_claims_not_established"]) for r in clean)
        ),
        "schema_contract": {
            "version": 1,
            "desire_key": "intends_to_watch",
            "meaning": "likely willingness, not guaranteed future action",
        },
        "training_completed": False,
        "external_benchmark_used": False,
        "annotations_sha256": hashlib.sha256(
            (root / "reviews" / "annotations.txt").read_bytes()
        ).hexdigest(),
    }
    write_json(output / "manifest.json", manifest)
    checksums = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "checksums.json"
    }
    write_json(output / "checksums.json", checksums)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    print(json.dumps(run(output=a.output), indent=2))
