"""Explain benchmark errors and review behavior using saved records.

Run analyse.py first, then this script to create subgroup scores, parsing
checks, state-answer comparisons and an error casebook. The review-policy
replay selects between already saved answers and is an exploratory analysis,
not a newly evaluated or deployment-selected policy.
"""

from project_setup import ensure_analysis_inputs

ensure_analysis_inputs()
from pathlib import Path
import collections, json, re
import numpy as np
import pandas as pd
from analyse import *

CLAIMS = {
    "A": {
        "proposer": "recommender",
        "seen": "no",
        "recommendation_response": "declined",
    },
    "B": {
        "proposer": "recommender",
        "seen": "no",
        "recommendation_response": "accepted",
    },
    "C": {
        "proposer": "recommender",
        "seen": "yes",
        "recommendation_response": "declined",
    },
    "D": {
        "proposer": "recommender",
        "seen": "yes",
        "recommendation_response": "accepted",
    },
    "E": {"proposer": "seeker", "seen": "no", "appraisal": "likes"},
    "F": {"proposer": "seeker", "seen": "yes", "appraisal": "dislikes"},
    "G": {"proposer": "seeker", "seen": "yes", "appraisal": "likes"},
}


def run(root):
    """Analyze frozen predictions by subgroup, inspect state-answer agreement and
    parsing, and collect cases with specified disagreements or changes. Replay
    an uncertainty-triggered selection of already-saved review branches as an
    exploratory CPU analysis, then save its tables and casebook without new
    inference.
    """
    out = root / "results"
    subgroup = []
    replay = []
    parsers = []
    agreements = []
    cases = []
    for kind in ["rectom", "opentom"]:
        df, structured = load_dataset(root, kind)
        inp = index(
            read_rows(
                root
                / (
                    "inputs/preparation/evaluation/test_inputs.jsonl"
                    if kind == "rectom"
                    else "inputs/opentom/opentom_transfer/final_inputs_corrected.jsonl"
                )
            )
        )
        proc = pd.read_csv(out / f"{kind}_process_records.csv").set_index("record_id")
        if kind == "rectom":
            df["order"] = "belief_or_desire_not_directly_comparable_to_opentom_orders"
        else:
            df["order"] = df.family.map(
                lambda s: (
                    "first_order"
                    if "_fo" in s
                    else "second_order" if "_so" in s else "attitude"
                )
            )
        for group_col in ["family", "order"]:
            for group, part in df.groupby(group_col):
                for sys in SYSTEMS:
                    subgroup.append(
                        {
                            "dataset": kind,
                            "grouping": group_col,
                            "group": group,
                            "system": sys,
                            "questions": len(part),
                            "clusters": part.cluster.nunique(),
                            "correct": int((part[sys] == part.reference).sum()),
                            "accuracy": float((part[sys] == part.reference).mean()),
                            "out_of_target": int(
                                sum(
                                    (
                                        p not in LABELS[f]
                                        for p, f in zip(part[sys], part.family)
                                    )
                                )
                            ),
                        }
                    )
        df["trigger"] = False
        for idx, row in df.iterrows():
            rid = row.record_id
            r = structured[rid]
            before = json_object(r.get("raw_initial")) or {}
            if kind == "rectom":
                df.loc[idx, "trigger"] = any(
                    (v.get("value") == "unknown" for v in r["state_before"].values())
                )
                for stage, key, pred in [
                    ("initial", "state_before", row.subjectesis_no_review),
                    ("reviewed", "state_after", row.subjectesis_review),
                ]:
                    expected = (
                        CLAIMS[pred]
                        if row.family == "belief"
                        else {"intends_to_watch": "yes" if pred == "A" else "no"}
                    )
                    known_conflict = [
                        f
                        for f, v in expected.items()
                        if r[key][f]["value"] not in ["unknown", v]
                    ]
                    unknown = [f for f in expected if r[key][f]["value"] == "unknown"]
                    agreements.append(
                        {
                            "dataset": kind,
                            "record_id": rid,
                            "stage": stage,
                            "family": row.family,
                            "answer": pred,
                            "reference": row.reference,
                            "correct": pred == row.reference,
                            "known_conflict_fields": "|".join(known_conflict),
                            "known_conflicts": len(known_conflict),
                            "unknown_fields": "|".join(unknown),
                            "unsupported_fields": len(unknown),
                        }
                    )
            else:
                df.loc[idx, "trigger"] = (
                    str(before.get("mental_state_claim", "")).strip().lower()
                    == "unknown"
                )
                for sys in SYSTEMS:
                    val = row[sys]
                    parsers.append(
                        {
                            "dataset": kind,
                            "record_id": rid,
                            "family": row.family,
                            "system": sys,
                            "raw": row[sys + "_raw"],
                            "parsed": val,
                            "out_of_target": val not in LABELS[row.family],
                            "saved_corrupt_flag": val == "-1",
                            "ambiguous_location": row.family.startswith("location_fg")
                            and val == "3",
                        }
                    )
            # Select all cases satisfying these error/change criteria for inspection.
            # This casebook is deliberately selected, so its case frequencies do not
            # estimate prevalence in an unselected population.
            changed = proc.loc[rid, "transition"] in [
                "right_to_wrong",
                "wrong_to_right",
            ]
            discord = (row.subjectesis_direct == row.reference) != (
                row.answer_only_direct == row.reference
            )
            if (
                changed
                or discord
                or (kind == "rectom" and bool(proc.loc[rid, "any_state_change"]))
            ):
                cases.append(
                    {
                        "dataset": kind,
                        "record_id": rid,
                        "cluster": row.cluster,
                        "family": row.family,
                        "selection_reason": {
                            "review_correctness_changed": bool(changed),
                            "training_conditions_discordant": bool(discord),
                            "rectom_state_changed": bool(
                                proc.loc[rid].get("any_state_change", False)
                            ),
                        },
                        "input": inp[rid],
                        "reference": row.reference,
                        "predictions": {s: row[s + "_raw"] for s in SYSTEMS},
                        "structured": r,
                    }
                )
        # Replay a rule by choosing among answers already produced in the study.
        # This does not run an adaptive controller or validate a new deployment policy.
        pred = np.where(df.trigger, df.subjectesis_review, df.subjectesis_no_review)
        groups = sorted(df.cluster.unique())
        counts = bootstrap_counts(len(groups))
        den = np.array([(df.cluster == g).sum() for g in groups], float)
        diff = (pred == df.reference).astype(float) - (
            df.subjectesis_no_review == df.reference
        ).astype(float)
        delta = np.array([diff[df.cluster == g].sum() for g in groups])
        boot = counts @ delta / (counts @ den)
        replay.append(
            {
                "dataset": kind,
                "policy": (
                    "review_if_any_unknown_field"
                    if kind == "rectom"
                    else "review_if_claim_exactly_unknown"
                ),
                "triggered_questions": int(df.trigger.sum()),
                "correct": int((pred == df.reference).sum()),
                "accuracy": float((pred == df.reference).mean()),
                "difference_from_no_review": float(diff.mean()),
                "difference_low": interval(boot)[0],
                "difference_high": interval(boot)[1],
                "status": "post-hoc offline branch replay; not new inference or validation-selected deployment policy",
            }
        )
    pd.DataFrame(subgroup).to_csv(out / "subgroup_metrics.csv", index=False)
    pd.DataFrame(replay).to_csv(out / "offline_review_policy_replay.csv", index=False)
    pd.DataFrame(parsers).to_csv(out / "opentom_parser_audit.csv", index=False)
    pd.DataFrame(agreements).to_csv(
        out / "rectom_state_answer_agreement.csv", index=False
    )
    with (out / "error_casebook.jsonl").open("w") as f:
        for r in cases:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(
        out / "error_casebook_scope.json",
        {
            "selected_records": len(cases),
            "selection": "All review correctness transitions, direct condition correctness disagreements, and RecToM final state changes. Selection is exhaustive for these criteria, not a prevalence sample.",
            "semantic_validation": "Casebook generation is automated. Separate interpretive annotations identify records actually read; no independent human validation or inter-rater agreement is claimed.",
        },
    )
    print(pd.DataFrame(replay).to_string(index=False))
    print(
        "RecToM state-answer agreements",
        pd.DataFrame(agreements)
        .groupby("stage")[["known_conflicts", "unsupported_fields"]]
        .agg(["sum", lambda x: int((x > 0).sum())]),
    )


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
