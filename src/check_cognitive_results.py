"""Check the saved cognitive outputs without loading Qwen or using a GPU.

Counterfactual scores and process counts are recomputed from individual saved
records. Probe and RSA tables are checked for consistency only: independently
refitting them requires the hidden-state arrays saved in the original run.
"""

from collections import Counter
from pathlib import Path
import ast
import json

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from project_setup import ensure_analysis_inputs

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/cognitive"
SYSTEMS = [
    "base_direct",
    "answer_only_direct",
    "subjectesis_direct",
    "subjectesis_no_review",
    "subjectesis_review",
]


def read_rows(path):
    """Read non-empty JSONL lines into records in file order. These saved records
    provide the individual observations used to check the cognitive summaries.
    """
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def check_counterfactuals():
    """Recalculate accuracy, success on both members of a pair, and
    unobserved-location leakage from the 64 saved cases. Check that the cases
    match the frozen diagnostic and that the recomputed table matches the
    supplied results; no model is run.
    """
    frame = pd.read_csv(RESULTS / "counterfactual_predictions.csv")
    frame["options"] = frame.options.map(ast.literal_eval)
    frozen = {
        row["record_id"]: row
        for row in read_rows(RESULTS / "counterfactual_diagnostic_frozen.jsonl")
    }
    assert len(frame) == len(frozen) == 64
    assert frame.record_id.is_unique
    for row in frame.to_dict("records"):
        for key, value in frozen[row["record_id"]].items():
            assert row[key] == value, (row["record_id"], key)

    metrics = []
    for system in SYSTEMS:
        valid = frame[system].notna()
        correct = frame[system] == frame.gold
        for order in ["all", "first_order", "second_order"]:
            mask = valid.copy()
            if order != "all":
                mask &= frame.order == order
            metrics.append(
                {
                    "system": system,
                    "order": order,
                    "n": int(mask.sum()),
                    "accuracy": float(correct[mask].mean()),
                    "invalid_rate": float((~valid).mean()),
                }
            )

        # A pair succeeds only when both observed and unobserved cases are right.
        pair_success = []
        for _, group in frame.groupby("pair_id"):
            assert len(group) == 2
            if group[system].notna().all():
                pair_success.append((group[system] == group.gold).all())

        leakage = []
        for row in frame[valid & ~frame.observed].to_dict("records"):
            new_letter = next(
                k for k, v in row["options"].items() if v == row["new_location"]
            )
            leakage.append(row[system] == new_letter)
        metrics.append(
            {
                "system": system,
                "order": "paired_counterfactual",
                "n": len(pair_success),
                "accuracy": float(np.mean(pair_success)),
                "invalid_rate": float((~valid).mean()),
                "unobserved_world_state_leakage_rate": float(np.mean(leakage)),
            }
        )
    expected = pd.read_csv(RESULTS / "counterfactual_metrics.csv")
    assert_frame_equal(
        pd.DataFrame(metrics), expected, check_exact=False, rtol=1e-12, atol=1e-12
    )
    return {"questions": 64, "pairs": 32, "systems": 5, "metrics_match": True}


def check_process_counts():
    """Recount review decisions and changes to states, evidence and answers from
    the saved RecToM and OpenToM records. Compare the totals with the cognitive
    process summary, keeping a state revision distinct from an improvement in
    correctness.
    """
    expected = json.loads(
        (RESULTS / "monitoring_control_process_summary.json").read_text()
    )
    rectom = read_rows(
        ROOT / "inputs/rectom/final_test/structured_test_predictions.jsonl"
    )
    decisions, fields = Counter(), Counter()
    changes = {key: Counter() for key in ["value", "basis", "evidence_turns"]}
    preserve_changed = 0
    for row in rectom:
        before, after = row["state_before"], row["state_after"]
        for field in set(before) | set(after):
            for key in changes:
                if before.get(field, {}).get(key) != after.get(field, {}).get(key):
                    changes[key][field] += 1
        for review in row["reviews"]:
            field = review["field"]
            parsed = review.get("parsed") or {}
            decision = parsed.get("decision", "invalid")
            fields[field] += 1
            decisions[decision] += 1
            if decision == "preserve" and field in before:
                preserve_changed += (parsed.get("updated_claim") or {}).get(
                    "value"
                ) != before[field].get("value")
    actual = {
        "questions": len(rectom),
        "review_calls": sum(fields.values()),
        "answer_changes": sum(
            r["no_review_prediction"] != r["reviewed_prediction"] for r in rectom
        ),
        "decision_counts": dict(decisions),
        "field_review_counts": dict(fields),
        "field_value_changes": dict(changes["value"]),
        "field_basis_changes": dict(changes["basis"]),
        "field_evidence_turn_changes": dict(changes["evidence_turns"]),
        "preserve_decision_with_changed_value": preserve_changed,
    }
    assert actual == expected["rectom"]

    opentom = read_rows(
        ROOT / "inputs/opentom/opentom_transfer/subjectesis_structured.jsonl"
    )
    decisions = Counter()
    actual = {
        "questions": len(opentom),
        "invalid_initial": 0,
        "invalid_review": 0,
        "mental_state_claim_changes": 0,
        "evidence_sentence_changes": 0,
        "uncertainty_changes": 0,
        "raw_answer_changes": sum(
            str(r["no_review_prediction"]).strip()
            != str(r["review_prediction"]).strip()
            for r in opentom
        ),
    }
    for row in opentom:
        if not row.get("initial_valid", False):
            actual["invalid_initial"] += 1
            continue
        try:
            initial = json.loads(row["raw_initial"])
            review = json.loads(row["raw_review"])
        except (json.JSONDecodeError, TypeError):
            actual["invalid_review"] += 1
            continue
        if not row.get("review_valid", False) or not isinstance(review, dict):
            actual["invalid_review"] += 1
            continue
        decisions[review.get("decision", "unknown")] += 1
        for result_key, before_key, after_key in [
            (
                "mental_state_claim_changes",
                "mental_state_claim",
                "updated_mental_state_claim",
            ),
            ("evidence_sentence_changes", "evidence_sentences", "evidence_sentences"),
            ("uncertainty_changes", "uncertainty", "uncertainty"),
        ]:
            actual[result_key] += initial.get(before_key) != review.get(after_key)
    actual["decision_counts"] = dict(decisions)
    assert actual == expected["opentom"]
    return {"rectom_counts_match": True, "opentom_counts_match": True}


def check_representation_tables():
    """Check dialogue separation and agreement between the layer-wise tables and
    their final-layer summaries. This verifies saved-table consistency only; it
    does not refit probes or recompute RSA from hidden-state arrays.
    """
    index = pd.read_csv(RESULTS / "probe_dataset_index.csv")
    assert index.groupby("split").size().to_dict() == {
        "train": 2235,
        "validation": 319,
        "test": 621,
    }
    assert index.groupby("dialogue_id").split.nunique().max() == 1
    # Compare supplied summaries with other supplied tables. Recomputing
    # representations, fitting probes and rebuilding RSA require the external arrays.
    probes = pd.read_csv(RESULTS / "layerwise_linear_probe_results.csv")
    final = probes[probes.layer == 32].set_index(["field", "system"])
    for row in pd.read_csv(
        RESULTS / "probe_final_layer_dialogue_bootstrap.csv"
    ).itertuples():
        first, second = row.comparison.split(" - ")
        difference = (
            final.loc[(row.field, first), "test_macro_f1"]
            - final.loc[(row.field, second), "test_macro_f1"]
        )
        np.testing.assert_allclose(
            difference, row.point_macro_f1_difference, atol=1e-12
        )
    rsa = pd.read_csv(RESULTS / "rsa_belief_state_geometry.csv")
    final_rsa = rsa[rsa.layer == 32].set_index("system")
    for row in pd.read_csv(
        RESULTS / "rsa_final_layer_dialogue_jackknife.csv"
    ).itertuples():
        np.testing.assert_allclose(
            final_rsa.loc[row.system, "spearman_rho"], row.rho, atol=1e-12
        )
    return {
        "split_disjoint": True,
        "final_layer_tables_consistent": True,
        "probes_refitted": False,
        "rsa_recomputed_from_hidden_states": False,
    }


if __name__ == "__main__":
    ensure_analysis_inputs()
    report = {
        "counterfactual": check_counterfactuals(),
        "process": check_process_counts(),
        "representations": check_representation_tables(),
    }
    (ROOT / "results/cognitive_checks.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
