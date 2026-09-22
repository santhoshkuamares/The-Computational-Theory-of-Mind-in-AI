"""Frozen-output CPU analysis: analyse.
Run from the repository root. No model inference or adapter training occurs."""

from __future__ import annotations
from project_setup import ensure_analysis_inputs

ensure_analysis_inputs()
import argparse, collections, hashlib, json, math, random, re
from pathlib import Path
import numpy as np
import pandas as pd

SYSTEMS = [
    "base_direct",
    "answer_only_direct",
    "subjectesis_direct",
    "subjectesis_no_review",
    "subjectesis_review",
]
COMPARISONS = [(1, 0), (2, 0), (2, 1), (3, 2), (4, 3)]
LABELS = {
    "belief": list("ABCDEFG"),
    "desire": list("AB"),
    "location_cg_fo": ["0", "1"],
    "location_cg_so": ["0", "1"],
    "location_fg_fo": ["1", "2"],
    "location_fg_so": ["1", "2"],
    **{
        f"multihop_{o}_{t}": ["1", "2", "3"]
        for o in ["fo", "so"]
        for t in ["fullness", "accessibility"]
    },
    "attitude": ["1", "2", "3"],
}


def read_json(p):
    """Read one JSON object from disk."""
    return json.loads(Path(p).read_text(encoding="utf8"))


def read_rows(p):
    """Load the saved per-example records without modifying predictions."""
    with Path(p).open(encoding="utf8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_json(p, obj):
    """Write a JSON result using the original serialization convention."""
    Path(p).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf8",
    )


def sha(p):
    """Calculate a file checksum so that a changed input is detected before reuse."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def index(rows):
    """Index records by ID and reject duplicates before pairing systems."""
    out = {r["record_id"]: r for r in rows}
    assert len(out) == len(rows), "Duplicate record IDs"
    return out


def determiner(x):
    return re.sub("^(?:a|an|the)\\s+", "", str(x).strip().lower())


def overlap(pred, loc):
    a = str(pred).lower().replace("_", " ").replace("'s", "").replace(".", "").split()
    b = str(loc).lower().replace("_", " ").replace("'s", "").split()
    return len(set(a) & set(b)) / len(b) if b else 0.0


def parse_open(pred, ref):
    """Frozen corrected scoring convention; an out-of-target label stays an error."""
    p = "" if pred is None else str(pred).strip()
    a = ref["answer"]
    f = ref["family"]
    d = ref["family_detail"]
    if f.startswith("location_fg"):
        orig, moved = (
            determiner(ref["original_place"]),
            determiner(ref["move_to_place"]),
        )
        gt = (
            {"original": "1", "moved": "2"}[
                "original" if determiner(a) == orig else "moved"
            ]
            if determiner(a) in [orig, moved]
            else None
        )
        os, ms = (overlap(p, orig), overlap(p, moved))
        return (gt, "3" if os == ms else "1" if os > ms else "2")
    if f.startswith("location_cg"):
        pl = (
            "0"
            if "no" in p.lower() and "yes" not in p.lower()
            else "1" if "yes" in p.lower() and "no" not in p.lower() else "-1"
        )
        gt = "0" if "no" in str(a).lower() else "1" if "yes" in str(a).lower() else None
        return (gt, pl)
    if "fullness" in d:
        p = p.lower().replace(".", "")
        pl = (
            "1"
            if any((w in p for w in ["less full", "emptier", "more empty"]))
            else (
                "2"
                if any((w in p for w in ["more full", "fuller"]))
                else "3" if "equally full" in p else "-1"
            )
        )
        return (
            {"less full": "1", "more full": "2", "equally full": "3"}.get(
                str(a).strip().lower()
            ),
            pl,
        )
    if "accessibility" in d:
        p = p.lower().replace(".", "")
        pl = (
            "1"
            if "more accessible" in p
            else (
                "2"
                if "less accessible" in p
                else "3" if "equally accessible" in p else "-1"
            )
        )
        if isinstance(a, list):
            a = [v for v in a if v != "corrupted"]
            a = a[0] if len(a) == 1 else None
        if a is not None and "|" in str(a):
            a = "equally accessible"
        return (
            {
                "more accessible": "1",
                "less accessible": "2",
                "equally accessible": "3",
            }.get(str(a).strip().lower()),
            pl,
        )
    if f == "attitude":
        token = p.lower().split("\n\n")[-1].split(":")[-1].split(".")[0].strip()
        token = {"a": "positive", "b": "neutral", "c": "negative"}.get(token, token)
        found = [k for k in ["positive", "negative", "neutral"] if k in token]
        vals = {"positive": "1", "negative": "2", "neutral": "3"}
        return (
            vals.get(str(a).strip().lower()),
            vals[found[0]] if len(found) == 1 else "-1",
        )
    raise ValueError(d)


# Resample complete source clusters, keeping the systems paired within each draw.
def bootstrap_counts(n, reps=5000, seed=42):
    """Sample whole dialogue/story clusters and return their multiplicities."""
    rng = random.Random(seed)
    return np.array(
        [
            np.bincount([rng.randrange(n) for _ in range(n)], minlength=n)
            for _ in range(reps)
        ],
        dtype=float,
    )


def interval(x):
    """Return the central 95 percent interval using the saved quantile convention."""
    return np.quantile(x, [0.025, 0.975]).tolist()


def cluster_statistics(df):
    """Aggregate correct counts and class statistics within each source cluster."""
    groups = sorted(df.cluster.unique())
    families = [f for f in LABELS if f in set(df.family)]
    gindex = {v: i for i, v in enumerate(groups)}
    n = np.zeros((len(groups), len(families)))
    correct = np.zeros((len(SYSTEMS), len(groups), len(families)))
    classes = {}
    for j, f in enumerate(families):
        labs = LABELS[f]
        support = np.zeros((len(groups), len(labs)))
        pred = np.zeros((5, len(groups), len(labs)))
        tp = pred.copy()
        for _, r in df[df.family == f].iterrows():
            i = gindex[r.cluster]
            n[i, j] += 1
            k = labs.index(r.reference)
            support[i, k] += 1
            for s, sys in enumerate(SYSTEMS):
                correct[s, i, j] += int(r[sys] == r.reference)
                if r[sys] in labs:
                    l = labs.index(r[sys])
                    pred[s, i, l] += 1
                    if k == l:
                        tp[s, i, l] += 1
        classes[f] = (support, pred, tp)
    return (groups, families, n, correct, classes)


def metrics_for_weights(weights, n, correct, classes, families):
    """Sufficient-statistics bootstrap, also usable for leave-one-cluster-out."""
    den = weights @ n
    accuracy = []
    task_mean = []
    f1 = []
    facc = []
    ff1 = []
    for s in range(5):
        num = weights @ correct[s]
        by = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        accuracy.append(num.sum(1) / den.sum(1))
        task_mean.append(by.mean(1))
        facc.append(by)
        fs = []
        for f in families:
            sup, pre, tp = classes[f]
            d = weights @ (sup + pre[s])
            t = weights @ tp[s]
            fs.append(np.divide(2 * t, d, out=np.zeros_like(t), where=d > 0).mean(1))
        ff1.append(np.stack(fs, axis=1))
        f1.append(np.mean(fs, axis=0))
    return dict(
        accuracy=np.array(accuracy),
        mean_family_accuracy=np.array(task_mean),
        mean_family_f1=np.array(f1),
        family_accuracy=np.array(facc),
        family_f1=np.array(ff1),
    )


# Swap the two systems at cluster level to preserve within-story dependence.
def signflip_p(contributions, reps=50000, seed=20260922):
    """Paired cluster label swaps. Conditional exchangeability is an assumption.

    Zero-contribution clusters can be removed exactly for a linear statistic.
    Enumerate all assignments up to 18 nonzero clusters; otherwise Monte Carlo
    with the add-one correction. This is post-hoc, not a randomized experiment."""
    d = np.asarray(contributions, dtype=float)
    d = d[np.abs(d) > 1e-14]
    obs = abs(d.sum())
    if not len(d):
        return (1.0, 1, "exact", 0)
    if len(d) <= 18:
        nums = np.arange(2 ** len(d), dtype=np.uint64)[:, None]
        signs = 2 * (nums >> np.arange(len(d), dtype=np.uint64) & 1).astype(float) - 1
        p = float(np.mean(np.abs(signs @ d) >= obs - 1e-12))
        return (p, len(signs), "exact", len(d))
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, reps, 2000):
        signs = rng.choice([-1.0, 1.0], size=(min(2000, reps - start), len(d)))
        hits += int((np.abs(signs @ d) >= obs - 1e-12).sum())
    return ((hits + 1) / (reps + 1), reps, "monte_carlo", len(d))


def holm(pvalues):
    """Adjust the specified family of p-values using the Holm step-down procedure."""
    p = np.asarray(pvalues)
    order = np.argsort(p)
    out = np.zeros(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def load_dataset(root, kind):
    """Join raw predictions to corrected references by ID, checking complete paired coverage."""
    if kind == "rectom":
        d = root / "inputs/rectom/final_test"
        refs = index(
            read_rows(root / "inputs/preparation/scoring_only/test_references.jsonl")
        )
        raw = [index(read_rows(d / (s + "_predictions.jsonl"))) for s in SYSTEMS[:3]]
        structured = index(read_rows(d / "structured_test_predictions.jsonl"))
    else:
        d = root / "inputs/opentom/opentom_transfer"
        refs = index(read_rows(d / "final_references_corrected.jsonl"))
        raw = [index(read_rows(d / (s + ".jsonl"))) for s in SYSTEMS[:3]]
        structured = index(read_rows(d / "subjectesis_structured.jsonl"))
    assert len(refs) == 621
    for r in raw + [structured]:
        assert set(r) == set(refs), "Prediction/reference coverage differs"
    rows = []
    for rid, ref in refs.items():
        f = ref["task"] if kind == "rectom" else ref["family_detail"]
        rec = {
            "record_id": rid,
            "cluster": str(ref["dialogue_id"] if kind == "rectom" else ref["story_id"]),
            "family": f,
        }
        predictions = [r[rid]["prediction"] for r in raw] + [
            structured[rid]["no_review_prediction"],
            structured[rid][
                "reviewed_prediction" if kind == "rectom" else "review_prediction"
            ],
        ]
        for s, p in zip(SYSTEMS, predictions):
            gt, pr = (ref["answer"], p) if kind == "rectom" else parse_open(p, ref)
            assert gt is not None and gt in LABELS[f], f"Unscorable {rid}"
            rec["reference"] = gt
            rec[s] = pr
            rec[s + "_raw"] = p
        rows.append(rec)
    df = pd.DataFrame(rows)
    assert df.cluster.nunique() == (67 if kind == "rectom" else 27)
    return (df, structured)


def analyse_dataset(root, kind):
    """Recompute benchmark tables, cluster uncertainty and paired comparisons."""
    df, structured = load_dataset(root, kind)
    out = root / "results"
    df.to_csv(out / (kind + "_prediction_matrix.csv"), index=False)
    groups, families, n, correct, classes = cluster_statistics(df)
    point = metrics_for_weights(
        np.ones((1, len(groups))), n, correct, classes, families
    )
    counts = bootstrap_counts(len(groups))
    boot = metrics_for_weights(counts, n, correct, classes, families)
    loo = metrics_for_weights(
        np.ones((len(groups), len(groups))) - np.eye(len(groups)),
        n,
        correct,
        classes,
        families,
    )
    primary = "mean_family_accuracy" if kind == "rectom" else "accuracy"
    system_rows = []
    family_rows = []
    class_rows = []
    pairs = []
    influence = []
    sens = []
    for s, sys in enumerate(SYSTEMS):
        row = {
            "dataset": kind,
            "system": sys,
            "questions": len(df),
            "clusters": len(groups),
        }
        for metric in ["accuracy", "mean_family_accuracy", "mean_family_f1"]:
            row[metric] = float(point[metric][s, 0])
            row[metric + "_low"], row[metric + "_high"] = interval(boot[metric][s])
        row["correct"] = int(correct[s].sum())
        row["out_of_target"] = sum(
            (pr not in LABELS[f] for pr, f in zip(df[sys], df.family))
        )
        system_rows.append(row)
        for j, f in enumerate(families):
            frow = {
                "dataset": kind,
                "system": sys,
                "family": f,
                "questions": int(n[:, j].sum()),
                "clusters": int((n[:, j] > 0).sum()),
            }
            for metric in ["accuracy", "f1"]:
                frow[metric] = float(point["family_" + metric][s, 0, j])
                frow[metric + "_low"], frow[metric + "_high"] = interval(
                    boot["family_" + metric][s, :, j]
                )
            family_rows.append(frow)
            sup, pre, tp = classes[f]
            for k, lab in enumerate(LABELS[f]):
                support = sup[:, k].sum()
                predicted = pre[s, :, k].sum()
                hit = tp[s, :, k].sum()
                class_rows.append(
                    {
                        "dataset": kind,
                        "system": sys,
                        "family": f,
                        "label": lab,
                        "support": int(support),
                        "supporting_clusters": int((sup[:, k] > 0).sum()),
                        "predicted": int(predicted),
                        "tp": int(hit),
                        "recall": hit / support if support else 0.0,
                        "precision": hit / predicted if predicted else 0.0,
                        "f1": (
                            2 * hit / (support + predicted)
                            if support + predicted
                            else 0.0
                        ),
                    }
                )
        cluster_acc = correct[s].sum(1) / n.sum(1)
        sens.append(
            {
                "dataset": kind,
                "system": sys,
                "equal_cluster_accuracy": float(cluster_acc.mean()),
                "question_accuracy": float(point["accuracy"][s, 0]),
                "mean_family_accuracy": float(point["mean_family_accuracy"][s, 0]),
                "target_macro_f1": float(point["mean_family_f1"][s, 0]),
            }
        )
    for a, b in COMPARISONS:
        d = correct[a] - correct[b]
        contrib = (d / n.sum(0)).mean(1) if kind == "rectom" else d.sum(1) / n.sum()
        p, reps, method, nonzero = signflip_p(contrib)
        row = {
            "dataset": kind,
            "a": SYSTEMS[a],
            "b": SYSTEMS[b],
            "primary_metric": primary,
            "p_cluster_swap": p,
            "permutation_method": method,
            "permutations": reps,
            "nonzero_clusters": nonzero,
            "a_only_correct": int(
                (
                    (df[SYSTEMS[a]] == df.reference) & (df[SYSTEMS[b]] != df.reference)
                ).sum()
            ),
            "b_only_correct": int(
                (
                    (df[SYSTEMS[b]] == df.reference) & (df[SYSTEMS[a]] != df.reference)
                ).sum()
            ),
        }
        for metric in ["accuracy", "mean_family_accuracy", "mean_family_f1"]:
            row[metric + "_difference"] = float(
                point[metric][a, 0] - point[metric][b, 0]
            )
            row[metric + "_low"], row[metric + "_high"] = interval(
                boot[metric][a] - boot[metric][b]
            )
        for j, g in enumerate(groups):
            influence.append(
                {
                    "dataset": kind,
                    "a": SYSTEMS[a],
                    "b": SYSTEMS[b],
                    "omitted_cluster": g,
                    "metric": primary,
                    "difference": float(loo[primary][a, j] - loo[primary][b, j]),
                }
            )
        pairs.append(row)
    pd.DataFrame(system_rows).to_csv(out / (kind + "_systems.csv"), index=False)
    pd.DataFrame(family_rows).to_csv(out / (kind + "_families.csv"), index=False)
    pd.DataFrame(class_rows).to_csv(out / (kind + "_classes.csv"), index=False)
    pd.DataFrame(influence).to_csv(
        out / (kind + "_leave_one_cluster_out.csv"), index=False
    )
    pd.DataFrame(sens).to_csv(out / (kind + "_weighting_sensitivity.csv"), index=False)
    audit_process(root, kind, df, structured)
    verify_saved(root, kind, point, boot, families, df)
    return pairs


def verify_saved(root, kind, point, boot, families, df):
    """Compare recalculated metrics and intervals with the frozen saved outputs."""
    comparisons = []
    if kind == "rectom":
        saved = read_json(root / "inputs/rectom/final_test/final_metrics.json")
        for s, entry in enumerate(saved["systems"]):
            assert entry["system"] == SYSTEMS[s]
            for j, f in enumerate(families):
                for metric, key in [("accuracy", "accuracy"), ("f1", "macro_f1")]:
                    actual = point["family_" + metric][s, 0, j]
                    expected = entry[f + "_" + key]
                    assert abs(actual - expected) < 1e-12
                    comparisons.append(
                        {
                            "system": SYSTEMS[s],
                            "metric": f + "_" + key,
                            "absolute_difference": abs(actual - expected),
                        }
                    )
            vals = np.sort(boot["mean_family_accuracy"][s])
            ci = [vals[125], vals[4875]]
            assert np.allclose(
                ci,
                saved["system_bootstrap_ci95"][SYSTEMS[s]]["selection_score"],
                atol=1e-12,
            )
    else:
        saved = pd.read_csv(
            root
            / "inputs/opentom/opentom_transfer/corrected_final_metrics/system_summary.csv"
        )
        for s, sys in enumerate(SYSTEMS):
            entry = saved[saved.system == sys].iloc[0]
            for metric, key, lo, hi in [
                (
                    "accuracy",
                    "strict_overall_accuracy",
                    "accuracy_ci_low",
                    "accuracy_ci_high",
                ),
                (
                    "mean_family_f1",
                    "mean_family_strict_macro_f1",
                    "macro_f1_ci_low",
                    "macro_f1_ci_high",
                ),
            ]:
                assert abs(point[metric][s, 0] - entry[key]) < 1e-12
                assert np.allclose(
                    interval(boot[metric][s]), [entry[lo], entry[hi]], atol=1e-12
                )
                comparisons.append(
                    {
                        "system": sys,
                        "metric": metric,
                        "absolute_difference": abs(point[metric][s, 0] - entry[key]),
                    }
                )
        original = pd.read_csv(
            root
            / "inputs/opentom/opentom_transfer/corrected_final_metrics/prediction_matrix.csv"
        )
        assert set(original.record_id) == set(df.record_id)
    write_json(
        root / "results" / f"{kind}_reconciliation.json",
        {
            "saved_point_estimates_reproduced": True,
            "saved_cluster_intervals_reproduced": True,
            "checks": comparisons,
            "new_intervals_use_numpy_linear_quantiles": True,
        },
    )


def json_object(raw):
    decoder = json.JSONDecoder()
    for pos, c in enumerate(raw or ""):
        if c == "{":
            try:
                obj, _ = decoder.raw_decode(raw[pos:])
                return obj
            except json.JSONDecodeError:
                pass
    return None


def audit_process(root, kind, df, structured):
    """Count recorded state revisions and answer transitions separately."""
    records = []
    events = []
    tot = collections.Counter()
    lookup = df.set_index("record_id")
    for rid, r in structured.items():
        row = lookup.loc[rid]
        before = row.subjectesis_no_review == row.reference
        after = row.subjectesis_review == row.reference
        transition = (
            ("right" if before else "wrong") + "_to_" + ("right" if after else "wrong")
        )
        tot[transition] += 1
        rec = {
            "record_id": rid,
            "cluster": row.cluster,
            "family": row.family,
            "transition": transition,
            "raw_answer_changed": row.subjectesis_no_review_raw
            != row.subjectesis_review_raw,
            "parsed_answer_changed": row.subjectesis_no_review
            != row.subjectesis_review,
        }
        if kind == "rectom":
            a, b = (r["state_before"], r["state_after"])
            initial = json_object(r["raw_initial"]) or {}
            rec.update(
                value_changes=sum((a[k].get("value") != b[k].get("value") for k in a)),
                basis_changes=sum((a[k].get("basis") != b[k].get("basis") for k in a)),
                citation_changes=sum(
                    (
                        a[k].get("evidence_turns") != b[k].get("evidence_turns")
                        for k in a
                    )
                ),
                any_state_change=a != b,
                review_calls=len(r["reviews"]),
                initial_finalizer_disagreement=r.get("initial_json_answer")
                != r["no_review_prediction"],
                unknown_fields=sum((v.get("value") == "unknown" for v in a.values())),
                unsupported_answer_claim_count=len(
                    initial.get("answer_claims_not_established", [])
                ),
            )
            current = {k: dict(v) for k, v in a.items()}
            for rev in r["reviews"]:
                parsed = rev.get("parsed") or {}
                field = rev["field"]
                old = current.get(field, {})
                new = parsed.get("updated_claim", old)
                events.append(
                    {
                        "record_id": rid,
                        "field": field,
                        "decision": parsed.get("decision"),
                        "valid": rev["valid"],
                        "old_value": old.get("value"),
                        "new_value": new.get("value"),
                        "value_changed": old.get("value") != new.get("value"),
                        "claim_changed": old != new,
                        "citation_changed": old.get("evidence_turns")
                        != new.get("evidence_turns"),
                        "withdrawal": old.get("value") != "unknown"
                        and new.get("value") == "unknown",
                    }
                )
                if rev["valid"]:
                    current[field] = new
                tot["review_calls"] += 1
                tot["invalid_review"] += int(not rev["valid"])
            tot["initial_evidence_items"] += r["initial_evidence_binding"]["items"]
            tot["initial_evidence_bound"] += r["initial_evidence_binding"][
                "valid_items"
            ]
            for rev in r["reviews"]:
                ev = rev.get("evidence_binding", {})
                tot["review_evidence_items"] += ev.get("items", 0)
                tot["review_evidence_bound"] += ev.get("valid_items", 0)
        else:
            a = json_object(r.get("raw_initial"))
            b = json_object(r.get("raw_review"))
            rec.update(
                initial_valid=r.get("initial_valid"),
                review_valid=r.get("review_valid"),
                decision=b.get("decision") if b else None,
                claim_changed=(
                    a.get("mental_state_claim") != b.get("updated_mental_state_claim")
                    if a and b
                    else None
                ),
                citation_changed=(
                    a.get("evidence_sentences") != b.get("evidence_sentences")
                    if a and b
                    else None
                ),
                initial_answer_finalizer_disagreement=(
                    a.get("answer") != r["no_review_prediction"] if a else None
                ),
            )
            tot["review_calls"] += 1
            tot["invalid_review"] += int(not r.get("review_valid", False))
            tot["invalid_initial"] += int(not r.get("initial_valid", False))
        records.append(rec)
    process = pd.DataFrame(records)
    process.to_csv(root / "results" / f"{kind}_process_records.csv", index=False)
    if events:
        pd.DataFrame(events).to_csv(
            root / "results" / f"{kind}_review_events.csv", index=False
        )
    for c in process.columns:
        if c not in ["record_id", "cluster", "family", "transition", "decision"]:
            tot[c] = int(process[c].fillna(False).sum())
    write_json(root / "results" / f"{kind}_process_summary.json", dict(tot))


def training_audit(root):
    """Recover selected-checkpoint exposure, token counts and validation history from run logs."""
    records = []
    valid = []
    for arm, condition in [
        ("answer_training", "ordinary"),
        ("subjectesis_training", "subjectesis"),
    ]:
        p = root / "inputs" / arm
        run = p / "runs" / condition
        c = read_json(run / "completed.json")
        log = read_rows(run / "training_log.jsonl")
        unique = {r["step"]: r for r in log}
        assert set(unique) == set(range(1, c["step"] + 1))
        best = c["best_step"]
        b = unique[best]
        token = read_json(p / "reports/tokenization.json")["conditions"][condition]
        for key in ["loss", "grad_norm"]:
            assert all((math.isfinite(r[key]) for r in log))
        records.append(
            {
                "condition": condition,
                "selected_step": best,
                "selected_examples_seen": b["examples_seen"],
                "selected_target_tokens_seen": sum(
                    (unique[i]["target_tokens"] for i in range(1, best + 1))
                ),
                "completed_steps": c["step"],
                "completed_examples": c["examples_seen"],
                "completed_target_tokens": c["target_tokens_seen"],
                "full_total_tokens": token["total_tokens"],
                "log_rows": len(log),
                "repeated_step_rows": len(log) - len(unique),
                "recorded_step_seconds_last_per_step": sum(
                    (r["step_seconds"] for r in unique.values())
                ),
                "median_step_seconds_last_per_step": float(
                    np.median([r["step_seconds"] for r in unique.values()])
                ),
                "reported_peak_allocated_gib": max(
                    (r.get("a100_peak_vram_gib", 0) for r in log)
                ),
                "trainable_parameters": read_json(p / "reports/model_rank0.json")[
                    "trainable_parameters"
                ],
            }
        )
        assert (
            sum((unique[i]["target_tokens"] for i in unique)) == c["target_tokens_seen"]
        )
        for f in sorted(run.glob("validation_*.json")):
            v = read_json(f)
            valid.append(
                {
                    "condition": condition,
                    "step": int(f.stem.split("_")[1]),
                    **{k: x for k, x in v.items() if isinstance(x, (float, int))},
                }
            )
    pd.DataFrame(records).to_csv(root / "results/training_audit.csv", index=False)
    pd.DataFrame(valid).to_csv(root / "results/validation_history.csv", index=False)
    write_json(
        root / "results/training_interpretation.json",
        {
            "full_schedule_exposure_matched": True,
            "evaluated_checkpoints_exposure_matched": False,
            "target_token_or_compute_matched": False,
            "time_note": "Summed step_seconds includes validation/checkpoint overhead at some boundaries and cannot establish pure training time or billed wall time. Resumed duplicate step entries are not independent completed steps.",
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    root = args.root
    (root / "results").mkdir(exist_ok=True)
    m = read_json(root / "inputs/preparation/manifest.json")
    groups = m["dialogue_ids"]
    assert not (
        set(groups["train"]) & set(groups["validation"])
        or set(groups["train"]) & set(groups["test"])
        or set(groups["validation"]) & set(groups["test"])
    )
    pairs = []
    for kind in ["rectom", "opentom"]:
        pairs.extend(analyse_dataset(root, kind))
    p = pd.DataFrame(pairs)
    p["p_holm_all_ten_accuracy_contrasts"] = holm(p.p_cluster_swap.values)
    p.to_csv(root / "results/paired_comparisons.csv", index=False)
    training_audit(root)
    provenance = {
        str(p.relative_to(root)): sha(p)
        for p in sorted((root / "inputs").rglob("*"))
        if p.is_file()
    }
    write_json(root / "results/input_hashes.json", provenance)
    write_json(
        root / "results/analysis_run.json",
        {
            "bootstrap_repetitions": 5000,
            "bootstrap_seed": 42,
            "cluster_swap_seed": 20260922,
            "monte_carlo_swaps": 50000,
            "holm_family": "ten post-hoc accuracy contrasts across two benchmarks",
            "new_inference": False,
            "new_training": False,
            "completed": True,
        },
    )
    print(
        p[
            [
                "dataset",
                "a",
                "b",
                "accuracy_difference",
                "mean_family_accuracy_difference",
                "p_cluster_swap",
                "p_holm_all_ten_accuracy_contrasts",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
