"""Reproduce corrected OpenToM scores from already saved predictions.

Rebuild location metadata from explicit keys in the pinned dataset, then parse
answers, calculate strict metrics and estimate story-level bootstrap intervals.
The correction changes scoring metadata rather than model answers, and uses CPU
calculations with the original Drive file layout.
"""

from project_setup import mount_drive, prepare_project

mount_drive()
from pathlib import Path
import json, hashlib, os, shutil, subprocess, sys

ROOT = Path("/content/subjectesis_opentom_rescore")
DRIVE_ROOT = Path("/content/drive/MyDrive/Subjectesis")
TRANSFER_DIR = DRIVE_ROOT / "opentom_transfer"
CORRECTED_DIR = TRANSFER_DIR / "corrected_final_metrics"
ROOT.mkdir(parents=True, exist_ok=True)
CORRECTED_DIR.mkdir(parents=True, exist_ok=True)
REQUIRED = [
    "base_direct.jsonl",
    "answer_only_direct.jsonl",
    "subjectesis_direct.jsonl",
    "subjectesis_structured.jsonl",
    "PREDICTIONS_FROZEN.json",
    "opentom_protocol_frozen.json",
    "selection_manifest.json",
]
for name in REQUIRED:
    p = TRANSFER_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Missing required frozen artifact: {p}")
selection = json.loads((TRANSFER_DIR / "selection_manifest.json").read_text())
freeze = json.loads((TRANSFER_DIR / "PREDICTIONS_FROZEN.json").read_text())
protocol = json.loads((TRANSFER_DIR / "opentom_protocol_frozen.json").read_text())
if freeze.get("labels_read") is not False:
    raise RuntimeError("Frozen-prediction metadata no longer states labels_read=false.")
if freeze["protocol_hash"] != protocol["protocol_hash"]:
    raise RuntimeError("Prediction/protocol hash mismatch.")
if selection["opentom_commit"] != protocol["config"]["opentom_commit"]:
    raise RuntimeError("Pinned OpenToM commit mismatch between saved manifests.")
if len(selection["final_story_ids"]) != 27 or selection["final_questions"] != 621:
    raise RuntimeError(
        "Frozen OpenToM subset no longer matches 27 stories / 621 questions."
    )
OPENTOM_COMMIT = selection["opentom_commit"]
PROTOCOL_HASH = protocol["protocol_hash"]
FINAL_IDS = [str(x) for x in selection["final_story_ids"]]
OLD_INPUT_SHA = selection["input_sha256"]
OLD_REFERENCE_SHA = selection["reference_sha256"]
print(" Frozen prediction record verified.")
print("Protocol hash:", PROTOCOL_HASH)
print("Pinned OpenToM commit:", OPENTOM_COMMIT)
print("Final stories:", len(FINAL_IDS))
print("No model inference will be performed.")
OTOM = ROOT / "OpenToM"
if OTOM.exists():
    shutil.rmtree(OTOM)
subprocess.run(
    ["git", "clone", "--quiet", "https://github.com/seacowx/OpenToM.git", str(OTOM)],
    check=True,
)
subprocess.run(
    ["git", "-C", str(OTOM), "checkout", "--quiet", OPENTOM_COMMIT], check=True
)
actual_commit = subprocess.check_output(
    ["git", "-C", str(OTOM), "rev-parse", "HEAD"], text=True
).strip()
if actual_commit != OPENTOM_COMMIT:
    raise RuntimeError("OpenToM checkout did not reproduce the pinned commit.")
DATA = OTOM / "data/opentom_data"
QUESTION_FILES = {
    "location_cg_fo": DATA / "location_cg_fo.json",
    "location_cg_so": DATA / "location_cg_so.json",
    "location_fg_fo": DATA / "location_fg_fo_new.json",
    "location_fg_so": DATA / "location_fg_so_new.json",
    "multihop_fo": DATA / "multihop_fo.json",
    "multihop_so": DATA / "multihop_so.json",
    "attitude": DATA / "attitude.json",
}
META_FILE = DATA / "meta_data.json"
meta = json.loads(META_FILE.read_text())
questions = {k: json.loads(v.read_text()) for k, v in QUESTION_FILES.items()}
EXPECTED_COUNTS = {
    "location_cg_fo": 2,
    "location_cg_so": 2,
    "location_fg_fo": 4,
    "location_fg_so": 2,
    "multihop_fo": 6,
    "multihop_so": 6,
    "attitude": 1,
}


def family_detail(family, q):
    """Split multihop questions into fullness and accessibility subfamilies using
    their question wording. Other family names pass through unchanged, giving
    the scorer the appropriate label set for each question.
    """
    if family.startswith("multihop"):
        low = q["question"].lower()
        if "fullness" in low:
            return family + "_fullness"
        if "accessibility" in low:
            return family + "_accessibility"
        raise RuntimeError(f"Unrecognized multihop question: {q['question']}")
    return family


def corrected_plot_info(story_id):
    """Require and read the five explicit plot metadata keys for one story.
    Returning named fields avoids the dictionary-order mistake that previously
    made some location references unscorable.
    """
    p = meta[str(story_id)]["plot_info"]
    required = {"mover", "observer", "eoi", "original_place", "move_to_place"}
    if not required.issubset(p):
        raise RuntimeError(f"Missing plot_info keys for story {story_id}: {p.keys()}")
    return {
        "mover": p["mover"],
        "observer": p["observer"],
        "eoi": p["eoi"],
        "original_place": p["original_place"],
        "move_to_place": p["move_to_place"],
    }


def build_records(story_ids, include_answers):
    """Reconstruct the originally selected stories and questions using corrected
    named location metadata. Verify family counts and optionally add reference
    answers, without regenerating or modifying any model predictions.
    """
    rows = []
    for sid in map(str, story_ids):
        if sid not in meta:
            raise RuntimeError(f"Story {sid} missing from pinned metadata.")
        narrative = meta[sid]["narrative"]
        info = corrected_plot_info(sid)
        for family, source in questions.items():
            qs = source[sid]
            if len(qs) != EXPECTED_COUNTS[family]:
                raise RuntimeError(
                    f"{sid}/{family}: expected {EXPECTED_COUNTS[family]}, got {len(qs)}"
                )
            for idx, q in enumerate(qs):
                r = {
                    "record_id": f"{sid}::{family}::{idx}",
                    "story_id": sid,
                    "family": family,
                    "family_detail": family_detail(family, q),
                    "question": q["question"],
                    "narrative": narrative,
                    "original_place": info["original_place"],
                    "move_to_place": info["move_to_place"],
                }
                if include_answers:
                    r["answer"] = q["answer"]
                rows.append(r)
    return rows


# Rebuild metadata for the same selected questions. The frozen prediction
# files remain the evidence being scored, so no new GPU inference is needed.
corrected_inputs = build_records(FINAL_IDS, include_answers=False)
corrected_refs_full = build_records(FINAL_IDS, include_answers=True)
if len(corrected_inputs) != 621 or len(corrected_refs_full) != 621:
    raise RuntimeError("Corrected reconstruction is not 621 questions.")
CORRECTED_INPUT_FILE = TRANSFER_DIR / "final_inputs_corrected.jsonl"
CORRECTED_REF_FILE = TRANSFER_DIR / "final_references_corrected.jsonl"


def write_jsonl(path, rows):
    """Write corrected records in order as one UTF-8 JSON object per line. These
    files preserve a readable record of the metadata and references used for
    rescoring.
    """
    with Path(path).open("w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


write_jsonl(CORRECTED_INPUT_FILE, corrected_inputs)
write_jsonl(
    CORRECTED_REF_FILE,
    [
        {
            "record_id": r["record_id"],
            "story_id": r["story_id"],
            "family": r["family"],
            "family_detail": r["family_detail"],
            "answer": r["answer"],
            "original_place": r["original_place"],
            "move_to_place": r["move_to_place"],
        }
        for r in corrected_refs_full
    ],
)
CORRECTED_INPUT_SHA = hashlib.sha256(CORRECTED_INPUT_FILE.read_bytes()).hexdigest()
CORRECTED_REF_SHA = hashlib.sha256(CORRECTED_REF_FILE.read_bytes()).hexdigest()
correction_manifest = {
    "correction_id": "opentom_plot_info_explicit_keys_v1",
    "reason": "Original transfer notebook unpacked plot_info through dict value order. This rescoring reconstructs evaluator metadata by explicit keys: mover, observer, eoi, original_place, move_to_place.",
    "opentom_commit": OPENTOM_COMMIT,
    "protocol_hash": PROTOCOL_HASH,
    "story_ids": FINAL_IDS,
    "questions": 621,
    "old_input_sha256": OLD_INPUT_SHA,
    "old_reference_sha256": OLD_REFERENCE_SHA,
    "corrected_input_sha256": CORRECTED_INPUT_SHA,
    "corrected_reference_sha256": CORRECTED_REF_SHA,
    "predictions_regenerated": False,
    "model_inference_rerun": False,
    "retraining_rerun": False,
    "model_prompt_semantics_changed": False,
    "correction_scope": "evaluation-only original/moved location metadata and downstream scoring",
}
(TRANSFER_DIR / "scoring_correction_manifest.json").write_text(
    json.dumps(correction_manifest, indent=2) + "\n"
)
print(" Corrected input/reference files reconstructed.")
print("Corrected input SHA:", CORRECTED_INPUT_SHA)
print("Corrected reference SHA:", CORRECTED_REF_SHA)
print("Predictions regenerated: NO")
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.metrics import f1_score


def read_jsonl(path):
    """Load each non-empty JSONL line as a record. The rescoring stage uses these
    saved inputs and predictions rather than calling the model again.
    """
    return [
        json.loads(x)
        for x in Path(path).read_text(encoding="utf8").splitlines()
        if x.strip()
    ]


BASE_RAW = read_jsonl(TRANSFER_DIR / "base_direct.jsonl")
ANSWER_RAW = read_jsonl(TRANSFER_DIR / "answer_only_direct.jsonl")
SUBJECTESIS_RAW = read_jsonl(TRANSFER_DIR / "subjectesis_direct.jsonl")
STRUCTURED_RAW = read_jsonl(TRANSFER_DIR / "subjectesis_structured.jsonl")
REF_ROWS = read_jsonl(CORRECTED_REF_FILE)
REFS = {r["record_id"]: r for r in REF_ROWS}
for name, rows in {
    "base_direct": BASE_RAW,
    "answer_only_direct": ANSWER_RAW,
    "subjectesis_direct": SUBJECTESIS_RAW,
    "subjectesis_structured": STRUCTURED_RAW,
}.items():
    if len(rows) != 621:
        raise RuntimeError(f"{name}: expected 621 saved predictions, found {len(rows)}")
expected_ids = set(REFS)
for name, rows in {
    "base_direct": BASE_RAW,
    "answer_only_direct": ANSWER_RAW,
    "subjectesis_direct": SUBJECTESIS_RAW,
    "subjectesis_structured": STRUCTURED_RAW,
}.items():
    if {r["record_id"] for r in rows} != expected_ids:
        raise RuntimeError(f"{name}: record IDs do not match corrected references.")


def remove_determiner(text):
    """Lowercase location text and remove a leading article. This normalizes
    reference place names before comparing them with the original and moved
    locations.
    """
    text = str(text).strip().lower()
    for det in ["a ", "an ", "the "]:
        if text.startswith(det):
            return text[len(det) :].strip()
    return text


def lexical_overlap(pred, location):
    """Return the fraction of location words matched by distinct prediction words
    after basic text normalization. The fine-location parser compares this
    score for the original and moved places and retains a tie as an ambiguous
    answer.
    """
    pred = (
        str(pred).lower().replace("_", " ").replace("'s", "").replace(".", "").split()
    )
    loc = str(location).lower().replace("_", " ").replace("'s", "").split()
    if not loc:
        return 0.0
    seen = set()
    score = 0
    for w in pred:
        if w in loc and w not in seen:
            score += 1
            seen.add(w)
    return score / len(loc)


def normalize_accessibility_reference(answer):
    """Normalize the recorded accessibility reference format, including singleton
    lists and the pipe-separated equality convention. Return None when a list
    cannot be resolved to one usable reference so the scorer can flag it as
    unscorable.
    """
    if isinstance(answer, list):
        answer = [x for x in answer if x != "corrupted"]
        if len(answer) != 1:
            return None
        answer = answer[0]
    answer = str(answer).strip()
    if "|" in answer:
        return "equally accessible"
    return answer


def parse_prediction(prediction, ref):
    """Map raw answer text and its reference to numeric labels using the fixed
    family-specific parsing rules. Return the reference label, prediction label
    and allowed labels, keeping malformed or ambiguous predictions distinct
    from valid answers.
    """
    family = ref["family"]
    detail = ref["family_detail"]
    pred = "" if prediction is None else str(prediction).strip()
    answer = ref["answer"]
    if family.startswith("location_fg"):
        original = remove_determiner(ref["original_place"])
        moved = remove_determiner(ref["move_to_place"])
        gt_answer = remove_determiner(answer)
        oscore = lexical_overlap(pred, original)
        mscore = lexical_overlap(pred, moved)
        if oscore == mscore:
            pl = 3
        elif oscore > mscore:
            pl = 1
        else:
            pl = 2
        if gt_answer == original:
            gt = 1
        elif gt_answer == moved:
            gt = 2
        else:
            gt = None
        return (gt, pl, [1, 2])
    if family.startswith("location_cg"):
        p = pred.lower()
        a = str(answer).lower()
        if "no" in p and "yes" not in p:
            pl = 0
        elif "yes" in p and "no" not in p:
            pl = 1
        else:
            pl = -1
        if "no" in a:
            gt = 0
        elif "yes" in a:
            gt = 1
        else:
            gt = None
        return (gt, pl, [0, 1])
    if "fullness" in detail:
        p = pred.replace(".", "").lower()
        if any((x in p for x in ["less full", "emptier", "more empty"])):
            pl = 1
        elif any((x in p for x in ["more full", "fuller"])):
            pl = 2
        elif "equally full" in p:
            pl = 3
        else:
            pl = -1
        gt = {"less full": 1, "more full": 2, "equally full": 3}.get(
            str(answer).strip().lower()
        )
        return (gt, pl, [1, 2, 3])
    if "accessibility" in detail:
        p = pred.replace(".", "").lower()
        if "more accessible" in p:
            pl = 1
        elif "less accessible" in p:
            pl = 2
        elif "equally accessible" in p:
            pl = 3
        else:
            pl = -1
        a = normalize_accessibility_reference(answer)
        if a is None:
            gt = None
        else:
            gt = {
                "more accessible": 1,
                "less accessible": 2,
                "equally accessible": 3,
            }.get(a.lower())
        return (gt, pl, [1, 2, 3])
    if family == "attitude":
        token = pred.lower().split("\n\n")[-1].split(":")[-1].split(".")[0].strip()
        letter_map = {"a": "positive", "b": "neutral", "c": "negative"}
        token = letter_map.get(token, token)
        flags = ["positive" in token, "neutral" in token, "negative" in token]
        if sum(flags) != 1:
            pl = -1
        elif flags[0]:
            pl = 1
        elif flags[2]:
            pl = 2
        else:
            pl = 3
        gt = {"positive": 1, "negative": 2, "neutral": 3}.get(
            str(answer).strip().lower()
        )
        return (gt, pl, [1, 2, 3])
    raise RuntimeError(f"Unknown family/detail: {family}/{detail}")


def score_rows(rows, system, prediction_key="prediction"):
    """Join each saved prediction to the corrected reference by ID and record
    parsed labels, correctness and parser status. Retain raw answers alongside
    these fields so the corrected scores can be traced to the original
    generation.
    """
    out = []
    for r in rows:
        ref = REFS[r["record_id"]]
        gt, pl, labels = parse_prediction(r.get(prediction_key), ref)
        out.append(
            {
                "system": system,
                "record_id": r["record_id"],
                "story_id": str(r["story_id"]),
                "family": r["family"],
                "family_detail": r["family_detail"],
                "raw_prediction": r.get(prediction_key),
                "reference": ref["answer"],
                "gt": gt,
                "pred": pl,
                "target_labels": labels,
                "scorable": gt is not None,
                "corrupt": pl == -1,
                "correct": gt is not None and pl == gt,
            }
        )
    return out


BASE_S = score_rows(BASE_RAW, "base_direct")
ANSWER_S = score_rows(ANSWER_RAW, "answer_only_direct")
SUBJECTESIS_S = score_rows(SUBJECTESIS_RAW, "subjectesis_direct")
NO_REVIEW_S = score_rows(
    STRUCTURED_RAW, "subjectesis_no_review", "no_review_prediction"
)
REVIEW_S = score_rows(STRUCTURED_RAW, "subjectesis_review", "review_prediction")
SYSTEMS = {
    "base_direct": BASE_S,
    "answer_only_direct": ANSWER_S,
    "subjectesis_direct": SUBJECTESIS_S,
    "subjectesis_no_review": NO_REVIEW_S,
    "subjectesis_review": REVIEW_S,
}
FAMILY_ORDER = [
    "location_cg_fo",
    "location_cg_so",
    "location_fg_fo",
    "location_fg_so",
    "multihop_fo_fullness",
    "multihop_fo_accessibility",
    "multihop_so_fullness",
    "multihop_so_accessibility",
    "attitude",
]
FAMILY_LABELS = {
    "location_cg_fo": [0, 1],
    "location_cg_so": [0, 1],
    "location_fg_fo": [1, 2],
    "location_fg_so": [1, 2],
    "multihop_fo_fullness": [1, 2, 3],
    "multihop_fo_accessibility": [1, 2, 3],
    "multihop_so_fullness": [1, 2, 3],
    "multihop_so_accessibility": [1, 2, 3],
    "attitude": [1, 2, 3],
}


def strict_target_macro_f1(rows, labels):
    """Compute per-class F1 over the specified target labels and average them,
    assigning zero to an undefined class score. Only unscorable references are
    excluded; malformed predictions still produce false negatives for their
    true class.
    """
    rows = [r for r in rows if r["scorable"]]
    vals = []
    for lab in labels:
        tp = sum((r["gt"] == lab and r["pred"] == lab for r in rows))
        fp = sum((r["gt"] != lab and r["pred"] == lab for r in rows))
        fn = sum((r["gt"] == lab and r["pred"] != lab for r in rows))
        den = 2 * tp + fp + fn
        vals.append(2 * tp / den if den else 0.0)
    return float(np.mean(vals))


def metric_block(rows, family):
    """Summarize one question family with strict accuracy, fixed-target macro F1
    and parser counts. Also retain historical valid-only comparison scores,
    while strict metrics continue to count malformed predictions as errors.
    """
    sc = [r for r in rows if r["scorable"]]
    # The valid-only figures reproduce a historical comparison convention.
    # The strict metrics below retain malformed predictions as errors.
    valid = [r for r in sc if not r["corrupt"]]
    labels = FAMILY_LABELS[family]
    strict_acc = sum((r["correct"] for r in sc)) / len(sc) if sc else None
    strict_f1 = strict_target_macro_f1(sc, labels) if sc else None
    if valid:
        off_acc = sum((r["correct"] for r in valid)) / len(valid)
        off_f1 = float(
            f1_score(
                [int(r["gt"]) for r in valid],
                [int(r["pred"]) for r in valid],
                average="macro",
                zero_division=0,
            )
        )
    else:
        off_acc = off_f1 = None
    return {
        "total_rows": len(rows),
        "scorable_reference_rows": len(sc),
        "unscorable_reference_rows": len(rows) - len(sc),
        "corrupt_predictions": sum((r["corrupt"] for r in sc)),
        "corrupt_rate": sum((r["corrupt"] for r in sc)) / len(sc) if sc else None,
        "strict_accuracy": strict_acc,
        "strict_target_macro_f1": strict_f1,
        "official_valid_accuracy": off_acc,
        "official_valid_macro_f1": off_f1,
    }


def summarize(name, rows):
    """Combine family summaries into one system result with pooled strict accuracy
    and mean family macro F1. Include scorable-reference and corrupt-prediction
    counts so the denominator and parsing effects remain visible.
    """
    fam = {
        f: metric_block([r for r in rows if r["family_detail"] == f], f)
        for f in FAMILY_ORDER
    }
    sc = [r for r in rows if r["scorable"]]
    return {
        "system": name,
        "questions": len(rows),
        "scorable_questions": len(sc),
        "unscorable_reference_questions": len(rows) - len(sc),
        "strict_overall_accuracy": float(sum((r["correct"] for r in sc)) / len(sc)),
        "mean_family_strict_target_macro_f1": float(
            np.mean([fam[f]["strict_target_macro_f1"] for f in FAMILY_ORDER])
        ),
        "mean_family_official_valid_macro_f1": float(
            np.mean(
                [
                    fam[f]["official_valid_macro_f1"]
                    for f in FAMILY_ORDER
                    if fam[f]["official_valid_macro_f1"] is not None
                ]
            )
        ),
        "overall_corrupt_rate": float(sum((r["corrupt"] for r in sc)) / len(sc)),
        "families": fam,
    }


SYSTEM_METRICS = {n: summarize(n, r) for n, r in SYSTEMS.items()}
unscorable_by_family = {
    f: SYSTEM_METRICS["base_direct"]["families"][f]["unscorable_reference_rows"]
    for f in FAMILY_ORDER
}
print("Unscorable corrected references by family:", unscorable_by_family)
print(
    "Total unscorable corrected references:",
    SYSTEM_METRICS["base_direct"]["unscorable_reference_questions"],
)
print(" Point estimates recomputed from frozen predictions.")
import random

BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_SEED = 42
STORY_IDS = sorted({r["story_id"] for r in BASE_S})
if len(STORY_IDS) != 27:
    raise RuntimeError(f"Expected 27 stories, found {len(STORY_IDS)}")
story_index = {sid: i for i, sid in enumerate(STORY_IDS)}
N_STORIES = len(STORY_IDS)
rng = random.Random(BOOTSTRAP_SEED)
# One shared matrix of story multiplicities is applied to every system.
# Paired differences therefore compare systems on exactly the same resamples.
BOOT_COUNTS = np.zeros((BOOTSTRAP_REPETITIONS, N_STORIES), dtype=np.int16)
for b in range(BOOTSTRAP_REPETITIONS):
    sampled = [rng.randrange(N_STORIES) for _ in range(N_STORIES)]
    BOOT_COUNTS[b] = np.bincount(sampled, minlength=N_STORIES)


def build_story_stats(rows):
    """Aggregate correct counts and per-class reference, prediction and
    true-positive counts by story. These compact arrays support story-level
    bootstrap calculations without repeatedly copying individual question rows.
    """
    correct = np.zeros(N_STORIES, float)
    n = np.zeros(N_STORIES, float)
    fam = {}
    for family in FAMILY_ORDER:
        k = len(FAMILY_LABELS[family])
        fam[family] = {
            "support": np.zeros((N_STORIES, k), float),
            "predicted": np.zeros((N_STORIES, k), float),
            "tp": np.zeros((N_STORIES, k), float),
        }
    for r in rows:
        if not r["scorable"]:
            continue
        s = story_index[r["story_id"]]
        n[s] += 1
        correct[s] += float(r["correct"])
        family = r["family_detail"]
        labels = FAMILY_LABELS[family]
        fs = fam[family]
        for j, lab in enumerate(labels):
            if r["gt"] == lab:
                fs["support"][s, j] += 1
            if r["pred"] == lab:
                fs["predicted"][s, j] += 1
            if r["gt"] == lab and r["pred"] == lab:
                fs["tp"][s, j] += 1
    return {"correct": correct, "n": n, "families": fam}


STORY_STATS = {name: build_story_stats(rows) for name, rows in SYSTEMS.items()}


def bootstrap_system(stats):
    """Apply the shared bootstrap story-count matrix to one system aggregated
    statistics. Return sampled accuracy and mean family macro F1, preserving
    within-story dependence and paired resampling across systems.
    """
    correct = BOOT_COUNTS @ stats["correct"]
    n = BOOT_COUNTS @ stats["n"]
    accuracy = np.divide(correct, n, out=np.zeros_like(correct, float), where=n > 0)
    family_f1 = []
    for family in FAMILY_ORDER:
        fs = stats["families"][family]
        support = BOOT_COUNTS @ fs["support"]
        predicted = BOOT_COUNTS @ fs["predicted"]
        tp = BOOT_COUNTS @ fs["tp"]
        den = support + predicted
        class_f1 = np.divide(2 * tp, den, out=np.zeros_like(tp, float), where=den > 0)
        family_f1.append(class_f1.mean(axis=1))
    return {
        "accuracy": accuracy,
        "mean_family_f1": np.stack(family_f1, axis=1).mean(axis=1),
    }


BOOT_SYSTEM = {name: bootstrap_system(STORY_STATS[name]) for name in SYSTEMS}


def ci(x):
    """Return the 2.5th and 97.5th percentiles of a sampled metric or difference.
    These are the central 95 percent story-bootstrap bounds used in the
    corrected result tables.
    """
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


SYSTEM_BOOTSTRAP_CI = {}
for name in SYSTEMS:
    SYSTEM_BOOTSTRAP_CI[name] = {
        "strict_overall_accuracy": {
            "point": SYSTEM_METRICS[name]["strict_overall_accuracy"],
            "ci95_story_bootstrap": ci(BOOT_SYSTEM[name]["accuracy"]),
        },
        "mean_family_strict_target_macro_f1": {
            "point": SYSTEM_METRICS[name]["mean_family_strict_target_macro_f1"],
            "ci95_story_bootstrap": ci(BOOT_SYSTEM[name]["mean_family_f1"]),
        },
        "repetitions": BOOTSTRAP_REPETITIONS,
        "seed": BOOTSTRAP_SEED,
    }
COMPARISONS = [
    ("answer_only_direct", "base_direct", "Answer-only direct - Base direct"),
    ("subjectesis_direct", "base_direct", "Subjectesis direct - Base direct"),
    (
        "subjectesis_direct",
        "answer_only_direct",
        "Subjectesis direct - Answer-only direct",
    ),
    (
        "subjectesis_no_review",
        "answer_only_direct",
        "Subjectesis no-review - Answer-only direct",
    ),
    (
        "subjectesis_review",
        "subjectesis_no_review",
        "Subjectesis review - Subjectesis no-review",
    ),
    (
        "subjectesis_review",
        "answer_only_direct",
        "Subjectesis review - Answer-only direct",
    ),
]
PAIRED_BOOTSTRAP = []
for a, b, label in COMPARISONS:
    ad = BOOT_SYSTEM[a]["accuracy"] - BOOT_SYSTEM[b]["accuracy"]
    fd = BOOT_SYSTEM[a]["mean_family_f1"] - BOOT_SYSTEM[b]["mean_family_f1"]
    PAIRED_BOOTSTRAP.append(
        {
            "comparison": label,
            "repetitions": BOOTSTRAP_REPETITIONS,
            "seed": BOOTSTRAP_SEED,
            "strict_overall_accuracy": {
                "point_difference": SYSTEM_METRICS[a]["strict_overall_accuracy"]
                - SYSTEM_METRICS[b]["strict_overall_accuracy"],
                "ci95_story_bootstrap": ci(ad),
            },
            "mean_family_strict_target_macro_f1": {
                "point_difference": SYSTEM_METRICS[a][
                    "mean_family_strict_target_macro_f1"
                ]
                - SYSTEM_METRICS[b]["mean_family_strict_target_macro_f1"],
                "ci95_story_bootstrap": ci(fd),
            },
        }
    )
transitions = Counter()
for before, after in zip(NO_REVIEW_S, REVIEW_S):
    if not before["scorable"]:
        continue
    if before["correct"] and after["correct"]:
        transitions["right_to_right"] += 1
    elif before["correct"] and (not after["correct"]):
        transitions["right_to_wrong"] += 1
    elif not before["correct"] and after["correct"]:
        transitions["wrong_to_right"] += 1
    else:
        transitions["wrong_to_wrong"] += 1
STRUCTURED_METRICS = {
    "invalid_initial_outputs": sum(
        (not r.get("initial_valid", False) for r in STRUCTURED_RAW)
    ),
    "invalid_initial_rate": sum(
        (not r.get("initial_valid", False) for r in STRUCTURED_RAW)
    )
    / 621,
    "review_attempts": sum((r.get("initial_valid", False) for r in STRUCTURED_RAW)),
    "invalid_reviews": sum(
        (
            r.get("initial_valid", False) and (not r.get("review_valid", False))
            for r in STRUCTURED_RAW
        )
    ),
    "raw_answer_changes": sum(
        (
            str(a["raw_prediction"]).strip() != str(b["raw_prediction"]).strip()
            for a, b in zip(NO_REVIEW_S, REVIEW_S)
        )
    ),
    "review_transitions": dict(transitions),
}
if STRUCTURED_METRICS["review_attempts"]:
    STRUCTURED_METRICS["invalid_review_rate"] = (
        STRUCTURED_METRICS["invalid_reviews"] / STRUCTURED_METRICS["review_attempts"]
    )
else:
    STRUCTURED_METRICS["invalid_review_rate"] = None
print(" 5,000 paired story-level bootstrap repetitions complete.")
print("Review transitions:", STRUCTURED_METRICS["review_transitions"])
import matplotlib.pyplot as plt

FINAL_RESULT = {
    "stage": "opentom_frozen_transfer_corrected_rescoring",
    "scoring_version": "explicit_plot_info_keys_v1",
    "fine_tuning_on_opentom": False,
    "model_inference_rerun": False,
    "predictions_regenerated": False,
    "opentom_commit": OPENTOM_COMMIT,
    "protocol_hash": PROTOCOL_HASH,
    "selection": selection,
    "correction_manifest": correction_manifest,
    "systems": SYSTEM_METRICS,
    "system_bootstrap_ci95": SYSTEM_BOOTSTRAP_CI,
    "structured": STRUCTURED_METRICS,
    "paired_story_bootstrap_differences": PAIRED_BOOTSTRAP,
    "bootstrap_unit": "OpenToM story",
    "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "metric_notes": {
        "primary_metric": "Macro-F1 is emphasized because OpenToM labels are imbalanced.",
        "strict_metric": "Malformed/corrupted predictions remain errors in strict metrics.",
        "correction": "Fine-location original/moved locations come from explicit metadata keys.",
        "no_new_experiment": "All metrics reuse the already-frozen prediction files.",
    },
}
(CORRECTED_DIR / "metrics.json").write_text(json.dumps(FINAL_RESULT, indent=2) + "\n")
summary_rows = []
for n, m in SYSTEM_METRICS.items():
    summary_rows.append(
        {
            "system": n,
            "questions": m["questions"],
            "scorable_questions": m["scorable_questions"],
            "unscorable_reference_questions": m["unscorable_reference_questions"],
            "strict_overall_accuracy": m["strict_overall_accuracy"],
            "accuracy_ci_low": SYSTEM_BOOTSTRAP_CI[n]["strict_overall_accuracy"][
                "ci95_story_bootstrap"
            ][0],
            "accuracy_ci_high": SYSTEM_BOOTSTRAP_CI[n]["strict_overall_accuracy"][
                "ci95_story_bootstrap"
            ][1],
            "mean_family_strict_macro_f1": m["mean_family_strict_target_macro_f1"],
            "macro_f1_ci_low": SYSTEM_BOOTSTRAP_CI[n][
                "mean_family_strict_target_macro_f1"
            ]["ci95_story_bootstrap"][0],
            "macro_f1_ci_high": SYSTEM_BOOTSTRAP_CI[n][
                "mean_family_strict_target_macro_f1"
            ]["ci95_story_bootstrap"][1],
            "overall_corrupt_rate": m["overall_corrupt_rate"],
        }
    )
pd.DataFrame(summary_rows).to_csv(CORRECTED_DIR / "system_summary.csv", index=False)
family_rows = []
for n, m in SYSTEM_METRICS.items():
    for f in FAMILY_ORDER:
        family_rows.append({"system": n, "family": f, **m["families"][f]})
pd.DataFrame(family_rows).to_csv(CORRECTED_DIR / "family_metrics.csv", index=False)
pair_rows = []
for r in PAIRED_BOOTSTRAP:
    pair_rows.append(
        {
            "comparison": r["comparison"],
            "accuracy_difference": r["strict_overall_accuracy"]["point_difference"],
            "accuracy_ci_low": r["strict_overall_accuracy"]["ci95_story_bootstrap"][0],
            "accuracy_ci_high": r["strict_overall_accuracy"]["ci95_story_bootstrap"][1],
            "macro_f1_difference": r["mean_family_strict_target_macro_f1"][
                "point_difference"
            ],
            "macro_f1_ci_low": r["mean_family_strict_target_macro_f1"][
                "ci95_story_bootstrap"
            ][0],
            "macro_f1_ci_high": r["mean_family_strict_target_macro_f1"][
                "ci95_story_bootstrap"
            ][1],
            "bootstrap_repetitions": r["repetitions"],
        }
    )
pd.DataFrame(pair_rows).to_csv(
    CORRECTED_DIR / "paired_bootstrap_comparisons.csv", index=False
)
maps = {n: {r["record_id"]: r for r in rows} for n, rows in SYSTEMS.items()}
matrix = []
for rid in sorted(expected_ids):
    base = maps["base_direct"][rid]
    rec = {
        "record_id": rid,
        "story_id": base["story_id"],
        "family": base["family"],
        "family_detail": base["family_detail"],
        "reference": base["reference"],
        "scorable_reference": base["scorable"],
    }
    for n in SYSTEMS:
        rr = maps[n][rid]
        rec[n + "_prediction"] = rr["raw_prediction"]
        rec[n + "_correct"] = rr["correct"]
        rec[n + "_corrupt"] = rr["corrupt"]
    matrix.append(rec)
pd.DataFrame(matrix).to_csv(CORRECTED_DIR / "prediction_matrix.csv", index=False)
CONF = CORRECTED_DIR / "confusion_matrices"
CONF.mkdir(exist_ok=True)
for n, rows in SYSTEMS.items():
    for f in FAMILY_ORDER:
        x = [r for r in rows if r["family_detail"] == f and r["scorable"]]
        tab = pd.crosstab(
            pd.Series([r["gt"] for r in x], name="reference"),
            pd.Series([r["pred"] for r in x], name="prediction"),
            dropna=False,
        )
        tab.to_csv(CONF / f"{n}__{f}.csv")
names = list(SYSTEMS)
points = np.array([SYSTEM_METRICS[n]["strict_overall_accuracy"] for n in names])
low = np.array(
    [
        SYSTEM_BOOTSTRAP_CI[n]["strict_overall_accuracy"]["ci95_story_bootstrap"][0]
        for n in names
    ]
)
high = np.array(
    [
        SYSTEM_BOOTSTRAP_CI[n]["strict_overall_accuracy"]["ci95_story_bootstrap"][1]
        for n in names
    ]
)
fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(names))
ax.bar(x, points)
ax.errorbar(
    x, points, yerr=np.vstack([points - low, high - points]), fmt="none", capsize=4
)
ax.set_ylim(0, 1)
ax.set_ylabel("Corrected strict OpenToM accuracy")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=25, ha="right")
fig.tight_layout()
fig.savefig(CORRECTED_DIR / "corrected_accuracy_with_ci.png", dpi=200)
plt.show()
f1p = np.array([SYSTEM_METRICS[n]["mean_family_strict_target_macro_f1"] for n in names])
f1l = np.array(
    [
        SYSTEM_BOOTSTRAP_CI[n]["mean_family_strict_target_macro_f1"][
            "ci95_story_bootstrap"
        ][0]
        for n in names
    ]
)
f1h = np.array(
    [
        SYSTEM_BOOTSTRAP_CI[n]["mean_family_strict_target_macro_f1"][
            "ci95_story_bootstrap"
        ][1]
        for n in names
    ]
)
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.bar(x, f1p)
ax.errorbar(x, f1p, yerr=np.vstack([f1p - f1l, f1h - f1p]), fmt="none", capsize=4)
ax.set_ylim(0, 1)
ax.set_ylabel("Corrected mean family macro-F1")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=25, ha="right")
fig.tight_layout()
fig.savefig(CORRECTED_DIR / "corrected_macro_f1_with_ci.png", dpi=200)
plt.show()
old_metrics_file = TRANSFER_DIR / "final_metrics" / "metrics.json"
if old_metrics_file.exists():
    old = json.loads(old_metrics_file.read_text())
    old_systems = old.get("systems", {})
    comparison = []
    for n in names:
        old_m = old_systems.get(n, {})
        comparison.append(
            {
                "system": n,
                "old_accuracy": old_m.get("strict_overall_accuracy"),
                "corrected_accuracy": SYSTEM_METRICS[n]["strict_overall_accuracy"],
                "old_macro_f1": old_m.get("mean_family_strict_target_macro_f1"),
                "corrected_macro_f1": SYSTEM_METRICS[n][
                    "mean_family_strict_target_macro_f1"
                ],
                "old_unscorable": old_m.get("unscorable_reference_questions"),
                "corrected_unscorable": SYSTEM_METRICS[n][
                    "unscorable_reference_questions"
                ],
            }
        )
    pd.DataFrame(comparison).to_csv(
        CORRECTED_DIR / "old_vs_corrected_summary.csv", index=False
    )
report_hashes = {}
for p in CORRECTED_DIR.rglob("*"):
    if p.is_file():
        report_hashes[str(p.relative_to(CORRECTED_DIR))] = hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
complete = {
    "status": "completed",
    "experiment": "OpenToM frozen-transfer corrected rescoring",
    "correction_id": correction_manifest["correction_id"],
    "opentom_commit": OPENTOM_COMMIT,
    "protocol_hash": PROTOCOL_HASH,
    "stories": 27,
    "questions": 621,
    "bootstrap_unit": "story",
    "bootstrap_repetitions": 5000,
    "bootstrap_seed": 42,
    "corrected_input_sha256": CORRECTED_INPUT_SHA,
    "corrected_reference_sha256": CORRECTED_REF_SHA,
    "fine_tuning_on_opentom": False,
    "model_inference_rerun": False,
    "predictions_regenerated": False,
    "report_hashes": report_hashes,
}
(TRANSFER_DIR / "OPENTOM_CORRECTED_RESCORING_COMPLETE.json").write_text(
    json.dumps(complete, indent=2) + "\n"
)
print("\n=== CORRECTED OPENTOM SUMMARY ===")
print(pd.DataFrame(summary_rows).to_string(index=False))
print("\n Corrected rescoring complete. No model inference or retraining occurred.")
print("Saved to:", CORRECTED_DIR)
print("Completion record:", TRANSFER_DIR / "OPENTOM_CORRECTED_RESCORING_COMPLETE.json")
