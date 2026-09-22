"""Generate frozen-model OpenToM transfer predictions.

Select the recorded smoke and final stories, run the three direct model
conditions, and produce paired no-review/review answers from shared initial
states. This script performs no OpenToM fine-tuning. Run rescore_opentom.py
afterward to reproduce the corrected metrics from the saved predictions.
"""

from project_setup import mount_drive, prepare_project

mount_drive()
from pathlib import Path, PurePosixPath
import os, sys, json, hashlib, zipfile, shutil, subprocess

ROOT = Path("/content/subjectesis_opentom_transfer")
DRIVE = Path("/content/drive/MyDrive/Subjectesis")
OUT = DRIVE / "opentom_transfer"
SUB_ZIP = DRIVE / "subjectesis_a100_latest.zip"
ANS_ZIP = DRIVE / "answer_only_a100_latest.zip"
OTOM = ROOT / "OpenToM"
OTOM_COMMIT = "3f22b66276b2d7ca5fe573c28c79cc0d077aafc5"
SELECTION_SEED = "subjectesis-opentom-transfer-v1"
N_SMOKE = 5
N_FINAL = 27
ROOT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
for p in [SUB_ZIP, ANS_ZIP]:
    if not p.exists():
        raise FileNotFoundError(p)


def verify_archive(path, condition):
    """Verify the completion record and selected adapter files against the archive
    manifest. Require completed training with the expected example exposure and
    no recorded test scoring, then return the checkpoint metadata used by
    transfer evaluation.
    """
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("recovery_manifest.json"))["files"]
        cpath = f"runs/{condition}/completed.json"
        raw = z.read(cpath)
        if hashlib.sha256(raw).hexdigest() != manifest[cpath]:
            raise RuntimeError("Checkpoint manifest mismatch.")
        completed = json.loads(raw)
        assert completed["status"] == "completed"
        assert completed["examples_seen"] == 30266
        assert completed["test_scored"] is False
        for name in [
            f"runs/{condition}/best_adapter/adapter_config.json",
            f"runs/{condition}/best_adapter/adapter_model.safetensors",
        ]:
            raw = z.read(name)
            if hashlib.sha256(raw).hexdigest() != manifest[name]:
                raise RuntimeError(f"Adapter hash mismatch: {name}")
        return completed


ans_completed = verify_archive(ANS_ZIP, "ordinary")
sub_completed = verify_archive(SUB_ZIP, "subjectesis")
print("Answer-only best step:", ans_completed["best_step"])
print("Subjectesis best step:", sub_completed["best_step"])
PROJECT = ROOT / "project"
prepare_project(PROJECT)
settings = {
    "model_id": "Qwen/Qwen3.5-4B",
    "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "training_method": "NF4_QLoRA",
    "train_baseline": False,
    "epochs": 1,
    "rank": 8,
    "alpha": 16,
    "learning_rate": 0.0001,
    "seed": 42,
    "effective_batch_size": 8,
    "max_length": 2048,
    "warmup_fraction": 0.03,
    "loss_chunk_size": 32,
    "save_steps": 25,
    "eval_steps": 500,
    "max_session_hours": 0.0,
}
(PROJECT / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
if OTOM.exists():
    shutil.rmtree(OTOM)
subprocess.run(
    ["git", "clone", "--quiet", "https://github.com/seacowx/OpenToM.git", str(OTOM)],
    check=True,
)
subprocess.run(["git", "-C", str(OTOM), "checkout", "--quiet", OTOM_COMMIT], check=True)
actual = subprocess.check_output(
    ["git", "-C", str(OTOM), "rev-parse", "HEAD"], text=True
).strip()
assert actual == OTOM_COMMIT
print("OpenToM commit:", actual)
DATA = OTOM / "data/opentom_data"
FILES = {
    "location_cg_fo": DATA / "location_cg_fo.json",
    "location_cg_so": DATA / "location_cg_so.json",
    "location_fg_fo": DATA / "location_fg_fo_new.json",
    "location_fg_so": DATA / "location_fg_so_new.json",
    "multihop_fo": DATA / "multihop_fo.json",
    "multihop_so": DATA / "multihop_so.json",
    "attitude": DATA / "attitude.json",
}
meta = json.loads((DATA / "meta_data.json").read_text())
questions = {k: json.loads(v.read_text()) for k, v in FILES.items()}
story_ids = sorted(meta)
assert len(story_ids) == 596


def rank_story(sid):
    """Hash the fixed selection seed together with a story ID to obtain a
    repeatable ordering key. Sorting these keys selects smoke and final stories
    without using their answers or observed model performance.
    """
    return hashlib.sha256((SELECTION_SEED + "|" + str(sid)).encode()).hexdigest()


# The story subset is determined by the fixed hash ordering. Smoke stories
# are excluded from the final 27-story, 621-question transfer set.
ranked = sorted(story_ids, key=lambda sid: (rank_story(sid), sid))
SMOKE_IDS = ranked[:N_SMOKE]
FINAL_IDS = ranked[N_SMOKE : N_SMOKE + N_FINAL]
assert not set(SMOKE_IDS) & set(FINAL_IDS)
COUNTS = {
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
        text = q["question"].lower()
        if "fullness" in text:
            return family + "_fullness"
        if "accessibility" in text:
            return family + "_accessibility"
        raise RuntimeError("Unknown multihop type.")
    return family


# Corrected lookup: dictionary insertion order is not a metadata schema.
def plot_info(story_id):
    """Read the mover, observer, entity and two locations by their explicit
    metadata keys. This corrects the earlier assumption that dictionary
    insertion order identified those fields reliably.
    """
    p = meta[story_id]["plot_info"]
    return (
        p["mover"],
        p["observer"],
        p["eoi"],
        p["original_place"],
        p["move_to_place"],
    )


def build(ids, include_answers):
    """Expand selected stories into question records with narrative, family and
    corrected location metadata. Include answer labels only when requested for
    the separate reference records, and verify each story family has its
    expected number of questions.
    """
    rows = []
    for sid in ids:
        narrative = meta[sid]["narrative"]
        mover, observer, entity, original_place, moved_place = plot_info(sid)
        for family, source in questions.items():
            qs = source[sid]
            assert len(qs) == COUNTS[family]
            for idx, q in enumerate(qs):
                item = {
                    "record_id": f"{sid}::{family}::{idx}",
                    "story_id": sid,
                    "family": family,
                    "family_detail": family_detail(family, q),
                    "question": q["question"],
                    "narrative": narrative,
                    "original_place": original_place,
                    "move_to_place": moved_place,
                }
                if include_answers:
                    item["answer"] = q["answer"]
                rows.append(item)
    return rows


SMOKE_ROWS = build(SMOKE_IDS, False)
TRANSFER_ROWS = build(FINAL_IDS, False)
REF_ROWS = build(FINAL_IDS, True)
assert len(SMOKE_ROWS) == 115
assert len(TRANSFER_ROWS) == 621
assert len(REF_ROWS) == 621
INPUT_DIR = ROOT / "opentom_inputs"
INPUT_DIR.mkdir(exist_ok=True)
INPUT_FILE = INPUT_DIR / "final_inputs.jsonl"
REF_FILE = INPUT_DIR / "final_references.jsonl"


def write_jsonl(path, rows):
    """Write each input or reference record as one UTF-8 JSON line in its existing
    order. Keeping these records separate allows inference to consume the
    answer-free input file.
    """
    with path.open("w", encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


write_jsonl(INPUT_FILE, TRANSFER_ROWS)
write_jsonl(
    REF_FILE,
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
        for r in REF_ROWS
    ],
)
INPUT_SHA = hashlib.sha256(INPUT_FILE.read_bytes()).hexdigest()
REF_SHA = hashlib.sha256(REF_FILE.read_bytes()).hexdigest()
del REF_ROWS
selection = {
    "opentom_commit": OTOM_COMMIT,
    "selection_rule": "sha256(seed|story_id)",
    "seed": SELECTION_SEED,
    "smoke_story_ids": SMOKE_IDS,
    "final_story_ids": FINAL_IDS,
    "final_questions": 621,
    "input_sha256": INPUT_SHA,
    "reference_sha256": REF_SHA,
    "fine_tuning": False,
}
(OUT / "selection_manifest.json").write_text(json.dumps(selection, indent=2) + "\n")
print("Smoke stories:", len(SMOKE_IDS))
print("Final transfer:", len(FINAL_IDS), "stories /", len(TRANSFER_ROWS), "questions")
print("References separated.")
from pathlib import Path, PurePosixPath
import subprocess
import sys
import os
import json
import zipfile
import shutil


def run(cmd, env=None):
    """Print and execute one setup command, raising an error if it fails. This
    keeps dependency and model-download failures from being mistaken for a
    completed evaluation stage.
    """
    print("\n>>>", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, env=env)


subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "transformers==5.13.0",
        "peft==0.21.0",
        "bitsandbytes==0.49.2",
        "accelerate==1.12.0",
        "safetensors==0.8.0",
        "huggingface-hub==1.11.0",
        "tokenizers==0.22.2",
        "scikit-learn",
        "pandas",
        "matplotlib",
    ]
)
run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--no-deps",
        "torchao==0.16.0",
    ]
)
env = os.environ.copy()
env["PYTHONPATH"] = str(PROJECT / "code") + ":" + env.get("PYTHONPATH", "")
env["TOKENIZERS_PARALLELISM"] = "false"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
run(
    [sys.executable, PROJECT / "code" / "download_model.py", "--root", PROJECT], env=env
)
ADAPTER_DIR = ROOT / "adapters"
ADAPTER_DIR.mkdir(exist_ok=True)


def extract_adapter(archive, condition, destination):
    """Copy the selected condition best-adapter directory from its recovery
    archive into a fresh local folder. The transfer comparison loads these
    completed checkpoints rather than fitting adapters on OpenToM.
    """
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    prefix = f"runs/{condition}/best_adapter/"
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not (name.startswith(prefix) and (not name.endswith("/"))):
                continue
            rel = PurePosixPath(name).relative_to(PurePosixPath(prefix))
            target = destination.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))


SUB_ADAPTER = ADAPTER_DIR / "subjectesis_best"
ANS_ADAPTER = ADAPTER_DIR / "answer_only_best"
extract_adapter(SUB_ZIP, "subjectesis", SUB_ADAPTER)
extract_adapter(ANS_ZIP, "ordinary", ANS_ADAPTER)
print("Adapters ready.")
import json
import sys
import re
import copy
import hashlib
import torch

sys.path.insert(0, str(PROJECT / "code"))
from common import read_json
from model_runtime import build_network
from safetensors.torch import load_file
from peft import set_peft_model_state_dict
from transformers import AutoTokenizer

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
if not torch.cuda.is_available():
    raise RuntimeError("Use A100 GPU.")
print("GPU:", torch.cuda.get_device_name(0))
config = read_json(PROJECT / "settings.json")
model_path = Path(read_json(PROJECT / "reports" / "model_download.json")["snapshot"])
tok = AutoTokenizer.from_pretrained(
    model_path, local_files_only=True, trust_remote_code=False
)
tok.padding_side = "left"
network = build_network(PROJECT, config, 0, model_path)
network.eval()
try:
    network.gradient_checkpointing_disable()
except Exception:
    pass
network.config.use_cache = True


def load_adapter(path):
    """Load one saved adapter into the existing model and switch to evaluation
    mode. This reuses the same base backbone for the trained direct and
    structured transfer conditions.
    """
    state = load_file(Path(path) / "adapter_model.safetensors")
    set_peft_model_state_dict(network, state, adapter_name="default")
    network.eval()


def instruction(row):
    """Choose the required answer format for a question family, such as yes/no,
    location, fullness or attitude. Restricting the requested wording makes raw
    generations compatible with the recorded OpenToM parser.
    """
    fam = row["family"]
    q = row["question"].lower()
    if fam.startswith("location_cg"):
        return 'Answer only "Yes" or "No".'
    if fam.startswith("location_fg"):
        return "Answer only with the precise location phrase."
    if fam.startswith("multihop"):
        if "fullness" in q:
            return 'Answer only "more full", "equally full", or "less full".'
        return (
            'Answer only "more accessible", "equally accessible", or "less accessible".'
        )
    if fam == "attitude":
        return 'Answer only "positive", "neutral", or "negative".'
    raise RuntimeError(fam)


def direct_prompt(row):
    """Combine the story, question and family-specific answer instruction into a
    direct prompt. The base and both trained adapters receive this same prompt
    format for the direct transfer comparison.
    """
    return (
        "Story:\n"
        + row["narrative"]
        + "\n\nQuestion:\n"
        + row["question"]
        + "\n\n"
        + instruction(row)
        + " Do not explain."
    )


INITIAL = '\nBuild a subject-relative mental-state record for the question.\n\nReturn exactly one JSON object:\n{\n  "perspective_holder": string,\n  "nested_perspective": string or null,\n  "mental_state_claim": string,\n  "evidence_sentences": [integer sentence numbers],\n  "uncertainty": string,\n  "answer": string\n}\n\nUse only the story.\nTrack what the queried perspective could know, believe, perceive, or feel.\nFor second-order questions, use nested_perspective for the inner mental-state holder.\nDo not invent unsupported facts.\n'
REVIEW = '\nReview the current mental-state record against the story.\n\nReturn exactly one JSON object:\n{\n  "decision": "revise" or "preserve" or "retain_unknown",\n  "updated_mental_state_claim": string,\n  "evidence_sentences": [integer sentence numbers],\n  "uncertainty": string,\n  "reason": string\n}\n\nRevise unsupported claims.\nDo not use information unavailable to the queried perspective.\n'
FINALIZER = "\nUse the mental-state record only as evidence bookkeeping.\nAnswer the original question using the requested answer format.\nDo not explain.\n"
# OpenToM uses one whole-record review with no additional fine-tuning.
# Its state fields and output formats differ from the RecToM controller.
PROTOCOL = {
    "version": "subjectesis_opentom_transfer_v1",
    "opentom_commit": OTOM_COMMIT,
    "selection_seed": SELECTION_SEED,
    "smoke_stories": N_SMOKE,
    "final_stories": N_FINAL,
    "initial_instruction": INITIAL,
    "review_instruction": REVIEW,
    "finalizer_instruction": FINALIZER,
    "initial_max_new_tokens": 220,
    "review_max_new_tokens": 180,
    "answer_max_new_tokens": 24,
    "review_calls": 1,
    "same_initial_state": True,
    "labels_used_by_controller": False,
    "fine_tuning": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
print("OpenToM protocol:", PROTOCOL_HASH)


def numbered_story(text):
    """Split a narrative at sentence-ending punctuation and number the resulting
    sentences. Return both the numbered text and sentence count so structured
    prompts can cite evidence and validators can check citation bounds.
    """
    parts = [x.strip() for x in re.split("(?<=[.!?])\\s+", text.strip()) if x.strip()]
    return ("\n".join((f"{i + 1}: {s}" for i, s in enumerate(parts))), len(parts))


def state_prompt(row):
    """Ask the frozen model to build a subject-relative mental-state record from a
    numbered story. Include the question and answer-format instruction so the
    state addresses the queried perspective, including nested perspectives when
    relevant.
    """
    story, _ = numbered_story(row["narrative"])
    return (
        "Numbered story:\n"
        + story
        + "\n\nQuestion:\n"
        + row["question"]
        + "\n\n"
        + instruction(row)
        + "\n\n"
        + INITIAL
    )


def review_prompt(row, state):
    """Present the numbered story, question and current mental-state record for
    one review. This OpenToM protocol reviews the whole claim once, unlike the
    bounded sequence of field reviews used for RecToM.
    """
    story, _ = numbered_story(row["narrative"])
    return (
        "Numbered story:\n"
        + story
        + "\n\nQuestion:\n"
        + row["question"]
        + "\n\nCurrent state:\n"
        + json.dumps(state, ensure_ascii=False)
        + "\n\n"
        + REVIEW
    )


def final_prompt(row, state):
    """Append a mental-state record to the direct question prompt and request the
    final short answer. Using this same format before and after review makes
    the two structured conditions comparable.
    """
    return (
        direct_prompt(row)
        + "\n\nMental-state record:\n"
        + json.dumps(state, ensure_ascii=False)
        + "\n\n"
        + FINALIZER
    )


def chat(prompt):
    """Apply the Qwen user-message template with a generation marker and thinking
    disabled. Return text for tokenization so all generated responses use the
    same chat formatting.
    """
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate(prompts, max_new_tokens, preferred=8):
    """Generate ordered greedy completions from left-padded, attention-masked
    prompt batches. Halve the batch after GPU out-of-memory errors and return
    only continuation text, preserving the frozen prompts and output-token
    budget.
    """
    results = []
    start = 0
    while start < len(prompts):
        batch = min(preferred, len(prompts) - start)
        while True:
            try:
                encoded = [
                    tok.encode(chat(p), add_special_tokens=False)
                    for p in prompts[start : start + batch]
                ]
                pad = tok.pad_token_id or tok.eos_token_id
                maxlen = max(map(len, encoded))
                ids = torch.full(
                    (batch, maxlen), int(pad), dtype=torch.long, device="cuda"
                )
                mask = torch.zeros((batch, maxlen), dtype=torch.long, device="cuda")
                for i, seq in enumerate(encoded):
                    n = len(seq)
                    ids[i, -n:] = torch.tensor(seq, device="cuda")
                    mask[i, -n:] = 1
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.float16
                ):
                    output = network.generate(
                        input_ids=ids,
                        attention_mask=mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=int(pad),
                        eos_token_id=tok.eos_token_id,
                    )
                text = [
                    tok.decode(row[maxlen:], skip_special_tokens=True).strip()
                    for row in output
                ]
                results.extend(text)
                start += batch
                del ids
                del mask
                del output
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch == 1:
                    raise
                batch = max(1, batch // 2)
                print("OOM -> batch", batch)
    return results


def first_json(text):
    """Extract and decode the first balanced JSON object while respecting quoted
    braces and escapes. Return None if the object is absent or cannot be
    decoded, allowing malformed generations to be recorded.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    quoted = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if quoted:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                quoted = False
            continue
        if ch == '"':
            quoted = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except:
                    return None
    return None


def valid_initial(obj, row):
    """Check that an initial record contains the required keys and a list of
    in-range sentence citations. This is a limited structural check; it does
    not validate every field type or establish that the mental-state claim
    follows from the story.
    """
    if not isinstance(obj, dict):
        return False
    keys = {
        "perspective_holder",
        "nested_perspective",
        "mental_state_claim",
        "evidence_sentences",
        "uncertainty",
        "answer",
    }
    if not keys.issubset(obj):
        return False
    if not isinstance(obj["evidence_sentences"], list):
        return False
    _, n = numbered_story(row["narrative"])
    return all((isinstance(x, int) and 1 <= x <= n for x in obj["evidence_sentences"]))


def valid_review(obj, row):
    """Check the review keys, decision and sentence-number bounds before applying
    its update. This validator assumes evidence_sentences is iterable and
    checks structural form rather than whether the cited evidence supports the
    claim.
    """
    if not isinstance(obj, dict):
        return False
    keys = {
        "decision",
        "updated_mental_state_claim",
        "evidence_sentences",
        "uncertainty",
        "reason",
    }
    if not keys.issubset(obj):
        return False
    if obj["decision"] not in {"revise", "preserve", "retain_unknown"}:
        return False
    _, n = numbered_story(row["narrative"])
    return all((isinstance(x, int) and 1 <= x <= n for x in obj["evidence_sentences"]))


load_adapter(SUB_ADAPTER)
details = {}
for row in SMOKE_ROWS:
    details.setdefault(row["family_detail"], row)
expected = {
    "location_cg_fo",
    "location_cg_so",
    "location_fg_fo",
    "location_fg_so",
    "multihop_fo_fullness",
    "multihop_fo_accessibility",
    "multihop_so_fullness",
    "multihop_so_accessibility",
    "attitude",
}
assert set(details) == expected
smoke = [details[k] for k in sorted(expected)]
raw = generate([state_prompt(r) for r in smoke], PROTOCOL["initial_max_new_tokens"])
bad_initial = 0
bad_review = 0
for row, output in zip(smoke, raw):
    state = first_json(output)
    if not valid_initial(state, row):
        bad_initial += 1
        continue
    review_output = generate(
        [review_prompt(row, state)], PROTOCOL["review_max_new_tokens"], preferred=1
    )[0]
    review_state = first_json(review_output)
    if not valid_review(review_state, row):
        bad_review += 1
print("Smoke invalid initial:", bad_initial, "/ 9")
print("Smoke invalid review:", bad_review, "/ 9")
if bad_initial > 1 or bad_review > 1:
    raise RuntimeError("Structural smoke failed. Do not run final transfer.")
frozen = {
    "protocol_hash": PROTOCOL_HASH,
    "config": PROTOCOL,
    "selection": selection,
    "labels_read": False,
    "fine_tuning": False,
}
(OUT / "opentom_protocol_frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
print("Protocol frozen.")
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PRED_DIR = OUT
PRED_DIR.mkdir(exist_ok=True)


def save_jsonl(path, rows):
    """Write records to a temporary JSONL file and replace the destination after
    the write completes. Transfer inference uses this to persist completed
    chunks without leaving a half-written result file.
    """
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_jsonl(path):
    """Load non-empty JSONL records, returning an empty list when the file does
    not exist. This supports both a fresh transfer run and continuation from
    saved predictions.
    """
    if not path.exists():
        return []
    return [
        json.loads(x) for x in path.read_text(encoding="utf8").splitlines() if x.strip()
    ]


def direct_system(name, adapter=None, base=False):
    """Generate the missing direct predictions for one model condition and save
    them after each chunk. Reuse saved record IDs, optionally disable adapters
    for the base condition, and require all 621 predictions before returning.
    """
    path = PRED_DIR / f"{name}.jsonl"
    done_rows = load_jsonl(path)
    done = {r["record_id"]: r for r in done_rows}
    if adapter is not None:
        load_adapter(adapter)
    ctx = network.disable_adapter() if base else None
    if ctx is not None:
        ctx.__enter__()
    try:
        remaining = [r for r in TRANSFER_ROWS if r["record_id"] not in done]
        for start in range(0, len(remaining), 32):
            chunk = remaining[start : start + 32]
            preds = generate(
                [direct_prompt(r) for r in chunk], PROTOCOL["answer_max_new_tokens"]
            )
            for row, pred in zip(chunk, preds):
                done[row["record_id"]] = {
                    "system": name,
                    "record_id": row["record_id"],
                    "story_id": row["story_id"],
                    "family": row["family"],
                    "family_detail": row["family_detail"],
                    "prediction": pred,
                }
            ordered = [
                done[r["record_id"]] for r in TRANSFER_ROWS if r["record_id"] in done
            ]
            save_jsonl(path, ordered)
            print(name, len(done), "/ 621")
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    assert len(done) == 621
    return [done[r["record_id"]] for r in TRANSFER_ROWS]


load_adapter(SUB_ADAPTER)
BASE = direct_system("base_direct", base=True)
ANSWER = direct_system("answer_only_direct", adapter=ANS_ADAPTER)
SUBJECTESIS = direct_system("subjectesis_direct", adapter=SUB_ADAPTER)
load_adapter(SUB_ADAPTER)
STRUCTURED_PATH = PRED_DIR / "subjectesis_structured.jsonl"
structured_done = {r["record_id"]: r for r in load_jsonl(STRUCTURED_PATH)}


def initial_state_only(obj):
    """Copy perspective, claim, evidence and uncertainty fields from an initial
    response while excluding its proposed answer. The finalizer then selects
    its answer from the story and this state representation.
    """
    return {
        "perspective_holder": obj["perspective_holder"],
        "nested_perspective": obj["nested_perspective"],
        "mental_state_claim": obj["mental_state_claim"],
        "evidence_sentences": list(obj["evidence_sentences"]),
        "uncertainty": obj["uncertainty"],
    }


def apply_review(state, review):
    """Deep-copy the current state and replace its claim, evidence sentences and
    uncertainty with the valid review output. Perspective-holder fields are
    retained, and even a preserve decision can update the record contents.
    """
    out = copy.deepcopy(state)
    # Merge content from a valid review regardless of the decision label.
    # A preserve label is not a guarantee that the generated text is unchanged.
    out["mental_state_claim"] = review["updated_mental_state_claim"]
    out["evidence_sentences"] = list(review["evidence_sentences"])
    out["uncertainty"] = review["uncertainty"]
    return out


remaining = [r for r in TRANSFER_ROWS if r["record_id"] not in structured_done]
started = time.time()
new_count = 0
for start in range(0, len(remaining), 16):
    chunk = remaining[start : start + 16]
    initial_raw = generate(
        [state_prompt(r) for r in chunk], PROTOCOL["initial_max_new_tokens"]
    )
    work = []
    for row, raw in zip(chunk, initial_raw):
        obj = first_json(raw)
        ok = valid_initial(obj, row)
        state = initial_state_only(obj) if ok else None
        work.append(
            {"row": row, "initial_valid": ok, "raw_initial": raw, "state": state}
        )
    valid = [w for w in work if w["state"] is not None]
    no_review_preds = generate(
        [final_prompt(w["row"], w["state"]) for w in valid],
        PROTOCOL["answer_max_new_tokens"],
    )
    review_raw = generate(
        [review_prompt(w["row"], w["state"]) for w in valid],
        PROTOCOL["review_max_new_tokens"],
    )
    reviewed = []
    for w, no_pred, rr in zip(valid, no_review_preds, review_raw):
        robj = first_json(rr)
        rok = valid_review(robj, w["row"])
        state_after = (
            apply_review(w["state"], robj) if rok else copy.deepcopy(w["state"])
        )
        reviewed.append((w, no_pred, rr, rok, state_after))
    review_preds = generate(
        [
            final_prompt(w["row"], state_after)
            for w, no_pred, rr, rok, state_after in reviewed
        ],
        PROTOCOL["answer_max_new_tokens"],
    )
    review_map = {}
    for item, review_pred in zip(reviewed, review_preds):
        w, no_pred, rr, rok, state_after = item
        review_map[w["row"]["record_id"]] = {
            "no_review": no_pred,
            "review": review_pred,
            "raw_review": rr,
            "review_valid": rok,
            "state_after": state_after,
        }
    for w in work:
        row = w["row"]
        extra = review_map.get(row["record_id"], {})
        structured_done[row["record_id"]] = {
            "record_id": row["record_id"],
            "story_id": row["story_id"],
            "family": row["family"],
            "family_detail": row["family_detail"],
            "initial_valid": w["initial_valid"],
            "raw_initial": w["raw_initial"],
            "no_review_prediction": extra.get("no_review"),
            "review_prediction": extra.get("review"),
            "raw_review": extra.get("raw_review"),
            "review_valid": extra.get("review_valid", False),
        }
    save_jsonl(
        STRUCTURED_PATH,
        [
            structured_done[r["record_id"]]
            for r in TRANSFER_ROWS
            if r["record_id"] in structured_done
        ],
    )
    new_count += len(chunk)
    elapsed = time.time() - started
    eta = elapsed / max(1, new_count) * (len(remaining) - new_count) / 60
    print("Structured", len(structured_done), "/ 621 | ETA", round(eta, 1), "min")
assert len(structured_done) == 621
STRUCTURED = [structured_done[r["record_id"]] for r in TRANSFER_ROWS]
freeze = {
    "protocol_hash": PROTOCOL_HASH,
    "input_sha": INPUT_SHA,
    "base": 621,
    "answer_only": 621,
    "subjectesis": 621,
    "structured": 621,
    "labels_read": False,
}
(OUT / "PREDICTIONS_FROZEN.json").write_text(json.dumps(freeze, indent=2) + "\n")
print("Predictions frozen.")
print(
    "Run src/rescore_opentom.py to calculate corrected scores from these frozen predictions."
)
