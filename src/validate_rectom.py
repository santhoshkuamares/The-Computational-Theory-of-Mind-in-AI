"""Evaluate the selected adapters and develop the controller on validation.

The 319 RecToM validation questions are used to reproduce saved checkpoint
answers and compare direct, no-review and reviewed predictions. The script
records its controller protocol for the later sealed test. It expects the
recorded Colab environment and completed adapter archives.
"""

from project_setup import mount_drive, prepare_project

mount_drive()
from pathlib import Path, PurePosixPath
import base64, io, zipfile, json, hashlib, os, shutil, sys

ROOT = Path("/content/subjectesis_dev_eval_v1")
DRIVE_DIR = Path("/content/drive/MyDrive/Subjectesis")
SUBJECTESIS_ZIP = DRIVE_DIR / "subjectesis_a100_latest.zip"
ANSWER_ONLY_ZIP = DRIVE_DIR / "answer_only_a100_latest.zip"
if not SUBJECTESIS_ZIP.exists():
    raise FileNotFoundError(f"Missing {SUBJECTESIS_ZIP}")
if not ANSWER_ONLY_ZIP.exists():
    raise FileNotFoundError(f"Missing {ANSWER_ONLY_ZIP}")
ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / "reports").mkdir(exist_ok=True)


def sha256_bytes(data):
    """Return a SHA-256 checksum for bytes read from an archive member. Comparing
    it with the recovery manifest confirms which saved training artifact is
    being evaluated.
    """
    return hashlib.sha256(data).hexdigest()


def verify_recovery_archive(path, condition):
    """Verify all manifested archive files and require completed training with the
    expected exposure and no recorded test scoring. Check that the selected
    validation report agrees with completion metadata, then return the
    completion record, best validation result and manifest.
    """
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"ZIP corruption in {path.name}: {bad}")
        names = set(z.namelist())
        if "recovery_manifest.json" not in names:
            raise RuntimeError(f"{path.name} has no recovery manifest")
        manifest = json.loads(z.read("recovery_manifest.json"))
        for name, expected in manifest["files"].items():
            if name not in names:
                raise RuntimeError(f"{path.name} missing {name}")
            if sha256_bytes(z.read(name)) != expected:
                raise RuntimeError(f"Hash mismatch: {path.name} :: {name}")
        completed_name = f"runs/{condition}/completed.json"
        completed = json.loads(z.read(completed_name))
        if completed.get("status") != "completed":
            raise RuntimeError(f"{path.name} is not a completed run")
        if completed.get("examples_seen") != 30266:
            raise RuntimeError(f"{path.name} exposure count changed")
        if completed.get("test_scored") is not False:
            raise RuntimeError(f"{path.name} unexpectedly reports test scoring")
        best_step = int(completed["best_step"])
        best_name = f"runs/{condition}/validation_{best_step:07d}.json"
        best = json.loads(z.read(best_name))
        if abs(float(best["selection_score"]) - float(completed["best_score"])) > 1e-12:
            raise RuntimeError("Best-checkpoint metadata disagreement")
        return (completed, best, manifest)


ordinary_completed, ordinary_saved_best, _ = verify_recovery_archive(
    ANSWER_ONLY_ZIP, "ordinary"
)
subjectesis_completed, subjectesis_saved_best, _ = verify_recovery_archive(
    SUBJECTESIS_ZIP, "subjectesis"
)
print(
    "Answer-only archive verified:",
    "best step",
    ordinary_completed["best_step"],
    "score",
    ordinary_completed["best_score"],
)
print(
    " Subjectesis archive verified:",
    "best step",
    subjectesis_completed["best_step"],
    "score",
    subjectesis_completed["best_score"],
)

prepare_project(ROOT)
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
(ROOT / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
import subprocess

env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT / "code") + ":" + env.get("PYTHONPATH", "")
subprocess.run(
    [sys.executable, str(ROOT / "code/prepare.py"), "--root", str(ROOT)],
    check=True,
    env=env,
)
VALIDATION_INPUTS = ROOT / "preparation/data/evaluation/validation_inputs.jsonl"
VALIDATION_REFERENCES = (
    ROOT / "preparation/data/scoring_only/validation_references.jsonl"
)
assert VALIDATION_INPUTS.exists()
assert VALIDATION_REFERENCES.exists()
print(" Revision-4 validation data reconstructed.")
print(" Sealed test is NOT read by this notebook.")
from pathlib import Path, PurePosixPath
import subprocess, sys, os, json, zipfile, shutil, hashlib

ROOT = Path("/content/subjectesis_dev_eval_v1")


def run(cmd, env=None):
    """Print and execute a setup command, using the supplied environment when
    provided. Stop on a nonzero exit code so later evaluation stages cannot
    silently continue after failed preparation.
    """
    cmd = [str(x) for x in cmd]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
    check=False,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
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
env["PYTHONPATH"] = str(ROOT / "code") + ":" + env.get("PYTHONPATH", "")
env["TOKENIZERS_PARALLELISM"] = "false"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
run([sys.executable, ROOT / "code/download_model.py", "--root", ROOT], env=env)
ADAPTER_ROOT = ROOT / "adapters"
ADAPTER_ROOT.mkdir(exist_ok=True)


def extract_best_adapter(archive, condition, destination):
    """Extract the selected condition best-adapter files into a fresh local
    directory. Require both configuration and weights and reject
    parent-directory paths so inference uses the intended saved checkpoint.
    """
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    prefix = f"runs/{condition}/best_adapter/"
    with zipfile.ZipFile(archive) as z:
        names = [
            n for n in z.namelist() if n.startswith(prefix) and (not n.endswith("/"))
        ]
        required = {
            prefix + "adapter_config.json",
            prefix + "adapter_model.safetensors",
        }
        if not required.issubset(set(names)):
            raise RuntimeError(f"Best adapter missing from {archive}")
        for name in names:
            rel = PurePosixPath(name).relative_to(PurePosixPath(prefix))
            if ".." in rel.parts:
                raise RuntimeError("Unsafe adapter path")
            target = destination.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))


SUBJECTESIS_ADAPTER = ADAPTER_ROOT / "subjectesis_best"
ANSWER_ONLY_ADAPTER = ADAPTER_ROOT / "answer_only_best"
extract_best_adapter(SUBJECTESIS_ZIP, "subjectesis", SUBJECTESIS_ADAPTER)
extract_best_adapter(ANSWER_ONLY_ZIP, "ordinary", ANSWER_ONLY_ADAPTER)
print(
    " Best Subjectesis adapter extracted from step", subjectesis_completed["best_step"]
)
print(" Best answer-only adapter extracted from step", ordinary_completed["best_step"])
from pathlib import Path
import json, sys, os, time, math
import numpy as np
import torch

ROOT = Path("/content/subjectesis_dev_eval_v1")
sys.path.insert(0, str(ROOT / "code"))
from common import read_json, read_jsonl
from model_runtime import build_network
from engine import CompletionLoss
from safetensors.torch import load_file
from peft import set_peft_model_state_dict
from transformers import AutoTokenizer

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
if not torch.cuda.is_available():
    raise RuntimeError("Use an A100 GPU runtime.")
if "A100" not in torch.cuda.get_device_name(0).upper():
    raise RuntimeError("This evaluation notebook is configured for A100.")
c = read_json(ROOT / "settings.json")
model_path = Path(read_json(ROOT / "reports/model_download.json")["snapshot"])
tok = AutoTokenizer.from_pretrained(
    model_path, local_files_only=True, trust_remote_code=False
)
tok.padding_side = "right"
network = build_network(ROOT, c, 0, model_path)
network.eval()
try:
    network.gradient_checkpointing_disable()
except Exception:
    pass
network.config.use_cache = True
wrapper = CompletionLoss(network, c["loss_chunk_size"])
wrapper.eval()
validation_rows = read_jsonl(VALIDATION_INPUTS)
validation_refs = {r["record_id"]: r for r in read_jsonl(VALIDATION_REFERENCES)}
if len(validation_rows) != 319 or len(validation_refs) != 319:
    raise RuntimeError("Validation coverage changed")
letter_ids = {k: tok.encode(k, add_special_tokens=False) for k in "ABCDEFG"}
if any((len(v) != 1 for v in letter_ids.values())):
    raise RuntimeError("Answer letters are no longer single Qwen tokens")


def chat_ids(user_prompt):
    """Apply the Qwen chat template with thinking disabled and return token IDs
    for one user prompt. Reject overlength inputs instead of truncating
    dialogue evidence before answer selection.
    """
    text = tok.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) > c["max_length"]:
        raise RuntimeError("Validation prompt exceeds max length")
    return ids


def score_one_prompt(prompt, allowed):
    """Compute next-token vocabulary logits and choose the highest-scoring allowed
    option letter. This gives a constrained forced-choice answer without
    sampling text or consulting the reference label.
    """
    ids = torch.tensor([chat_ids(prompt)], device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        logits = wrapper.next_logits(ids)[0]
    vals = [float(logits[letter_ids[k][0]]) for k in allowed]
    return allowed[int(np.argmax(vals))]


def load_adapter(adapter_dir):
    """Load saved LoRA tensors into the existing default adapter and switch the
    network to evaluation mode. Return the tensor count and loading message so
    validation can record which adapter was loaded.
    """
    state = load_file(Path(adapter_dir) / "adapter_model.safetensors")
    result = set_peft_model_state_dict(network, state, adapter_name="default")
    network.eval()
    return (len(state), str(result))


def direct_eval(system_name, adapter_dir=None, disable_adapter=False):
    """Run constrained answer-only validation using the base model or a selected
    adapter. Compare each prediction with its validation reference and return
    per-question records used to reproduce the original checkpoint scores.
    """
    if adapter_dir is not None:
        n, result = load_adapter(adapter_dir)
        print(f"{system_name}: loaded {n} adapter tensors")
    out = []
    started = time.time()
    ctx = network.disable_adapter() if disable_adapter else None
    if ctx is not None:
        ctx.__enter__()
    try:
        for i, row in enumerate(validation_rows, 1):
            pred = score_one_prompt(
                row["answer_only_prompt"], list(row["choices"].keys())
            )
            ref = validation_refs[row["record_id"]]["answer"]
            out.append(
                {
                    "system": system_name,
                    "record_id": row["record_id"],
                    "dialogue_id": row["dialogue_id"],
                    "task": row["task"],
                    "prediction": pred,
                    "reference": ref,
                    "correct": pred == ref,
                }
            )
            if i % 50 == 0:
                print(f"{system_name}: {i}/319")
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    print(f"{system_name}: {time.time() - started:.1f}s")
    return out


base_direct = direct_eval("base_direct", disable_adapter=True)
answer_direct = direct_eval("answer_only_direct", adapter_dir=ANSWER_ONLY_ADAPTER)
subjectesis_direct = direct_eval("subjectesis_direct", adapter_dir=SUBJECTESIS_ADAPTER)


def task_accuracy(rows, task):
    """Return the fraction of correct predictions for one task family. Separating
    belief and desire makes their unequal question counts visible before
    calculating the equal-family mean.
    """
    x = [r for r in rows if r["task"] == task]
    return sum((r["correct"] for r in x)) / len(x)


def selection_score(rows):
    """Average belief accuracy and desire accuracy with equal weight. The
    historical name refers to validation checkpoint selection; on the final
    test it is only a reported task-mean score.
    """
    return (task_accuracy(rows, "belief") + task_accuracy(rows, "desire")) / 2


def macro_f1(rows, task):
    """Compute F1 for each label appearing in the references or predictions of the
    selected task, then average those values. This preserves the original
    evaluator convention; the later CPU analysis explicitly fixes the
    target-class set.
    """
    x = [r for r in rows if r["task"] == task]
    labels = sorted(
        set((r["reference"] for r in x)) | set((r["prediction"] for r in x))
    )
    f1s = []
    for lab in labels:
        tp = sum((r["reference"] == lab and r["prediction"] == lab for r in x))
        fp = sum((r["reference"] != lab and r["prediction"] == lab for r in x))
        fn = sum((r["reference"] == lab and r["prediction"] != lab for r in x))
        p = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * rec / (p + rec) if p + rec else 0.0)
    return sum(f1s) / len(f1s)


def direct_summary(name, rows):
    """Collect belief and desire accuracy, their equal-weight mean and task macro
    F1 for one direct condition. Return a named summary suitable for comparing
    the base and trained adapters.
    """
    return {
        "system": name,
        "belief_accuracy": task_accuracy(rows, "belief"),
        "desire_accuracy": task_accuracy(rows, "desire"),
        "selection_score": selection_score(rows),
        "belief_macro_f1": macro_f1(rows, "belief"),
        "desire_macro_f1": macro_f1(rows, "desire"),
    }


direct_metrics = [
    direct_summary("base_direct", base_direct),
    direct_summary("answer_only_direct", answer_direct),
    direct_summary("subjectesis_direct", subjectesis_direct),
]
print("\n=== DIRECT VALIDATION METRICS ===")
for m in direct_metrics:
    print(m)


def saved_prediction_map(saved):
    """Index the selected checkpoint validation predictions by record ID. This
    lets the evaluation compare answers question by question without depending
    on stored row order.
    """
    return {r["record_id"]: r["prediction"] for r in saved["predictions"]}


# Before developing the structured controller, check that direct answers
# match the best-checkpoint validation predictions saved during training.
for name, observed, saved in [
    ("answer-only", answer_direct, ordinary_saved_best),
    ("Subjectesis", subjectesis_direct, subjectesis_saved_best),
]:
    obs = {r["record_id"]: r["prediction"] for r in observed}
    exp = saved_prediction_map(saved)
    mismatch = [rid for rid in exp if obs[rid] != exp[rid]]
    print(f"{name} reproduction mismatches:", len(mismatch))
    if mismatch:
        raise RuntimeError(
            f"{name} best-adapter validation did not reproduce. First mismatches: {mismatch[:5]}"
        )
print(
    "Both trained best adapters exactly reproduce their saved validation predictions."
)
import json, re, hashlib, copy, time
from collections import Counter, defaultdict

load_adapter(SUBJECTESIS_ADAPTER)
network.eval()
network.config.use_cache = True
REVIEW_INSTRUCTION = "Review the supplied evidence. Return JSON with field, decision, updated_claim, evidence and stop_reason. Revise supported mistakes. If neither known value is supported, withdraw the claim to unknown; otherwise preserve a supported claim or retain unknown. Do not add facts from later turns."
FINALIZER_INSTRUCTION = "Use this perspective state as evidence bookkeeping. Do not invent facts that are not in the supplied dialogue. Choose the forced-choice option best supported by the dialogue and this state. Return only the selected option letter."
FIELD_GAPS = {
    "proposer": "Check who introduced the target movie into the discussion.",
    "seen": "Check whether the seeker has watched the target movie.",
    "recommendation_response": "Check whether the seeker accepts or declines this recommendation.",
    "appraisal": "Check the seeker's positive or negative attitude toward the target movie.",
    "intends_to_watch": "Check the seeker's likely willingness to watch, a prediction rather than a guaranteed future action.",
}
ALLOWED_VALUES = {
    "proposer": {"recommender", "seeker", "unknown"},
    "seen": {"yes", "no", "unknown"},
    "recommendation_response": {"accepted", "declined", "unknown"},
    "appraisal": {"likes", "dislikes", "unknown"},
    "intends_to_watch": {"yes", "no", "unknown"},
}
REQUIRED_FIELDS = {
    "belief": ["proposer", "seen", "recommendation_response", "appraisal"],
    "desire": ["intends_to_watch"],
}
PROTOCOL_CONFIG = {
    "version": "subjectesis_dev_controller_v1",
    "initial_prompt_source": "revision4 validation_inputs.subjectesis_prompt",
    "review_instruction": REVIEW_INSTRUCTION,
    "field_gaps": FIELD_GAPS,
    "finalizer_instruction": FINALIZER_INSTRUCTION,
    "initial_max_new_tokens": 700,
    "review_max_new_tokens": 320,
    "generation": "greedy_do_sample_false_enable_thinking_false",
    "belief_review_order": "proposer -> seen -> recommendation_response if reviewed proposer=recommender; appraisal if proposer=seeker; both if proposer remains unknown",
    "desire_review_order": "intends_to_watch",
    "review_merge": "always apply valid updated_claim, including decision=preserve",
    "invalid_initial_policy": "count invalid; no label-based repair; denominator retained",
    "invalid_review_policy": "record invalid review; keep prior claim; continue bounded plan",
    "final_answer": "same constrained next-letter finalizer before and after review",
    "labels_used_by_controller": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL_CONFIG, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
print("Protocol candidate hash:", PROTOCOL_HASH)


def extract_first_json_object(text):
    """Locate the first complete brace-delimited object while tracking quoted
    strings and escaped characters. Decode it as JSON or return None, so
    surrounding prose can be tolerated without repairing malformed model
    content.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                raw = text[start : i + 1]
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return None
    return None


def generate_text(user_prompt, max_new_tokens):
    """Generate one structured response greedily with the frozen Qwen model and
    decode only its continuation. Enforce the input-length limit and requested
    output-token budget without using gradient updates.
    """
    text = tok.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) > c["max_length"]:
        raise RuntimeError("Structured prompt exceeds training max length")
    x = torch.tensor([ids], device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        out = network.generate(
            input_ids=x,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0, len(ids) :], skip_special_tokens=True).strip()


def validate_initial(obj, task, allowed_letters):
    """Check required top-level keys, task fields, allowed values and basic types
    in an initial generated state. Return a validity flag and reason; passing
    this schema check does not establish that the belief or its evidence is
    true.
    """
    if not isinstance(obj, dict):
        return (False, "not_object")
    required_top = {
        "perspective",
        "evidence",
        "state",
        "missing_information",
        "answer",
        "answer_claims_not_established",
    }
    if not required_top.issubset(obj):
        return (False, "missing_top_level_key")
    if obj["answer"] not in allowed_letters:
        return (False, "invalid_answer_letter")
    if not isinstance(obj["state"], dict):
        return (False, "state_not_object")
    for field in REQUIRED_FIELDS[task]:
        claim = obj["state"].get(field)
        if not isinstance(claim, dict):
            return (False, f"missing_claim:{field}")
        if claim.get("value") not in ALLOWED_VALUES[field]:
            return (False, f"invalid_value:{field}")
        if not isinstance(claim.get("evidence_turns"), list):
            return (False, f"evidence_turns_not_list:{field}")
        if claim.get("basis") not in {"explicit", "inferred", "not_established"}:
            return (False, f"invalid_basis:{field}")
        if not isinstance(claim.get("reason"), str):
            return (False, f"missing_reason:{field}")
    if not isinstance(obj["evidence"], list):
        return (False, "evidence_not_list")
    return (True, "ok")


def validate_review(obj, field):
    """Check that a generated review addresses the selected field and has an
    allowed decision, updated claim and evidence list. Return a validity flag
    and reason so malformed updates can be recorded while retaining the
    previous state.
    """
    if not isinstance(obj, dict):
        return (False, "not_object")
    required = {"field", "decision", "updated_claim", "evidence", "stop_reason"}
    if not required.issubset(obj):
        return (False, "missing_key")
    if obj["field"] != field:
        return (False, "wrong_field")
    if obj["decision"] not in {"revise", "preserve", "retain_unknown"}:
        return (False, "invalid_decision")
    claim = obj["updated_claim"]
    if not isinstance(claim, dict):
        return (False, "claim_not_object")
    if claim.get("value") not in ALLOWED_VALUES[field]:
        return (False, "invalid_value")
    if not isinstance(claim.get("evidence_turns"), list):
        return (False, "evidence_turns_not_list")
    if claim.get("basis") not in {"explicit", "inferred", "not_established"}:
        return (False, "invalid_basis")
    if not isinstance(claim.get("reason"), str):
        return (False, "missing_reason")
    if not isinstance(obj["evidence"], list):
        return (False, "evidence_not_list")
    return (True, "ok")


def current_state_for_review(state, task):
    """Deep-copy only the fields required by the current question task. Review and
    finalizer prompts use this task-specific view without sharing mutable state
    objects or adding reference answers.
    """
    return {k: copy.deepcopy(state[k]) for k in REQUIRED_FIELDS[task] if k in state}


def make_review_prompt(row, state, field):
    """Combine the original question, current task state and instruction for one
    selected field. The same frozen wording is used throughout the review
    sequence, making each update traceable to the supplied dialogue and prior
    state.
    """
    return (
        row["prompt"]
        + "\nCurrent perspective state (may contain mistakes): "
        + json.dumps(current_state_for_review(state, row["task"]), ensure_ascii=False)
        + "\nSelected gap: "
        + FIELD_GAPS[field]
        + "\n"
        + REVIEW_INSTRUCTION
    )


def finalizer_prediction(row, state):
    """Build a prompt containing the question and the current state, then select
    an allowed answer letter from model logits. Applying the same rule before
    and after review isolates the effect of the updated state on the final
    answer.
    """
    prompt = (
        row["prompt"]
        + "\nPerspective state: "
        + json.dumps(current_state_for_review(state, row["task"]), ensure_ascii=False)
        + "\n"
        + FINALIZER_INSTRUCTION
    )
    return score_one_prompt(prompt, list(row["choices"].keys()))


def review_fields(task, state):
    """Choose the task-relevant review fields from the current proposer value.
    Desire uses intends_to_watch; belief uses proposer and seen plus
    recommendation_response, appraisal, or both when the proposer remains
    unknown.
    """
    if task == "desire":
        return ["intends_to_watch"]
    fields = ["proposer", "seen"]
    prop = state.get("proposer", {}).get("value")
    if prop == "recommender":
        fields.append("recommendation_response")
    elif prop == "seeker":
        fields.append("appraisal")
    else:
        fields.extend(["recommendation_response", "appraisal"])
    return fields


def normalize_ws(x):
    """Collapse whitespace in a value after converting it to text. Citation checks
    use this to ignore formatting differences between a generated quotation and
    its source turn.
    """
    return " ".join(str(x).split())


def dialogue_turns(row):
    """Extract numbered dialogue lines from the common prompt into a turn-number
    lookup. Evidence checks use this lookup to match a generated citation with
    the actual supplied turn.
    """
    block = row["prompt"].split("Dialogue:\n", 1)[1].split("\nQuestion:", 1)[0]
    turns = {}
    for line in block.splitlines():
        m = re.match("^(\\d+):\\s*(.*)$", line)
        if m:
            turns[int(m.group(1))] = m.group(2)
    return turns


def evidence_binding_audit(evidence, turns):
    """Count citations whose turn exists and whose normalized quote equals or
    contains that source-turn text. This records textual source binding after
    generation; despite the historical all_exact key, it allows containing text
    and does not test semantic entailment.
    """
    if not isinstance(evidence, list):
        return {"items": 0, "valid_items": 0, "all_exact": False}
    valid = 0
    for item in evidence:
        if not isinstance(item, dict):
            continue
        turn = item.get("turn")
        quote = item.get("quote")
        if (
            not isinstance(turn, int)
            or turn not in turns
            or (not isinstance(quote, str))
        ):
            continue
        q = normalize_ws(quote)
        source = normalize_ws(turns[turn])
        # The recorded rule accepts a quote containing the full source turn.
        # This checks textual binding only, not whether the turn supports the claim.
        if q == source or q.endswith(source) or source in q:
            valid += 1
    return {
        "items": len(evidence),
        "valid_items": valid,
        "all_exact": valid == len(evidence),
    }


def state_contract_audit(state, task, turns):
    """Check each field against the known/unknown evidence rules and legal turn
    numbers. Return per-field flags without altering predictions, keeping
    structural consistency separate from semantic correctness.
    """
    results = {}
    for field in REQUIRED_FIELDS[task]:
        claim = state.get(field, {})
        value = claim.get("value")
        evidence_turns = claim.get("evidence_turns", [])
        basis = claim.get("basis")
        valid_turns = isinstance(evidence_turns, list) and all(
            (isinstance(t, int) and t in turns for t in evidence_turns)
        )
        if value == "unknown":
            contract = (
                evidence_turns == [] and basis == "not_established" and valid_turns
            )
        else:
            contract = (
                bool(evidence_turns)
                and basis in {"explicit", "inferred"}
                and valid_turns
            )
        results[field] = bool(contract)
    return results


def run_structured_case(row):
    """Run one sequential validation case through initial-state construction,
    no-review finalization and bounded field review. Update only valid claims,
    select later belief fields from the reviewed proposer, and return both
    answers plus the complete state/evidence trace.
    """
    ref = validation_refs[row["record_id"]]["answer"]
    allowed = list(row["choices"].keys())
    raw_initial = generate_text(
        row["subjectesis_prompt"], PROTOCOL_CONFIG["initial_max_new_tokens"]
    )
    initial = extract_first_json_object(raw_initial)
    valid_initial, initial_error = validate_initial(initial, row["task"], allowed)
    result = {
        "record_id": row["record_id"],
        "dialogue_id": row["dialogue_id"],
        "task": row["task"],
        "reference": ref,
        "initial_valid": valid_initial,
        "initial_error": initial_error,
        "raw_initial": raw_initial,
        "initial_json_answer": initial.get("answer") if valid_initial else None,
        "no_review_prediction": None,
        "reviewed_prediction": None,
        "state_before": None,
        "state_after": None,
        "reviews": [],
    }
    if not valid_initial:
        return result
    state = copy.deepcopy(initial["state"])
    result["state_before"] = copy.deepcopy(state)
    result["no_review_prediction"] = finalizer_prediction(row, state)
    if row["task"] == "desire":
        plan = ["intends_to_watch"]
    else:
        plan = ["proposer", "seen"]
    idx = 0
    while idx < len(plan):
        field = plan[idx]
        prompt = make_review_prompt(row, state, field)
        raw = generate_text(prompt, PROTOCOL_CONFIG["review_max_new_tokens"])
        obj = extract_first_json_object(raw)
        valid, err = validate_review(obj, field)
        review_record = {
            "field": field,
            "valid": valid,
            "error": err,
            "raw": raw,
            "parsed": obj if valid else None,
        }
        if valid:
            state[field] = copy.deepcopy(obj["updated_claim"])
        result["reviews"].append(review_record)
        if row["task"] == "belief" and field == "proposer":
            prop = state.get("proposer", {}).get("value")
            if prop == "recommender":
                plan.append("recommendation_response")
            elif prop == "seeker":
                plan.append("appraisal")
            else:
                plan.extend(["recommendation_response", "appraisal"])
        idx += 1
    result["state_after"] = copy.deepcopy(state)
    result["reviewed_prediction"] = finalizer_prediction(row, state)
    turns = dialogue_turns(row)
    result["initial_evidence_binding"] = evidence_binding_audit(
        initial["evidence"], turns
    )
    result["initial_state_contract"] = state_contract_audit(
        result["state_before"], row["task"], turns
    )
    for rr in result["reviews"]:
        if rr["valid"]:
            rr["evidence_binding"] = evidence_binding_audit(
                rr["parsed"]["evidence"], turns
            )
    return result


SMOKE_N = 8
structured_smoke = []
for i, row in enumerate(validation_rows[:SMOKE_N], 1):
    print(f"Structured smoke {i}/{SMOKE_N}: {row['record_id']}")
    structured_smoke.append(run_structured_case(row))
smoke_invalid = sum((not r["initial_valid"] for r in structured_smoke))
print("Smoke invalid initial outputs:", smoke_invalid, "/", SMOKE_N)
if smoke_invalid > SMOKE_N // 2:
    raise RuntimeError(
        "More than half of structured smoke outputs were invalid. Do not run the full evaluation; inspect raw outputs first."
    )
print("Structured controller smoke completed.")
from pathlib import Path
import json, os, shutil, time, copy, torch

EXPECTED_PROTOCOL_HASH = (
    "ffa671e7007de68947288d3043d7975917ec2435c4366b7c212a8e9fcbc090ef"
)
if PROTOCOL_HASH != EXPECTED_PROTOCOL_HASH:
    raise RuntimeError(
        f"Protocol changed: {PROTOCOL_HASH}\nExpected: {EXPECTED_PROTOCOL_HASH}\nSTOP. Do not evaluate under a different protocol."
    )
network.eval()
network.config.use_cache = True
DRIVE_REPORT_DIR = DRIVE_DIR / "dev_evaluation"
DRIVE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
PARTIAL_DRIVE = DRIVE_REPORT_DIR / "structured_validation_partial.jsonl"
META_DRIVE = DRIVE_REPORT_DIR / "structured_validation_partial.meta.json"
PARTIAL_LOCAL = ROOT / "reports" / "structured_validation_partial.jsonl"
CHUNK_SIZE = 16
PREFERRED_GENERATION_BATCH = 8
validation_by_id = {r["record_id"]: r for r in validation_rows}
validation_order = [r["record_id"] for r in validation_rows]


def _chat_input_ids(user_prompt):
    """Apply the frozen chat template and encode one prompt with thinking
    disabled. Enforce the input-length limit so batching never silently removes
    dialogue or review context.
    """
    text = tok.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) > c["max_length"]:
        raise RuntimeError(
            f"Prompt has {len(ids)} input tokens; max_length={c['max_length']}"
        )
    return ids


def _generate_batch_once(prompts, max_new_tokens):
    """Left-pad independent prompts, mask the padding and run one greedy
    generation batch. Return only newly generated text in prompt order, keeping
    each example context separate while sharing GPU work.
    """
    encoded = [_chat_input_ids(p) for p in prompts]
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id
    if pad_id is None:
        raise RuntimeError("Tokenizer has no pad/eos token.")
    max_input = max((len(x) for x in encoded))
    batch_size = len(encoded)
    input_ids = torch.full(
        (batch_size, max_input), int(pad_id), dtype=torch.long, device="cuda"
    )
    attention_mask = torch.zeros(
        (batch_size, max_input), dtype=torch.long, device="cuda"
    )
    for i, ids in enumerate(encoded):
        n = len(ids)
        input_ids[i, -n:] = torch.as_tensor(ids, dtype=torch.long, device="cuda")
        attention_mask[i, -n:] = 1
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        generated = network.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=int(pad_id),
            eos_token_id=tok.eos_token_id,
        )
    continuation = generated[:, max_input:]
    texts = [tok.decode(row, skip_special_tokens=True).strip() for row in continuation]
    del (input_ids, attention_mask, generated, continuation)
    return texts


def batched_generate_text(
    prompts, max_new_tokens, preferred=PREFERRED_GENERATION_BATCH
):
    """Generate responses in batches and halve a batch after an out-of-memory
    error until it fits. Preserve prompt order and decoding settings, and raise
    the error if even one example cannot fit.
    """
    if not prompts:
        return []
    outputs = [None] * len(prompts)
    index = 0
    current_batch = min(preferred, len(prompts))
    while index < len(prompts):
        size = min(current_batch, len(prompts) - index)
        while True:
            try:
                result = _generate_batch_once(
                    prompts[index : index + size], max_new_tokens
                )
                outputs[index : index + size] = result
                index += size
                current_batch = preferred
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if size == 1:
                    raise
                size = max(1, size // 2)
                current_batch = size
                print(f"Generation OOM -> retry batch {size}", flush=True)
    return outputs


if len(structured_smoke) >= 2:
    probe_rows = [
        validation_by_id[structured_smoke[0]["record_id"]],
        validation_by_id[structured_smoke[1]["record_id"]],
    ]
    probe_initial = batched_generate_text(
        [r["subjectesis_prompt"] for r in probe_rows],
        PROTOCOL_CONFIG["initial_max_new_tokens"],
        preferred=2,
    )
    for produced, smoke in zip(probe_initial, structured_smoke[:2]):
        produced_json = extract_first_json_object(produced)
        smoke_json = extract_first_json_object(smoke["raw_initial"])
        if produced_json != smoke_json:
            raise RuntimeError(
                "Batched initial generation did not reproduce the sequential smoke result."
            )
    review_prompts = []
    review_expected = []
    for smoke in structured_smoke:
        if smoke["initial_valid"] and smoke["reviews"]:
            row = validation_by_id[smoke["record_id"]]
            first_field = smoke["reviews"][0]["field"]
            review_prompts.append(
                make_review_prompt(
                    row, copy.deepcopy(smoke["state_before"]), first_field
                )
            )
            review_expected.append(smoke["reviews"][0]["raw"])
        if len(review_prompts) == 2:
            break
    if len(review_prompts) == 2:
        produced_reviews = batched_generate_text(
            review_prompts, PROTOCOL_CONFIG["review_max_new_tokens"], preferred=2
        )
        for produced, expected in zip(produced_reviews, review_expected):
            if extract_first_json_object(produced) != extract_first_json_object(
                expected
            ):
                raise RuntimeError(
                    "Batched review generation did not reproduce the sequential smoke result."
                )
print("Batched execution reproduced the smoke-tested protocol.")
done = {}
if PARTIAL_DRIVE.exists():
    if not META_DRIVE.exists():
        raise RuntimeError("Found partial results but no protocol metadata.")
    meta = json.loads(META_DRIVE.read_text())
    if meta.get("protocol_hash") != PROTOCOL_HASH:
        raise RuntimeError("Saved progress belongs to a different protocol.")
    if int(meta.get("subjectesis_best_step", -1)) != int(
        subjectesis_completed["best_step"]
    ):
        raise RuntimeError("Saved progress belongs to a different Subjectesis adapter.")
    for line in PARTIAL_DRIVE.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        result = json.loads(line)
        rid = result["record_id"]
        if rid not in validation_by_id:
            raise RuntimeError(f"Unknown record: {rid}")
        if rid in done:
            raise RuntimeError(f"Duplicate record: {rid}")
        done[rid] = result
    print(f" Resumed {len(done)}/319 cases from Google Drive.")
for result in structured_smoke:
    done.setdefault(result["record_id"], result)


def save_progress():
    """Save completed validation records in their original order to local storage
    and Google Drive using temporary files. Store protocol and checkpoint
    metadata alongside them so a later session can resume the same evaluation.
    """
    ordered = [done[rid] for rid in validation_order if rid in done]
    PARTIAL_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    local_tmp = PARTIAL_LOCAL.with_suffix(".tmp")
    with local_tmp.open("w", encoding="utf8") as f:
        for result in ordered:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    os.replace(local_tmp, PARTIAL_LOCAL)
    drive_tmp = PARTIAL_DRIVE.with_suffix(".tmp")
    shutil.copy2(PARTIAL_LOCAL, drive_tmp)
    os.replace(drive_tmp, PARTIAL_DRIVE)
    metadata = {
        "protocol_hash": PROTOCOL_HASH,
        "subjectesis_best_step": int(subjectesis_completed["best_step"]),
        "completed_records": len(ordered),
        "validation_total": len(validation_rows),
        "sealed_test_scored": False,
    }
    meta_tmp = META_DRIVE.with_suffix(".tmp")
    meta_tmp.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(meta_tmp, META_DRIVE)


save_progress()
print(f"Persistent starting progress: {len(done)}/319")


def process_chunk(rows):
    """Run the same bounded controller on several independent validation
    questions, batching one review step at a time. Preserve each initial state
    for the no-review comparison, apply valid updates and return answers,
    review traces and evidence checks with validation references for reporting.
    """
    raw_initials = batched_generate_text(
        [row["subjectesis_prompt"] for row in rows],
        PROTOCOL_CONFIG["initial_max_new_tokens"],
    )
    work = []
    for row, raw_initial in zip(rows, raw_initials):
        ref = validation_refs[row["record_id"]]["answer"]
        allowed = list(row["choices"].keys())
        initial = extract_first_json_object(raw_initial)
        valid_initial, initial_error = validate_initial(initial, row["task"], allowed)
        result = {
            "record_id": row["record_id"],
            "dialogue_id": row["dialogue_id"],
            "task": row["task"],
            "reference": ref,
            "initial_valid": valid_initial,
            "initial_error": initial_error,
            "raw_initial": raw_initial,
            "initial_json_answer": initial.get("answer") if valid_initial else None,
            "no_review_prediction": None,
            "reviewed_prediction": None,
            "state_before": None,
            "state_after": None,
            "reviews": [],
        }
        # Keep the invalid case in the result set with missing predictions.
        # Do not repair its state using a reference answer.
        if not valid_initial:
            work.append(
                {"row": row, "result": result, "state": None, "plan": [], "index": 0}
            )
            continue
        state = copy.deepcopy(initial["state"])
        result["state_before"] = copy.deepcopy(state)
        result["no_review_prediction"] = finalizer_prediction(row, state)
        if row["task"] == "desire":
            plan = ["intends_to_watch"]
        else:
            plan = ["proposer", "seen"]
        work.append(
            {"row": row, "result": result, "state": state, "plan": plan, "index": 0}
        )
    while True:
        active = [
            w for w in work if w["state"] is not None and w["index"] < len(w["plan"])
        ]
        if not active:
            break
        prompts = []
        fields = []
        for w in active:
            field = w["plan"][w["index"]]
            fields.append(field)
            prompts.append(make_review_prompt(w["row"], w["state"], field))
        raws = batched_generate_text(prompts, PROTOCOL_CONFIG["review_max_new_tokens"])
        for w, field, raw in zip(active, fields, raws):
            obj = extract_first_json_object(raw)
            valid, error = validate_review(obj, field)
            review_record = {
                "field": field,
                "valid": valid,
                "error": error,
                "raw": raw,
                "parsed": obj if valid else None,
            }
            # Apply every schema-valid updated claim, even when the decision is
            # preserve, because evidence or rationale can change without changing its value.
            if valid:
                w["state"][field] = copy.deepcopy(obj["updated_claim"])
            w["result"]["reviews"].append(review_record)
            # Choose later fields from the reviewed proposer. A recommender
            # requires response review; a seeker requires appraisal; unknown requires both.
            if w["row"]["task"] == "belief" and field == "proposer":
                prop = w["state"].get("proposer", {}).get("value")
                if prop == "recommender":
                    w["plan"].append("recommendation_response")
                elif prop == "seeker":
                    w["plan"].append("appraisal")
                else:
                    w["plan"].extend(["recommendation_response", "appraisal"])
            w["index"] += 1
    output = []
    for w in work:
        row = w["row"]
        result = w["result"]
        if w["state"] is not None:
            state = w["state"]
            result["state_after"] = copy.deepcopy(state)
            result["reviewed_prediction"] = finalizer_prediction(row, state)
            turns = dialogue_turns(row)
            initial_obj = extract_first_json_object(result["raw_initial"])
            result["initial_evidence_binding"] = evidence_binding_audit(
                initial_obj["evidence"], turns
            )
            result["initial_state_contract"] = state_contract_audit(
                result["state_before"], row["task"], turns
            )
            for rr in result["reviews"]:
                if rr["valid"]:
                    rr["evidence_binding"] = evidence_binding_audit(
                        rr["parsed"]["evidence"], turns
                    )
        output.append(result)
    return output


remaining = [row for row in validation_rows if row["record_id"] not in done]
print(f"Remaining validation cases: {len(remaining)}")
started = time.time()
new_completed = 0
for start in range(0, len(remaining), CHUNK_SIZE):
    chunk = remaining[start : start + CHUNK_SIZE]
    chunk_results = process_chunk(chunk)
    for result in chunk_results:
        done[result["record_id"]] = result
    new_completed += len(chunk_results)
    save_progress()
    elapsed = time.time() - started
    seconds_per_case = elapsed / max(1, new_completed)
    left = len(remaining) - new_completed
    eta_minutes = seconds_per_case * left / 60
    print(
        f" Persistent progress {len(done)}/319 | this session {new_completed}/{len(remaining)} | ETA ~{eta_minutes:.1f} min",
        flush=True,
    )
structured_results = [done[row["record_id"]] for row in validation_rows]
if len(structured_results) != len(validation_rows):
    raise RuntimeError("Structured evaluation coverage is incomplete.")
print("\nFULL STRUCTURED VALIDATION COMPLETE")
print(f"{len(structured_results)}/319")
print("Persistent trace:", PARTIAL_DRIVE)
print(" Sealed RecToM test remains untouched.")
from pathlib import Path
from collections import Counter, defaultdict
import json, math, random, statistics, hashlib
import pandas as pd
import matplotlib.pyplot as plt

REPORT_DIR = ROOT / "reports/dev_evaluation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_REPORT_DIR = DRIVE_DIR / "dev_evaluation"
DRIVE_REPORT_DIR.mkdir(parents=True, exist_ok=True)


def accuracy_from_structured(rows, key, task):
    """Calculate task accuracy using the requested answer field in the structured
    records. Missing or invalid predictions compare unequal to the reference
    and remain in the denominator.
    """
    x = [r for r in rows if r["task"] == task]
    return sum((r.get(key) == r["reference"] for r in x)) / len(x)


def structured_summary(key):
    """Summarize one structured answer stage with belief accuracy, desire accuracy
    and their mean. Selecting the before-review or after-review key provides
    directly comparable stage results.
    """
    return {
        "system": key,
        "belief_accuracy": accuracy_from_structured(structured_results, key, "belief"),
        "desire_accuracy": accuracy_from_structured(structured_results, key, "desire"),
        "selection_score": (
            accuracy_from_structured(structured_results, key, "belief")
            + accuracy_from_structured(structured_results, key, "desire")
        )
        / 2,
    }


no_review_metrics = structured_summary("no_review_prediction")
review_metrics = structured_summary("reviewed_prediction")
invalid_initial = sum((not r["initial_valid"] for r in structured_results))
review_calls = sum((len(r["reviews"]) for r in structured_results))
invalid_reviews = sum(
    (not rr["valid"] for r in structured_results for rr in r["reviews"])
)
transitions = Counter()
for r in structured_results:
    before = r.get("no_review_prediction") == r["reference"]
    after = r.get("reviewed_prediction") == r["reference"]
    if before and after:
        transitions["right_to_right"] += 1
    elif before and (not after):
        transitions["right_to_wrong"] += 1
    elif not before and after:
        transitions["wrong_to_right"] += 1
    else:
        transitions["wrong_to_wrong"] += 1
initial_evidence_items = 0
initial_evidence_valid = 0
initial_state_contracts = []
review_evidence_items = 0
review_evidence_valid = 0
for r in structured_results:
    if not r["initial_valid"]:
        continue
    e = r.get("initial_evidence_binding", {})
    initial_evidence_items += e.get("items", 0)
    initial_evidence_valid += e.get("valid_items", 0)
    initial_state_contracts.extend(r.get("initial_state_contract", {}).values())
    for rr in r["reviews"]:
        if rr["valid"] and "evidence_binding" in rr:
            e2 = rr["evidence_binding"]
            review_evidence_items += e2.get("items", 0)
            review_evidence_valid += e2.get("valid_items", 0)
source_binding = {
    "initial_evidence_exact_turn_binding_rate": (
        initial_evidence_valid / initial_evidence_items
        if initial_evidence_items
        else None
    ),
    "initial_state_known_unknown_contract_rate": (
        sum(initial_state_contracts) / len(initial_state_contracts)
        if initial_state_contracts
        else None
    ),
    "review_evidence_exact_turn_binding_rate": (
        review_evidence_valid / review_evidence_items if review_evidence_items else None
    ),
    "note": "These are structural source-binding checks only. They do not establish that a quotation semantically entails the claim.",
}


def dialogue_bootstrap_difference(rows_a, rows_b, metric_name, reps=5000, seed=42):
    """Pair two validation conditions by record ID and resample complete dialogues
    to estimate their task-mean accuracy difference. Return the observed
    difference and original sorted-percentile bounds from valid samples,
    preserving dependence between questions from the same dialogue.
    """
    a = {r["record_id"]: r for r in rows_a}
    b = {r["record_id"]: r for r in rows_b}
    if set(a) != set(b):
        raise RuntimeError("Paired systems have different validation coverage")
    joined = []
    for rid in a:
        joined.append(
            {
                "record_id": rid,
                "dialogue_id": a[rid]["dialogue_id"],
                "task": a[rid]["task"],
                "a_correct": bool(a[rid]["correct"]),
                "b_correct": bool(b[rid]["correct"]),
            }
        )
    by_dialogue = defaultdict(list)
    for r in joined:
        by_dialogue[r["dialogue_id"]].append(r)
    dialogues = sorted(by_dialogue)

    def score(sample, key):
        """Calculate equal-weight belief/desire accuracy for one resampled set and
        one system correctness field. Return None if either task is absent,
        allowing the outer bootstrap to skip that incomplete draw.
        """
        vals = {}
        for task in ["belief", "desire"]:
            x = [r for r in sample if r["task"] == task]
            if not x:
                return None
            vals[task] = sum((r[key] for r in x)) / len(x)
        return (vals["belief"] + vals["desire"]) / 2

    rng = random.Random(seed)
    diffs = []
    for _ in range(reps):
        sampled_ids = [rng.choice(dialogues) for _ in dialogues]
        sample = []
        for did in sampled_ids:
            sample.extend(by_dialogue[did])
        sa = score(sample, "a_correct")
        sb = score(sample, "b_correct")
        if sa is not None and sb is not None:
            diffs.append(sa - sb)
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))]
    point = selection_score(rows_a) - selection_score(rows_b)
    return {
        "comparison": metric_name,
        "point_difference": point,
        "ci95_dialogue_bootstrap": [lo, hi],
        "repetitions": len(diffs),
        "seed": seed,
    }


def structured_as_direct(key, name):
    """Convert a chosen structured answer stage into the common per-question
    prediction format. This lets the same paired comparison code evaluate
    direct answers, no-review answers and reviewed answers.
    """
    out = []
    for r in structured_results:
        pred = r.get(key)
        out.append(
            {
                "system": name,
                "record_id": r["record_id"],
                "dialogue_id": r["dialogue_id"],
                "task": r["task"],
                "prediction": pred,
                "reference": r["reference"],
                "correct": pred == r["reference"],
            }
        )
    return out


no_review_rows = structured_as_direct("no_review_prediction", "subjectesis_no_review")
review_rows = structured_as_direct("reviewed_prediction", "subjectesis_review")
bootstrap = [
    dialogue_bootstrap_difference(
        subjectesis_direct, answer_direct, "Subjectesis direct - Answer-only direct"
    ),
    dialogue_bootstrap_difference(
        review_rows, no_review_rows, "Subjectesis review - Subjectesis no-review"
    ),
]
metrics = {
    "direct": direct_metrics,
    "structured_no_review": no_review_metrics,
    "structured_review": review_metrics,
    "invalid_initial_outputs": invalid_initial,
    "invalid_initial_rate": invalid_initial / len(structured_results),
    "review_calls": review_calls,
    "invalid_reviews": invalid_reviews,
    "invalid_review_rate": invalid_reviews / review_calls if review_calls else None,
    "review_transitions": dict(transitions),
    "source_binding": source_binding,
    "bootstrap": bootstrap,
    "validation_examples": len(validation_rows),
    "protocol_hash": PROTOCOL_HASH,
    "sealed_test_scored": False,
}
print("\n=== DEVELOPMENT RESULTS ===")
print(json.dumps(metrics, indent=2))
with (REPORT_DIR / "metrics.json").open("w", encoding="utf8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
with (REPORT_DIR / "protocol_candidate.json").open("w", encoding="utf8") as f:
    json.dump(
        {
            "protocol_hash": PROTOCOL_HASH,
            "config": PROTOCOL_CONFIG,
            "status": "development_candidate_not_yet_sealed_tested",
            "sealed_test_scored": False,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )
pd.DataFrame(base_direct + answer_direct + subjectesis_direct).to_csv(
    REPORT_DIR / "direct_predictions.csv", index=False
)
with (REPORT_DIR / "structured_results.jsonl").open("w", encoding="utf8") as f:
    for r in structured_results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
for system_name, rows in [
    ("base_direct", base_direct),
    ("answer_only_direct", answer_direct),
    ("subjectesis_direct", subjectesis_direct),
    ("subjectesis_no_review", no_review_rows),
    ("subjectesis_review", review_rows),
]:
    for task in ["belief", "desire"]:
        x = [r for r in rows if r["task"] == task]
        table = pd.crosstab(
            pd.Series([r["reference"] for r in x], name="reference"),
            pd.Series([r["prediction"] for r in x], name="prediction"),
            dropna=False,
        )
        table.to_csv(REPORT_DIR / f"confusion_{system_name}_{task}.csv")
plot_rows = []
for m in direct_metrics:
    plot_rows.append((m["system"], "belief", m["belief_accuracy"]))
    plot_rows.append((m["system"], "desire", m["desire_accuracy"]))
plot_rows += [
    ("subjectesis_no_review", "belief", no_review_metrics["belief_accuracy"]),
    ("subjectesis_no_review", "desire", no_review_metrics["desire_accuracy"]),
    ("subjectesis_review", "belief", review_metrics["belief_accuracy"]),
    ("subjectesis_review", "desire", review_metrics["desire_accuracy"]),
]
plot_df = pd.DataFrame(plot_rows, columns=["system", "task", "accuracy"])
fig, ax = plt.subplots(figsize=(11, 5))
systems = list(dict.fromkeys(plot_df["system"]))
x = np.arange(len(systems))
width = 0.36
belief_vals = [
    float(
        plot_df[(plot_df.system == s) & (plot_df.task == "belief")]["accuracy"].iloc[0]
    )
    for s in systems
]
desire_vals = [
    float(
        plot_df[(plot_df.system == s) & (plot_df.task == "desire")]["accuracy"].iloc[0]
    )
    for s in systems
]
ax.bar(x - width / 2, belief_vals, width, label="belief")
ax.bar(x + width / 2, desire_vals, width, label="desire")
ax.set_ylim(0, 1)
ax.set_ylabel("Validation accuracy")
ax.set_xticks(x)
ax.set_xticklabels(systems, rotation=25, ha="right")
ax.legend()
fig.tight_layout()
fig.savefig(REPORT_DIR / "validation_system_comparison.png", dpi=180)
plt.show()
if DRIVE_REPORT_DIR.exists():
    pass
for p in REPORT_DIR.rglob("*"):
    if p.is_file():
        rel = p.relative_to(REPORT_DIR)
        target = DRIVE_REPORT_DIR / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
print("\n Development evaluation saved to:", DRIVE_REPORT_DIR)
print("Protocol candidate hash:", PROTOCOL_HASH)
print(" Sealed test remains untouched.")
