"""Run the frozen five-condition RecToM test on 621 questions.

The script verifies the development protocol and selected adapters, saves
direct predictions, then runs initial-state construction and bounded field
review. Reference answers are attached after predictions are frozen. Colab
paths and the original completion guards are retained; inspect the functions
below to follow the controller and scoring stages.
"""

from project_setup import mount_drive, prepare_project

mount_drive()
from pathlib import Path, PurePosixPath
import ast, base64, io, zipfile, json, hashlib, os, shutil, sys, subprocess

ROOT = Path("/content/subjectesis_final_test_v1")
DRIVE_DIR = Path("/content/drive/MyDrive/Subjectesis")
FINAL_DIR = DRIVE_DIR / "final_test"
FINAL_DIR.mkdir(parents=True, exist_ok=True)
SUBJECTESIS_ZIP = DRIVE_DIR / "subjectesis_a100_latest.zip"
ANSWER_ONLY_ZIP = DRIVE_DIR / "answer_only_a100_latest.zip"
DEV_PROTOCOL_FILE = DRIVE_DIR / "dev_evaluation/protocol_candidate.json"
COMPLETE_MARKER = FINAL_DIR / "FINAL_TEST_COMPLETE.json"
EXPECTED_PROTOCOL_HASH = (
    "ffa671e7007de68947288d3043d7975917ec2435c4366b7c212a8e9fcbc090ef"
)
EXPECTED_PROTOCOL_CONFIG = '{"version": "subjectesis_dev_controller_v1", "initial_prompt_source": "revision4 validation_inputs.subjectesis_prompt", "review_instruction": "Review the supplied evidence. Return JSON with field, decision, updated_claim, evidence and stop_reason. Revise supported mistakes. If neither known value is supported, withdraw the claim to unknown; otherwise preserve a supported claim or retain unknown. Do not add facts from later turns.", "field_gaps": {"proposer": "Check who introduced the target movie into the discussion.", "seen": "Check whether the seeker has watched the target movie.", "recommendation_response": "Check whether the seeker accepts or declines this recommendation.", "appraisal": "Check the seeker\'s positive or negative attitude toward the target movie.", "intends_to_watch": "Check the seeker\'s likely willingness to watch, a prediction rather than a guaranteed future action."}, "finalizer_instruction": "Use this perspective state as evidence bookkeeping. Do not invent facts that are not in the supplied dialogue. Choose the forced-choice option best supported by the dialogue and this state. Return only the selected option letter.", "initial_max_new_tokens": 700, "review_max_new_tokens": 320, "generation": "greedy_do_sample_false_enable_thinking_false", "belief_review_order": "proposer -> seen -> recommendation_response if reviewed proposer=recommender; appraisal if proposer=seeker; both if proposer remains unknown", "desire_review_order": "intends_to_watch", "review_merge": "always apply valid updated_claim, including decision=preserve", "invalid_initial_policy": "count invalid; no label-based repair; denominator retained", "invalid_review_policy": "record invalid review; keep prior claim; continue bounded plan", "final_answer": "same constrained next-letter finalizer before and after review", "labels_used_by_controller": false}'
EXPECTED_PROTOCOL_CONFIG = json.loads(EXPECTED_PROTOCOL_CONFIG)
for required in [SUBJECTESIS_ZIP, ANSWER_ONLY_ZIP, DEV_PROTOCOL_FILE]:
    if not required.exists():
        raise FileNotFoundError(f"Missing required Drive artifact: {required}")
if COMPLETE_MARKER.exists():
    FINAL_TEST_ALREADY_COMPLETE = True
    print(" FINAL TEST ALREADY COMPLETED.")
    print(json.loads(COMPLETE_MARKER.read_text()))
    print("Prediction cells will refuse to run again.")
else:
    FINAL_TEST_ALREADY_COMPLETE = False


def canonical_hash(obj):
    """Serialize protocol settings with sorted keys and compact separators, then
    return their checksum. The evaluator uses this stable identity to verify
    the exact development protocol before sealed-test inference.
    """
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


dev_protocol = json.loads(DEV_PROTOCOL_FILE.read_text())
if dev_protocol["protocol_hash"] != EXPECTED_PROTOCOL_HASH:
    raise RuntimeError("Drive development protocol hash changed.")
if dev_protocol["config"] != EXPECTED_PROTOCOL_CONFIG:
    raise RuntimeError("Drive development protocol config changed.")
if canonical_hash(dev_protocol["config"]) != EXPECTED_PROTOCOL_HASH:
    raise RuntimeError("Frozen protocol does not reproduce its own hash.")
if dev_protocol.get("sealed_test_scored") is not False:
    raise RuntimeError("Development protocol unexpectedly says test was scored.")
print("Frozen development protocol verified:", EXPECTED_PROTOCOL_HASH)


def critical_archive_check(path, condition):
    """Verify completion metadata and the best adapter configuration and weights
    against the recovery manifest. Require the recorded training exposure and
    absence of test scoring, then return the checkpoint-selection metadata.
    """
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        if "recovery_manifest.json" not in names:
            raise RuntimeError(f"{path.name} has no manifest.")
        manifest = json.loads(z.read("recovery_manifest.json"))["files"]
        completed_name = f"runs/{condition}/completed.json"
        if completed_name not in names:
            raise RuntimeError(f"{path.name} is not completed.")
        completed_bytes = z.read(completed_name)
        if hashlib.sha256(completed_bytes).hexdigest() != manifest[completed_name]:
            raise RuntimeError(f"Completion metadata hash mismatch in {path.name}")
        completed = json.loads(completed_bytes)
        if completed.get("status") != "completed":
            raise RuntimeError(f"{path.name} status is not completed.")
        if int(completed.get("examples_seen", -1)) != 30266:
            raise RuntimeError(f"{path.name} has wrong exposure count.")
        if completed.get("test_scored") is not False:
            raise RuntimeError(f"Training archive unexpectedly reports test scoring.")
        best_step = int(completed["best_step"])
        prefix = f"runs/{condition}/best_adapter/"
        for rel in ["adapter_config.json", "adapter_model.safetensors"]:
            name = prefix + rel
            if name not in names or name not in manifest:
                raise RuntimeError(f"Missing best adapter member {name}")
            data = z.read(name)
            if hashlib.sha256(data).hexdigest() != manifest[name]:
                raise RuntimeError(f"Best adapter hash mismatch: {name}")
        return completed


ordinary_completed = critical_archive_check(ANSWER_ONLY_ZIP, "ordinary")
subjectesis_completed = critical_archive_check(SUBJECTESIS_ZIP, "subjectesis")
print(" Answer-only completed; best step:", ordinary_completed["best_step"])
print(" Subjectesis completed; best step:", subjectesis_completed["best_step"])
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
env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT / "code") + ":" + env.get("PYTHONPATH", "")
subprocess.run(
    [sys.executable, str(ROOT / "code/prepare.py"), "--root", str(ROOT)],
    check=True,
    env=env,
)
TEST_INPUTS = ROOT / "preparation/data/evaluation/test_inputs.jsonl"
TEST_REFERENCES = ROOT / "preparation/data/scoring_only/test_references.jsonl"
if not TEST_INPUTS.exists() or not TEST_REFERENCES.exists():
    raise RuntimeError("Revision-4 test split was not reconstructed.")


def read_jsonl_local(path):
    """Read non-empty JSONL lines in file order. The evaluator uses this for
    frozen test inputs and saved records without rewriting their content.
    """
    with Path(path).open("r", encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


test_rows = read_jsonl_local(TEST_INPUTS)
if len(test_rows) != 621:
    raise RuntimeError(f"Expected 621 test questions, found {len(test_rows)}")
if len({str(r["dialogue_id"]) for r in test_rows}) != 67:
    raise RuntimeError("Expected 67 sealed test dialogues.")
TEST_INPUT_SHA256 = hashlib.sha256(TEST_INPUTS.read_bytes()).hexdigest()
TEST_REFERENCE_SHA256 = hashlib.sha256(TEST_REFERENCES.read_bytes()).hexdigest()
print("Sealed test inputs ready:", len(test_rows), "questions / 67 dialogues")
print(" Reference bytes were hashed; answers are not used by the prediction cells.")
from pathlib import Path, PurePosixPath
import subprocess, sys, os, json, zipfile, shutil

ROOT = Path("/content/subjectesis_final_test_v1")


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
print("Subjectesis best adapter:", subjectesis_completed["best_step"])
print("Answer-only best adapter:", ordinary_completed["best_step"])
from pathlib import Path
import json, sys, os, time, hashlib
import numpy as np
import torch

if FINAL_TEST_ALREADY_COMPLETE:
    raise RuntimeError(
        "Final test is already complete; refusing to regenerate predictions."
    )
ROOT = Path("/content/subjectesis_final_test_v1")
sys.path.insert(0, str(ROOT / "code"))
from common import read_json
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
    raise RuntimeError("Select an A100 GPU runtime.")
if "A100" not in torch.cuda.get_device_name(0).upper():
    raise RuntimeError(f"Expected A100, found {torch.cuda.get_device_name(0)}")
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
letter_ids = {k: tok.encode(k, add_special_tokens=False) for k in "ABCDEFG"}
if any((len(v) != 1 for v in letter_ids.values())):
    raise RuntimeError("Answer letters are no longer single Qwen tokens.")


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
        raise RuntimeError("Test prompt exceeds max length.")
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
    """Load one saved LoRA checkpoint into the default adapter and set evaluation
    mode. Reusing the same base network lets the direct conditions differ in
    adapter weights while retaining the model-loading setup.
    """
    state = load_file(Path(adapter_dir) / "adapter_model.safetensors")
    set_peft_model_state_dict(network, state, adapter_name="default")
    network.eval()


def prediction_meta(system):
    """Describe the system, frozen protocol, test-input hash, model revision and
    selected checkpoints for a prediction file. This metadata is checked before
    partial predictions are reused.
    """
    return {
        "system": system,
        "protocol_hash": EXPECTED_PROTOCOL_HASH,
        "test_input_sha256": TEST_INPUT_SHA256,
        "model_id": c["model_id"],
        "model_revision": c["model_revision"],
        "answer_only_best_step": int(ordinary_completed["best_step"]),
        "subjectesis_best_step": int(subjectesis_completed["best_step"]),
        "labels_read": False,
    }


def direct_paths(system):
    """Return the prediction and metadata paths for one direct system. Keeping the
    pair together makes saved-answer reuse conditional on the matching
    experiment description.
    """
    return (
        FINAL_DIR / f"{system}_predictions.jsonl",
        FINAL_DIR / f"{system}_predictions.meta.json",
    )


def load_existing_direct(system):
    """Load a partial direct prediction file only when its metadata exactly
    matches the current frozen run. Reject duplicate record IDs and return an
    ID-indexed dictionary so completed questions can be skipped.
    """
    path, meta_path = direct_paths(system)
    if not path.exists():
        return {}
    if not meta_path.exists():
        raise RuntimeError(f"{system}: prediction file exists without metadata.")
    if json.loads(meta_path.read_text()) != prediction_meta(system):
        raise RuntimeError(f"{system}: saved prediction metadata changed.")
    done = {}
    for line in path.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["record_id"] in done:
            raise RuntimeError(f"{system}: duplicate saved record.")
        done[r["record_id"]] = r
    return done


def save_direct(system, done):
    """Write completed direct predictions in test-input order and save their
    metadata through temporary files. Regular saves let interrupted inference
    resume without repeating completed questions.
    """
    path, meta_path = direct_paths(system)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf8") as f:
        for row in test_rows:
            rid = row["record_id"]
            if rid in done:
                f.write(json.dumps(done[rid], ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    meta_tmp = meta_path.with_suffix(".tmp")
    meta_tmp.write_text(json.dumps(prediction_meta(system), indent=2) + "\n")
    os.replace(meta_tmp, meta_path)


def run_direct(system, adapter_dir=None, disable_adapter=False):
    """Generate missing direct answers using the selected adapter or
    disabled-adapter base model and periodically save progress. Require
    complete coverage of 621 questions before returning the frozen predictions,
    without reading answer labels for generation.
    """
    if adapter_dir is not None:
        load_adapter(adapter_dir)
    done = load_existing_direct(system)
    print(f"{system}: resumed {len(done)}/621")
    ctx = network.disable_adapter() if disable_adapter else None
    if ctx is not None:
        ctx.__enter__()
    try:
        new_since_save = 0
        for i, row in enumerate(test_rows, 1):
            rid = row["record_id"]
            if rid in done:
                continue
            pred = score_one_prompt(
                row["answer_only_prompt"], list(row["choices"].keys())
            )
            done[rid] = {
                "system": system,
                "record_id": rid,
                "dialogue_id": str(row["dialogue_id"]),
                "task": row["task"],
                "prediction": pred,
            }
            new_since_save += 1
            if new_since_save >= 50:
                save_direct(system, done)
                print(f"{system}: persistent {len(done)}/621")
                new_since_save = 0
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    save_direct(system, done)
    if len(done) != 621:
        raise RuntimeError(f"{system}: incomplete prediction coverage.")
    print(f" {system}: 621/621 predictions frozen.")
    return [done[row["record_id"]] for row in test_rows]


base_direct = run_direct("base_direct", disable_adapter=True)
answer_direct = run_direct("answer_only_direct", adapter_dir=ANSWER_ONLY_ADAPTER)
subjectesis_direct = run_direct("subjectesis_direct", adapter_dir=SUBJECTESIS_ADAPTER)
load_adapter(SUBJECTESIS_ADAPTER)
print(" Direct predictions complete. Test labels still have NOT been read.")
import json, re, hashlib, copy, time
from collections import Counter, defaultdict

if FINAL_TEST_ALREADY_COMPLETE:
    raise RuntimeError(
        "Final test is already complete; refusing to rerun structured inference."
    )
PROTOCOL_CONFIG = '{"version": "subjectesis_dev_controller_v1", "initial_prompt_source": "revision4 validation_inputs.subjectesis_prompt", "review_instruction": "Review the supplied evidence. Return JSON with field, decision, updated_claim, evidence and stop_reason. Revise supported mistakes. If neither known value is supported, withdraw the claim to unknown; otherwise preserve a supported claim or retain unknown. Do not add facts from later turns.", "field_gaps": {"proposer": "Check who introduced the target movie into the discussion.", "seen": "Check whether the seeker has watched the target movie.", "recommendation_response": "Check whether the seeker accepts or declines this recommendation.", "appraisal": "Check the seeker\'s positive or negative attitude toward the target movie.", "intends_to_watch": "Check the seeker\'s likely willingness to watch, a prediction rather than a guaranteed future action."}, "finalizer_instruction": "Use this perspective state as evidence bookkeeping. Do not invent facts that are not in the supplied dialogue. Choose the forced-choice option best supported by the dialogue and this state. Return only the selected option letter.", "initial_max_new_tokens": 700, "review_max_new_tokens": 320, "generation": "greedy_do_sample_false_enable_thinking_false", "belief_review_order": "proposer -> seen -> recommendation_response if reviewed proposer=recommender; appraisal if proposer=seeker; both if proposer remains unknown", "desire_review_order": "intends_to_watch", "review_merge": "always apply valid updated_claim, including decision=preserve", "invalid_initial_policy": "count invalid; no label-based repair; denominator retained", "invalid_review_policy": "record invalid review; keep prior claim; continue bounded plan", "final_answer": "same constrained next-letter finalizer before and after review", "labels_used_by_controller": false}'
PROTOCOL_CONFIG = json.loads(PROTOCOL_CONFIG)


def canonical_hash(obj):
    """Serialize protocol settings with sorted keys and compact separators, then
    return their checksum. The evaluator uses this stable identity to verify
    the exact development protocol before sealed-test inference.
    """
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


PROTOCOL_HASH = canonical_hash(PROTOCOL_CONFIG)
if PROTOCOL_HASH != EXPECTED_PROTOCOL_HASH:
    raise RuntimeError("Frozen controller protocol hash mismatch.")
REVIEW_INSTRUCTION = PROTOCOL_CONFIG["review_instruction"]
FINALIZER_INSTRUCTION = PROTOCOL_CONFIG["finalizer_instruction"]
FIELD_GAPS = PROTOCOL_CONFIG["field_gaps"]
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


# Text binding is a structural audit; it does not decide semantic support.
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


print(" Frozen controller loaded.")
print("Protocol hash:", PROTOCOL_HASH)
print(" Test reference bytes were hashed; answers are not used for generation.")
from pathlib import Path
import json, os, shutil, time, copy, torch

if FINAL_TEST_ALREADY_COMPLETE:
    raise RuntimeError(
        "Final test already complete; refusing to regenerate predictions."
    )
STRUCTURED_PATH = FINAL_DIR / "structured_test_predictions.jsonl"
STRUCTURED_META = FINAL_DIR / "structured_test_predictions.meta.json"
CHUNK_SIZE = 16
PREFERRED_GENERATION_BATCH = 8
test_by_id = {r["record_id"]: r for r in test_rows}
test_order = [r["record_id"] for r in test_rows]


def structured_meta():
    """Record the identity of the paired no-review/review evaluation, including
    protocol, model, input hash and selected adapter. The saved metadata
    prevents partial structured traces from being mixed across incompatible
    runs.
    """
    return {
        "system_pair": ["subjectesis_no_review", "subjectesis_review"],
        "protocol_hash": PROTOCOL_HASH,
        "test_input_sha256": TEST_INPUT_SHA256,
        "model_id": c["model_id"],
        "model_revision": c["model_revision"],
        "subjectesis_best_step": int(subjectesis_completed["best_step"]),
        "generation_batching": "independent inference only; 8->4->2->1 OOM fallback",
        "labels_read": False,
    }


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
    b = len(encoded)
    input_ids = torch.full((b, max_input), int(pad_id), dtype=torch.long, device="cuda")
    attention_mask = torch.zeros((b, max_input), dtype=torch.long, device="cuda")
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
    del input_ids, attention_mask, generated, continuation
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


def load_structured_progress():
    """Read saved structured traces after checking their metadata, test membership
    and unique IDs. Return completed cases by ID so the frozen controller can
    resume only the remaining questions.
    """
    if not STRUCTURED_PATH.exists():
        return {}
    if not STRUCTURED_META.exists():
        raise RuntimeError("Structured partial exists without metadata.")
    if json.loads(STRUCTURED_META.read_text()) != structured_meta():
        raise RuntimeError("Structured saved metadata changed.")
    done = {}
    for line in STRUCTURED_PATH.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rid = r["record_id"]
        if rid not in test_by_id:
            raise RuntimeError(f"Unknown saved test record {rid}")
        if rid in done:
            raise RuntimeError(f"Duplicate saved record {rid}")
        done[rid] = r
    return done


def save_structured_progress(done):
    """Save completed structured traces in test order together with the expected
    run metadata. Temporary-file replacement reduces the chance of leaving a
    partial record file during an interruption.
    """
    tmp = STRUCTURED_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf8") as f:
        for rid in test_order:
            if rid in done:
                f.write(json.dumps(done[rid], ensure_ascii=False) + "\n")
    os.replace(tmp, STRUCTURED_PATH)
    meta_tmp = STRUCTURED_META.with_suffix(".tmp")
    meta_tmp.write_text(json.dumps(structured_meta(), indent=2) + "\n")
    os.replace(meta_tmp, STRUCTURED_META)


# Construct one initial state, then run the fixed field-review plan on that state.
# The no-review and review conditions share the initial state and answer rule.
def process_chunk(rows):
    """Construct an initial state for each test question, finalize its no-review
    answer, and execute the bounded field-review plan. Valid updates modify
    only the selected field; the reviewed proposer chooses later belief fields,
    and both final answers and complete traces are returned without gold-label
    access.
    """
    raw_initials = batched_generate_text(
        [row["subjectesis_prompt"] for row in rows],
        PROTOCOL_CONFIG["initial_max_new_tokens"],
    )
    work = []
    for row, raw_initial in zip(rows, raw_initials):
        allowed = list(row["choices"].keys())
        initial = extract_first_json_object(raw_initial)
        valid_initial, initial_error = validate_initial(initial, row["task"], allowed)
        result = {
            "record_id": row["record_id"],
            "dialogue_id": str(row["dialogue_id"]),
            "task": row["task"],
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
        plan = ["intends_to_watch"] if row["task"] == "desire" else ["proposer", "seen"]
        work.append(
            {"row": row, "result": result, "state": state, "plan": plan, "index": 0}
        )
    while True:
        active = [
            w for w in work if w["state"] is not None and w["index"] < len(w["plan"])
        ]
        if not active:
            break
        fields = []
        prompts = []
        for w in active:
            field = w["plan"][w["index"]]
            fields.append(field)
            prompts.append(make_review_prompt(w["row"], w["state"], field))
        raws = batched_generate_text(prompts, PROTOCOL_CONFIG["review_max_new_tokens"])
        for w, field, raw in zip(active, fields, raws):
            obj = extract_first_json_object(raw)
            valid, error = validate_review(obj, field)
            rr = {
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
            w["result"]["reviews"].append(rr)
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


done = load_structured_progress()
print(f"Structured persistent starting progress: {len(done)}/621")
remaining = [row for row in test_rows if row["record_id"] not in done]
started = time.time()
new_completed = 0
for start in range(0, len(remaining), CHUNK_SIZE):
    chunk = remaining[start : start + CHUNK_SIZE]
    chunk_results = process_chunk(chunk)
    for result in chunk_results:
        done[result["record_id"]] = result
    new_completed += len(chunk_results)
    save_structured_progress(done)
    elapsed = time.time() - started
    seconds_per_case = elapsed / max(1, new_completed)
    left = len(remaining) - new_completed
    eta_minutes = seconds_per_case * left / 60
    print(
        f" Persistent structured progress {len(done)}/621 | this session {new_completed}/{len(remaining)} | ETA ~{eta_minutes:.1f} min",
        flush=True,
    )
if len(done) != 621:
    raise RuntimeError("Structured test prediction coverage incomplete.")
structured_results = [done[row["record_id"]] for row in test_rows]
prediction_freeze = {
    "protocol_hash": PROTOCOL_HASH,
    "test_input_sha256": TEST_INPUT_SHA256,
    "base_direct_records": len(base_direct),
    "answer_only_direct_records": len(answer_direct),
    "subjectesis_direct_records": len(subjectesis_direct),
    "structured_records": len(structured_results),
    "labels_read": False,
}
(FINAL_DIR / "PREDICTIONS_FROZEN.json").write_text(
    json.dumps(prediction_freeze, indent=2) + "\n"
)
print("\n ALL FIVE SYSTEMS' TEST PREDICTIONS ARE FROZEN.")
print(" Test references still have NOT been opened.")
from pathlib import Path
from collections import Counter, defaultdict
import json, random, math, shutil, hashlib, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PREDICTION_FREEZE = FINAL_DIR / "PREDICTIONS_FROZEN.json"
if not PREDICTION_FREEZE.exists():
    raise RuntimeError("Predictions are not frozen; refusing to open test references.")
freeze = json.loads(PREDICTION_FREEZE.read_text())
if freeze["protocol_hash"] != EXPECTED_PROTOCOL_HASH:
    raise RuntimeError("Prediction freeze protocol changed.")
if freeze["test_input_sha256"] != TEST_INPUT_SHA256:
    raise RuntimeError("Prediction freeze test input changed.")
if any(
    (
        freeze[k] != 621
        for k in [
            "base_direct_records",
            "answer_only_direct_records",
            "subjectesis_direct_records",
            "structured_records",
        ]
    )
):
    raise RuntimeError("Prediction coverage is not complete.")
with TEST_REFERENCES.open("r", encoding="utf8") as f:
    test_reference_rows = [json.loads(line) for line in f if line.strip()]
if len(test_reference_rows) != 621:
    raise RuntimeError("Expected 621 sealed references.")
# Only now attach gold answers, after the saved prediction freeze.
# All metric and transition calculations below operate on those fixed answers.
test_refs = {r["record_id"]: r for r in test_reference_rows}
if set(test_refs) != {r["record_id"] for r in test_rows}:
    raise RuntimeError("Test input/reference record coverage mismatch.")
print(" Sealed references opened AFTER prediction freeze.")
print("Now computing final metrics. No protocol/model changes are permitted.")


def score_direct(predictions):
    """Attach reference labels and correctness flags to already-frozen direct
    predictions by record ID. This scoring step runs after inference, keeping
    the answer labels outside the prediction procedure.
    """
    rows = []
    for p in predictions:
        ref = test_refs[p["record_id"]]["answer"]
        rows.append({**p, "reference": ref, "correct": p["prediction"] == ref})
    return rows


base_scored = score_direct(base_direct)
answer_scored = score_direct(answer_direct)
subjectesis_direct_scored = score_direct(subjectesis_direct)


def score_structured(key, name):
    """Read one answer stage from each saved structured trace and compare it with
    the sealed reference. Return the same scored-row format as direct
    evaluation, retaining invalid or missing answers as incorrect.
    """
    rows = []
    for r in structured_results:
        ref = test_refs[r["record_id"]]["answer"]
        pred = r.get(key)
        rows.append(
            {
                "system": name,
                "record_id": r["record_id"],
                "dialogue_id": r["dialogue_id"],
                "task": r["task"],
                "prediction": pred,
                "reference": ref,
                "correct": pred == ref,
            }
        )
    return rows


no_review_scored = score_structured("no_review_prediction", "subjectesis_no_review")
review_scored = score_structured("reviewed_prediction", "subjectesis_review")
SYSTEM_ROWS = {
    "base_direct": base_scored,
    "answer_only_direct": answer_scored,
    "subjectesis_direct": subjectesis_direct_scored,
    "subjectesis_no_review": no_review_scored,
    "subjectesis_review": review_scored,
}


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
        set((r["reference"] for r in x)) | set((r["prediction"] for r in x)),
        key=lambda v: str(v),
    )
    f1s = []
    for lab in labels:
        tp = sum((r["reference"] == lab and r["prediction"] == lab for r in x))
        fp = sum((r["reference"] != lab and r["prediction"] == lab for r in x))
        fn = sum((r["reference"] == lab and r["prediction"] != lab for r in x))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return sum(f1s) / len(f1s)


def summarize_system(name, rows):
    """Return one named condition summary with task accuracies, their equal-weight
    mean and task macro F1. These values feed the final comparison table after
    all predictions have been frozen.
    """
    return {
        "system": name,
        "belief_accuracy": task_accuracy(rows, "belief"),
        "desire_accuracy": task_accuracy(rows, "desire"),
        "selection_score": selection_score(rows),
        "belief_macro_f1": macro_f1(rows, "belief"),
        "desire_macro_f1": macro_f1(rows, "desire"),
    }


system_metrics = [summarize_system(name, rows) for name, rows in SYSTEM_ROWS.items()]
invalid_initial = sum((not r["initial_valid"] for r in structured_results))
review_calls = sum((len(r["reviews"]) for r in structured_results))
invalid_reviews = sum(
    (not rr["valid"] for r in structured_results for rr in r["reviews"])
)
transitions = Counter()
for r in structured_results:
    ref = test_refs[r["record_id"]]["answer"]
    before = r.get("no_review_prediction") == ref
    after = r.get("reviewed_prediction") == ref
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
    "note": "Structural source-binding checks only; they do not establish semantic entailment.",
}


def score_metric(sample, key, metric):
    """Calculate belief accuracy, desire accuracy or their equal-weight mean from
    a sampled correctness field. The bootstrap uses this shared metric
    definition for both systems in every paired draw.
    """
    if metric == "belief_accuracy":
        x = [r for r in sample if r["task"] == "belief"]
        return sum((r[key] for r in x)) / len(x)
    if metric == "desire_accuracy":
        x = [r for r in sample if r["task"] == "desire"]
        return sum((r[key] for r in x)) / len(x)
    if metric == "selection_score":
        b = score_metric(sample, key, "belief_accuracy")
        d = score_metric(sample, key, "desire_accuracy")
        return (b + d) / 2
    raise ValueError(metric)


def paired_bootstrap_diff(rows_a, rows_b, comparison, reps=5000, seed=42):
    """Join two conditions by question ID and resample whole dialogues with
    replacement using the same draw for both. Return observed task differences
    and the original sorted-percentile intervals, so within-dialogue dependence
    is retained.
    """
    a = {r["record_id"]: r for r in rows_a}
    b = {r["record_id"]: r for r in rows_b}
    if set(a) != set(b):
        raise RuntimeError("Paired systems have different test coverage.")
    joined = []
    for rid in a:
        joined.append(
            {
                "record_id": rid,
                "dialogue_id": str(a[rid]["dialogue_id"]),
                "task": a[rid]["task"],
                "a_correct": bool(a[rid]["correct"]),
                "b_correct": bool(b[rid]["correct"]),
            }
        )
    by_dialogue = defaultdict(list)
    for r in joined:
        by_dialogue[r["dialogue_id"]].append(r)
    dialogues = sorted(by_dialogue)
    result = {"comparison": comparison, "repetitions": reps, "seed": seed}
    # Use the same seeded dialogue draws for both systems in each metric.
    # This estimates sample uncertainty while keeping paired questions together.
    for metric in ["belief_accuracy", "desire_accuracy", "selection_score"]:
        rng = random.Random(seed)
        diffs = []
        for _ in range(reps):
            sampled_ids = [rng.choice(dialogues) for _ in dialogues]
            sample = []
            for did in sampled_ids:
                sample.extend(by_dialogue[did])
            diffs.append(
                score_metric(sample, "a_correct", metric)
                - score_metric(sample, "b_correct", metric)
            )
        diffs.sort()
        lo = diffs[int(0.025 * len(diffs))]
        hi = diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))]
        if metric == "belief_accuracy":
            point = task_accuracy(rows_a, "belief") - task_accuracy(rows_b, "belief")
        elif metric == "desire_accuracy":
            point = task_accuracy(rows_a, "desire") - task_accuracy(rows_b, "desire")
        else:
            point = selection_score(rows_a) - selection_score(rows_b)
        result[metric] = {
            "point_difference": point,
            "ci95_dialogue_bootstrap": [lo, hi],
        }
    return result


bootstrap_differences = [
    paired_bootstrap_diff(
        answer_scored, base_scored, "Answer-only direct - Base direct"
    ),
    paired_bootstrap_diff(
        subjectesis_direct_scored, base_scored, "Subjectesis direct - Base direct"
    ),
    paired_bootstrap_diff(
        subjectesis_direct_scored,
        answer_scored,
        "Subjectesis direct - Answer-only direct",
    ),
    paired_bootstrap_diff(
        no_review_scored, answer_scored, "Subjectesis no-review - Answer-only direct"
    ),
    paired_bootstrap_diff(
        review_scored, no_review_scored, "Subjectesis review - Subjectesis no-review"
    ),
    paired_bootstrap_diff(
        review_scored, answer_scored, "Subjectesis review - Answer-only direct"
    ),
]


def system_bootstrap_ci(rows, reps=5000, seed=42):
    """Resample complete dialogues for one condition and recalculate both task
    accuracies and their mean. Return the original sorted-percentile interval
    bounds for each metric rather than treating every question as independent.
    """
    by_dialogue = defaultdict(list)
    for r in rows:
        by_dialogue[str(r["dialogue_id"])].append(r)
    dialogues = sorted(by_dialogue)
    rng = random.Random(seed)
    samples = {"belief_accuracy": [], "desire_accuracy": [], "selection_score": []}
    for _ in range(reps):
        sample = []
        for did in [rng.choice(dialogues) for _ in dialogues]:
            sample.extend(by_dialogue[did])
        b = task_accuracy(sample, "belief")
        d = task_accuracy(sample, "desire")
        samples["belief_accuracy"].append(b)
        samples["desire_accuracy"].append(d)
        samples["selection_score"].append((b + d) / 2)
    out = {}
    for key, vals in samples.items():
        vals.sort()
        out[key] = [
            vals[int(0.025 * len(vals))],
            vals[min(len(vals) - 1, int(0.975 * len(vals)))],
        ]
    return out


system_cis = {name: system_bootstrap_ci(rows) for name, rows in SYSTEM_ROWS.items()}
final_metrics = {
    "stage": "sealed_rectom_test",
    "protocol_hash": EXPECTED_PROTOCOL_HASH,
    "model_id": c["model_id"],
    "model_revision": c["model_revision"],
    "answer_only_best_step": int(ordinary_completed["best_step"]),
    "subjectesis_best_step": int(subjectesis_completed["best_step"]),
    "test_questions": 621,
    "test_dialogues": 67,
    "test_input_sha256": TEST_INPUT_SHA256,
    "test_reference_sha256": TEST_REFERENCE_SHA256,
    "systems": system_metrics,
    "system_bootstrap_ci95": system_cis,
    "structured": {
        "invalid_initial_outputs": invalid_initial,
        "invalid_initial_rate": invalid_initial / 621,
        "review_calls": review_calls,
        "invalid_reviews": invalid_reviews,
        "invalid_review_rate": invalid_reviews / review_calls if review_calls else None,
        "review_transitions": dict(transitions),
        "source_binding": source_binding,
    },
    "paired_dialogue_bootstrap_differences": bootstrap_differences,
    "predictions_frozen_before_label_read": True,
    "sealed_test_scored": True,
}
print("\n=== FINAL SEALED RECTOM TEST RESULTS ===")
print(json.dumps(final_metrics, indent=2))
LOCAL_REPORT = ROOT / "reports/final_test"
LOCAL_REPORT.mkdir(parents=True, exist_ok=True)
(LOCAL_REPORT / "final_metrics.json").write_text(
    json.dumps(final_metrics, indent=2) + "\n"
)
pd.DataFrame(base_scored + answer_scored + subjectesis_direct_scored).to_csv(
    LOCAL_REPORT / "direct_predictions_scored.csv", index=False
)
with (LOCAL_REPORT / "structured_results_scored.jsonl").open("w", encoding="utf8") as f:
    for r in structured_results:
        rr = dict(r)
        rr["reference"] = test_refs[r["record_id"]]["answer"]
        f.write(json.dumps(rr, ensure_ascii=False) + "\n")
error_rows = []
pred_maps = {
    name: {r["record_id"]: r["prediction"] for r in rows}
    for name, rows in SYSTEM_ROWS.items()
}
for row in test_rows:
    rid = row["record_id"]
    ref = test_refs[rid]["answer"]
    record = {
        "record_id": rid,
        "dialogue_id": str(row["dialogue_id"]),
        "task": row["task"],
        "reference": ref,
    }
    for name in SYSTEM_ROWS:
        pred = pred_maps[name][rid]
        record[f"{name}_prediction"] = pred
        record[f"{name}_correct"] = pred == ref
    error_rows.append(record)
pd.DataFrame(error_rows).to_csv(LOCAL_REPORT / "error_matrix.csv", index=False)
for system_name, rows in SYSTEM_ROWS.items():
    for task in ["belief", "desire"]:
        x = [r for r in rows if r["task"] == task]
        table = pd.crosstab(
            pd.Series([r["reference"] for r in x], name="reference"),
            pd.Series([r["prediction"] for r in x], name="prediction"),
            dropna=False,
        )
        table.to_csv(LOCAL_REPORT / f"confusion_{system_name}_{task}.csv")
plot_df = pd.DataFrame(
    [
        {
            "system": m["system"],
            "belief": m["belief_accuracy"],
            "desire": m["desire_accuracy"],
        }
        for m in system_metrics
    ]
)
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(plot_df))
width = 0.36
ax.bar(x - width / 2, plot_df["belief"], width, label="belief")
ax.bar(x + width / 2, plot_df["desire"], width, label="desire")
ax.set_ylim(0, 1)
ax.set_ylabel("Test accuracy")
ax.set_xticks(x)
ax.set_xticklabels(plot_df["system"], rotation=25, ha="right")
ax.legend()
fig.tight_layout()
fig.savefig(LOCAL_REPORT / "test_task_accuracy.png", dpi=180)
plt.show()
names = [m["system"] for m in system_metrics]
points = [m["selection_score"] for m in system_metrics]
lower = [
    points[i] - system_cis[names[i]]["selection_score"][0] for i in range(len(names))
]
upper = [
    system_cis[names[i]]["selection_score"][1] - points[i] for i in range(len(names))
]
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(names))
ax.bar(x, points)
ax.errorbar(x, points, yerr=np.array([lower, upper]), fmt="none", capsize=4)
ax.set_ylim(0, 1)
ax.set_ylabel("Mean belief/desire accuracy")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=25, ha="right")
fig.tight_layout()
fig.savefig(LOCAL_REPORT / "test_selection_score_ci.png", dpi=180)
plt.show()
transition_names = [
    "wrong_to_right",
    "right_to_wrong",
    "right_to_right",
    "wrong_to_wrong",
]
transition_values = [transitions.get(k, 0) for k in transition_names]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(transition_names, transition_values)
ax.set_ylabel("Questions")
ax.set_xticklabels(transition_names, rotation=20, ha="right")
fig.tight_layout()
fig.savefig(LOCAL_REPORT / "review_transitions.png", dpi=180)
plt.show()
final_protocol = {
    "protocol_hash": EXPECTED_PROTOCOL_HASH,
    "config": EXPECTED_PROTOCOL_CONFIG,
    "status": "frozen_and_scored_on_sealed_rectom_test",
    "test_predictions_generated_before_reference_read": True,
    "sealed_test_scored": True,
}
(LOCAL_REPORT / "final_protocol.json").write_text(
    json.dumps(final_protocol, indent=2) + "\n"
)
for p in LOCAL_REPORT.rglob("*"):
    if p.is_file():
        target = FINAL_DIR / p.name
        shutil.copy2(p, target)
report_hashes = {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in LOCAL_REPORT.iterdir()
    if p.is_file()
}
complete = {
    "status": "completed",
    "protocol_hash": EXPECTED_PROTOCOL_HASH,
    "test_questions": 621,
    "test_dialogues": 67,
    "test_input_sha256": TEST_INPUT_SHA256,
    "test_reference_sha256": TEST_REFERENCE_SHA256,
    "predictions_frozen_before_label_read": True,
    "sealed_test_scored": True,
    "report_hashes": report_hashes,
}
marker_tmp = COMPLETE_MARKER.with_suffix(".tmp")
marker_tmp.write_text(json.dumps(complete, indent=2) + "\n")
os.replace(marker_tmp, COMPLETE_MARKER)
print("\n FINAL SEALED RECTOM TEST COMPLETE.")
print("Saved to:", FINAL_DIR)
print("Do not change the protocol or rerun model predictions based on these scores.")
