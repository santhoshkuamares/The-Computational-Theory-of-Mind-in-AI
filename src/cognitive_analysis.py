"""Exploratory frozen-model probes, RSA, process counts and 64 counterfactual cases.
Qwen and its adapters are frozen; the linear probe classifiers are fitted. Raw hidden-state arrays are external inputs for independent probe/RSA reproduction.
"""

from project_setup import mount_drive, prepare_project

mount_drive()
from pathlib import Path, PurePosixPath
import ast
import base64
import copy
import hashlib
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import zipfile
import numpy as np
import pandas as pd

ROOT = Path("/content/subjectesis_cognitive_analysis")
DRIVE_ROOT = Path("/content/drive/MyDrive/Subjectesis")
RESULT_DIR = DRIVE_ROOT / "cognitive_analysis"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
ROOT.mkdir(parents=True, exist_ok=True)
SUBJECTESIS_ZIP = DRIVE_ROOT / "subjectesis_a100_latest.zip"
ANSWER_ONLY_ZIP = DRIVE_ROOT / "answer_only_a100_latest.zip"
RECTOM_FINAL_DIR = DRIVE_ROOT / "final_test"
OPENTOM_DIR = DRIVE_ROOT / "opentom_transfer"
for path in [SUBJECTESIS_ZIP, ANSWER_ONLY_ZIP]:
    if not path.exists():
        raise FileNotFoundError(path)
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SEED = 42


def sha256_bytes(data):
    """Calculate the checksum of an archive member before loading it."""
    return hashlib.sha256(data).hexdigest()


def verify_training_archive(path, condition):
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("recovery_manifest.json"))["files"]
        completed_name = f"runs/{condition}/completed.json"
        raw = z.read(completed_name)
        if sha256_bytes(raw) != manifest[completed_name]:
            raise RuntimeError(f"Completion metadata hash mismatch in {path.name}")
        completed = json.loads(raw)
        if completed.get("status") != "completed":
            raise RuntimeError(f"{path.name} is not completed")
        if int(completed.get("examples_seen", -1)) != 30266:
            raise RuntimeError(f"{path.name} has unexpected full-run exposure")
        prefix = f"runs/{condition}/best_adapter/"
        for rel in ["adapter_config.json", "adapter_model.safetensors"]:
            name = prefix + rel
            raw = z.read(name)
            if sha256_bytes(raw) != manifest[name]:
                raise RuntimeError(f"Adapter hash mismatch: {name}")
        return completed


ordinary_completed = verify_training_archive(ANSWER_ONLY_ZIP, "ordinary")
subjectesis_completed = verify_training_archive(SUBJECTESIS_ZIP, "subjectesis")
print("Answer-only selected checkpoint step:", ordinary_completed["best_step"])
print("Subjectesis selected checkpoint step:", subjectesis_completed["best_step"])
print(
    "Approximate selected-checkpoint training exposures:",
    int(ordinary_completed["best_step"]) * 8,
    int(subjectesis_completed["best_step"]) * 8,
)
PROJECT = ROOT / "project"
prepare_project(PROJECT)
settings = {
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
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
env = os.environ.copy()
env["PYTHONPATH"] = str(PROJECT / "code") + ":" + env.get("PYTHONPATH", "")
subprocess.run(
    [sys.executable, str(PROJECT / "code/prepare.py"), "--root", str(PROJECT)],
    check=True,
    env=env,
)
PREP = PROJECT / "preparation/data"
print("Revision-4 data reconstructed.")
sys.path.insert(0, str(PROJECT / "preparation/code"))
from convert_v1 import input_text, benchmark_state


def read_jsonl(path):
    """Read one JSON record per non-empty line, preserving file order."""
    with Path(path).open("r", encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


train_records = read_jsonl(PREP / "review/all_training_records.jsonl")
train_records = [r for r in train_records if r.get("training_eligible") is True]
val_inputs = read_jsonl(PREP / "evaluation/validation_inputs.jsonl")
val_refs = {
    r["record_id"]: r
    for r in read_jsonl(PREP / "scoring_only/validation_references.jsonl")
}
test_inputs = read_jsonl(PREP / "evaluation/test_inputs.jsonl")
test_refs = {
    r["record_id"]: r for r in read_jsonl(PREP / "scoring_only/test_references.jsonl")
}


# All model conditions receive this same prompt during representation extraction.
def plain_prompt(row):
    return input_text(row) + "\nReturn only the selected option letter."


probe_rows = []
for r in train_records:
    inp = copy.deepcopy(r["input"])
    probe_rows.append(
        {
            "record_id": r["record_id"],
            "dialogue_id": str(r["dialogue_id"]),
            "split": "train",
            "task": r["task"],
            "prompt": plain_prompt(inp),
            "state": copy.deepcopy(r["official_answer_state"]),
        }
    )
for split, inputs, refs in [
    ("validation", val_inputs, val_refs),
    ("test", test_inputs, test_refs),
]:
    for inp in inputs:
        row = copy.deepcopy(inp)
        row["answer"] = [refs[row["record_id"]]["answer"]]
        state = benchmark_state(row)
        probe_rows.append(
            {
                "record_id": row["record_id"],
                "dialogue_id": str(row["dialogue_id"]),
                "split": split,
                "task": row["task"],
                "prompt": plain_prompt(row),
                "state": state,
            }
        )
probe_df = pd.DataFrame(probe_rows)
if len(probe_df[probe_df.split == "train"]) != 2235:
    raise RuntimeError("Unexpected eligible training count")
if len(probe_df[probe_df.split == "validation"]) != 319:
    raise RuntimeError("Unexpected validation count")
if len(probe_df[probe_df.split == "test"]) != 621:
    raise RuntimeError("Unexpected test count")
FIELDS = {
    "proposer": ["recommender", "seeker"],
    "seen": ["yes", "no"],
    "recommendation_response": ["accepted", "declined"],
    "appraisal": ["likes", "dislikes"],
    "intends_to_watch": ["yes", "no"],
}
for field, classes in FIELDS.items():
    probe_df[field] = probe_df["state"].apply(lambda s: s.get(field))
    counts = probe_df.groupby("split")[field].value_counts(dropna=True)
    print("\n", field)
    print(counts)
probe_df[["record_id", "dialogue_id", "split", "task"] + list(FIELDS)].to_csv(
    RESULT_DIR / "probe_dataset_index.csv", index=False
)
print("\nProbe dataset prepared. Training and evaluation remain dialogue-disjoint.")
import importlib.metadata as md

subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
    check=False,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
subprocess.run(
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
        "scipy",
        "matplotlib",
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--no-deps",
        "torchao==0.16.0",
    ],
    check=True,
)
import torch
from safetensors.torch import load_file
from peft import set_peft_model_state_dict
from transformers import AutoTokenizer

sys.path.insert(0, str(PROJECT / "code"))
from common import read_json
from model_runtime import build_network
from engine import CompletionLoss

if not torch.cuda.is_available():
    raise RuntimeError("A CUDA GPU is required for hidden-state extraction")
GPU_NAME = torch.cuda.get_device_name(0)
print("GPU:", GPU_NAME)
if "A100" in GPU_NAME.upper() or "H100" in GPU_NAME.upper():
    HIDDEN_BATCH = 8
    GENERATION_BATCH = 8
elif "L4" in GPU_NAME.upper():
    HIDDEN_BATCH = 4
    GENERATION_BATCH = 4
else:
    HIDDEN_BATCH = 2
    GENERATION_BATCH = 2
print("Initial hidden-state batch size:", HIDDEN_BATCH)
print("Initial generation batch size:", GENERATION_BATCH)
env = os.environ.copy()
env["PYTHONPATH"] = str(PROJECT / "code") + ":" + env.get("PYTHONPATH", "")
env["TOKENIZERS_PARALLELISM"] = "false"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
subprocess.run(
    [sys.executable, str(PROJECT / "code/download_model.py"), "--root", str(PROJECT)],
    check=True,
    env=env,
)
c = read_json(PROJECT / "settings.json")
model_path = Path(read_json(PROJECT / "reports/model_download.json")["snapshot"])
tok = AutoTokenizer.from_pretrained(
    model_path, local_files_only=True, trust_remote_code=False
)
tok.padding_side = "left"
network = build_network(PROJECT, c, 0, model_path)
network.eval()
try:
    network.gradient_checkpointing_disable()
except Exception:
    pass
network.config.use_cache = False
wrapper = CompletionLoss(network, c["loss_chunk_size"])
wrapper.eval()


def extract_best_adapter(archive, condition, destination):
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    prefix = f"runs/{condition}/best_adapter/"
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.startswith(prefix) and (not name.endswith("/")):
                rel = PurePosixPath(name).relative_to(PurePosixPath(prefix))
                target = destination.joinpath(*rel.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(name))


ADAPTER_DIR = ROOT / "adapters"
ADAPTER_DIR.mkdir(exist_ok=True)
ANSWER_ADAPTER = ADAPTER_DIR / "answer_only_best"
SUBJECTESIS_ADAPTER = ADAPTER_DIR / "subjectesis_best"
extract_best_adapter(ANSWER_ONLY_ZIP, "ordinary", ANSWER_ADAPTER)
extract_best_adapter(SUBJECTESIS_ZIP, "subjectesis", SUBJECTESIS_ADAPTER)


def load_adapter(path):
    state = load_file(Path(path) / "adapter_model.safetensors")
    set_peft_model_state_dict(network, state, adapter_name="default")
    network.eval()


load_adapter(SUBJECTESIS_ADAPTER)
print("Model and frozen adapters are ready.")
import time

REP_DIR = ROOT / "representations"
REP_DIR.mkdir(exist_ok=True)


def chat_ids(prompt):
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) > c["max_length"]:
        raise RuntimeError(f"Prompt exceeds max_length: {len(ids)}")
    return ids


# Select the last real prompt token rather than padding positions.
def forward_hidden_batch(prompts):
    """Extract the final real prompt-token hidden states for each layer."""
    encoded = [chat_ids(p) for p in prompts]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    max_len = max((len(x) for x in encoded))
    batch = len(encoded)
    input_ids = torch.full(
        (batch, max_len), int(pad_id), dtype=torch.long, device="cuda"
    )
    attention_mask = torch.zeros((batch, max_len), dtype=torch.long, device="cuda")
    for i, ids in enumerate(encoded):
        n = len(ids)
        input_ids[i, -n:] = torch.tensor(ids, dtype=torch.long, device="cuda")
        attention_mask[i, -n:] = 1
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        out = network(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hidden = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1)
    result = hidden.detach().to(dtype=torch.float16, device="cpu").numpy()
    del input_ids, attention_mask, out, hidden
    return result


def extract_system_representations(
    system_name, adapter_path=None, disable_adapter=False
):
    """Run frozen-model forward passes and save representation arrays for one condition."""
    path = REP_DIR / f"{system_name}.npy"
    meta_path = REP_DIR / f"{system_name}.meta.json"
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("records") == len(probe_df):
            print(system_name, "already extracted; reusing local file")
            return np.load(path, mmap_mode="r")
    if adapter_path is not None:
        load_adapter(adapter_path)
    prompts = probe_df["prompt"].tolist()
    context = network.disable_adapter() if disable_adapter else None
    if context is not None:
        context.__enter__()
    try:
        first = forward_hidden_batch(prompts[:1])
    finally:
        if context is not None:
            context.__exit__(None, None, None)
    _, n_layers, hidden_size = first.shape
    arr = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float16, shape=(len(prompts), n_layers, hidden_size)
    )
    arr[0:1] = first
    start_index = 1
    batch_size = HIDDEN_BATCH
    started = time.time()
    if adapter_path is not None:
        load_adapter(adapter_path)
    context = network.disable_adapter() if disable_adapter else None
    if context is not None:
        context.__enter__()
    try:
        i = start_index
        while i < len(prompts):
            size = min(batch_size, len(prompts) - i)
            try:
                block = forward_hidden_batch(prompts[i : i + size])
                arr[i : i + size] = block
                i += size
                if i % 128 < size or i == len(prompts):
                    elapsed = time.time() - started
                    rate = max(1e-09, (i - start_index) / elapsed)
                    remaining = (len(prompts) - i) / rate / 60
                    print(
                        f"{system_name}: {i}/{len(prompts)} | ETA {remaining:.1f} min"
                    )
                batch_size = HIDDEN_BATCH
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if size == 1:
                    raise
                batch_size = max(1, size // 2)
                print(system_name, "OOM; retrying with batch", batch_size)
    finally:
        if context is not None:
            context.__exit__(None, None, None)
    arr.flush()
    meta = {
        "system": system_name,
        "records": len(prompts),
        "representation_layers": int(n_layers),
        "hidden_size": int(hidden_size),
        "dtype": "float16",
        "representation": "final non-padding prompt token",
        "input_prompt": "plain answer-only prompt shared across systems",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return np.load(path, mmap_mode="r")


BASE_H = extract_system_representations("base", disable_adapter=True)
ANSWER_H = extract_system_representations("answer_only", adapter_path=ANSWER_ADAPTER)
SUBJECTESIS_H = extract_system_representations(
    "subjectesis", adapter_path=SUBJECTESIS_ADAPTER
)
print("Representation shapes:")
print("Base:", BASE_H.shape)
print("Answer-only:", ANSWER_H.shape)
print("Subjectesis:", SUBJECTESIS_H.shape)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score

SYSTEM_H = {"base": BASE_H, "answer_only": ANSWER_H, "subjectesis": SUBJECTESIS_H}
N_LAYERS = BASE_H.shape[1]
probe_results = []
final_layer_predictions = {}


def encode_binary(series, classes):
    mapping = {classes[0]: 0, classes[1]: 1}
    return np.array([mapping[x] for x in series], dtype=np.int64)


for field, classes in FIELDS.items():
    field_rows = probe_df[probe_df[field].isin(classes)].copy()
    train_idx = field_rows.index[field_rows.split == "train"].to_numpy()
    val_idx = field_rows.index[field_rows.split == "validation"].to_numpy()
    test_idx = field_rows.index[field_rows.split == "test"].to_numpy()
    if len(train_idx) == 0 or len(test_idx) == 0:
        continue
    y_all = probe_df[field]
    y_train = encode_binary(y_all.loc[train_idx], classes)
    y_val = (
        encode_binary(y_all.loc[val_idx], classes)
        if len(val_idx)
        else np.array([], dtype=int)
    )
    y_test = encode_binary(y_all.loc[test_idx], classes)
    print(
        "\nField:",
        field,
        "| train:",
        len(train_idx),
        "| validation:",
        len(val_idx),
        "| test:",
        len(test_idx),
    )
    for system_name, hidden in SYSTEM_H.items():
        for layer in range(N_LAYERS):
            X_train = np.asarray(hidden[train_idx, layer, :], dtype=np.float32)
            X_test = np.asarray(hidden[test_idx, layer, :], dtype=np.float32)
            X_val = (
                np.asarray(hidden[val_idx, layer, :], dtype=np.float32)
                if len(val_idx)
                else None
            )
            probe = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=SEED,
                ),
            )
            probe.fit(X_train, y_train)
            pred_test = probe.predict(X_test)
            row = {
                "field": field,
                "system": system_name,
                "layer": layer,
                "train_n": len(train_idx),
                "validation_n": len(val_idx),
                "test_n": len(test_idx),
                "test_balanced_accuracy": balanced_accuracy_score(y_test, pred_test),
                "test_macro_f1": f1_score(
                    y_test, pred_test, average="macro", zero_division=0
                ),
            }
            if len(val_idx):
                pred_val = probe.predict(X_val)
                row["validation_balanced_accuracy"] = balanced_accuracy_score(
                    y_val, pred_val
                )
                row["validation_macro_f1"] = f1_score(
                    y_val, pred_val, average="macro", zero_division=0
                )
            else:
                row["validation_balanced_accuracy"] = np.nan
                row["validation_macro_f1"] = np.nan
            probe_results.append(row)
            if layer == N_LAYERS - 1:
                final_layer_predictions[field, system_name] = {
                    "record_indices": test_idx.tolist(),
                    "y_true": y_test.tolist(),
                    "y_pred": pred_test.tolist(),
                }
            del X_train, X_test
probe_results_df = pd.DataFrame(probe_results)
probe_results_df.to_csv(RESULT_DIR / "layerwise_linear_probe_results.csv", index=False)
print("\nLayer-wise probing complete.")
import matplotlib.pyplot as plt

PLOT_DIR = RESULT_DIR / "figures"
PLOT_DIR.mkdir(exist_ok=True)
for field in probe_results_df.field.unique():
    fig, ax = plt.subplots(figsize=(9, 5))
    for system_name in ["base", "answer_only", "subjectesis"]:
        d = probe_results_df[
            (probe_results_df.field == field) & (probe_results_df.system == system_name)
        ].sort_values("layer")
        ax.plot(d.layer, d.test_macro_f1, marker="o", markersize=2.5, label=system_name)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Representation layer")
    ax.set_ylabel("Test macro-F1")
    ax.set_title(f"Linear decodability of {field}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"probe_{field}_layer_curve.png", dpi=200)
    plt.show()
BOOT_REPS = 5000
rng = random.Random(SEED)
bootstrap_rows = []
test_dialogues = probe_df.loc[probe_df.split == "test", ["dialogue_id"]]
for field, classes in FIELDS.items():
    key_subjectesis = (field, "subjectesis")
    key_answer = (field, "answer_only")
    key_base = (field, "base")
    if key_subjectesis not in final_layer_predictions:
        continue
    pred_records = {}
    for system_name in ["base", "answer_only", "subjectesis"]:
        info = final_layer_predictions[field, system_name]
        for idx, y, p in zip(info["record_indices"], info["y_true"], info["y_pred"]):
            pred_records.setdefault(
                int(idx), {"dialogue_id": probe_df.loc[idx, "dialogue_id"], "y": y}
            )
            pred_records[int(idx)][system_name] = p
    by_dialogue = {}
    for idx, row in pred_records.items():
        by_dialogue.setdefault(row["dialogue_id"], []).append(idx)
    dialogue_ids = sorted(by_dialogue)
    comparisons = [
        ("subjectesis", "answer_only"),
        ("subjectesis", "base"),
        ("answer_only", "base"),
    ]
    for a, b in comparisons:
        diffs = []
        for _ in range(BOOT_REPS):
            sampled = [rng.choice(dialogue_ids) for _ in dialogue_ids]
            ids = []
            for did in sampled:
                ids.extend(by_dialogue[did])
            y = [pred_records[i]["y"] for i in ids]
            pa = [pred_records[i][a] for i in ids]
            pb = [pred_records[i][b] for i in ids]
            fa = f1_score(y, pa, average="macro", zero_division=0)
            fb = f1_score(y, pb, average="macro", zero_division=0)
            diffs.append(fa - fb)
        actual_ids = sorted(pred_records)
        y = [pred_records[i]["y"] for i in actual_ids]
        pa = [pred_records[i][a] for i in actual_ids]
        pb = [pred_records[i][b] for i in actual_ids]
        point = f1_score(y, pa, average="macro", zero_division=0) - f1_score(
            y, pb, average="macro", zero_division=0
        )
        bootstrap_rows.append(
            {
                "field": field,
                "comparison": f"{a} - {b}",
                "layer": N_LAYERS - 1,
                "point_macro_f1_difference": point,
                "ci95_low": float(np.percentile(diffs, 2.5)),
                "ci95_high": float(np.percentile(diffs, 97.5)),
                "bootstrap_repetitions": BOOT_REPS,
                "bootstrap_unit": "dialogue",
            }
        )
probe_bootstrap_df = pd.DataFrame(bootstrap_rows)
probe_bootstrap_df.to_csv(
    RESULT_DIR / "probe_final_layer_dialogue_bootstrap.csv", index=False
)
print(probe_bootstrap_df.to_string(index=False))
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_distances

belief_test = probe_df[(probe_df.split == "test") & (probe_df.task == "belief")].copy()
belief_indices = belief_test.index.to_numpy()
STATE_FIELDS = ["proposer", "seen", "recommendation_response", "appraisal"]


# This target encodes benchmark variables, not independently measured human beliefs.
def state_distance_matrix(frame):
    """Build the benchmark-variable distance matrix used as the RSA target."""
    n = len(frame)
    D = np.zeros((n, n), dtype=np.float32)
    states = frame["state"].tolist()
    for i in range(n):
        for j in range(i + 1, n):
            comparable = []
            for field in STATE_FIELDS:
                a = states[i].get(field)
                b = states[j].get(field)
                if a is not None and b is not None:
                    comparable.append(0.0 if a == b else 1.0)
            value = float(np.mean(comparable)) if comparable else np.nan
            D[i, j] = value
            D[j, i] = value
    return D


THEORY_D = state_distance_matrix(belief_test)
tri = np.triu_indices(len(belief_test), k=1)
theory_vec = THEORY_D[tri]
valid_pair_mask = ~np.isnan(theory_vec)
rsa_rows = []
for system_name, hidden in SYSTEM_H.items():
    for layer in range(N_LAYERS):
        X = np.asarray(hidden[belief_indices, layer, :], dtype=np.float32)
        rep_D = cosine_distances(X)
        rep_vec = rep_D[tri]
        rho, _ = spearmanr(theory_vec[valid_pair_mask], rep_vec[valid_pair_mask])
        rsa_rows.append(
            {
                "system": system_name,
                "layer": layer,
                "spearman_rho": float(rho),
                "belief_test_questions": len(belief_test),
                "valid_pairs": int(valid_pair_mask.sum()),
            }
        )
rsa_df = pd.DataFrame(rsa_rows)
rsa_df.to_csv(RESULT_DIR / "rsa_belief_state_geometry.csv", index=False)
fig, ax = plt.subplots(figsize=(9, 5))
for system_name in ["base", "answer_only", "subjectesis"]:
    d = rsa_df[rsa_df.system == system_name].sort_values("layer")
    ax.plot(d.layer, d.spearman_rho, marker="o", markersize=2.5, label=system_name)
ax.axhline(0, linewidth=1)
ax.set_xlabel("Representation layer")
ax.set_ylabel("Spearman correlation")
ax.set_title("RSA: hidden-state geometry and benchmark-encoded belief-state geometry")
ax.legend()
fig.tight_layout()
fig.savefig(PLOT_DIR / "rsa_belief_state_layer_curve.png", dpi=200)
plt.show()
jackknife_rows = []
for system_name, hidden in SYSTEM_H.items():
    X_all = np.asarray(hidden[belief_indices, N_LAYERS - 1, :], dtype=np.float32)
    frame = belief_test.reset_index(drop=True)
    dialogues = sorted(frame.dialogue_id.unique())
    estimates = []
    for did in dialogues:
        keep = frame.dialogue_id.to_numpy() != did
        sub = frame.loc[keep].reset_index(drop=True)
        if len(sub) < 3:
            continue
        theory = state_distance_matrix(sub)
        tri_sub = np.triu_indices(len(sub), k=1)
        tv = theory[tri_sub]
        valid = ~np.isnan(tv)
        rep = cosine_distances(X_all[keep])
        rv = rep[tri_sub]
        rho, _ = spearmanr(tv[valid], rv[valid])
        estimates.append(float(rho))
    point = (
        rsa_df[(rsa_df.system == system_name) & (rsa_df.layer == N_LAYERS - 1)]
        .iloc[0]
        .spearman_rho
    )
    estimates = np.array(estimates, dtype=float)
    m = len(estimates)
    mean_j = estimates.mean()
    se = np.sqrt((m - 1) / m * np.sum((estimates - mean_j) ** 2))
    jackknife_rows.append(
        {
            "system": system_name,
            "layer": N_LAYERS - 1,
            "rho": point,
            "jackknife_se": float(se),
            "approx_ci95_low": float(point - 1.96 * se),
            "approx_ci95_high": float(point + 1.96 * se),
            "jackknife_unit": "dialogue",
        }
    )
rsa_jackknife_df = pd.DataFrame(jackknife_rows)
rsa_jackknife_df.to_csv(
    RESULT_DIR / "rsa_final_layer_dialogue_jackknife.csv", index=False
)
print(rsa_jackknife_df.to_string(index=False))
from collections import Counter

# Count state edits separately from final-answer edits.
process_summary = {}
rectom_path = RECTOM_FINAL_DIR / "structured_results_scored.jsonl"
if rectom_path.exists():
    rectom_rows = read_jsonl(rectom_path)
    decision_counts = Counter()
    field_review_counts = Counter()
    field_value_changes = Counter()
    field_basis_changes = Counter()
    evidence_turn_changes = Counter()
    preserve_with_changed_value = 0
    total_reviews = 0
    for row in rectom_rows:
        before = row.get("state_before") or {}
        after = row.get("state_after") or {}
        for field in set(before) | set(after):
            b = before.get(field, {})
            a = after.get(field, {})
            if b.get("value") != a.get("value"):
                field_value_changes[field] += 1
            if b.get("basis") != a.get("basis"):
                field_basis_changes[field] += 1
            if b.get("evidence_turns") != a.get("evidence_turns"):
                evidence_turn_changes[field] += 1
        for review in row.get("reviews", []):
            total_reviews += 1
            field = review.get("field")
            field_review_counts[field] += 1
            parsed = review.get("parsed") or {}
            decision = parsed.get("decision", "invalid")
            decision_counts[decision] += 1
            if decision == "preserve" and field in before:
                updated = parsed.get("updated_claim") or {}
                if updated.get("value") != before[field].get("value"):
                    preserve_with_changed_value += 1
    answer_changes = sum(
        (
            r.get("no_review_prediction") != r.get("reviewed_prediction")
            for r in rectom_rows
        )
    )
    process_summary["rectom"] = {
        "questions": len(rectom_rows),
        "review_calls": total_reviews,
        "answer_changes": answer_changes,
        "decision_counts": dict(decision_counts),
        "field_review_counts": dict(field_review_counts),
        "field_value_changes": dict(field_value_changes),
        "field_basis_changes": dict(field_basis_changes),
        "field_evidence_turn_changes": dict(evidence_turn_changes),
        "preserve_decision_with_changed_value": preserve_with_changed_value,
    }
opentom_structured_path = OPENTOM_DIR / "subjectesis_structured.jsonl"
if opentom_structured_path.exists():
    ot_rows = read_jsonl(opentom_structured_path)
    decision_counts = Counter()
    claim_changes = 0
    evidence_changes = 0
    uncertainty_changes = 0
    invalid_initial = 0
    invalid_review = 0
    for row in ot_rows:
        if not row.get("initial_valid", False):
            invalid_initial += 1
            continue
        initial = None
        review = None
        try:
            initial = json.loads(row.get("raw_initial", ""))
        except Exception:
            pass
        try:
            review = json.loads(row.get("raw_review", ""))
        except Exception:
            pass
        if not row.get("review_valid", False) or not isinstance(review, dict):
            invalid_review += 1
            continue
        decision_counts[review.get("decision", "unknown")] += 1
        if isinstance(initial, dict):
            if initial.get("mental_state_claim") != review.get(
                "updated_mental_state_claim"
            ):
                claim_changes += 1
            if initial.get("evidence_sentences") != review.get("evidence_sentences"):
                evidence_changes += 1
            if initial.get("uncertainty") != review.get("uncertainty"):
                uncertainty_changes += 1
    raw_answer_changes = sum(
        (
            str(r.get("no_review_prediction")).strip()
            != str(r.get("review_prediction")).strip()
            for r in ot_rows
        )
    )
    process_summary["opentom"] = {
        "questions": len(ot_rows),
        "invalid_initial": invalid_initial,
        "invalid_review": invalid_review,
        "decision_counts": dict(decision_counts),
        "mental_state_claim_changes": claim_changes,
        "evidence_sentence_changes": evidence_changes,
        "uncertainty_changes": uncertainty_changes,
        "raw_answer_changes": raw_answer_changes,
    }
(RESULT_DIR / "monitoring_control_process_summary.json").write_text(
    json.dumps(process_summary, indent=2) + "\n"
)
print(json.dumps(process_summary, indent=2))
SCENARIOS = [
    ("Maya", "Liam", "book", "drawer", "shelf"),
    ("Nora", "Ethan", "key", "box", "cabinet"),
    ("Aisha", "Daniel", "letter", "folder", "desk"),
    ("Elena", "Noah", "toy", "basket", "cupboard"),
    ("Priya", "Lucas", "phone", "bag", "table"),
    ("Sofia", "Owen", "watch", "case", "locker"),
    ("Amira", "Henry", "ticket", "envelope", "drawer"),
    ("Clara", "Adam", "map", "folder", "suitcase"),
    ("Leah", "Samuel", "coin", "jar", "box"),
    ("Rina", "Jacob", "notebook", "cabinet", "backpack"),
    ("Tara", "Leo", "photo", "album", "drawer"),
    ("Mina", "Arthur", "ring", "case", "safe"),
    ("Layla", "Ben", "card", "wallet", "desk"),
    ("Eva", "Ryan", "pen", "cup", "drawer"),
    ("Anya", "David", "scarf", "chair", "closet"),
    ("Iris", "Tom", "ball", "basket", "garage"),
]
counterfactual_rows = []
for i, (mover, observer, obj, initial, new) in enumerate(SCENARIOS):
    if i % 2 == 0:
        options = {"A": initial, "B": new}
        initial_letter, new_letter = ("A", "B")
    else:
        options = {"A": new, "B": initial}
        initial_letter, new_letter = ("B", "A")
    for order in ["first_order", "second_order"]:
        for observed in [True, False]:
            if observed:
                story = f"{mover} and {observer} are in the room. {mover} puts the {obj} in the {initial}. {observer} sees the {obj} in the {initial}. {mover} moves the {obj} to the {new} while {observer} watches. Both remain in the room."
                gold = new_letter
            else:
                story = f"{mover} and {observer} are in the room. {mover} puts the {obj} in the {initial}. {observer} sees the {obj} in the {initial}. {mover} sees {observer} leave the room. While {observer} is away, {mover} moves the {obj} to the {new}. {observer} does not see the move."
                gold = initial_letter
            if order == "first_order":
                question = f"Where does {observer} believe the {obj} is now?"
                outer = observer
                target = observer
            else:
                question = (
                    f"Where does {mover} think {observer} believes the {obj} is now?"
                )
                outer = mover
                target = observer
            counterfactual_rows.append(
                {
                    "record_id": f"scenario_{i:02d}_{order}_{('observed' if observed else 'unobserved')}",
                    "pair_id": f"scenario_{i:02d}_{order}",
                    "scenario_id": i,
                    "order": order,
                    "observed": observed,
                    "mover": mover,
                    "observer": observer,
                    "object": obj,
                    "initial_location": initial,
                    "new_location": new,
                    "story": story,
                    "question": question,
                    "outer_perspective": outer,
                    "target_belief_holder": target,
                    "options": options,
                    "gold": gold,
                }
            )
counterfactual_df = pd.DataFrame(counterfactual_rows)
counterfactual_df.to_json(
    RESULT_DIR / "counterfactual_diagnostic_frozen.jsonl",
    orient="records",
    lines=True,
    force_ascii=False,
)
print("Counterfactual cases:", len(counterfactual_df))
print(counterfactual_df.groupby(["order", "observed"]).size())
letter_ids = {k: tok.encode(k, add_special_tokens=False) for k in ["A", "B"]}
if any((len(v) != 1 for v in letter_ids.values())):
    raise RuntimeError("A/B are not single tokens under the pinned tokenizer")


def direct_counterfactual_prompt(row):
    return (
        "Story:\n"
        + row["story"]
        + "\n\nQuestion:\n"
        + row["question"]
        + "\n\nOptions:\n"
        + "\n".join((f"{k}: {v}" for k, v in row["options"].items()))
        + "\nReturn only A or B."
    )


def score_letter(prompt):
    """Choose among the allowed answer tokens using the frozen model logits."""
    ids = chat_ids(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        out = network(input_ids=input_ids, use_cache=False, return_dict=True)
    logits = out.logits[0, -1]
    values = [float(logits[letter_ids[k][0]]) for k in ["A", "B"]]
    return ["A", "B"][int(np.argmax(values))]


def numbered_story(text):
    parts = [x.strip() for x in re.split("(?<=[.!?])\\s+", text.strip()) if x.strip()]
    return ("\n".join((f"{i + 1}: {s}" for i, s in enumerate(parts))), len(parts))


STATE_INSTRUCTION = "Build a perspective-specific belief record for this location question.\nReturn exactly one JSON object with keys outer_perspective, target_belief_holder, believed_location, evidence_sentences, uncertainty, and answer.\nUse only information available to the perspective described in the question. If the target agent did not observe a move, do not give that agent knowledge of the move. The answer must be A or B."
REVIEW_INSTRUCTION = "Review the current perspective-specific belief record against the numbered story.\nReturn exactly one JSON object with keys decision, believed_location, evidence_sentences, uncertainty, reason.\nDecision must be revise, preserve, or retain_unknown. Do not use information unavailable to the target belief holder."


def state_prompt_counterfactual(row):
    story, _ = numbered_story(row["story"])
    return (
        "Numbered story:\n"
        + story
        + "\n\nQuestion:\n"
        + row["question"]
        + "\n\nOptions:\n"
        + "\n".join((f"{k}: {v}" for k, v in row["options"].items()))
        + "\n\n"
        + STATE_INSTRUCTION
    )


def review_prompt_counterfactual(row, state):
    story, _ = numbered_story(row["story"])
    return (
        "Numbered story:\n"
        + story
        + "\n\nQuestion:\n"
        + row["question"]
        + "\n\nCurrent belief record:\n"
        + json.dumps(state, ensure_ascii=False)
        + "\n\n"
        + REVIEW_INSTRUCTION
    )


def finalizer_prompt_counterfactual(row, state):
    return (
        direct_counterfactual_prompt(row)
        + "\n\nPerspective-specific belief record:\n"
        + json.dumps(state, ensure_ascii=False)
        + "\nUse the record as evidence bookkeeping and return only A or B."
    )


def apply_chat(prompt):
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_text_batch(prompts, max_new_tokens, preferred=None):
    """Generate greedy completions with the saved prompt and batching settings."""
    if not prompts:
        return []
    preferred = preferred or GENERATION_BATCH
    outputs = [None] * len(prompts)
    index = 0
    while index < len(prompts):
        size = min(preferred, len(prompts) - index)
        while True:
            try:
                encoded = [
                    tok.encode(apply_chat(p), add_special_tokens=False)
                    for p in prompts[index : index + size]
                ]
                pad_id = (
                    tok.pad_token_id
                    if tok.pad_token_id is not None
                    else tok.eos_token_id
                )
                max_len = max((len(x) for x in encoded))
                input_ids = torch.full(
                    (size, max_len), int(pad_id), dtype=torch.long, device="cuda"
                )
                attention_mask = torch.zeros(
                    (size, max_len), dtype=torch.long, device="cuda"
                )
                for i, ids in enumerate(encoded):
                    n = len(ids)
                    input_ids[i, -n:] = torch.tensor(
                        ids, dtype=torch.long, device="cuda"
                    )
                    attention_mask[i, -n:] = 1
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.float16
                ):
                    generated = network.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=int(pad_id),
                        eos_token_id=tok.eos_token_id,
                    )
                continuation = generated[:, max_len:]
                text = [
                    tok.decode(x, skip_special_tokens=True).strip()
                    for x in continuation
                ]
                outputs[index : index + size] = text
                index += size
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if size == 1:
                    raise
                size = max(1, size // 2)
                preferred = size
    return outputs


def first_json(text):
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
                except json.JSONDecodeError:
                    return None
    return None


def run_direct_system(name, adapter_path=None, disable_adapter=False):
    if adapter_path is not None:
        load_adapter(adapter_path)
    ctx = network.disable_adapter() if disable_adapter else None
    if ctx is not None:
        ctx.__enter__()
    try:
        preds = []
        for _, row in counterfactual_df.iterrows():
            preds.append(score_letter(direct_counterfactual_prompt(row)))
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    return preds


base_pred = run_direct_system("base", disable_adapter=True)
answer_pred = run_direct_system("answer_only", adapter_path=ANSWER_ADAPTER)
subjectesis_pred = run_direct_system("subjectesis", adapter_path=SUBJECTESIS_ADAPTER)
load_adapter(SUBJECTESIS_ADAPTER)
rows = [r for _, r in counterfactual_df.iterrows()]
raw_initial = generate_text_batch([state_prompt_counterfactual(r) for r in rows], 220)
states = []
for row, raw in zip(rows, raw_initial):
    obj = first_json(raw)
    valid = (
        isinstance(obj, dict)
        and obj.get("answer") in ["A", "B"]
        and isinstance(obj.get("evidence_sentences"), list)
    )
    states.append(obj if valid else None)
no_review_pred = []
for row, state in zip(rows, states):
    no_review_pred.append(
        score_letter(finalizer_prompt_counterfactual(row, state))
        if state is not None
        else None
    )
review_prompts = [
    review_prompt_counterfactual(row, state)
    for row, state in zip(rows, states)
    if state is not None
]
raw_reviews_valid = generate_text_batch(review_prompts, 180)
raw_review_iter = iter(raw_reviews_valid)
reviewed_states = []
review_valid_flags = []
for row, state in zip(rows, states):
    if state is None:
        reviewed_states.append(None)
        review_valid_flags.append(False)
        continue
    raw = next(raw_review_iter)
    obj = first_json(raw)
    valid = (
        isinstance(obj, dict)
        and obj.get("decision") in ["revise", "preserve", "retain_unknown"]
        and isinstance(obj.get("evidence_sentences"), list)
    )
    if valid:
        updated = copy.deepcopy(state)
        updated["believed_location"] = obj.get("believed_location")
        updated["evidence_sentences"] = obj.get("evidence_sentences")
        updated["uncertainty"] = obj.get("uncertainty")
        reviewed_states.append(updated)
    else:
        reviewed_states.append(copy.deepcopy(state))
    review_valid_flags.append(valid)
review_pred = []
for row, state in zip(rows, reviewed_states):
    review_pred.append(
        score_letter(finalizer_prompt_counterfactual(row, state))
        if state is not None
        else None
    )
counterfactual_out = counterfactual_df.copy()
counterfactual_out["base_direct"] = base_pred
counterfactual_out["answer_only_direct"] = answer_pred
counterfactual_out["subjectesis_direct"] = subjectesis_pred
counterfactual_out["subjectesis_no_review"] = no_review_pred
counterfactual_out["subjectesis_review"] = review_pred
counterfactual_out["initial_state_valid"] = [s is not None for s in states]
counterfactual_out["review_valid"] = review_valid_flags
counterfactual_out.to_csv(RESULT_DIR / "counterfactual_predictions.csv", index=False)
print("Counterfactual model inference complete.")
SYSTEM_COLUMNS = [
    "base_direct",
    "answer_only_direct",
    "subjectesis_direct",
    "subjectesis_no_review",
    "subjectesis_review",
]
# Preserve the original diagnostic denominators; all saved predictions are valid.
metric_rows = []
for system in SYSTEM_COLUMNS:
    pred = counterfactual_out[system]
    valid = pred.notna()
    correct = pred == counterfactual_out.gold
    for order in ["all", "first_order", "second_order"]:
        mask = valid.copy()
        if order != "all":
            mask &= counterfactual_out.order == order
        metric_rows.append(
            {
                "system": system,
                "order": order,
                "n": int(mask.sum()),
                "accuracy": float(correct[mask].mean()) if mask.sum() else np.nan,
                "invalid_rate": float((~valid).mean()),
            }
        )
    unobserved = valid & (counterfactual_out.observed == False)
    if unobserved.any():
        leakage = []
        for _, row in counterfactual_out.loc[unobserved].iterrows():
            actual_new_letter = [
                k for k, v in row.options.items() if v == row.new_location
            ][0]
            leakage.append(row[system] == actual_new_letter)
        leakage_rate = float(np.mean(leakage))
    else:
        leakage_rate = np.nan
    pair_success = []
    for pair_id, group in counterfactual_out.groupby("pair_id"):
        if len(group) != 2 or group[system].isna().any():
            continue
        pair_success.append(bool((group[system] == group.gold).all()))
    metric_rows.append(
        {
            "system": system,
            "order": "paired_counterfactual",
            "n": len(pair_success),
            "accuracy": float(np.mean(pair_success)) if pair_success else np.nan,
            "invalid_rate": float((~valid).mean()),
            "unobserved_world_state_leakage_rate": leakage_rate,
        }
    )
counterfactual_metrics = pd.DataFrame(metric_rows)
counterfactual_metrics.to_csv(RESULT_DIR / "counterfactual_metrics.csv", index=False)
print(counterfactual_metrics.to_string(index=False))
fig, ax = plt.subplots(figsize=(10, 5))
summary = counterfactual_metrics[counterfactual_metrics.order == "all"]
ax.bar(summary.system, summary.accuracy)
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.set_title("Exploratory counterfactual information-access diagnostic")
ax.tick_params(axis="x", rotation=25)
fig.tight_layout()
fig.savefig(PLOT_DIR / "counterfactual_overall_accuracy.png", dpi=200)
plt.show()
analysis_manifest = {
    "analysis_type": "exploratory cognitive and representational analysis",
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "base_model_changed": False,
    "retraining_performed": False,
    "answer_only_best_step": int(ordinary_completed["best_step"]),
    "subjectesis_best_step": int(subjectesis_completed["best_step"]),
    "answer_only_selected_checkpoint_exposures_approx": int(
        ordinary_completed["best_step"]
    )
    * 8,
    "subjectesis_selected_checkpoint_exposures_approx": int(
        subjectesis_completed["best_step"]
    )
    * 8,
    "probe_train_questions": int((probe_df.split == "train").sum()),
    "probe_validation_questions": int((probe_df.split == "validation").sum()),
    "probe_test_questions": int((probe_df.split == "test").sum()),
    "probe_split_unit": "dialogue",
    "probe_input": "same plain answer-only prompt for all systems",
    "representation": "final prompt-token hidden state from every layer including embedding output",
    "linear_probe": "standardized logistic regression, C=1.0, class_weight=balanced",
    "rsa_target": "benchmark-encoded belief-state geometry",
    "counterfactual_cases": int(len(counterfactual_out)),
    "counterfactual_status": "exploratory post-hoc diagnostic",
    "interpretation_limits": [
        "No claim of consciousness or biological equivalence.",
        "Probe targets are benchmark-encoded variables, not independent cognitive annotations.",
        "Adapter checkpoints differ in validation-selected training exposure.",
        "Counterfactual diagnostic was designed after the main benchmark results and is exploratory.",
    ],
}
(RESULT_DIR / "analysis_manifest.json").write_text(
    json.dumps(analysis_manifest, indent=2) + "\n"
)
summary = {
    "analysis_manifest": analysis_manifest,
    "process_summary": process_summary,
    "probe_final_layer": probe_results_df[
        probe_results_df.layer == N_LAYERS - 1
    ].to_dict(orient="records"),
    "probe_bootstrap": probe_bootstrap_df.to_dict(orient="records"),
    "rsa_final_layer": rsa_jackknife_df.to_dict(orient="records"),
    "counterfactual_metrics": counterfactual_metrics.to_dict(orient="records"),
}
(RESULT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("Analysis complete.")
print("Saved to:", RESULT_DIR)
