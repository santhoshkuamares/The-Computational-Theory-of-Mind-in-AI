"""Train or resume the ordinary answer-only control with the recorded A100 settings."""

from project_setup import mount_drive, prepare_project, restore_training_source

mount_drive()
from pathlib import Path, PurePosixPath
import base64, io, zipfile, json, os

ROOT = Path("/content/answer_only_qwen35_4b_a100_v1")
DRIVE_DIR = Path("/content/drive/MyDrive/Subjectesis")
RESUME_ARCHIVE = DRIVE_DIR / "answer_only_a100_latest.zip"
import torch

if not torch.cuda.is_available():
    raise RuntimeError("Select Runtime -> Change runtime type -> A100 GPU.")
gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
print("GPU:", gpu_name, f"{gpu_mem:.1f} GiB")
if "A100" not in gpu_name.upper():
    raise RuntimeError(f"This notebook is configured for A100; found {gpu_name}.")
ROOT.mkdir(parents=True, exist_ok=True)
DRIVE_DIR.mkdir(parents=True, exist_ok=True)

prepare_project(ROOT)
settings = {
    "model_id": "Qwen/Qwen3.5-4B",
    "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "training_method": "NF4_QLoRA",
    "train_baseline": True,
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
if RESUME_ARCHIVE.exists() and (not (ROOT / "runs/ordinary/latest.json").exists()):
    import sys

    sys.path.insert(0, str(ROOT / "code"))
    restore_training_source(RESUME_ARCHIVE, ROOT, "ordinary")
    print(" Restored baseline recovery archive:", RESUME_ARCHIVE)
elif (ROOT / "runs/ordinary/latest.json").exists():
    print(" Existing local baseline run retained.")
else:
    print(" Fresh answer-only baseline. No Subjectesis adapter is loaded.")
print("ROOT:", ROOT)
print("Drive recovery:", RESUME_ARCHIVE)
from pathlib import Path
import subprocess, sys, os, json, importlib.metadata as md

ROOT = Path("/content/answer_only_qwen35_4b_a100_v1")


def run(cmd, env=None):
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
import torch

packages = [
    "torch",
    "torchao",
    "transformers",
    "peft",
    "bitsandbytes",
    "accelerate",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
]
versions = {}
for p in packages:
    try:
        versions[p] = md.version(p)
    except Exception:
        versions[p] = "unavailable"
versions["torch"] = torch.__version__
(ROOT / "environment_lock.json").write_text(json.dumps(versions, indent=2) + "\n")
print("Environment lock:", versions)
env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT / "code") + ":" + env.get("PYTHONPATH", "")
env["TOKENIZERS_PARALLELISM"] = "false"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
run([sys.executable, ROOT / "code/prepare.py", "--root", ROOT], env=env)
run([sys.executable, ROOT / "code/tokenize_data.py", "--root", ROOT], env=env)
run([sys.executable, ROOT / "code/download_model.py", "--root", ROOT], env=env)
report = json.loads((ROOT / "tokenized/ordinary.json").read_text())
assert report["examples"] == 30266
assert report["truncations"] == 0
print("\n BASELINE DATA + TOKENIZATION + MODEL READY")
print("Answer-only examples:", report["examples"])
print("Answer-only total tokens:", report["total_tokens"])
print("Answer-only supervised tokens:", report["supervised_tokens"])
print("Answer-only max tokens:", report["max_tokens"])
from pathlib import Path
import os, sys, json, subprocess, threading, shutil, time, zipfile, hashlib

ROOT = Path("/content/answer_only_qwen35_4b_a100_v1")
DRIVE_DIR = Path("/content/drive/MyDrive/Subjectesis")
WRAPPER = Path("/content/a100_answer_only_fast.py")
sys.path.insert(0, str(ROOT / "code"))
from common import read_json
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
WRAPPER = Path(__file__).resolve().parent / "training/a100_answer_only.py"


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_baseline(
    root, destination=Path("/content/answer_only_qwen35_4b_outputs.zip")
):
    root = Path(root)
    destination = Path(destination)
    tmp = destination.with_suffix(".tmp")
    files = []
    ordinary = root / "runs/ordinary"
    if ordinary.exists():
        files += [
            p
            for p in ordinary.rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and (not any((part.endswith(".tmp") for part in p.parts)))
            and (not p.is_symlink())
        ]
    for folder in ["logs", "reports", "code"]:
        d = root / folder
        if d.exists():
            files += [
                p
                for p in d.rglob("*")
                if p.is_file()
                and "__pycache__" not in p.parts
                and (not any((part.endswith(".tmp") for part in p.parts)))
                and (not p.is_symlink())
            ]
    files += [
        p
        for p in [
            root / "settings.json",
            root / "README.md",
            root / "environment_lock.json",
        ]
        if p.is_file()
    ]
    unique, seen = ([], set())
    for p in files:
        rp = str(p.relative_to(root))
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        manifest = {}
        for p in unique:
            rp = str(p.relative_to(root))
            digest = file_sha(p)
            z.write(p, rp)
            manifest[rp] = digest
        z.writestr(
            "recovery_manifest.json",
            json.dumps({"version": 1, "files": manifest}, indent=2),
        )
    os.replace(tmp, destination)
    with zipfile.ZipFile(destination) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"Recovery ZIP corruption at {bad}")
    return destination


def backup_to_drive():
    for attempt in range(3):
        try:
            archive = package_baseline(ROOT)
            partial = DRIVE_DIR / "answer_only_a100_latest.partial"
            target = DRIVE_DIR / "answer_only_a100_latest.zip"
            shutil.copy2(archive, partial)
            os.replace(partial, target)
            latest_path = ROOT / "runs/ordinary/latest.json"
            checkpoint = (
                read_json(latest_path)["folder"] if latest_path.exists() else "none"
            )
            print(f"\n Drive recovery updated: {checkpoint}", flush=True)
            return
        except FileNotFoundError as exc:
            if attempt == 2:
                print("\nBackup warning after retries:", repr(exc), flush=True)
                return
            time.sleep(5)
        except Exception as exc:
            print("\nBackup warning:", repr(exc), flush=True)
            return


stop_backup = threading.Event()


def backup_loop():
    while not stop_backup.wait(1800):
        backup_to_drive()


threading.Thread(target=backup_loop, daemon=True).start()
env = os.environ.copy()
env["ANSWER_ONLY_ROOT"] = str(ROOT)
env["PYTHONPATH"] = str(ROOT / "code") + ":" + env.get("PYTHONPATH", "")
env["TOKENIZERS_PARALLELISM"] = "false"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
cmd = [
    sys.executable,
    "-u",
    "-m",
    "torch.distributed.run",
    "--standalone",
    "--nnodes=1",
    "--nproc_per_node=1",
    str(WRAPPER),
    "--root",
    str(ROOT),
    "--condition",
    "ordinary",
]
print("\n" + "=" * 68)
print("STARTING / RESUMING MATCHED ANSWER-ONLY BASELINE")
print("Fresh base model: Qwen/Qwen3.5-4B")
print("Condition: ordinary / answer-only")
print("Effective batch: 8")
print("Sealed test: NOT SCORED")
print("=" * 68 + "\n")
try:
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"Baseline training exited with code {code}")
finally:
    stop_backup.set()
    backup_to_drive()
from pathlib import Path
import json, os, shutil, zipfile, hashlib

ROOT = Path("/content/answer_only_qwen35_4b_a100_v1")
DRIVE_DIR = Path("/content/drive/MyDrive/Subjectesis")
RUN = ROOT / "runs/ordinary"


def read_json_local(path):
    return json.loads(Path(path).read_text())


latest_file = RUN / "latest.json"
completed_file = RUN / "completed.json"
status_file = RUN / "status.json"
if not latest_file.exists():
    raise RuntimeError("No durable answer-only checkpoint exists yet.")
latest = read_json_local(latest_file)
checkpoint = RUN / "checkpoints" / latest["folder"]
progress = read_json_local(checkpoint / "progress.json")
print("\n=== ANSWER-ONLY BASELINE STATE ===")
print("Durable checkpoint:", latest["folder"])
print("Step:", progress["step"])
print("Examples seen:", progress["examples_seen"])
print("Next group:", progress["next_group"])
print("Best step:", progress["best_step"])
print("Best validation score:", progress["best_score"])
if status_file.exists():
    print("Status:", read_json_local(status_file))
if completed_file.exists():
    completed = read_json_local(completed_file)
    print("\n ONE EPOCH COMPLETED")
    print(json.dumps(completed, indent=2))
    if completed.get("test_scored") is not False:
        raise RuntimeError("Unexpected test-scoring state.")
else:
    print("\n Not complete yet. The archive below is resumable.")
if "package_baseline" not in globals():

    def file_sha(path):
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def package_baseline(
        root, destination=Path("/content/answer_only_qwen35_4b_outputs.zip")
    ):
        root = Path(root)
        destination = Path(destination)
        tmp = destination.with_suffix(".tmp")
        files = []
        for folder in ["runs/ordinary", "logs", "reports", "code"]:
            d = root / folder
            if d.exists():
                files += [
                    p
                    for p in d.rglob("*")
                    if p.is_file()
                    and "__pycache__" not in p.parts
                    and (not any((part.endswith(".tmp") for part in p.parts)))
                    and (not p.is_symlink())
                ]
        files += [
            p
            for p in [
                root / "settings.json",
                root / "README.md",
                root / "environment_lock.json",
            ]
            if p.is_file()
        ]
        unique, seen = ([], set())
        for p in files:
            rp = str(p.relative_to(root))
            if rp not in seen:
                seen.add(rp)
                unique.append(p)
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            manifest = {}
            for p in unique:
                rp = str(p.relative_to(root))
                digest = file_sha(p)
                z.write(p, rp)
                manifest[rp] = digest
            z.writestr(
                "recovery_manifest.json",
                json.dumps({"version": 1, "files": manifest}, indent=2),
            )
        os.replace(tmp, destination)
        with zipfile.ZipFile(destination) as z:
            bad = z.testzip()
            if bad:
                raise RuntimeError(f"Corrupt ZIP member: {bad}")
        return destination


archive = package_baseline(ROOT)
target = DRIVE_DIR / "answer_only_a100_latest.zip"
partial = DRIVE_DIR / "answer_only_a100_latest.partial"
shutil.copy2(archive, partial)
os.replace(partial, target)
print("\n VERIFIED DRIVE COPY")
print("Archive:", target)
print("Size MB:", round(target.stat().st_size / 1024**2, 1))
best_step = progress["best_step"]
best_file = RUN / f"validation_{best_step:07d}.json"
if best_file.exists():
    best = read_json_local(best_file)
    print("\nBest validation:")
    print(" step:", best["step"])
    print(" belief:", best["scores"]["belief"])
    print(" desire:", best["scores"]["desire"])
    print(" selection score:", best["selection_score"])
print("\nSealed RecToM test remains untouched. Do not score it yet.")
