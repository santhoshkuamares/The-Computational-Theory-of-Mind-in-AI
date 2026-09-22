"""Shared file, dataset, hashing and run-recovery helpers."""

from __future__ import annotations
import hashlib, json, os, random, shutil, stat, zipfile
from pathlib import Path, PurePosixPath

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TRAIN_FILES = {
    "subjectesis": "subjectesis_full_train.jsonl",
    "ordinary": "answer_only_matched_train.jsonl",
}


def sha(path):
    """Return a file checksum by reading the file in one-megabyte chunks. Chunked
    reading also works for large checkpoints without loading the whole file
    into memory.
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def identity(value):
    """Convert a Python value to JSON with a fixed key order and hash that text.
    Equivalent settings therefore receive the same identifier, which is used to
    check whether a saved run can be reused.
    """
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_json(path):
    """Load one UTF-8 JSON file into a Python value. Training uses this helper to
    read settings, manifests and progress consistently.
    """
    return json.loads(Path(path).read_text(encoding="utf8"))


def read_jsonl(path):
    """Load each non-empty line as a separate JSON record. The original order is
    retained so training schedules and saved predictions remain traceable.
    """
    with Path(path).open(encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    """Write readable JSON to a temporary file and then replace the destination.
    This avoids leaving a partly written settings or progress file if a write
    is interrupted.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf8",
    )
    os.replace(tmp, path)


def safe_extract(archive, destination, overwrite=False):
    """Check ZIP members for damaged content, duplicate names, unsafe paths and
    conflicting existing files before extracting them. This restores experiment
    inputs without silently replacing different local data unless overwrite is
    explicitly requested.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP CRC failure")
        members = z.infolist()
        if len({x.filename for x in members}) != len(members):
            raise ValueError("Duplicate ZIP members")
        for m in members:
            p = PurePosixPath(m.filename)
            if (
                p.is_absolute()
                or ".." in p.parts
                or "\\" in m.filename
                or stat.S_ISLNK(m.external_attr >> 16)
            ):
                raise ValueError("Unsafe ZIP member: " + m.filename)
            if m.is_dir():
                continue
            target = destination.joinpath(*p.parts)
            if target.is_symlink():
                raise ValueError("Refusing symlink destination")
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Path escape")
            data = z.read(m)
            if target.exists() and target.read_bytes() != data and (not overwrite):
                raise ValueError("Existing file differs: " + str(target))
        for m in members:
            if m.is_dir():
                continue
            target = destination.joinpath(*PurePosixPath(m.filename).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or overwrite:
                target.write_bytes(z.read(m))


def load_config(root):
    """Read settings.json and verify the pinned model, NF4 QLoRA method and basic
    training limits. Return the settings only when they match the assumptions
    used by this implementation.
    """
    c = read_json(Path(root) / "settings.json")
    if c["model_id"] != MODEL_ID or c["model_revision"] != MODEL_REVISION:
        raise ValueError("Model identity changed")
    if c["training_method"] != "NF4_QLoRA":
        raise ValueError("This release uses explicit NF4 QLoRA")
    if (
        c["epochs"] < 1
        or c["effective_batch_size"] % 2
        or c["effective_batch_size"] < 2
    ):
        raise ValueError("Invalid training settings")
    if c["max_length"] < 64 or c["save_steps"] < 1 or c["eval_steps"] < 1:
        raise ValueError("Invalid limits")
    return c


def load_train(root, condition):
    """Load one allowed training condition and verify its checksum, 30,266
    examples, unique IDs and training-only dialogue membership. Checking
    message roles and non-empty text prevents malformed or held-out records
    from entering the training loop.
    """
    if condition not in TRAIN_FILES:
        raise ValueError("Select one allowlisted training condition")
    data = Path(root) / "preparation/data"
    m = read_json(data / "manifest.json")
    checks = read_json(data / "checksums.json")
    path = data / "sft" / TRAIN_FILES[condition]
    if sha(path) != checks["sft/" + path.name]:
        raise ValueError("Training hash mismatch")
    rows = read_jsonl(path)
    if len(rows) != 30266:
        raise ValueError("Expected 30,266 examples")
    train = set(m["dialogue_ids"]["train"])
    val = set(m["dialogue_ids"]["validation"])
    test = set(m["dialogue_ids"]["test"])
    if train & val or train & test or val & test:
        raise ValueError("Split overlap")
    if len({r["example_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate examples")
    for r in rows:
        if r["split"] != "train" or str(r["dialogue_id"]) not in train:
            raise ValueError("Non-training row")
        if [x["role"] for x in r["messages"]] != ["user", "assistant"]:
            raise ValueError("Unexpected role sequence")
        if any(
            (
                not isinstance(x["content"], str) or not x["content"]
                for x in r["messages"]
            )
        ):
            raise ValueError("Invalid messages")
    return rows


def epoch_groups(n, batch_size, seed, epoch):
    """Shuffle example indices with seed plus epoch, then divide them into logical
    batches without dropping the final group. Even sizes support the original
    two-GPU partition while ensuring every scheduled example is used exactly
    once per epoch.
    """
    if n % 2 or batch_size % 2 or batch_size < 2:
        raise ValueError(
            "Two-rank exact exposure requires an even row count and batch size"
        )
    order = list(range(n))
    random.Random(seed + epoch).shuffle(order)
    return [order[i : i + batch_size] for i in range(0, n, batch_size)]


def package_outputs(root):
    """Collect run files, logs, reports, source and settings into a recovery ZIP
    and return its path. A manifest stores file checksums, while temporary
    files and caches are omitted so the archive contains completed artifacts.
    """
    root = Path(root)
    target = root.parent / "subjectesis_qwen35_4b_outputs.zip"
    files = []
    for folder in ["runs", "logs", "reports", "code"]:
        if (root / folder).exists():
            files += [
                p
                for p in (root / folder).rglob("*")
                if p.is_file()
                and (not any((part.endswith(".tmp") for part in p.parts)))
                and ("__pycache__" not in p.parts)
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
    tmp = target.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in files:
            z.write(p, str(p.relative_to(root)))
        z.writestr(
            "recovery_manifest.json",
            json.dumps(
                {
                    "version": 1,
                    "files": {str(p.relative_to(root)): sha(p) for p in files},
                }
            ),
        )
    os.replace(tmp, target)
    return target


def restore_runs(archive, root):
    """Verify archive checksums and matching settings, then restore only files
    under runs/. Reject path escapes and conflicting local files so recovery
    cannot silently mix different experiments or replace executable source.
    """
    root = Path(root)
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read("recovery_manifest.json"))
        if json.loads(z.read("settings.json")) != read_json(root / "settings.json"):
            raise ValueError("Recovery settings differ")
        for n, digest in manifest["files"].items():
            if hashlib.sha256(z.read(n)).hexdigest() != digest:
                raise ValueError("Recovery checksum failure: " + n)
        for n in manifest["files"]:
            p = PurePosixPath(n)
            if p.is_absolute() or ".." in p.parts or "\\" in n:
                raise ValueError("Unsafe recovery path")
            if p.parts[0] != "runs":
                continue
            dest = root.joinpath(*p.parts)
            if not dest.resolve().is_relative_to(root.resolve()):
                raise ValueError("Recovery path escape")
            if dest.exists() and dest.read_bytes() != z.read(n):
                raise ValueError("Recovery conflicts with local run")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(n))
