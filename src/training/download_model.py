"""Download and verify the pinned Qwen model and tokenizer."""

from pathlib import Path, PurePosixPath
import argparse, os, shutil
from common import load_config, read_json, write_json, sha

RESERVE_BYTES = 4 * 2**30
FALLBACK_TENSOR_BYTES = 9319737856


def required_space(weight_bytes, cached_bytes, reserve=RESERVE_BYTES):
    return max(0, int(weight_bytes) - int(cached_bytes)) + int(reserve)


def shard_name(name):
    p = PurePosixPath(name)
    if (
        p.is_absolute()
        or ".." in p.parts
        or len(p.parts) != 1
        or (not name.endswith(".safetensors"))
    ):
        raise ValueError("Invalid checkpoint shard name: " + name)
    return name


def run(root):
    from huggingface_hub import (
        hf_hub_download,
        snapshot_download,
        get_hf_file_metadata,
        hf_hub_url,
    )
    from safetensors import safe_open

    root = Path(root)
    c = load_config(root)
    cache = root / "hf_cache/hub"
    cache.mkdir(parents=True, exist_ok=True)
    idx = Path(
        hf_hub_download(
            c["model_id"],
            "model.safetensors.index.json",
            revision=c["model_revision"],
            cache_dir=str(cache),
        )
    )
    index = read_json(idx)
    wm = index["weight_map"]
    shards = sorted({shard_name(x) for x in wm.values()})
    metadata = {}
    metadata_errors = []
    for name in shards:
        try:
            m = get_hf_file_metadata(
                hf_hub_url(c["model_id"], name, revision=c["model_revision"]),
                token=os.environ.get("HF_TOKEN") or None,
                timeout=30,
            )
            metadata[name] = {"size": m.size, "etag": m.etag}
        except Exception as exc:
            metadata_errors.append({"shard": name, "type": type(exc).__name__})
    total = sum((m["size"] for m in metadata.values() if m.get("size")))
    exact_sizes = len(metadata) == len(shards) and all(
        (m.get("size") for m in metadata.values())
    )
    if not exact_sizes:
        total = (
            int(index.get("metadata", {}).get("total_size", FALLBACK_TENSOR_BYTES))
            + 16 * 2**20
        )
    cached = 0
    for name in shards:
        path = idx.parent / name
        if not path.is_file():
            continue
        size = path.stat().st_size
        if (
            name in metadata
            and metadata[name].get("size")
            and (size != metadata[name]["size"])
        ):
            raise RuntimeError(
                "Cached shard size mismatch: "
                + name
                + ". No files deleted; inspect this model cache."
            )
        cached += size
    need = required_space(total, cached)
    free = shutil.disk_usage(cache).free
    storage = {
        "expected_weight_bytes": total,
        "cached_weight_bytes": cached,
        "reserve_bytes": RESERVE_BYTES,
        "free_bytes": free,
        "required_free_bytes": need,
        "shortfall_bytes": max(0, need - free),
        "sizes_from_remote_metadata": bool(exact_sizes),
        "metadata_errors": metadata_errors,
    }
    write_json(root / "reports/storage_preflight.json", storage)
    print(
        f"4B disk check: {free / 2 ** 30:.2f} GiB free; {need / 2 ** 30:.2f} GiB required (includes {RESERVE_BYTES / 2 ** 30:.0f} GiB checkpoint/output reserve).",
        flush=True,
    )
    if free < need:
        raise RuntimeError(
            f"Disk is short by {(need - free) / 2 ** 30:.2f} GiB. Use a fresh Kaggle session without old Gemma/9B caches, or remove only an old unneeded cache. This notebook never deletes your model caches, checkpoints or dataset automatically."
        )
    path = Path(
        snapshot_download(
            c["model_id"],
            revision=c["model_revision"],
            cache_dir=str(cache),
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "vocab.json",
                "merges.txt",
                "chat_template.jinja",
                "model.safetensors.index.json",
                *shards,
            ],
            max_workers=2,
        )
    )
    checked = []
    for name in shards:
        file = path / name
        with safe_open(file, framework="pt", device="cpu") as f:
            expected = {k for k, v in wm.items() if v == name}
            if set(f.keys()) != expected:
                raise RuntimeError(
                    "Checkpoint index and shard tensor keys differ: " + name
                )
        etag = str(metadata.get(name, {}).get("etag") or "").strip('"')
        verified = False
        if len(etag) == 64 and all((x in "0123456789abcdef" for x in etag.lower())):
            if sha(file) != etag.lower():
                raise RuntimeError("Downloaded weight checksum mismatch: " + name)
            verified = True
        checked.append(
            {"file": name, "bytes": file.stat().st_size, "sha256_checked": verified}
        )
    cfg = read_json(path / "config.json")
    t = cfg.get("text_config", cfg)
    if (
        not t.get("tie_word_embeddings")
        or t["hidden_size"] != 2560
        or t["num_hidden_layers"] != 32
    ):
        raise RuntimeError(
            "Unexpected checkpoint configuration; refusing to change the architecture."
        )
    report = {
        "model_id": c["model_id"],
        "revision": c["model_revision"],
        "snapshot": str(path),
        "shards": len(shards),
        "indexed_tensors": len(wm),
        "weight_files": checked,
        "storage": storage,
        "tied_embeddings": True,
    }
    write_json(root / "reports/model_download.json", report)
    print("Official pinned Qwen3.5-4B checkpoint ready.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    run(p.parse_args().root)
