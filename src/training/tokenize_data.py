"""Tokenize the fixed supervision and mask prompt tokens from the loss."""

from pathlib import Path
import argparse, json, hashlib
from common import *


def encode(row, tok, max_length):
    """Apply the Qwen chat template to a user message and its supervised assistant
    completion. Verify that the prompt tokens are an unchanged prefix and that
    nothing needs truncation, then return all token IDs and the boundary where
    supervision starts.
    """
    messages = row["messages"]
    if [m["role"] for m in messages] != ["user", "assistant"]:
        raise ValueError("Expected user + assistant")
    prefix = tok.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    if not full.startswith(prefix):
        raise ValueError("Chat-template prefix mismatch: " + row["example_id"])
    prompt_ids = tok.encode(prefix, add_special_tokens=False)
    ids = tok.encode(full, add_special_tokens=False)
    # The loss mask depends on an exact prompt/completion boundary.
    # Do not guess a boundary if joint tokenization changes the prefix.
    if ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Token boundary mismatch: " + row["example_id"])
    if len(ids) > max_length:
        raise ValueError(
            f"{row['example_id']}: {len(ids)} tokens exceeds {max_length}. Nothing was truncated."
        )
    if len(ids) <= len(prompt_ids):
        raise ValueError("Empty target")
    if not full[len(prefix) :].startswith(messages[-1]["content"]):
        raise ValueError("Target text was altered by template")
    return (ids, len(prompt_ids))


def run(root):
    """Create or reuse checked token arrays for each requested training condition
    and save the tokenizer. Also tokenize answer-only validation prompts,
    keeping reference labels out of those model inputs and recording lengths
    and supervised-token counts.
    """
    import numpy as np
    from transformers import AutoTokenizer

    root = Path(root)
    c = load_config(root)
    cache = root / "tokenized"
    cache.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(
        c["model_id"], revision=c["model_revision"], trust_remote_code=False
    )
    tok.padding_side = "right"
    tok.save_pretrained(root / "tokenizer")
    th = identity({"template": tok.chat_template, "vocab": tok.get_vocab()})
    meta = {
        "model": c["model_id"],
        "revision": c["model_revision"],
        "enable_thinking": False,
        "tokenizer_hash": th,
        "max_length": c["max_length"],
        "conditions": {},
    }
    conditions = ["subjectesis", "ordinary"] if c["train_baseline"] else ["subjectesis"]
    for name in conditions:
        source = root / "preparation/data/sft" / TRAIN_FILES[name]
        fingerprint = identity(
            {"hash": sha(source), "tokenizer": th, "max_length": c["max_length"]}
        )
        mp = cache / (name + ".json")
        npfile = cache / (name + ".npz")
        if mp.exists() and npfile.exists():
            saved = read_json(mp)
            if saved.get("fingerprint") == fingerprint and saved.get(
                "array_sha256"
            ) == sha(npfile):
                meta["conditions"][name] = saved
                print("Reusing verified tokenization:", name, flush=True)
                continue
        ids_flat = []
        offsets = [0]
        prefixes = []
        rows = load_train(root, name)
        for i, r in enumerate(rows):
            ids, prefix = encode(r, tok, c["max_length"])
            ids_flat.extend(ids)
            offsets.append(len(ids_flat))
            prefixes.append(prefix)
            if (i + 1) % 5000 == 0:
                print(f"Tokenized {name}: {i + 1}/{len(rows)}", flush=True)
        lengths = np.diff(offsets)
        with npfile.with_suffix(".tmp").open("wb") as f:
            np.savez_compressed(
                f,
                ids=np.array(ids_flat, dtype=np.int32),
                offsets=np.array(offsets, dtype=np.int64),
                prefixes=np.array(prefixes, dtype=np.int32),
            )
        npfile.with_suffix(".tmp").replace(npfile)
        report = {
            "fingerprint": fingerprint,
            "source_sha256": sha(source),
            "array_sha256": sha(npfile),
            "examples": len(rows),
            "total_tokens": len(ids_flat),
            "supervised_tokens": int(sum(lengths) - sum(prefixes)),
            "max_tokens": int(max(lengths)),
            "longest_index": int(lengths.argmax()),
            "truncations": 0,
        }
        write_json(mp, report)
        meta["conditions"][name] = report
        print(name, report, flush=True)
        del rows, ids_flat
    # Validation needs input tokens and option letters, not supervised
    # completion labels. Reference answers are opened by the scoring stage.
    vpath = root / "preparation/data/evaluation/validation_inputs.jsonl"
    expected = read_json(root / "preparation/data/checksums.json")[
        "evaluation/validation_inputs.jsonl"
    ]
    if sha(vpath) != expected:
        raise ValueError("Validation inputs changed")
    val = []
    for r in read_jsonl(vpath):
        text = tok.apply_chat_template(
            [{"role": "user", "content": r["answer_only_prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) > c["max_length"]:
            raise ValueError("Validation input exceeds length limit; no truncation")
        val.append(
            {
                "record_id": r["record_id"],
                "task": r["task"],
                "choices": list(r["choices"]),
                "input_ids": ids,
            }
        )
    write_json(cache / "validation_inputs.json", val)
    write_json(root / "reports/tokenization.json", meta)
    print(
        "Exact Qwen tokenizer and response masks checked. Test labels not opened.",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    run(p.parse_args().root)
