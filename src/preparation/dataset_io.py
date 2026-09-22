"""Revision-4 data preparation: dataset io."""

import json
import hashlib
from pathlib import Path

TRAIN_FILES = {
    "subjectesis_full_train.jsonl",
    "answer_only_matched_train.jsonl",
    "subjectesis_state_only_train.jsonl",
    "subjectesis_auxiliary_train.jsonl",
    "answer_only_clean_train.jsonl",
}


def load_training(data_root, filename):
    root = Path(data_root)
    if filename not in TRAIN_FILES:
        raise ValueError(
            "Select an explicit SFT training file; evaluation and review queues cannot be loaded for training"
        )
    manifest = json.loads((root / "manifest.json").read_text())
    expected = json.loads((root / "checksums.json").read_text())["sft/" + filename]
    raw = (root / "sft" / filename).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(
            "Training file differs from the validated conversion; rebuild from reviewed source inputs"
        )
    train_ids = set(manifest["dialogue_ids"]["train"])
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    for row in rows:
        if row.get("split") != "train" or str(row["dialogue_id"]) not in train_ids:
            raise ValueError("Non-training dialogue in SFT file")
        messages = row["messages"]
        if [m["role"] for m in messages] != ["user", "assistant"]:
            raise ValueError(
                "SFT examples must be one user prompt and one assistant completion"
            )
        if not messages[1]["content"]:
            raise ValueError("Empty training target")
    return rows


def encode_training_example(row, tokenizer, max_tokens=4096):
    messages = row["messages"]
    if [m["role"] for m in messages] != ["user", "assistant"]:
        raise ValueError("Invalid role sequence")
    prefix = (
        "<start_of_turn>user\n"
        + messages[0]["content"]
        + "<end_of_turn>\n<start_of_turn>model\n"
    )
    full = prefix + messages[1]["content"] + "<end_of_turn>\n"
    prompt_ids = tokenizer.encode(prefix, add_bos=True)
    input_ids = tokenizer.encode(full, add_bos=True)
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Tokenizer changed the prompt/response boundary; do not train with a guessed loss mask"
        )
    if len(input_ids) > max_tokens:
        raise ValueError(
            f"{row['example_id']} exceeds {max_tokens} tokens; no truncation was performed"
        )
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    if all((x == -100 for x in labels)):
        raise ValueError("No supervised completion tokens")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(input_ids) - len(prompt_ids),
    }


def pad_batch(examples, pad_token_id, length=None):
    length = max((len(x["input_ids"]) for x in examples)) if length is None else length
    result = {"input_ids": [], "attention_mask": [], "labels": []}
    for example in examples:
        remaining = length - len(example["input_ids"])
        if remaining < 0:
            raise ValueError("Padding length would truncate an example")
        result["input_ids"].append(example["input_ids"] + [pad_token_id] * remaining)
        result["attention_mask"].append(example["attention_mask"] + [0] * remaining)
        result["labels"].append(example["labels"] + [-100] * remaining)
    return result
