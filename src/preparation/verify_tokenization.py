"""Revision-4 data preparation: verify tokenization."""

import argparse
import hashlib
import json
from pathlib import Path
from dataset_io import load_training, encode_training_example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument(
        "--backend", choices=["gemma", "sentencepiece"], default="gemma"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.backend == "gemma":
        from gemma import gm

        tokenizer = gm.text.Gemma3Tokenizer(path=args.tokenizer_path)
    else:
        import sentencepiece as spm

        class Tokenizer:

            def __init__(self, path):
                self.sp = spm.SentencePieceProcessor(model_file=path)
                assert self.sp.bos_id() == 2 and self.sp.eos_id() == 1
                for token in ["<start_of_turn>", "<end_of_turn>"]:
                    assert self.sp.encode(token) == [self.sp.piece_to_id(token)]

            def encode(self, text, add_bos=False):
                return ([self.sp.bos_id()] if add_bos else []) + self.sp.encode(text)

        tokenizer = Tokenizer(args.tokenizer_path)
    report = {}
    for filename in ("subjectesis_full_train.jsonl", "answer_only_matched_train.jsonl"):
        rows = load_training(root / "data", filename)
        stats = []
        for row in rows:
            encoded = encode_training_example(row, tokenizer, max_tokens=4096)
            boundary = encoded["prompt_tokens"]
            assert all((x == -100 for x in encoded["labels"][:boundary]))
            assert encoded["labels"][boundary:] == encoded["input_ids"][boundary:]
            stats.append((len(encoded["input_ids"]), encoded["completion_tokens"]))
        report[filename] = {
            "backend": args.backend,
            "real_model_stack_executed": False,
            "examples": len(rows),
            "total_tokens": sum((x[0] for x in stats)),
            "supervised_tokens": sum((x[1] for x in stats)),
            "max_sequence_tokens": max((x[0] for x in stats)),
            "truncated": 0,
            "prompt_loss_mask_checks_passed": len(rows),
            "file_sha256": hashlib.sha256(
                (root / "data" / "sft" / filename).read_bytes()
            ).hexdigest(),
            "tokenizer_sha256": hashlib.sha256(
                Path(args.tokenizer_path).read_bytes()
            ).hexdigest(),
        }
    (root / "token_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
