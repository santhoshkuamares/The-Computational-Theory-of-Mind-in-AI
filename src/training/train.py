"""Training loop, validation selection and checkpoint recovery for the completed Qwen experiment."""

from __future__ import annotations
import argparse, gc, hashlib, json, math, os, random, shutil, sys, time, traceback
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from common import *
from engine import *
from model_runtime import build_network, frozen_sample


class Tokens:

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.ids = z["ids"]
        self.offsets = z["offsets"]
        self.prefixes = z["prefixes"]

    def __len__(self):
        return len(self.prefixes)

    # Prompt tokens use -100 so they do not become supervised targets.
    def __getitem__(self, i):
        ids = self.ids[self.offsets[i] : self.offsets[i + 1]].astype(np.int64).tolist()
        p = int(self.prefixes[i])
        return {"input_ids": ids, "labels": [-100] * p + ids[p:]}


def distributed_info():
    """Initialize the original two-process GPU training environment."""
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 2:
        raise RuntimeError("Launch exactly two processes with torchrun")
    torch.cuda.set_device(local)
    dist.init_process_group(
        "nccl", timeout=timedelta(minutes=30), device_id=torch.device("cuda", local)
    )
    return (rank, local, world, torch.device("cuda", local))


def rng_state(device):
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
        "python": random.getstate(),
    }


def restore_rng(s, device):
    torch.set_rng_state(s["cpu"])
    torch.cuda.set_rng_state(s["cuda"], device)
    random.setstate(s["python"])


def save_checkpoint(root, network, opt, scaler, progress, run_id, device):
    """Save adapter, optimizer, scaler, RNG and progress together before updating the latest pointer."""
    rank = dist.get_rank()
    states = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(rng_state(device), states, dst=0)
    if rank == 0:
        checkpoints = root / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        name = f"step_{progress['step']:07d}"
        dest = checkpoints / name
        tmp = checkpoints / (name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        network.save_pretrained(tmp / "adapter", safe_serialization=True)
        torch.save(
            {
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": states,
            },
            tmp / "optimizer.pt",
        )
        write_json(tmp / "progress.json", dict(progress, run_identity=run_id))
        files = {str(p.relative_to(tmp)): sha(p) for p in tmp.rglob("*") if p.is_file()}
        write_json(tmp / "files.json", files)
        if dest.exists():
            shutil.rmtree(dest)
        tmp.replace(dest)
        write_json(root / "latest.json", {"folder": name, "run_identity": run_id})
        complete = sorted(
            (
                p
                for p in checkpoints.glob("step_*")
                if p.is_dir() and (not p.name.endswith(".tmp"))
            )
        )
        for p in complete[:-2]:
            shutil.rmtree(p)
    dist.barrier()


def load_checkpoint(folder, network, opt, scaler, run_id, device):
    """Verify a checkpoint identity and restore its training state."""
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict

    for name, digest in read_json(folder / "files.json").items():
        if sha(folder / name) != digest:
            raise ValueError("Corrupt saved checkpoint: " + name)
    progress = read_json(folder / "progress.json")
    if progress["run_identity"] != run_id:
        raise ValueError(
            "Checkpoint model/data/code/settings differ; use a new RUN_FOLDER"
        )
    set_peft_model_state_dict(
        network, load_file(folder / "adapter/adapter_model.safetensors")
    )
    state = torch.load(folder / "optimizer.pt", map_location="cpu", weights_only=True)
    opt.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"][dist.get_rank()], device)
    return {k: v for k, v in progress.items() if k != "run_identity"}


@torch.no_grad()
def validate(wrapper, root, tokenizer, device, step):
    """Constrained first-answer-token validation, not a runtime Subjectesis-loop score."""
    rank = dist.get_rank()
    world = dist.get_world_size()
    rows = read_json(root / "tokenized/validation_inputs.json")
    refpath = root / "preparation/data/scoring_only/validation_references.jsonl"
    expected = read_json(root / "preparation/data/checksums.json")[
        "scoring_only/validation_references.jsonl"
    ]
    if sha(refpath) != expected:
        raise ValueError("Validation references changed")
    refs = {r["record_id"]: r["answer"] for r in read_jsonl(refpath)}
    codes = {k: tokenizer.encode(k, add_special_tokens=False) for k in "ABCDEFG"}
    if any((len(v) != 1 for v in codes.values())):
        raise ValueError("Answer option is not a single Qwen token")
    wrapper.eval()
    local = []
    for j in range(rank, len(rows), world):
        r = rows[j]
        inp = torch.tensor([r["input_ids"]], device=device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = wrapper.next_logits(inp)[0]
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite validation logits")
        allowed = r["choices"]
        values = [float(logits[codes[k][0]]) for k in allowed]
        pred = allowed[int(np.argmax(values))]
        local.append(
            {
                "record_id": r["record_id"],
                "task": r["task"],
                "prediction": pred,
                "reference": refs[r["record_id"]],
                "correct": pred == refs[r["record_id"]],
            }
        )
        if len(local) % 40 == 0 and rank == 0:
            print(
                "Validation progress:",
                min(j + 1, len(rows)),
                "/",
                len(rows),
                flush=True,
            )
    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    predictions = sorted(
        [r for part in gathered for r in part], key=lambda r: r["record_id"]
    )
    if len(predictions) != 319 or len({r["record_id"] for r in predictions}) != 319:
        raise ValueError("Validation coverage failure")
    scores = {
        task: sum((r["correct"] for r in predictions if r["task"] == task))
        / sum((r["task"] == task for r in predictions))
        for task in ["belief", "desire"]
    }
    result = {
        "step": step,
        "examples": 319,
        "scores": scores,
        "selection_score": sum(scores.values()) / 2,
        "protocol": "constrained next-letter accuracy; no review; validation only",
        "predictions": predictions,
    }
    wrapper.train()
    return result


def smoke(root, wrapper, ddp, network, data, c, opt, scaler, device):
    """Real longest-example backward, update, adapter reload. Discard every smoke update."""
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict

    rank = dist.get_rank()
    initial = adapter_copy(network)
    frozen = frozen_sample(network)
    lengths = np.diff(data.offsets)
    longest = int(lengths.argmax())
    ex = data[longest]
    print(
        f"[GPU {rank}] Longest-example gradient test: {len(ex['input_ids'])} tokens.",
        flush=True,
    )
    metrics = optimize_group(ddp, [ex], opt, scaler, device, 2, c["learning_rate"])
    changed = any(
        (
            not torch.equal(p.detach().cpu(), initial[n])
            for n, p in network.named_parameters()
            if n in initial
        )
    )
    if not changed:
        raise RuntimeError("No adapter weight changed in real-model test")
    if frozen_sample(network) != frozen:
        raise RuntimeError("Sampled frozen weights changed")
    probe = torch.tensor([ex["input_ids"][:16]], device=device)
    wrapper.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        before = wrapper.next_logits(probe).detach().cpu()
    smoke_path = root / "smoke_adapter"
    if rank == 0:
        network.save_pretrained(smoke_path, safe_serialization=True)
    dist.barrier()
    adapter_restore(network, initial)
    set_peft_model_state_dict(
        network, load_file(smoke_path / "adapter_model.safetensors")
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        after = wrapper.next_logits(probe).detach().cpu()
    if not torch.allclose(before, after, atol=0.0001, rtol=0.0001):
        raise RuntimeError("Adapter reload changed test logits")
    adapter_restore(network, initial)
    wrapper.train()
    opt.zero_grad(set_to_none=True)
    report = {
        "metrics": metrics,
        "sequence_tokens": len(ex["input_ids"]),
        "adapter_changed": True,
        "sampled_frozen_weights_unchanged": True,
        "saved_adapter_reload_logits_match": True,
        "smoke_updates_discarded": True,
        "peak_gpu_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    if rank == 0:
        write_json(root / "smoke_report.json", report)
        shutil.rmtree(smoke_path)
    dist.barrier()
    del initial, before, after
    gc.collect()
    torch.cuda.empty_cache()
    return report


def run(root, condition, smoke_only=False):
    root = Path(root)
    c = load_config(root)
    rank, local, world, device = distributed_info()
    out = root / "runs" / condition
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    torch.manual_seed(c["seed"])
    torch.cuda.manual_seed_all(c["seed"])
    random.seed(c["seed"])
    tok = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    data = Tokens(root / "tokenized" / f"{condition}.npz")
    if len(data) != 30266:
        raise ValueError("Training count changed")
    model_path = Path(read_json(root / "reports/model_download.json")["snapshot"])
    network = build_network(root, c, local, model_path)
    wrapper = CompletionLoss(network, c["loss_chunk_size"])
    fingerprint = identity(
        {
            k: hashlib.sha256(v.numpy().tobytes()).hexdigest()
            for k, v in adapter_copy(network).items()
        }
    )
    fingerprints = [None] * world
    dist.all_gather_object(fingerprints, fingerprint)
    if len(set(fingerprints)) != 1:
        raise RuntimeError("Adapter initializations differ across GPUs")
    ddp = DDP(
        wrapper,
        device_ids=[local],
        output_device=local,
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        init_sync=False,
    )
    params = trainable_parameters(network)
    if any(
        ("lora_" not in n for n, p in network.named_parameters() if p.requires_grad)
    ):
        raise RuntimeError("Unexpected training parameters")

    def optimizer():
        return torch.optim.AdamW(params, lr=c["learning_rate"], weight_decay=0.0)

    opt = optimizer()
    scaler = torch.amp.GradScaler("cuda", init_scale=16.0)
    methods = {
        k: c[k]
        for k in [
            "model_id",
            "model_revision",
            "training_method",
            "epochs",
            "rank",
            "alpha",
            "learning_rate",
            "seed",
            "effective_batch_size",
            "max_length",
            "warmup_fraction",
            "loss_chunk_size",
        ]
    }
    # A checkpoint is reusable only with the same data, settings, source and versions.
    run_id = identity(
        {
            "config": methods,
            "condition": condition,
            "tokens": read_json(root / "tokenized" / f"{condition}.json"),
            "code": {p.name: sha(p) for p in (root / "code").glob("*.py")},
            "versions": read_json(root / "environment_lock.json"),
        }
    )
    if (out / "completed.json").exists():
        done = read_json(out / "completed.json")
        if done["run_identity"] != run_id:
            raise ValueError("Completed run has different settings")
        if rank == 0:
            print(condition, "already completed; no retraining.", flush=True)
        dist.destroy_process_group()
        return
    latest = None
    if (out / "latest.json").exists():
        latest = out / "checkpoints" / read_json(out / "latest.json")["folder"]
    if not latest:
        smoke(out, wrapper, ddp, network, data, c, opt, scaler, device)
        opt = optimizer()
        scaler = torch.amp.GradScaler("cuda", init_scale=16.0)
    else:
        print(
            "Resuming verified checkpoint; prior real-model smoke report retained.",
            flush=True,
        )
    if smoke_only:
        if rank == 0:
            write_json(
                out / "status.json",
                {"status": "smoke_passed", "training_completed": False},
            )
        dist.destroy_process_group()
        return
    progress = {
        "step": 0,
        "epoch": 0,
        "next_group": 0,
        "examples_seen": 0,
        "target_tokens_seen": 0,
        "best_score": -1.0,
        "best_step": 0,
    }
    if latest:
        progress = load_checkpoint(latest, network, opt, scaler, run_id, device)

    def evaluate_and_select():
        result = validate(wrapper, root, tok, device, progress["step"])
        if result["selection_score"] > progress["best_score"]:
            progress["best_score"] = result["selection_score"]
            progress["best_step"] = progress["step"]
            if rank == 0:
                network.save_pretrained(out / "best_adapter", safe_serialization=True)
        if rank == 0:
            write_json(out / f"validation_{progress['step']:07d}.json", result)
            print(
                "Validation:",
                result["scores"],
                "best step:",
                progress["best_step"],
                flush=True,
            )
        dist.barrier()

    if not latest:
        evaluate_and_select()
        save_checkpoint(out, network, opt, scaler, progress, run_id, device)
    total_steps = math.ceil(len(data) / c["effective_batch_size"]) * c["epochs"]
    started = time.monotonic()
    previous = time.monotonic()
    paused = False
    session_start_step = progress["step"]
    for epoch in range(progress["epoch"], c["epochs"]):
        groups = epoch_groups(len(data), c["effective_batch_size"], c["seed"], epoch)
        begin = progress["next_group"] if epoch == progress["epoch"] else 0
        for gi in range(begin, len(groups)):
            group = groups[gi]
            local_examples = [data[i] for i in group[rank::world]]
            lr = learning_rate(
                progress["step"], total_steps, c["learning_rate"], c["warmup_fraction"]
            )
            metrics = optimize_group(
                ddp, local_examples, opt, scaler, device, world, lr
            )
            progress.update(
                step=progress["step"] + 1,
                epoch=epoch,
                next_group=gi + 1,
                examples_seen=progress["examples_seen"] + len(group),
                target_tokens_seen=progress["target_tokens_seen"]
                + metrics["target_tokens"],
            )
            if rank == 0:
                metrics.update(
                    step=progress["step"],
                    examples_seen=progress["examples_seen"],
                    lr=lr,
                    step_seconds=time.monotonic() - previous,
                )
                completed_here = max(1, progress["step"] - session_start_step)
                metrics["estimated_remaining_training_minutes"] = (
                    (time.monotonic() - started)
                    / completed_here
                    * max(0, total_steps - progress["step"])
                    / 60
                )
                with (out / "training_log.jsonl").open("a") as f:
                    f.write(json.dumps(metrics) + "\n")
                if progress["step"] <= 3 or progress["step"] % 10 == 0:
                    print(condition, metrics, flush=True)
            previous = time.monotonic()
            if progress["step"] % c["eval_steps"] == 0:
                evaluate_and_select()
            if progress["step"] % c["save_steps"] == 0:
                save_checkpoint(out, network, opt, scaler, progress, run_id, device)
            stop = torch.tensor(
                int(
                    (root / "STOP").exists()
                    or (
                        c["max_session_hours"] > 0
                        and time.monotonic() - started > c["max_session_hours"] * 3600
                    )
                ),
                device=device,
            )
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if stop.item():
                save_checkpoint(out, network, opt, scaler, progress, run_id, device)
                paused = True
                break
        if paused:
            break
        progress.update(epoch=epoch + 1, next_group=0)
    if not paused:
        evaluate_and_select()
        save_checkpoint(out, network, opt, scaler, progress, run_id, device)
        if progress["examples_seen"] != len(data) * c["epochs"]:
            raise RuntimeError("Final example exposure is incorrect")
        if rank == 0:
            network.save_pretrained(out / "final_adapter", safe_serialization=True)
            tok.save_pretrained(out / "final_adapter")
            write_json(
                out / "completed.json",
                dict(
                    progress,
                    run_identity=run_id,
                    model_id=MODEL_ID,
                    revision=MODEL_REVISION,
                    status="completed",
                    test_scored=False,
                ),
            )
    if rank == 0:
        write_json(
            out / "status.json",
            dict(
                progress,
                status="paused_resumable" if paused else "completed",
                training_completed=not paused,
            ),
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--condition", choices=["subjectesis", "ordinary"], required=True)
    p.add_argument("--smoke-only", action="store_true")
    a = p.parse_args()
    try:
        run(a.root, a.condition, a.smoke_only)
    except BaseException:
        f = Path(a.root) / "logs" / f"failure_rank{os.environ.get('RANK', '0')}.log"
        f.parent.mkdir(exist_ok=True)
        f.write_text(traceback.format_exc())
        raise
