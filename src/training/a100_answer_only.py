"""Executed single-A100 answer-only optimizer wrapper. Keeps gradient checkpointing enabled."""

import argparse
import os
import sys
import random
from pathlib import Path
from datetime import timedelta
from contextlib import nullcontext

root = Path(os.environ["ANSWER_ONLY_ROOT"])
sys.path.insert(0, str(root / "code"))
import torch
import torch.distributed as dist
import engine
import train as original_train


def single_gpu_distributed_info():
    """Initialize a one-process NCCL group for the answer-only run. Require one
    A100 and return the same rank/device interface as the original training
    loop so its checkpoint and validation code can be reused.
    """
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if (rank, local, world) != (0, 0, 1):
        raise RuntimeError(
            f"Expected one A100 process; got rank={rank}, local={local}, world={world}"
        )
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    device = torch.device("cuda", 0)
    name = torch.cuda.get_device_name(device)
    if "A100" not in name.upper():
        raise RuntimeError(f"Expected A100, found {name}")
    props = torch.cuda.get_device_properties(0)
    print(
        "ANSWER-ONLY A100 MODE:",
        {
            "gpu": name,
            "vram_GB": round(props.total_memory / 1024**3, 1),
            "world_size": 1,
            "effective_batch_size": 8,
            "precision": "FP16 autocast",
            "gradient_checkpointing": "ON",
            "strategy": "dynamic microbatch 8 -> 4 -> 2 -> 1",
        },
        flush=True,
    )
    return (rank, local, world, device)


original_train.distributed_info = single_gpu_distributed_info


def make_batch(examples, device):
    """Right-pad examples to the longest sequence in a microbatch and place the
    tensors on the GPU. Padding labels remain -100, so padded positions do not
    become supervised targets; the returned pair contains input IDs and labels.
    """
    if not examples:
        raise ValueError("Empty microbatch")
    max_len = max((len(ex["input_ids"]) for ex in examples))
    b = len(examples)
    ids = torch.zeros((b, max_len), dtype=torch.long, device=device)
    labels = torch.full((b, max_len), -100, dtype=torch.long, device=device)
    for i, ex in enumerate(examples):
        n = len(ex["input_ids"])
        if n != len(ex["labels"]):
            raise ValueError("input_ids/labels length mismatch")
        ids[i, :n] = torch.as_tensor(ex["input_ids"], dtype=torch.long, device=device)
        labels[i, :n] = torch.as_tensor(ex["labels"], dtype=torch.long, device=device)
    return (ids, labels)


def supervised_token_count(examples):
    """Count labels other than -100 after the causal one-token shift. This
    denominator gives each supervised target equal weight even when examples
    have different lengths or microbatch sizes.
    """
    return sum((sum((int(v != -100) for v in ex["labels"][1:])) for ex in examples))


def optimize_group_a100(
    ddp,
    local_examples,
    optimizer,
    scaler,
    device,
    world_size_argument,
    lr,
    max_grad_norm=1.0,
):
    """Apply one logical-batch update on the A100, accumulating losses divided by
    the same total target-token count. Try smaller microbatches after
    out-of-memory errors and bounded loss-scale retries for non-finite
    gradients, restoring random states before each attempt and returning the
    successful update metrics.
    """
    actual_world = dist.get_world_size()
    if actual_world != 1:
        raise RuntimeError(f"A100 baseline requires world_size=1, got {actual_world}")
    ddp.train()
    denominator = float(supervised_token_count(local_examples))
    if denominator <= 0:
        raise ValueError("No supervised target tokens in optimizer group")
    params = [p for p in ddp.parameters() if p.requires_grad]
    for group in optimizer.param_groups:
        group["lr"] = lr
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device)
    python_rng = random.getstate()
    candidates = [m for m in (8, 4, 2, 1) if m <= len(local_examples)]
    last_oom = None
    for microbatch_size in candidates:
        for overflow_attempt in range(5):
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, device)
            random.setstate(python_rng)
            # Start this retry with clean gradients. Losses from a failed
            # microbatch or overflow attempt must not carry into the next attempt.
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            oom = False
            try:
                chunks = (len(local_examples) + microbatch_size - 1) // microbatch_size
                for chunk_i, start in enumerate(
                    range(0, len(local_examples), microbatch_size)
                ):
                    examples = local_examples[start : start + microbatch_size]
                    ids, labels = make_batch(examples, device)
                    sync_ctx = (
                        ddp.no_sync()
                        if hasattr(ddp, "no_sync") and chunk_i < chunks - 1
                        else nullcontext()
                    )
                    with sync_ctx:
                        with torch.autocast(
                            device_type="cuda", dtype=torch.float16, enabled=True
                        ):
                            nll = ddp(ids, labels)
                            # Normalize every microbatch by the full logical-batch token count.
                            # Adding these gradients reproduces that token-weighted objective.
                            loss = nll / denominator
                        if not torch.isfinite(nll):
                            raise FloatingPointError("Non-finite forward loss")
                        loss_total += float(nll.detach())
                        scaler.scale(loss).backward()
                    del ids, labels, nll, loss
            except torch.cuda.OutOfMemoryError as exc:
                oom = True
                last_oom = exc
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(
                    f"A100 microbatch {microbatch_size} OOM; trying smaller microbatch...",
                    flush=True,
                )
            if oom:
                break
            scaler.unscale_(optimizer)
            finite = all(
                (p.grad is None or torch.isfinite(p.grad).all().item() for p in params)
            )
            if finite:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    params, max_grad_norm, error_if_nonfinite=True
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                return {
                    "loss": loss_total / denominator,
                    "target_tokens": int(denominator),
                    "grad_norm": float(grad_norm),
                    "scale": float(scaler.get_scale()),
                    "overflow_retries": overflow_attempt,
                    "a100_microbatch": microbatch_size,
                    "a100_peak_vram_gib": torch.cuda.max_memory_allocated(device)
                    / 1024**3,
                }
            if scaler.get_scale() <= 1:
                raise FloatingPointError(
                    "Non-finite adapter gradients at minimum scale"
                )
            # Start this retry with clean gradients. Losses from a failed
            # microbatch or overflow attempt must not carry into the next attempt.
            optimizer.zero_grad(set_to_none=True)
            scaler.update(new_scale=max(1.0, scaler.get_scale() / 2))
        if not oom:
            raise FloatingPointError(
                "Adapter gradients remained non-finite after bounded loss-scale retries"
            )
    raise RuntimeError("Even microbatch=1 exhausted A100 memory") from last_oom


# Replace the imported optimizer helper with the recorded A100 version.
# The training loop, validation selection and checkpoint code stay shared.
engine.optimize_group = optimize_group_a100
original_train.optimize_group = optimize_group_a100
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--smoke-only", action="store_true")
    a = p.parse_args()
    if a.condition != "ordinary":
        raise RuntimeError(
            "This notebook may train only the matched answer-only condition."
        )
    original_train.run(a.root, a.condition, a.smoke_only)
