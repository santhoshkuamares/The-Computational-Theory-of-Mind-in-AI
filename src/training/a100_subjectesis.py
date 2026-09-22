"""Recovered Subjectesis A100 continuation wrapper with the final gradient-checkpointing restoration.
The original run began on two T4 GPUs and resumed on one A100. This wrapper is not evidence of a fresh all-A100 rerun.
"""

import argparse
import os
import sys
import random
from pathlib import Path
from datetime import timedelta
from contextlib import nullcontext

root = Path(os.environ["SUBJECTESIS_ROOT"])
sys.path.insert(0, str(root / "code"))
import torch
import torch.distributed as dist
import engine
import train as original_train


def single_gpu_distributed_info():
    """Initialize the one-process A100 environment used by the continuation run."""
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if (rank, local, world) != (0, 0, 1):
        raise RuntimeError(
            f"Expected one process on one GPU, got rank={rank}, local={local}, world={world}"
        )
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    device = torch.device("cuda", 0)
    name = torch.cuda.get_device_name(device)
    if "A100" not in name.upper():
        raise RuntimeError(f"This optimized continuation expects A100; found: {name}")
    props = torch.cuda.get_device_properties(0)
    print(
        "\nA100 FAST MIGRATION MODE",
        {
            "gpu": name,
            "vram_GB": round(props.total_memory / 1024**3, 1),
            "world_size": 1,
            "effective_batch_size": 8,
            "strategy": "dynamic GPU microbatching",
        },
        flush=True,
    )
    return (rank, local, world, device)


original_train.distributed_info = single_gpu_distributed_info


def make_batch(examples, device):
    """Right-pad token sequences and mask padding labels out of the training loss."""
    if not examples:
        raise ValueError("Empty microbatch")
    max_len = max((len(ex["input_ids"]) for ex in examples))
    batch_size = len(examples)
    PAD_ID = 0
    input_ids = torch.full(
        (batch_size, max_len), PAD_ID, dtype=torch.long, device=device
    )
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
    for i, ex in enumerate(examples):
        ids = ex["input_ids"]
        labs = ex["labels"]
        if len(ids) != len(labs):
            raise ValueError("input_ids/labels length mismatch")
        n = len(ids)
        input_ids[i, :n] = torch.as_tensor(ids, dtype=torch.long, device=device)
        labels[i, :n] = torch.as_tensor(labs, dtype=torch.long, device=device)
    return (input_ids, labels)


def supervised_token_count(examples):
    """Count completion targets after the causal one-token shift."""
    return sum((sum((int(x != -100) for x in ex["labels"][1:])) for ex in examples))


def optimize_group_a100(
    ddp, local_examples, optimizer, scaler, device, world_size, lr, max_grad_norm=1.0
):
    """Process the same logical batch with smaller microbatches only when memory requires it."""
    if world_size != 1:
        raise RuntimeError("A100 optimized function requires world_size=1")
    ddp.train()
    base = ddp.module if hasattr(ddp, "module") else ddp
    for module_name, module in base.named_modules():
        enable_fn = getattr(module, "gradient_checkpointing_enable", None)
        if callable(enable_fn):
            enable_fn()
            print(
                "Gradient checkpointing enabled on:",
                module_name or "<root>",
                flush=True,
            )
            break
    count = supervised_token_count(local_examples)
    if count <= 0:
        raise ValueError("No supervised target tokens in group")
    denominator = float(count)
    params = [p for p in ddp.parameters() if p.requires_grad]
    for g in optimizer.param_groups:
        g["lr"] = lr
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device)
    python_rng = random.getstate()
    candidates = [8, 4, 2, 1]
    candidates = [x for x in candidates if x <= len(local_examples)]
    last_oom = None
    for microbatch_size in candidates:
        for overflow_attempt in range(5):
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, device)
            random.setstate(python_rng)
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            oom = False
            try:
                number_of_chunks = (
                    len(local_examples) + microbatch_size - 1
                ) // microbatch_size
                for chunk_i, start in enumerate(
                    range(0, len(local_examples), microbatch_size)
                ):
                    examples = local_examples[start : start + microbatch_size]
                    ids, labels = make_batch(examples, device)
                    sync_context = (
                        ddp.no_sync()
                        if hasattr(ddp, "no_sync") and chunk_i < number_of_chunks - 1
                        else nullcontext()
                    )
                    with sync_context:
                        with torch.autocast(
                            device_type="cuda", dtype=torch.float16, enabled=True
                        ):
                            nll = ddp(ids, labels)
                            loss = nll / denominator
                        if not torch.isfinite(nll):
                            raise FloatingPointError("Non-finite forward loss")
                        loss_total += float(nll.detach().item())
                        scaler.scale(loss).backward()
                    del ids, labels, nll, loss
            except torch.cuda.OutOfMemoryError as e:
                oom = True
                last_oom = e
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
            optimizer.zero_grad(set_to_none=True)
            scaler.update(new_scale=max(1.0, scaler.get_scale() / 2))
            print(
                "Non-finite gradients; retrying with scale",
                scaler.get_scale(),
                flush=True,
            )
        if not oom:
            raise FloatingPointError(
                "Adapter gradients remained non-finite after bounded loss-scale retries"
            )
    raise RuntimeError("Even microbatch=1 exhausted A100 memory") from last_oom


engine.optimize_group = optimize_group_a100
original_train.optimize_group = optimize_group_a100
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--smoke-only", action="store_true")
    a = p.parse_args()
    original_train.run(a.root, a.condition, a.smoke_only)
