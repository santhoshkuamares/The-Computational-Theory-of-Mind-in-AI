"""Checks and helpers used by the original training pipeline."""

import os, copy, json
from pathlib import Path
from datetime import timedelta
import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tests_cpu import Tiny, sample
from engine import CompletionLoss, trainable_parameters, adapter_copy, optimize_group


def main():
    torch.set_num_threads(1)
    rank = int(os.environ["RANK"])
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    torch.manual_seed(88)
    n = Tiny()
    reference = copy.deepcopy(n)
    w = CompletionLoss(n, 2)
    ddp = DDP(w, find_unused_parameters=False, init_sync=False, broadcast_buffers=False)
    opt = torch.optim.SGD(trainable_parameters(n), lr=0.02)
    refopt = torch.optim.SGD(trainable_parameters(reference), lr=0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    records = [sample(i) for i in range(10)]
    for group in [records[:8], records[8:]]:
        optimize_group(ddp, group[rank::2], opt, scaler, torch.device("cpu"), 2, 0.02)
        refopt.zero_grad()
        rw = CompletionLoss(reference, 100)
        den = sum((sum((t != -100 for t in r["labels"][1:])) for r in group))
        loss = (
            sum(
                (
                    rw(torch.tensor([r["input_ids"]]), torch.tensor([r["labels"]]))
                    for r in group
                )
            )
            / den
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters(reference), 1)
        refopt.step()
        for k, v in adapter_copy(n).items():
            torch.testing.assert_close(
                v, adapter_copy(reference)[k], atol=2e-07, rtol=2e-06
            )
    if rank == 0:
        print(
            "TWO-PROCESS CPU DDP CHECK PASSED: full group + short final group match serial gradients/updates.",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
