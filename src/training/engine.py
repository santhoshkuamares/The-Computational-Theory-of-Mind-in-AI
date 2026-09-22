"""Completion-only loss and optimizer helpers from the executed source package."""

from __future__ import annotations
from contextlib import nullcontext
import copy, math
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class CompletionLoss(torch.nn.Module):
    """Compute causal loss only at supervised completion tokens, in small vocabulary-head chunks."""

    def __init__(self, network, chunk_size=32):
        """Store the language model and the number of supervised positions
        processed in each vocabulary-head chunk. Smaller chunks reduce peak
        memory while the wrapper still sums loss across every supervised token.
        """
        super().__init__()
        self.network = network
        self.chunk_size = chunk_size

    def core(self):
        """Return the underlying language model, unwrapping the adapter container
        when present. The loss code needs direct access to its transformer
        backbone and output vocabulary head.
        """
        return (
            self.network.get_base_model()
            if hasattr(self.network, "get_base_model")
            else self.network
        )

    def forward(self, input_ids, labels):
        """Compute the summed next-token loss only where labels are not masked
        with -100. Run the backbone once, shift hidden states against their
        target tokens, then apply the vocabulary head in checkpointed chunks to
        reduce memory use.
        """
        core = self.core()
        hidden = core.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        # Position t predicts token t+1. Shift the labels and hidden states
        # together, then exclude prompt and padding targets using the -100 mask.
        mask = labels[:, 1:] != -100
        selected = hidden[:, :-1, :][mask]
        targets = labels[:, 1:][mask]
        if targets.numel() == 0:
            raise ValueError("Batch has no supervised next-token targets")
        losses = []

        def head_loss(h, y):
            """Apply the vocabulary head to one group of hidden states and sum its
            cross-entropy losses. Keeping this operation in a small function
            allows PyTorch to recompute it during backpropagation instead of
            storing the full vocabulary logits.
            """
            return F.cross_entropy(core.lm_head(h).float(), y, reduction="sum")

        for offset in range(0, len(targets), self.chunk_size):
            h = selected[offset : offset + self.chunk_size]
            y = targets[offset : offset + self.chunk_size]
            if torch.is_grad_enabled() and h.requires_grad:
                losses.append(checkpoint(head_loss, h, y, use_reentrant=False))
            else:
                losses.append(head_loss(h, y))
        return torch.stack(losses).sum()

    @torch.no_grad()
    def next_logits(self, input_ids):
        """Return vocabulary scores at the last input position without computing
        gradients. Validation uses these scores to select an allowed answer
        letter without generating a long response.
        """
        core = self.core()
        hidden = core.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        return core.lm_head(hidden[:, -1, :]).float()


def trainable_parameters(network):
    """Return parameters whose requires_grad flag is true and reject an empty set.
    Passing only these parameters to the optimizer limits updates to the
    intended trainable adapter weights.
    """
    params = [p for n, p in network.named_parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable adapter parameters")
    return params


def adapter_copy(network):
    """Make independent CPU copies of all trainable parameters, indexed by name.
    The smoke check uses this snapshot to undo its temporary trial update.
    """
    return {
        n: p.detach().cpu().clone()
        for n, p in network.named_parameters()
        if p.requires_grad
    }


def adapter_restore(network, values):
    """Copy a saved parameter snapshot back into the trainable weights after
    checking that the names match. Device and dtype conversion make the
    restoration work even though the snapshot was stored on CPU.
    """
    expected = {n for n, p in network.named_parameters() if p.requires_grad}
    if set(values) != expected:
        raise ValueError("Adapter parameter names differ")
    with torch.no_grad():
        for n, p in network.named_parameters():
            if n in values:
                p.copy_(values[n].to(p.device, p.dtype))


def learning_rate(step, total, peak, warmup_fraction):
    """Calculate the learning rate for one optimizer step using linear warmup
    followed by cosine decay. The function uses the total planned steps so
    interrupted sessions continue on the same schedule.
    """
    warmup = max(1, int(total * warmup_fraction))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * min(1, progress)))


def optimize_group(
    ddp, local_examples, optimizer, scaler, device, world_size, lr, max_grad_norm=1.0
):
    """Apply one optimizer update for a logical batch, normalizing by its total
    supervised-token count across GPUs. Accumulate example losses, synchronize
    gradients at the last example, clip finite gradients and retry bounded FP16
    overflows before returning training metrics.
    """
    # Validation switches to evaluation mode; restore training before the next update.
    ddp.train()
    wrapped = ddp.module if hasattr(ddp, "module") else ddp
    if isinstance(wrapped, CompletionLoss):
        layers = getattr(wrapped.core().model, "layers", ())
        if layers and any(hasattr(layer, "gradient_checkpointing") for layer in layers):
            active = sum(
                bool(getattr(layer, "gradient_checkpointing", False)) and layer.training
                for layer in layers
            )
            if active != len(layers):
                raise RuntimeError(
                    f"Decoder checkpointing is active on {active}/{len(layers)} layers"
                )
    import torch.distributed as dist

    use_dist = dist.is_available() and dist.is_initialized()
    count = sum((sum((v != -100 for v in ex["labels"][1:])) for ex in local_examples))
    # Count target tokens globally so unequal sequence lengths receive the right weight.
    denominator = torch.tensor(float(count), device=device, dtype=torch.float64)
    if use_dist:
        dist.all_reduce(denominator)
    if denominator.item() <= 0:
        raise ValueError("No target tokens in accumulation group")
    params = [p for p in ddp.parameters() if p.requires_grad]
    for g in optimizer.param_groups:
        g["lr"] = lr
    for attempt in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for i, ex in enumerate(local_examples):
            sync = (
                ddp.no_sync()
                if hasattr(ddp, "no_sync") and i < len(local_examples) - 1
                else nullcontext()
            )
            with sync:
                ids = torch.tensor([ex["input_ids"]], device=device, dtype=torch.long)
                labels = torch.tensor([ex["labels"]], device=device, dtype=torch.long)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    nll = ddp(ids, labels)
                    # DDP averages gradients across processes. Multiplying by world_size
                    # undoes that averaging so division uses the global target-token total.
                    loss = nll * (world_size / denominator.item())
                if not torch.isfinite(nll):
                    raise FloatingPointError(
                        "Non-finite forward loss. No optimizer update was applied."
                    )
                loss_total += float(nll.detach())
                scaler.scale(loss).backward()
        # Check and clip actual gradients after removing the FP16 loss scale.
        # An overflow must not be committed as a normal optimizer update.
        scaler.unscale_(optimizer)
        finite = torch.tensor(
            int(
                all(
                    (
                        p.grad is None or torch.isfinite(p.grad).all().item()
                        for p in params
                    )
                )
            ),
            device=device,
        )
        if use_dist:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item():
            grad_norm = torch.nn.utils.clip_grad_norm_(
                params, max_grad_norm, error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            metric = torch.tensor(loss_total, device=device, dtype=torch.float64)
            if use_dist:
                dist.all_reduce(metric)
            return {
                "loss": float(metric.item() / denominator.item()),
                "target_tokens": int(denominator.item()),
                "grad_norm": float(grad_norm),
                "scale": float(scaler.get_scale()),
                "overflow_retries": attempt,
            }
        if device.type != "cuda" or scaler.get_scale() <= 1:
            raise FloatingPointError("Non-finite adapter gradients")
        scaler.update(new_scale=max(1.0, scaler.get_scale() / 2))
    raise FloatingPointError(
        "Adapter gradients remained non-finite after bounded loss-scale retries"
    )
