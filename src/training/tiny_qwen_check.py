"""Checks and helpers used by the original training pipeline."""

from pathlib import Path
import argparse, tempfile, json, torch
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from engine import CompletionLoss, adapter_copy, adapter_restore
from common import write_json
from model_contract import verify_loading


def run(root):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qm

    for name in [
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "FusedRMSNormGated",
    ]:
        if not hasattr(qm, name):
            raise RuntimeError("Pinned Qwen interface changed: " + name)
        setattr(qm, name, None)
    qm.is_fast_path_available = False
    cfg = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        tie_word_embeddings=True,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [2, 1, 1],
        },
        use_cache=False,
    )
    base = Qwen3_5ForCausalLM(cfg)
    base.config._attn_implementation = "eager"
    if (
        base.get_input_embeddings().weight.data_ptr()
        != base.get_output_embeddings().weight.data_ptr()
    ):
        raise RuntimeError("Tiny model did not tie its embeddings")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cfg.save_pretrained(d)
        weights = {
            (
                "model.language_model." + k[len("model.") :]
                if k.startswith("model.")
                else k
            ): v.detach().clone()
            for k, v in base.state_dict().items()
            if k != "lm_head.weight"
        }
        shard = "model-00001-of-00001.safetensors"
        save_file(weights, str(d / shard), metadata={"format": "pt"})
        (d / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "total_size": sum(
                            (v.numel() * v.element_size() for v in weights.values())
                        )
                    },
                    "weight_map": {k: shard for k in weights},
                }
            )
        )
        loaded, info = Qwen3_5ForCausalLM.from_pretrained(
            d,
            config=cfg,
            dtype=torch.float32,
            key_mapping={"^model\\.language_model\\.": "model."},
            local_files_only=True,
            output_loading_info=True,
            attn_implementation="eager",
        )
        verify_loading(
            info,
            loaded.get_input_embeddings().weight.data_ptr()
            == loaded.get_output_embeddings().weight.data_ptr(),
        )
        for k, v in base.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[k], v, rtol=0, atol=0)
    net = get_peft_model(
        loaded,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=[
                "q_proj",
                "v_proj",
                "in_proj_qkv",
                "out_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
            lora_dropout=0.0,
        ),
    )
    w = CompletionLoss(net, 4)
    ids = torch.randint(2, 64, (1, 13))
    labels = ids.clone()
    labels[:, :7] = -100
    full = net(input_ids=ids, use_cache=False).logits
    reference = torch.nn.functional.cross_entropy(
        full[:, :-1].float().reshape(-1, 64),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    actual = w(ids, labels)
    torch.testing.assert_close(actual, reference, rtol=1e-05, atol=1e-05)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=0.001)
    before = adapter_copy(net)
    frozen = {
        n: p.detach().clone() for n, p in net.named_parameters() if not p.requires_grad
    }
    actual.backward()
    assert all(
        (p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    )
    opt.step()
    assert any(
        (
            not torch.equal(p.detach(), before[n])
            for n, p in net.named_parameters()
            if n in before
        )
    )
    assert all(
        (
            torch.equal(p.detach(), frozen[n])
            for n, p in net.named_parameters()
            if n in frozen
        )
    )
    adapter_restore(net, before)
    write_json(
        Path(root) / "reports/tiny_installed_qwen_check.json",
        {
            "passed": True,
            "pretrained_weights_used": False,
            "tied_checkpoint_remap_roundtrip": True,
            "hybrid_layers": True,
            "masked_loss_matches_full_logits": True,
            "finite_adapter_update": True,
            "frozen_parameters_unchanged": True,
        },
    )
    print(
        "Tiny installed Qwen3.5 tied-checkpoint and CPU gradient checks passed. This is not the pretrained 4B GPU test.",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    run(p.parse_args().root)
