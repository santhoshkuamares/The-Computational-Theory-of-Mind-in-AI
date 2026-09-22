"""Pinned Qwen loading, quantization and adapter construction."""

from pathlib import Path
import json, hashlib
import torch
from common import MODEL_ID, MODEL_REVISION, write_json

TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def build_network(root, c, local_rank, download_path):
    """Load the pinned Qwen text model in NF4 and attach the recorded FP32 LoRA parameters."""
    from transformers import AutoConfig, Qwen3_5ForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    for attr in [
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "FusedRMSNormGated",
    ]:
        if not hasattr(m, attr):
            raise RuntimeError(
                "Pinned Transformers DeltaNet interface changed: " + attr
            )
        setattr(m, attr, None)
    m.is_fast_path_available = False
    cfg = AutoConfig.from_pretrained(
        download_path, local_files_only=True, trust_remote_code=False
    ).get_text_config()
    cfg.use_cache = False
    if (
        not cfg.tie_word_embeddings
        or cfg.hidden_size != 2560
        or cfg.num_hidden_layers != 32
    ):
        raise RuntimeError(
            "Expected the pinned 4B text configuration, with tied input/output embeddings"
        )
    index = json.loads(
        (Path(download_path) / "model.safetensors.index.json").read_text()
    )["weight_map"]
    if any((k.startswith("model.language_model.") for k in index)):
        keymap = {"^model\\.language_model\\.": "model."}
    elif any((k.startswith("model.layers.") for k in index)):
        keymap = None
    else:
        raise RuntimeError(
            "Unexpected Qwen text-weight prefix. Refusing random initialization."
        )
    quant = BitsAndBytesConfig(
        llm_int8_skip_modules=["lm_head"],
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        download_path,
        config=cfg,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float16,
        quantization_config=quant,
        device_map={"": local_rank},
        attn_implementation="sdpa",
        key_mapping=keymap,
        output_loading_info=True,
    )
    write_json(
        Path(root) / "reports" / f"weight_loading_rank{local_rank}.json",
        json.loads(json.dumps(info, default=str)),
    )
    from model_contract import verify_loading

    loading_check = verify_loading(
        info,
        model.get_input_embeddings().weight.data_ptr()
        == model.get_output_embeddings().weight.data_ptr(),
    )
    write_json(
        Path(root) / "reports" / f"weight_contract_rank{local_rank}.json", loading_check
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.get_input_embeddings().to(dtype=torch.float16)
    model.get_output_embeddings().to(dtype=torch.float16)
    model.config.use_cache = False
    if (
        model.get_input_embeddings().weight.data_ptr()
        != model.get_output_embeddings().weight.data_ptr()
    ):
        raise RuntimeError("Embedding tying was lost during k-bit preparation")
    lora = LoraConfig(
        r=c["rank"],
        lora_alpha=c["alpha"],
        lora_dropout=0.0,
        target_modules=TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    torch.manual_seed(c["seed"])
    torch.cuda.manual_seed_all(c["seed"])
    network = get_peft_model(model, lora)
    network.peft_config["default"].base_model_name_or_path = MODEL_ID
    network.peft_config["default"].revision = MODEL_REVISION
    network.enable_input_require_grads()
    for name, p in network.named_parameters():
        if p.requires_grad:
            if "lora_" not in name:
                raise RuntimeError("Non-adapter weight was made trainable: " + name)
            p.data = p.data.float()
    if any(
        (
            p.device.type != "cuda" or p.device.index != local_rank
            for p in network.parameters()
        )
    ):
        raise RuntimeError("Model was offloaded or placed on the wrong GPU")
    report = {
        "rank": local_rank,
        "gpu": torch.cuda.get_device_name(local_rank),
        "base_storage": "NF4 double quantization; embeddings/head FP16; small norms/dynamics FP32",
        "adapter_dtype": "float32",
        "trainable_parameters": sum(
            (p.numel() for p in network.parameters() if p.requires_grad)
        ),
        "memory_allocated_gib": torch.cuda.memory_allocated(local_rank) / 2**30,
        "attention": "SDPA + PyTorch reference DeltaNet",
        "module_targets": TARGETS,
        "tied_embeddings_preserved": True,
        "text_only_backbone": True,
    }
    write_json(Path(root) / "reports" / f"model_rank{local_rank}.json", report)
    print("Model ready:", report, flush=True)
    return network


def frozen_sample(network):
    """Hash sampled frozen weights for a smoke diagnostic; this is not a full weight comparison."""
    h = hashlib.sha256()
    tensors = 0
    for name, p in network.named_parameters():
        if p.requires_grad:
            continue
        flat = p.detach().reshape(-1)
        if flat.numel():
            sample = (
                torch.cat([flat[:16], flat[-16:]]).cpu().contiguous().view(torch.uint8)
            )
            h.update(name.encode())
            h.update(sample.numpy().tobytes())
            tensors += 1
    return {
        "digest": h.hexdigest(),
        "sampled_tensors": tensors,
        "scope": "first/last 16 stored values per frozen parameter, not all weights",
    }
