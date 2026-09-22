"""Checks and helpers used by the original training pipeline."""


def verify_loading(info, tied_embeddings):
    if not tied_embeddings:
        raise RuntimeError("Qwen3.5-4B requires tied input/output embeddings.")
    missing = set(info.get("missing_keys") or [])
    unresolved = missing - {"lm_head.weight"}
    if unresolved or any(
        (info.get(k) for k in ["mismatched_keys", "error_msgs", "conversion_errors"])
    ):
        raise RuntimeError(
            "Pretrained text weights were not loaded completely; inspect the loading report."
        )
    unexpected = info.get("unexpected_keys") or []
    invalid = [
        k
        for k in unexpected
        if not k.startswith(("model.visual.", "visual.", "mtp.", "model.mtp."))
    ]
    if invalid:
        raise RuntimeError("Unrecognized unused checkpoint keys: " + str(invalid[:5]))
    return {
        "tied_embeddings_verified": True,
        "shared_head_alias_absent_from_file": "lm_head.weight" in missing,
        "unused_vision_or_mtp_keys": len(unexpected),
    }
