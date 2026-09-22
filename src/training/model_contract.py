"""Check that the intended pretrained text weights were loaded.

The Qwen checkpoint includes shared embedding/output weights and may also
contain unused non-text modules. The check distinguishes those expected cases
from missing or incompatible text-model weights.
"""


def verify_loading(info, tied_embeddings):
    """Check the weight-loading report and require genuinely shared input and
    output embeddings. Allow the shared output-head alias and unused vision or
    multi-token-prediction weights, but reject missing or mismatched text
    weights before returning a small verification summary.
    """
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
