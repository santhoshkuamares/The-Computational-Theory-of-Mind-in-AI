"""Checks and helpers used by the original training pipeline."""

from pathlib import Path
import tempfile, unittest, ast
from common import MODEL_ID, MODEL_REVISION, write_json, load_config
from download_model import (
    required_space,
    shard_name,
    RESERVE_BYTES,
    FALLBACK_TENSOR_BYTES,
)
from setup_runtime import PINS, TORCHAO_PIN
from model_contract import verify_loading
from launch import child_env


class Changes(unittest.TestCase):

    def test_official_identity(self):
        self.assertEqual(MODEL_ID, "Qwen/Qwen3.5-4B")
        self.assertEqual(MODEL_REVISION, "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")

    def test_uncached_4b_size(self):
        self.assertEqual(
            required_space(FALLBACK_TENSOR_BYTES, 0),
            FALLBACK_TENSOR_BYTES + RESERVE_BYTES,
        )
        self.assertLess(required_space(FALLBACK_TENSOR_BYTES, 0) / 2**30, 13)

    def test_partial_cache(self):
        self.assertEqual(required_space(100, 40, 20), 80)

    def test_complete_cache_keeps_reserve(self):
        self.assertEqual(required_space(100, 100, 20), 20)
        self.assertEqual(required_space(100, 101, 20), 20)

    def test_no_cache_path_escape(self):
        for s in [
            "../escape.safetensors",
            "/tmp/evil.safetensors",
            "a/part.safetensors",
            "model.bin",
        ]:
            with self.assertRaises(ValueError):
                shard_name(s)

    def test_valid_shard(self):
        self.assertEqual(
            shard_name("model.safetensors-00001-of-00002.safetensors"),
            "model.safetensors-00001-of-00002.safetensors",
        )

    def test_torchao_pin(self):
        self.assertEqual(TORCHAO_PIN, "torchao==0.16.0")
        self.assertIn("numpy==2.0.2", PINS)

    def test_no_pip_cache_or_jax(self):
        env = child_env("/tmp/project")
        self.assertEqual(env["PIP_NO_CACHE_DIR"], "1")
        self.assertNotIn("JAX_PLATFORMS", env)
        self.assertNotIn("XLA_FLAGS", env)

    def test_tie_must_be_real(self):
        with self.assertRaises(RuntimeError):
            verify_loading({}, False)

    def test_shared_head_alias_only(self):
        r = verify_loading({"missing_keys": ["lm_head.weight"]}, True)
        self.assertTrue(r["shared_head_alias_absent_from_file"])
        with self.assertRaises(RuntimeError):
            verify_loading({"missing_keys": ["model.embed_tokens.weight"]}, True)
        with self.assertRaises(RuntimeError):
            verify_loading(
                {"missing_keys": ["model.layers.0.mlp.up_proj.weight"]}, True
            )

    def test_unexpected_text_keys_rejected(self):
        with self.assertRaises(RuntimeError):
            verify_loading({"unexpected_keys": ["model.layers.0.foo"]}, True)
        self.assertEqual(
            verify_loading({"unexpected_keys": ["model.visual.foo", "mtp.foo"]}, True)[
                "unused_vision_or_mtp_keys"
            ],
            2,
        )

    def test_mismatch_rejected(self):
        for k in ["mismatched_keys", "error_msgs", "conversion_errors"]:
            with self.assertRaises(RuntimeError):
                verify_loading({k: ["failure"]}, True)

    def test_source_never_forces_untie(self):
        s = (Path(__file__).parent / "model_runtime.py").read_text()
        self.assertNotIn("cfg.tie_word_embeddings=False", s)
        self.assertIn("llm_int8_skip_modules=['lm_head']", s)

    def test_previous_9b_config_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            write_json(
                Path(d) / "settings.json",
                {"model_id": "Qwen/Qwen3.5-9B", "model_revision": MODEL_REVISION},
            )
            with self.assertRaises(ValueError):
                load_config(d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
