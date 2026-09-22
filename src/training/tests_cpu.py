"""Checks and helpers used by the original training pipeline."""

import copy, io, json, math, tempfile, unittest, zipfile
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from common import *
from engine import *
from tokenize_data import encode


class TinyBackbone(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 12)
        self.linear = torch.nn.Linear(12, 12, bias=False)
        for p in self.parameters():
            p.requires_grad = False
        self.lora_A = torch.nn.Parameter(torch.randn(12, 2) * 0.04)
        self.lora_B = torch.nn.Parameter(torch.zeros(2, 12))

    def forward(self, input_ids, **kwargs):
        """Sum next-token losses at the unmasked completion positions."""
        x = self.embedding(input_ids)
        return SimpleNamespace(
            last_hidden_state=torch.tanh(self.linear(x) + x @ self.lora_A @ self.lora_B)
        )


class Tiny(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = TinyBackbone()
        self.lm_head = torch.nn.Linear(12, 32, bias=False)
        self.lm_head.weight.requires_grad = False


def sample(i=0):
    ids = [2, 4 + i % 4, 8, 5, 3, 14, 9 + i % 4, 15]
    p = 3 + i % 3
    return {"input_ids": ids, "labels": [-100] * p + ids[p:]}


class ToyTokenizer:

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    ):
        p = "<user>" + messages[0]["content"] + "<assistant><think></think>"
        return p if add_generation_prompt else p + messages[1]["content"] + "<eos>"

    def encode(self, s, add_special_tokens=False):
        return list(s.encode())


class UnitTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(3)

    def test_chunked_loss_matches_full(self):
        net = Tiny()
        w = CompletionLoss(net, 2)
        ex = sample()
        ids = torch.tensor([ex["input_ids"]])
        labels = torch.tensor([ex["labels"]])
        hidden = net.model(ids).last_hidden_state
        logits = net.lm_head(hidden)
        ref = F.cross_entropy(
            logits[:, :-1].reshape(-1, 32),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        torch.testing.assert_close(w(ids, labels), ref)

    def test_chunked_gradients_match_full(self):
        a = Tiny()
        b = copy.deepcopy(a)
        ex = sample(1)
        ids = torch.tensor([ex["input_ids"]])
        labels = torch.tensor([ex["labels"]])
        CompletionLoss(a, 2)(ids, labels).backward()
        full = b.lm_head(b.model(ids).last_hidden_state)
        F.cross_entropy(
            full[:, :-1].reshape(-1, 32),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ).backward()
        for (n, p), (m, q) in zip(a.named_parameters(), b.named_parameters()):
            if p.requires_grad:
                torch.testing.assert_close(p.grad, q.grad)

    def test_frozen_parameters_not_updated(self):
        n = Tiny()
        f = {
            k: p.detach().clone()
            for k, p in n.named_parameters()
            if not p.requires_grad
        }
        a = adapter_copy(n)
        opt = torch.optim.AdamW(trainable_parameters(n), lr=0.01, weight_decay=0)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        r = optimize_group(
            CompletionLoss(n, 3),
            [sample(), sample(1)],
            opt,
            scaler,
            torch.device("cpu"),
            1,
            0.01,
        )
        self.assertTrue(math.isfinite(r["loss"]))
        self.assertGreater(r["target_tokens"], 0)
        self.assertTrue(
            any((not torch.equal(a[k], p) for k, p in n.named_parameters() if k in a))
        )
        for k, p in n.named_parameters():
            if k in f:
                self.assertTrue(torch.equal(p, f[k]))

    def test_accumulation_exact_tokens(self):
        a = Tiny()
        b = copy.deepcopy(a)
        exs = [sample(i) for i in range(4)]
        op = torch.optim.SGD(trainable_parameters(a), lr=0.02)
        op2 = torch.optim.SGD(trainable_parameters(b), lr=0.02)
        optimize_group(
            CompletionLoss(a, 4),
            exs,
            op,
            torch.amp.GradScaler("cuda", enabled=False),
            torch.device("cpu"),
            1,
            0.02,
        )
        w = CompletionLoss(b, 4)
        den = sum((sum((t != -100 for t in x["labels"][1:])) for x in exs))
        loss = (
            sum(
                (
                    w(torch.tensor([e["input_ids"]]), torch.tensor([e["labels"]]))
                    for e in exs
                )
            )
            / den
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters(b), 1)
        op2.step()
        for k, v in adapter_copy(a).items():
            torch.testing.assert_close(v, adapter_copy(b)[k])

    def test_adapter_restore(self):
        n = Tiny()
        a = adapter_copy(n)
        with torch.no_grad():
            for p in trainable_parameters(n):
                p.add_(1)
        adapter_restore(n, a)
        for k, v in adapter_copy(n).items():
            self.assertTrue(torch.equal(v, a[k]))

    def test_restore_name_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            adapter_restore(Tiny(), {})

    def test_empty_targets_rejected(self):
        with self.assertRaises(ValueError):
            CompletionLoss(Tiny())(
                torch.tensor([[1, 2, 3]]), torch.tensor([[-100] * 3])
            )

    def test_scheduler(self):
        self.assertGreater(learning_rate(0, 100, 0.0001, 0.03), 0)
        self.assertAlmostEqual(learning_rate(100, 100, 0.0001, 0.03), 0)

    def test_exact_epoch_exposure(self):
        groups = epoch_groups(30266, 8, 42, 0)
        flat = [i for g in groups for i in g]
        self.assertEqual(len(flat), 30266)
        self.assertEqual(set(flat), set(range(30266)))
        self.assertEqual(len(groups[-1]), 2)

    def test_rank_disjoint_and_complete(self):
        groups = epoch_groups(30266, 8, 42, 0)
        a = [i for g in groups for i in g[::2]]
        b = [i for g in groups for i in g[1::2]]
        self.assertFalse(set(a) & set(b))
        self.assertEqual(len(a), 15133)
        self.assertEqual(len(b), 15133)

    def test_epoch_seed(self):
        self.assertEqual(epoch_groups(100, 8, 42, 1), epoch_groups(100, 8, 42, 1))
        self.assertNotEqual(epoch_groups(100, 8, 42, 1), epoch_groups(100, 8, 42, 0))

    def test_odd_count_rejected(self):
        with self.assertRaises(ValueError):
            epoch_groups(11, 8, 42, 1)

    def test_token_boundary(self):
        r = {
            "example_id": "x",
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
        }
        ids, p = encode(r, ToyTokenizer(), 1024)
        self.assertEqual(bytes(ids[p:]).decode(), "answer<eos>")

    def test_no_truncation(self):
        r = {
            "example_id": "x",
            "messages": [
                {"role": "user", "content": "long question"},
                {"role": "assistant", "content": "answer"},
            ],
        }
        with self.assertRaises(ValueError):
            encode(r, ToyTokenizer(), 5)

    def test_wrong_roles(self):
        with self.assertRaises(ValueError):
            encode({"example_id": "x", "messages": []}, ToyTokenizer(), 2048)

    def test_token_prefix_change_rejected(self):

        class Broken(ToyTokenizer):

            def encode(self, s, add_special_tokens=False):
                return super().encode(s) + ([1] if not s.endswith("<eos>") else [])

        r = {
            "example_id": "x",
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        }
        with self.assertRaises(ValueError):
            encode(r, Broken(), 2048)

    def test_zip_path_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.zip"
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("../escape", "x")
            with self.assertRaises(ValueError):
                safe_extract(p, Path(d) / "out")

    def test_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.zip"
            out = Path(d) / "out"
            out.mkdir()
            (out / "x").write_text("old")
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("x", "new")
            with self.assertRaises(ValueError):
                safe_extract(p, out)

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            write_json(p, {"x": 1})
            self.assertEqual(read_json(p), {"x": 1})
            self.assertFalse(p.with_name("x.json.tmp").exists())

    def test_export_and_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "a"
            root.mkdir()
            write_json(root / "settings.json", {"x": 1})
            write_json(root / "runs/subjectesis/progress.json", {"step": 10})
            z = package_outputs(root)
            new = Path(d) / "b"
            new.mkdir()
            write_json(new / "settings.json", {"x": 1})
            restore_runs(z, new)
            self.assertEqual(
                read_json(new / "runs/subjectesis/progress.json"), {"step": 10}
            )

    def test_recovery_config_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "a"
            root.mkdir()
            write_json(root / "settings.json", {"x": 1})
            z = package_outputs(root)
            new = Path(d) / "b"
            new.mkdir()
            write_json(new / "settings.json", {"x": 2})
            with self.assertRaises(ValueError):
                restore_runs(z, new)

    def test_checkpoint_optimizer_roundtrip(self):
        n = Tiny()
        w = CompletionLoss(n)
        opt = torch.optim.AdamW(trainable_parameters(n), lr=0.01, weight_decay=0)
        sc = torch.amp.GradScaler("cuda", enabled=False)
        optimize_group(w, [sample(0)], opt, sc, torch.device("cpu"), 1, 0.01)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "state.pt"
            torch.save(
                {
                    "adapter": adapter_copy(n),
                    "opt": opt.state_dict(),
                    "rng": random.getstate(),
                },
                path,
            )
            optimize_group(w, [sample(1)], opt, sc, torch.device("cpu"), 1, 0.01)
            expected = adapter_copy(n)
            saved = torch.load(path, weights_only=True)
            adapter_restore(n, saved["adapter"])
            opt.load_state_dict(saved["opt"])
            optimize_group(w, [sample(1)], opt, sc, torch.device("cpu"), 1, 0.01)
            for k, v in adapter_copy(n).items():
                torch.testing.assert_close(v, expected[k], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
