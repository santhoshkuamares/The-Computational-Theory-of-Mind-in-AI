"""Revision-4 data preparation: test full."""

import copy, hashlib, json, tempfile, unittest
from collections import Counter
from pathlib import Path
from build_full import (
    ROOT,
    annotations,
    source_state,
    target_for,
    examples,
    run,
    check_state,
)
from convert_v1 import load_sources, digest, DOMAINS, input_text
from dataset_io import load_training, encode_training_example, pad_batch
from packets import groups


def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


class DataTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rows, cls.splits = load_sources(ROOT)
        cls.by = {r["record_id"]: r for r in cls.rows}
        cls.ledger = read(ROOT / "data/review/all_training_records.jsonl")
        cls.reviews = {r["record_id"]: r for r in cls.ledger}
        cls.subject = load_training(ROOT / "data", "subjectesis_full_train.jsonl")
        cls.control = load_training(ROOT / "data", "answer_only_matched_train.jsonl")

    def test_complete_annotation_coverage(self):
        self.assertEqual(len(self.ledger), 2270)
        self.assertEqual(
            {r["record_id"] for r in self.ledger},
            {
                r["record_id"]
                for r in self.rows
                if str(r["dialogue_id"]) in self.splits["train"]
            },
        )
        self.assertEqual(len(annotations(ROOT)), 1306)

    def test_split_disjoint(self):
        a, b, c = [self.splits[k] for k in ["train", "validation", "test"]]
        self.assertEqual([len(a), len(b), len(c)], [235, 34, 67])
        self.assertFalse(a & b or a & c or b & c)

    def test_source_binding(self):
        for r in self.ledger:
            self.assertEqual(r["source_record_sha256"], digest(self.by[r["record_id"]]))

    def test_claims_and_exact_citations(self):
        for item in self.ledger:
            target = item["grounded_target"]
            row = self.by[item["record_id"]]
            self.assertEqual(target["evidence"], check_state(row, target["state"]))
            for e in target["evidence"]:
                self.assertEqual(
                    e["quote"], row["utterance_context"].splitlines()[e["turn"] - 1]
                )

    def test_future_evidence_rejected(self):
        row = self.by["belief_10"]
        a = {
            "claims": {
                "seen": {
                    "value": "yes",
                    "basis": "explicit",
                    "reason": "test",
                    "evidence_turns": [999],
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "Future"):
            source_state(row, a)

    def test_other_person_only_evidence_rejected(self):
        row = self.by["belief_10"]
        a = {
            "claims": {
                "seen": {
                    "value": "yes",
                    "basis": "explicit",
                    "reason": "test",
                    "evidence_turns": [1],
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "other-speaker"):
            source_state(row, a)

    def test_state_independent_of_gold(self):
        ann = annotations(ROOT)
        for item in self.ledger:
            row = copy.deepcopy(self.by[item["record_id"]])
            row["answer"] = ["WRONG"]
            row["choices"] = {}
            self.assertEqual(
                source_state(row, ann[item["annotation_key"]]),
                item["grounded_target"]["state"],
            )

    def test_original_labels_preserved(self):
        for item in self.ledger:
            self.assertEqual(
                item["official_answer"], self.by[item["record_id"]]["answer"][0]
            )
        original = read(ROOT / "data/sft/answer_only_original_train.jsonl")
        self.assertEqual(len(original), 2270)
        for r in original:
            self.assertEqual(
                r["messages"][1]["content"], self.by[r["record_id"]]["answer"][0]
            )

    def test_quarantine_excluded_from_both_arms(self):
        bad = {r["record_id"] for r in self.ledger if not r["training_eligible"]}
        good = {r["record_id"] for r in self.ledger if r["training_eligible"]}
        self.assertTrue(bad)
        self.assertFalse(bad & {r["record_id"] for r in self.subject + self.control})
        self.assertEqual(good, {r["record_id"] for r in self.subject})
        for item in self.ledger:
            if item["grounded_target"]["answer_claims_conflicting"]:
                with self.assertRaises(ValueError):
                    examples(self.by[item["record_id"]], item["grounded_target"])

    def test_matched_question_exposure(self):
        self.assertEqual(
            Counter((r["record_id"] for r in self.subject)),
            Counter((r["record_id"] for r in self.control)),
        )

    def test_unique_examples(self):
        for rows in [self.subject, self.control]:
            self.assertEqual(len(rows), len({r["example_id"] for r in rows}))

    def test_training_membership(self):
        for r in self.subject + self.control:
            self.assertIn(str(r["dialogue_id"]), self.splits["train"])
            self.assertEqual(r["split"], "train")

    def test_eval_inputs_public_only(self):
        for split in ["validation", "test"]:
            for row in read(ROOT / f"data/evaluation/{split}_inputs.jsonl"):
                self.assertNotIn("answer", row)
                self.assertNotIn("state", row)
                self.assertNotIn("grounded_target", row)
                self.assertEqual(row["prompt"], input_text(self.by[row["record_id"]]))
                self.assertIn(str(row["dialogue_id"]), self.splits[split])

    def test_eval_and_references_unchanged(self):
        for split in ["validation", "test"]:
            rr = read(ROOT / f"data/scoring_only/{split}_references.jsonl")
            self.assertEqual(len(rr), 319 if split == "validation" else 621)
            for r in rr:
                self.assertEqual(r["answer"], self.by[r["record_id"]]["answer"][0])

    def test_audit_comments_never_become_model_targets(self):
        for r in self.subject:
            if r["kind"] == "answer_only":
                continue
            target = json.loads(r["messages"][1]["content"])
            self.assertNotIn("audit_note", target)
            self.assertNotIn("official_answer_state", target)
            if r["kind"] == "build_state":
                self.assertEqual(
                    set(target),
                    {
                        "perspective",
                        "evidence",
                        "state",
                        "missing_information",
                        "answer",
                        "answer_claims_not_established",
                    },
                )

    def test_no_answer_in_revision_state(self):
        for r in self.subject:
            if r["kind"] != "review_state":
                continue
            prompt = r["messages"][0]["content"]
            state = json.loads(
                prompt.split("Current perspective state (may contain mistakes): ")[
                    1
                ].split("\nSelected gap:")[0]
            )
            self.assertEqual(set(state), set(DOMAINS[r["task"]]))
            for c in state.values():
                self.assertEqual(set(c), {"value", "evidence_turns"})

    def test_balanced_revision_controls(self):
        c = Counter((r.get("construction", {}).get("case") for r in self.subject))
        self.assertEqual(c["masked"], c["wrong"])
        self.assertEqual(c["wrong"], c["preserve"])
        self.assertGreater(c["retain_unknown"], 0)

    def test_withdrawal_coverage_and_targets(self):
        actual = Counter()
        for r in self.subject:
            c = r.get("construction", {})
            if c.get("case") != "withdraw_unsupported":
                continue
            field = c["selected_field"]
            value = c["unsupported_value"]
            state = json.loads(
                r["messages"][0]["content"]
                .split("Current perspective state (may contain mistakes): ")[1]
                .split("\nSelected gap:")[0]
            )
            self.assertEqual(state[field], {"value": value, "evidence_turns": []})
            target = json.loads(r["messages"][1]["content"])
            self.assertEqual(target["decision"], "revise")
            self.assertEqual(target["updated_claim"]["value"], "unknown")
            self.assertEqual(target["updated_claim"]["basis"], "not_established")
            self.assertEqual(target["updated_claim"]["evidence_turns"], [])
            self.assertEqual(target["evidence"], [])
            self.assertEqual(target["stop_reason"], "evidence_not_sufficient")
            self.assertNotIn("withdraw_unsupported", r["messages"][0]["content"])
            actual[r["record_id"], field, value] += 1
        expected = Counter(
            (
                (r["record_id"], f, v)
                for r in self.ledger
                if r["training_eligible"]
                for f, c in r["grounded_target"]["state"].items()
                if c["value"] == "unknown"
                for v in DOMAINS[r["task"]][f]
                if v != "unknown"
            )
        )
        self.assertTrue(actual)
        self.assertEqual(actual, expected)

    def test_appraisal_adjudications(self):
        for rid in ["belief_812", "belief_1200", "belief_117", "belief_1027"]:
            c = self.reviews[rid]["grounded_target"]["state"]["appraisal"]
            self.assertEqual((c["value"], c["basis"]), ("likes", "inferred"))
        for rid in [
            "belief_1481",
            "belief_72",
            "belief_1181",
            "belief_719",
            "belief_38",
        ]:
            c = self.reviews[rid]["grounded_target"]["state"]["appraisal"]
            self.assertEqual((c["value"], c["basis"]), ("dislikes", "inferred"))

    def test_reviewed_title_alias_and_quarantine(self):
        item = self.reviews["belief_1199"]
        c = item["grounded_target"]["state"]["proposer"]
        self.assertEqual((c["value"], c["evidence_turns"]), ("recommender", [4]))
        self.assertEqual(item["official_answer"], "G")
        self.assertFalse(item["training_eligible"])
        self.assertIn("proposer", item["grounded_target"]["answer_claims_conflicting"])
        self.assertEqual(
            self.reviews["belief_1124"]["grounded_target"]["state"]["proposer"][
                "evidence_turns"
            ],
            [3],
        )
        row = copy.deepcopy(self.by["belief_1199"])
        row["utterance_context"] = (
            "RECOMMENDER says: The Fast and the Furious (2009) is good.\nSEEKER says: The Fast and the Furious (2001) is good."
        )
        self.assertEqual(
            source_state(row, {"claims": {}})["proposer"]["value"], "seeker"
        )

    def test_desire_contract_frozen(self):
        from convert_v1 import FIELDS

        self.assertEqual(set(DOMAINS["desire"]), {"intends_to_watch"})
        self.assertIn("likely willingness", FIELDS["intends_to_watch"])
        for r in self.ledger:
            if r["task"] == "desire":
                self.assertEqual(
                    set(r["grounded_target"]["state"]), {"intends_to_watch"}
                )

    def test_revision_two_heldout_bytes(self):
        baseline = json.loads((ROOT / "reviews/revision2_baseline.json").read_text())
        for rel, h in baseline["heldout_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), h)

    def test_recover_citations_for_all_known_fields(self):
        from audit_citation_controls import previous_state

        actual = Counter()
        for r in self.subject:
            c = r.get("construction", {})
            if c.get("case") != "recover_citation":
                continue
            field = c["selected_field"]
            target = self.reviews[r["record_id"]]["grounded_target"]["state"][field]
            before = previous_state(r)[field]
            after = json.loads(r["messages"][1]["content"])
            self.assertEqual(before, {"value": target["value"], "evidence_turns": []})
            self.assertNotEqual(before["value"], "unknown")
            self.assertEqual(after["updated_claim"], target)
            self.assertEqual(after["decision"], "preserve")
            self.assertTrue(after["evidence"])
            self.assertNotIn("recover_citation", r["messages"][0]["content"])
            actual[r["record_id"], field] += 1
        expected = Counter(
            (
                (r["record_id"], f)
                for r in self.ledger
                if r["training_eligible"]
                for f, c in r["grounded_target"]["state"].items()
                if c["value"] != "unknown"
            )
        )
        self.assertEqual(actual, expected)

    def test_misleading_citations_cover_unknown_values(self):
        from audit_citation_controls import previous_state

        actual = Counter()
        for r in self.subject:
            c = r.get("construction", {})
            if c.get("case") != "withdraw_misleading_citation":
                continue
            field = c["selected_field"]
            before = previous_state(r)[field]
            after = json.loads(r["messages"][1]["content"])
            correct = self.reviews[r["record_id"]]["grounded_target"]["state"][field]
            self.assertEqual(correct["value"], "unknown")
            self.assertEqual(after["updated_claim"], correct)
            self.assertEqual(after["decision"], "revise")
            self.assertFalse(after["evidence"])
            self.assertEqual(len(before["evidence_turns"]), 1)
            turn = before["evidence_turns"][0]
            lines = self.by[r["record_id"]]["utterance_context"].splitlines()
            self.assertTrue(1 <= turn <= len(lines))
            self.assertEqual(c["distractor"]["quote"], lines[turn - 1])
            self.assertNotIn(
                "withdraw_misleading_citation", r["messages"][0]["content"]
            )
            actual[r["record_id"], field, before["value"]] += 1
        expected = Counter(
            (
                (r["record_id"], f, v)
                for r in self.ledger
                if r["training_eligible"]
                for f, c in r["grounded_target"]["state"].items()
                if c["value"] == "unknown"
                for v in DOMAINS[r["task"]][f]
                if v != "unknown"
            )
        )
        self.assertEqual(actual, expected)

    def test_citation_presence_does_not_determine_unknown(self):
        from audit_citation_controls import previous_state

        outcomes = {False: set(), True: set()}
        for r in self.subject:
            if r["kind"] != "review_state":
                continue
            after = json.loads(r["messages"][1]["content"])
            before = previous_state(r)[after["field"]]
            if before["value"] != "unknown":
                outcomes[bool(before["evidence_turns"])].add(
                    after["updated_claim"]["value"] == "unknown"
                )
        self.assertEqual(outcomes, {False: {False, True}, True: {False, True}})

    def test_revision_three_protected_bytes(self):
        baseline = json.loads((ROOT / "reviews/revision3_baseline.json").read_text())
        for rel, h in baseline["protected_sha256"].items():
            self.assertEqual(
                hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), h, rel
            )

    def test_restricted_loader(self):
        for filename in [
            "../scoring_only/test_references.jsonl",
            "all_training_records.jsonl",
            "answer_only_original_train.jsonl",
        ]:
            with self.assertRaises(ValueError):
                load_training(ROOT / "data", filename)

    def test_checksum_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "sft").mkdir()
            for name in ["manifest.json", "checksums.json"]:
                (root / name).write_bytes((ROOT / "data" / name).read_bytes())
            (root / "sft/subjectesis_full_train.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                load_training(root, "subjectesis_full_train.jsonl")

    def test_assistant_mask_and_padding(self):

        class CharacterTokenizer:

            def encode(self, text, add_bos=True):
                return ([1] if add_bos else []) + [ord(c) + 2 for c in text]

        e = encode_training_example(
            self.control[0], CharacterTokenizer(), max_tokens=10000
        )
        self.assertTrue(all((x == -100 for x in e["labels"][: e["prompt_tokens"]])))
        self.assertEqual(
            e["labels"][e["prompt_tokens"] :], e["input_ids"][e["prompt_tokens"] :]
        )
        padded = pad_batch([e], 0, length=len(e["input_ids"]) + 3)
        self.assertEqual(padded["labels"][0][-3:], [-100] * 3)
        with self.assertRaises(ValueError):
            encode_training_example(self.control[0], CharacterTokenizer(), max_tokens=1)

    def test_all_hashes(self):
        for rel, h in json.loads((ROOT / "data/checksums.json").read_text()).items():
            self.assertEqual(
                hashlib.sha256((ROOT / "data" / rel).read_bytes()).hexdigest(), h
            )

    def test_rebuild_is_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "data"
            run(output=out)
            self.assertEqual(
                (out / "checksums.json").read_bytes(),
                (ROOT / "data/checksums.json").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
