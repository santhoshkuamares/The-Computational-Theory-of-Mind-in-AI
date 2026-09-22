"""Check the rebuilt supervision before it enters training.

These tests cover dialogue separation, source labels, annotation coverage,
quarantine, citation controls, loss masks and deterministic rebuilding.
training/prepare.py runs this suite as part of the preparation workflow.
"""

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
    """Read the preparation JSONL file into a list of records. The tests use this
    helper to inspect generated examples and compare them with the fixed source
    data.
    """
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


class DataTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load source records, split membership and both generated training
        conditions once for the test class. Shared lookup tables keep the
        subsequent checks focused on the same rebuilt dataset.
        """
        cls.rows, cls.splits = load_sources(ROOT)
        cls.by = {r["record_id"]: r for r in cls.rows}
        cls.ledger = read(ROOT / "data/review/all_training_records.jsonl")
        cls.reviews = {r["record_id"]: r for r in cls.ledger}
        cls.subject = load_training(ROOT / "data", "subjectesis_full_train.jsonl")
        cls.control = load_training(ROOT / "data", "answer_only_matched_train.jsonl")

    def test_complete_annotation_coverage(self):
        """Check that every original training question has an annotation ledger
        entry and that all 1,306 dialogue/movie targets are covered. This
        catches missing or extra records before supervision is exported.
        """
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
        """Check the expected 235/34/67 dialogue counts and require no overlap
        between training, validation and test. Whole-dialogue separation
        prevents related questions from crossing these splits.
        """
        a, b, c = [self.splits[k] for k in ["train", "validation", "test"]]
        self.assertEqual([len(a), len(b), len(c)], [235, 34, 67])
        self.assertFalse(a & b or a & c or b & c)

    def test_source_binding(self):
        """Recalculate the hash of each original question and compare it with its
        ledger record. This detects annotations attached to a changed or
        different source.
        """
        for r in self.ledger:
            self.assertEqual(r["source_record_sha256"], digest(self.by[r["record_id"]]))

    def test_claims_and_exact_citations(self):
        """Validate each annotated state and compare every exported quotation with
        its numbered source turn. This checks literal citation binding, not
        independent semantic support for the claim.
        """
        for item in self.ledger:
            target = item["grounded_target"]
            row = self.by[item["record_id"]]
            self.assertEqual(target["evidence"], check_state(row, target["state"]))
            for e in target["evidence"]:
                self.assertEqual(
                    e["quote"], row["utterance_context"].splitlines()[e["turn"] - 1]
                )

    def test_future_evidence_rejected(self):
        """Insert a citation beyond the available dialogue and require the state
        builder to reject it. This checks that later or nonexistent turns
        cannot support an earlier perspective state.
        """
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
        """Attach only recommender evidence to a claim about whether the seeker
        has watched the film and require rejection. This tests the implemented
        speaker-evidence constraint on that field.
        """
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
        """Replace answer labels and choices with unusable values and rebuild the
        annotated states. Matching states show that source_state does not
        derive its fields from the gold-answer input.
        """
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
        """Compare official ledger answers and ordinary training targets with the
        original benchmark labels. Quarantine and structured annotation must
        not rewrite the original reference answers.
        """
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
        """Check that flagged questions are absent from both training arms and
        that all eligible questions remain. Also require the example builder to
        reject a target with unresolved answer-state conflicts.
        """
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
        """Count how often each record appears in the two full training schedules
        and require equality. This matches example exposure by question,
        although it does not equalize target tokens or selected-checkpoint
        exposure.
        """
        self.assertEqual(
            Counter((r["record_id"] for r in self.subject)),
            Counter((r["record_id"] for r in self.control)),
        )

    def test_unique_examples(self):
        """Require every example ID to be unique within each training condition.
        Repeated question supervision is allowed, but its individual examples
        need distinct IDs for traceability.
        """
        for rows in [self.subject, self.control]:
            self.assertEqual(len(rows), len({r["example_id"] for r in rows}))

    def test_training_membership(self):
        """Check the train marker and dialogue membership of every exported
        training example. This guards against accidentally including validation
        or test dialogues.
        """
        for r in self.subject + self.control:
            self.assertIn(str(r["dialogue_id"]), self.splits["train"])
            self.assertEqual(r["split"], "train")

    def test_eval_inputs_public_only(self):
        """Check that held-out inputs contain the original public prompt and the
        correct split membership. Reject embedded answers, states or grounded
        targets that would expose scoring information to the model.
        """
        for split in ["validation", "test"]:
            for row in read(ROOT / f"data/evaluation/{split}_inputs.jsonl"):
                self.assertNotIn("answer", row)
                self.assertNotIn("state", row)
                self.assertNotIn("grounded_target", row)
                self.assertEqual(row["prompt"], input_text(self.by[row["record_id"]]))
                self.assertIn(str(row["dialogue_id"]), self.splits[split])

    def test_eval_and_references_unchanged(self):
        """Check held-out reference counts and compare every answer with the
        original source. This ensures training-data revisions did not change
        validation or test scoring labels.
        """
        for split in ["validation", "test"]:
            rr = read(ROOT / f"data/scoring_only/{split}_references.jsonl")
            self.assertEqual(len(rr), 319 if split == "validation" else 621)
            for r in rr:
                self.assertEqual(r["answer"], self.by[r["record_id"]]["answer"][0])

    def test_audit_comments_never_become_model_targets(self):
        """Inspect structured assistant completions and require the intended
        target keys only. Audit notes and label-state bookkeeping remain
        outside model supervision.
        """
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
        """Parse the state embedded in every review prompt and inspect its keys.
        Only task fields containing values and citation turns are allowed, so
        the revision input has no explicit reference-answer field.
        """
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
        """Check equal counts of masked, wrong and preserve review cases and the
        presence of retain_unknown examples. This verifies the intended
        synthetic review construction rather than a learned behavior.
        """
        c = Counter((r.get("construction", {}).get("case") for r in self.subject))
        self.assertEqual(c["masked"], c["wrong"])
        self.assertEqual(c["wrong"], c["preserve"])
        self.assertGreater(c["retain_unknown"], 0)

    def test_withdrawal_coverage_and_targets(self):
        """Verify one unsupported-value withdrawal example for every eligible
        unknown field and alternative known value. Check that its target
        revises to unknown with no supporting evidence and that the
        construction label is absent from the prompt.
        """
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
        """Check the recorded likes/dislikes decisions for selected previously
        reviewed questions. These fixed examples detect unintended changes to
        the annotation decisions used by the final dataset.
        """
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
        """Check specific movie-title and proposer decisions, including the
        question excluded for a label conflict. A synthetic pair of similarly
        named films also checks that the target title is not confused with
        another release.
        """
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
        """Require desire states to use only intends_to_watch with the documented
        likely-willingness meaning. This prevents later changes from silently
        treating that prediction as a guaranteed future action.
        """
        from convert_v1 import FIELDS

        self.assertEqual(set(DOMAINS["desire"]), {"intends_to_watch"})
        self.assertIn("likely willingness", FIELDS["intends_to_watch"])
        for r in self.ledger:
            if r["task"] == "desire":
                self.assertEqual(
                    set(r["grounded_target"]["state"]), {"intends_to_watch"}
                )

    def test_revision_two_heldout_bytes(self):
        """Compare held-out files with the recorded revision-two checksums. Byte
        equality shows that later annotation revisions left those evaluation
        inputs and references unchanged.
        """
        baseline = json.loads((ROOT / "reviews/revision2_baseline.json").read_text())
        for rel, h in baseline["heldout_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), h)

    def test_recover_citations_for_all_known_fields(self):
        """Verify that every eligible known field has a case where its correct
        value is preserved and missing citations are restored. This prevents
        absence of a citation from always implying that the value must become
        unknown.
        """
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
        """Check that each unknown field is paired with every alternative known
        value and a real but misleading citation. Require the target to
        withdraw the value to unknown and verify complete construction
        coverage.
        """
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
        """Collect known and unknown outcomes separately for prompts with and
        without citations. Requiring both outcomes in both groups checks that
        citation presence is not a deterministic target shortcut in the
        constructed examples.
        """
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
        """Compare protected source and data files with their recorded
        revision-three hashes. This detects unintended changes outside the
        later citation-control additions.
        """
        baseline = json.loads((ROOT / "reviews/revision3_baseline.json").read_text())
        for rel, h in baseline["protected_sha256"].items():
            self.assertEqual(
                hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), h, rel
            )

    def test_restricted_loader(self):
        """Try filenames outside the explicitly allowed supervision set and
        require rejection. The check prevents evaluation references or
        annotation ledgers from being loaded as training data through this
        helper.
        """
        for filename in [
            "../scoring_only/test_references.jsonl",
            "all_training_records.jsonl",
            "answer_only_original_train.jsonl",
        ]:
            with self.assertRaises(ValueError):
                load_training(ROOT / "data", filename)

    def test_checksum_tampering(self):
        """Create a temporary training file whose content disagrees with the saved
        checksum and require rejection. This checks that the loader verifies
        bytes rather than trusting the filename alone.
        """
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "sft").mkdir()
            for name in ["manifest.json", "checksums.json"]:
                (root / name).write_bytes((ROOT / "data" / name).read_bytes())
            (root / "sft/subjectesis_full_train.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                load_training(root, "subjectesis_full_train.jsonl")

    def test_assistant_mask_and_padding(self):

        """Use a simple character tokenizer to inspect prompt masking, completion
        labels and padded labels directly. Also require an overlength example
        to fail instead of silently truncating its supervision.
        """
        class CharacterTokenizer:

            def encode(self, text, add_bos=True):
                """Represent each character as a distinct integer and optionally
                prepend a beginning marker. This deliberately simple test
                tokenizer makes prompt boundaries predictable without loading a
                real model tokenizer.
                """
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
        """Recalculate every generated data-file checksum listed in the manifest.
        Matching hashes confirm that the tests are inspecting the intended
        frozen conversion outputs.
        """
        for rel, h in json.loads((ROOT / "data/checksums.json").read_text()).items():
            self.assertEqual(
                hashlib.sha256((ROOT / "data" / rel).read_bytes()).hexdigest(), h
            )

    def test_rebuild_is_identical(self):
        """Rebuild the full dataset in a temporary directory and compare its
        checksum manifest byte for byte. This checks deterministic preparation
        without replacing the existing experimental files.
        """
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "data"
            run(output=out)
            self.assertEqual(
                (out / "checksums.json").read_bytes(),
                (ROOT / "data/checksums.json").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
