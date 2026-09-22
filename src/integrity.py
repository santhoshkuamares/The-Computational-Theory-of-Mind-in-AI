"""Frozen-output CPU analysis: integrity.
Run from the repository root. No model inference or adapter training occurs."""

from project_setup import ensure_analysis_inputs

ensure_analysis_inputs()
import argparse, collections, json, random, zipfile
from pathlib import Path
from analyse import read_json, read_rows, sha, write_json


def run(root, preparation=None, uploads=None):
    report = {}
    prep = root / "inputs/preparation"
    m = read_json(prep / "manifest.json")
    checks = read_json(prep / "checksums.json")
    matched = []
    for rel, h in checks.items():
        p = prep / rel
        if p.exists():
            assert sha(p) == h
            matched.append(rel)
    report["preparation_hashes_verified"] = matched
    rect = root / "inputs/rectom/final_test"
    rm = read_json(rect / "final_metrics.json")
    assert sha(prep / "evaluation/test_inputs.jsonl") == rm["test_input_sha256"]
    assert (
        sha(prep / "scoring_only/test_references.jsonl") == rm["test_reference_sha256"]
    )
    for p in rect.glob("*.meta.json"):
        meta = read_json(p)
        assert meta["protocol_hash"] == rm["protocol_hash"]
        assert meta["test_input_sha256"] == rm["test_input_sha256"]
    report["rectom_protocol_and_inputs_match"] = True
    op = root / "inputs/opentom/opentom_transfer"
    c = read_json(op / "scoring_correction_manifest.json")
    assert sha(op / "final_inputs_corrected.jsonl") == c["corrected_input_sha256"]
    assert (
        sha(op / "final_references_corrected.jsonl") == c["corrected_reference_sha256"]
    )
    protocol = read_json(op / "opentom_protocol_frozen.json")
    assert protocol["protocol_hash"] == c["protocol_hash"]
    assert set(protocol["selection"]["final_story_ids"]) == set(c["story_ids"])
    assert not set(protocol["selection"]["final_story_ids"]) & set(
        protocol["selection"]["smoke_story_ids"]
    )
    inputs = read_rows(op / "final_inputs_corrected.jsonl")
    assert all(("answer" not in r for r in inputs))
    report["opentom_corrected_hashes_protocol_subset_and_answer_free_inputs_match"] = (
        True
    )
    if preparation:
        data = Path(preparation) / "data"
        arms = {}
        counts = {}
        exposures = {}
        kinds = {}
        for filename, cond, limit in [
            ("answer_only_matched_train.jsonl", "ordinary", 8000),
            ("subjectesis_full_train.jsonl", "subjectesis", 20000),
        ]:
            p = data / "sft" / filename
            rows = read_rows(p)
            assert sha(p) == checks["sft/" + filename]
            counts[cond] = collections.Counter((r["record_id"] for r in rows))
            assert len(rows) == 30266
            order = list(range(len(rows)))
            random.Random(42).shuffle(order)
            sel = [rows[i] for i in order[:limit]]
            exposures[cond] = {
                "selected_exposures": limit,
                "selected_unique_questions": len({r["record_id"] for r in sel}),
                "selected_dialogues": len({r["dialogue_id"] for r in sel}),
            }
            kinds[cond] = dict(collections.Counter((r["kind"] for r in rows)))
        assert counts["ordinary"] == counts["subjectesis"]
        report["full_training_question_exposure_identical"] = True
        report["selected_checkpoint_exposure"] = exposures
        report["training_example_types"] = kinds
    if uploads:
        archives = {}
        for p in Path(uploads).glob("*.zip"):
            if any(
                (
                    t in p.name
                    for t in [
                        "a100_latest(3)",
                        "final_test-20260921",
                        "dev_evaluation-20260921",
                        "opentom_transfer-20260922",
                    ]
                )
            ):
                with zipfile.ZipFile(p) as z:
                    assert z.testzip() is None
                archives[p.name] = {"sha256": sha(p), "zip_crc_passed": True}
        report["archives"] = archives
        with zipfile.ZipFile(
            next(Path(uploads).glob("opentom_transfer-20260921*.zip"))
        ) as old:
            same = {}
            for fn in [
                "base_direct.jsonl",
                "answer_only_direct.jsonl",
                "subjectesis_direct.jsonl",
                "subjectesis_structured.jsonl",
            ]:
                name = next((n for n in old.namelist() if n.endswith("/" + fn)))
                same[fn] = old.read(name) == (op / fn).read_bytes()
                assert same[fn]
            report["frozen_predictions_byte_identical_before_after_correction"] = same
    write_json(root / "results/integrity_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--preparation")
    p.add_argument("--uploads")
    a = p.parse_args()
    run(a.root, a.preparation, a.uploads)
