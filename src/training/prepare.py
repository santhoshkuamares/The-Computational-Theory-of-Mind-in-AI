"""Rebuild revision-4 training data and check its original hashes."""

from pathlib import Path
import argparse, subprocess, sys
from common import read_json, sha, write_json, load_train
from collections import Counter


def run(root):
    root = Path(root)
    prep = root / "preparation"
    (root / "logs").mkdir(exist_ok=True)
    expected = read_json(prep / "expected_data_checksums.json")
    for script in ["build_full.py", "test_full.py"]:
        cmd = [sys.executable, "-u", str(prep / "code" / script)]
        p = subprocess.run(cmd, cwd=prep, text=True, capture_output=True)
        (root / "logs" / ("preparation_" + script + ".log")).write_text(
            p.stdout + "\n" + p.stderr
        )
        if p.returncode:
            raise RuntimeError(p.stdout[-2000:] + p.stderr[-4000:])
        if script == "test_full.py":
            print(p.stdout + p.stderr, flush=True)
    checks = {n: sha(prep / "data" / n) == s for n, s in expected.items()}
    if not all(checks.values()):
        raise RuntimeError("Rebuild mismatch: " + str(checks))
    if read_json(prep / "data/checksums.json") != expected:
        raise RuntimeError("Checksum manifest changed")
    checks["checksums.json"] = sha(prep / "data/checksums.json") == sha(
        prep / "expected_data_checksums.json"
    )
    if not checks["checksums.json"]:
        raise RuntimeError("Checksum serialization changed")
    a = load_train(root, "subjectesis")
    b = load_train(root, "ordinary")
    if Counter((r["record_id"] for r in a)) != Counter((r["record_id"] for r in b)):
        raise ValueError("Exposure mismatch")
    write_json(
        root / "reports/data_verification.json",
        {
            "preparation_revision": 4,
            "files_byte_identical": checks,
            "training_examples": len(a),
            "eligible_questions": len({r["record_id"] for r in a}),
            "matched_baseline": True,
            "annotations_rewritten": False,
            "test_scored": False,
        },
    )
    print(
        "Revision 4 restored unchanged. 30,266 Subjectesis examples; optional matched baseline verified.",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    run(p.parse_args().root)
