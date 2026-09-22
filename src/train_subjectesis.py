"""Prepare and launch the Subjectesis condition using its recorded source.

The completed run started on two T4 GPUs and continued on one A100. A new run
is not expected to reproduce its weights bit for bit across hardware. The
saved best adapter is the source of the published evaluation results.

Use --prepare-only for the deterministic data rebuild without model loading.
Historical resume requires the original archive and keeps its identity checks.
"""

from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
from project_setup import prepare_project, restore_training_source, REPOSITORY


def run(command, env=None):
    """Execute one pipeline command using the requested environment and stop if it
    fails. Converting path objects to strings lets the same helper launch
    preparation, dependency setup and training stages.
    """
    subprocess.run([str(part) for part in command], check=True, env=env)


def main():
    """Read command-line options and prepare the Subjectesis working directory.
    Run the recorded T4 workflow or restore a historical archive for A100
    continuation, then launch training; preparation-only mode stops after
    rebuilding and checking the data.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["t4", "a100"], default="t4")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    root = prepare_project(args.root.resolve())
    settings = json.loads((REPOSITORY / "config/subjectesis_settings.json").read_text())
    (root / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "code") + os.pathsep + env.get("PYTHONPATH", "")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    run([sys.executable, root / "code/prepare.py", "--root", root], env)
    if args.prepare_only:
        return
    if args.resume:
        restore_training_source(args.resume, root, "subjectesis")
    if args.mode == "a100" and not args.resume:
        raise ValueError(
            "The recorded A100 path is a continuation. Supply --resume, or start with --mode t4."
        )
    if args.mode == "t4":
        run([sys.executable, root / "code/setup_runtime.py", "--root", root], env)
        python = root / "torch_env/bin/python"
    else:
        # Use the exact versions recorded in the recovered Colab environment.
        python = Path(sys.executable)
        run([python, "-m", "pip", "install", "-r", REPOSITORY / "requirements-gpu.txt"])
        run([python, "-m", "pip", "install", "--no-deps", "torchao==0.16.0"])
    run([python, root / "code/tokenize_data.py", "--root", root], env)
    run([python, root / "code/download_model.py", "--root", root], env)
    if args.mode == "t4":
        # Keep the original two-GPU environment settings and recovery packaging.
        command = [python, root / "code/launch.py", "--root", root]
        if args.smoke_only:
            command.append("--smoke-only")
        run(command, env)
        return
    entry = REPOSITORY / "src/training/a100_subjectesis.py"
    env["SUBJECTESIS_ROOT"] = str(root)
    command = [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=1",
        entry,
        "--root",
        root,
        "--condition",
        "subjectesis",
    ]
    if args.smoke_only:
        command.append("--smoke-only")
    run(command, env)


if __name__ == "__main__":
    main()
