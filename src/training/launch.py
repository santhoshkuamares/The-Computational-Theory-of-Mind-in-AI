"""Launch the original two-GPU run, stream progress and package recovery outputs."""

from pathlib import Path
import argparse, os, subprocess, sys, threading, time, signal
from common import *


def child_env(root):
    env = dict(os.environ)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DOWNLOAD_TIMEOUT": "180",
            "HF_HUB_ETAG_TIMEOUT": "60",
            "USE_TF": "0",
            "USE_FLAX": "0",
            "HF_HOME": str(Path(root) / "hf_cache"),
            "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "OMP_NUM_THREADS": "2",
            "PIP_NO_CACHE_DIR": "1",
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        }
    )
    for k in [
        "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME",
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "PYTHONPATH",
    ]:
        env.pop(k, None)
    return env


def stream(command, logfile, env):
    logfile = Path(logfile)
    logfile.parent.mkdir(exist_ok=True, parents=True)
    with logfile.open("a", buffering=1) as f:
        print("Starting:", " ".join(map(str, command)), flush=True)
        p = subprocess.Popen(
            list(map(str, command)),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
        done = threading.Event()

        def heartbeat():
            start = time.monotonic()
            while not done.wait(30):
                print(
                    f"Process alive: {int(time.monotonic() - start)}s. Full log: {logfile.name}",
                    flush=True,
                )

        t = threading.Thread(target=heartbeat, daemon=True)
        t.start()
        try:
            for line in p.stdout:
                print(line, end="", flush=True)
                f.write(line)
            code = p.wait()
            if code:
                raise RuntimeError(f"Process exited {code}; inspect {logfile}")
        except BaseException:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
            raise
        finally:
            done.set()
            t.join(timeout=2)


def run(root, smoke_only=False):
    root = Path(root)
    c = load_config(root)
    py = root / "torch_env/bin/python"
    conditions = ["subjectesis", "ordinary"] if c["train_baseline"] else ["subjectesis"]
    if (root / "STOP").exists():
        raise RuntimeError("Remove the STOP file to resume training")
    try:
        for condition in conditions:
            command = [
                py,
                "-u",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                "--nproc_per_node=2",
                root / "code/train.py",
                "--root",
                root,
                "--condition",
                condition,
            ]
            if smoke_only:
                command += ["--smoke-only"]
            stream(
                command, root / "logs" / f"{condition}_training.log", child_env(root)
            )
            status = root / "runs" / condition / "status.json"
            if (
                status.exists()
                and read_json(status).get("status") == "paused_resumable"
            ):
                break
    finally:
        print("OUTPUT ZIP:", package_outputs(root), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--smoke-only", action="store_true")
    a = p.parse_args()
    run(a.root, a.smoke_only)
