"""Install and record the original training runtime dependencies."""

from pathlib import Path
import argparse, json, os, subprocess, sys, venv
from common import read_json, write_json

PINS = [
    "transformers==5.13.0",
    "peft==0.21.0",
    "accelerate==1.12.0",
    "bitsandbytes==0.49.2",
    "safetensors==0.8.0",
    "numpy==2.0.2",
    "huggingface-hub==1.11.0",
    "tokenizers==0.22.2",
    "packaging==26.3",
    "jinja2==3.1.6",
]
TORCHAO_PIN = "torchao==0.16.0"


def run(root):
    root = Path(root)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    import torch

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"This notebook targets your Kaggle Python 3.12 runtime. Found {sys.version.split()[0]}; do not install into a different runtime silently."
        )
    if torch.__version__.split("+")[0] != "2.10.0":
        raise RuntimeError(
            f"This dependency lock targets the Torch 2.10.0 from your Kaggle logs. Found {torch.__version__}. Torch/CUDA were not changed."
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Select GPU T4 x2 and use a fresh Kaggle session.")
    info = []
    for i in range(2):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        info.append(
            dict(
                gpu=i,
                name=p.name,
                capability=[p.major, p.minor],
                free_gib=free / 2**30,
                total_gib=total / 2**30,
            )
        )
        if "T4" not in p.name:
            raise RuntimeError("This edition is configured for two T4 GPUs.")
        if free < 12 * 2**30:
            raise RuntimeError(
                "Less than 12 GiB free on a GPU. Stop old model processes or start a fresh session."
            )
    print("Attached GPUs:", json.dumps(info, indent=2), flush=True)
    env = root / "torch_env"
    py = env / "bin/python"
    if not py.exists():
        venv.EnvBuilder(system_site_packages=True, with_pip=False).create(env)
    constraints = root / "torch_constraints.txt"
    constraints.write_text("torch===" + torch.__version__ + "\n")
    request = dict(
        pins=PINS, torchao=TORCHAO_PIN, host_python=sys.version, torch=torch.__version__
    )
    marker = env / "subjectesis_setup.json"
    if not marker.exists() or read_json(marker) != request:
        prefix = [
            sys.executable,
            "-m",
            "pip",
            "--python",
            str(py),
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--only-binary=:all:",
        ]
        print("Installing project packages; retaining host Torch and CUDA.", flush=True)
        subprocess.run(
            prefix
            + [
                "--upgrade",
                "--upgrade-strategy",
                "only-if-needed",
                "-c",
                str(constraints),
                *PINS,
            ],
            check=True,
        )
        subprocess.run(prefix + ["--upgrade", "--no-deps", TORCHAO_PIN], check=True)
    probe = "\nimport json, sys, importlib.metadata as m\nfrom pathlib import Path\nimport torch, torchao\nfrom transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig, AutoTokenizer\nimport peft, bitsandbytes, accelerate\nfrom peft.import_utils import is_torchao_available\nassert torch.__version__ == sys.argv[1], 'Host Torch must remain unchanged'\nassert torch.cuda.is_available() and torch.cuda.device_count() == 2\nassert m.version('torchao') == '0.16.0' and is_torchao_available()\nassert Path(torchao.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())\nprint(json.dumps({x:m.version(x) for x in [\n 'torch','torchao','transformers','peft','bitsandbytes','accelerate',\n 'tokenizers','huggingface-hub','safetensors','numpy','packaging','jinja2'\n]}, sort_keys=True))\n"
    result = subprocess.run(
        [str(py), "-c", probe, torch.__version__],
        text=True,
        capture_output=True,
        timeout=180,
    )
    (root / "logs/environment_probe.log").write_text(
        result.stdout + "\n" + result.stderr
    )
    if result.returncode:
        raise RuntimeError(
            "Environment compatibility test failed:\n" + result.stderr[-6000:]
        )
    versions = json.loads(result.stdout.strip().splitlines()[-1])
    write_json(root / "environment_lock.json", versions)
    write_json(marker, request)
    write_json(
        root / "reports/hardware.json",
        dict(devices=info, torch=torch.__version__, cuda=torch.version.cuda),
    )
    print("Training environment:", versions, flush=True)
    print("Environment ready:", py, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    run(p.parse_args().root)
