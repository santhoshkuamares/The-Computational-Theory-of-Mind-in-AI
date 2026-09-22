"""Prepare local inputs and the working layout expected by the recorded scripts.

The repository stores code as readable Python files. Small ZIP files contain
data only. This module replaces the original notebooks' embedded source ZIPs;
it does not change prompts, labels, model settings or prediction algorithms.
"""

from pathlib import Path
import hashlib
import json
import shutil
import zipfile

REPOSITORY = Path(__file__).resolve().parents[1]


def mount_drive():
    """Mount Google Drive when running in Colab; do nothing outside Colab."""
    try:
        from google.colab import drive
    except ImportError:
        return
    drive.mount("/content/drive")


def extract_checked(archive, destination):
    """Extract data safely and refuse to overwrite a different existing file."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            if item.is_dir():
                continue
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(
                    "Archive path leaves the destination: " + item.filename
                )
            data = bundle.read(item)
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError(
                        "Existing input differs from the frozen data: " + str(target)
                    )
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)


def ensure_analysis_inputs():
    """Make the bundled frozen predictions and references available to CPU scripts."""
    extract_checked(REPOSITORY / "data/frozen_analysis_inputs.zip", REPOSITORY)


def prepare_project(root):
    """Stage the source and revision-4 data under the original working paths.

    The old notebooks called the training directory ``code``. Here it lives
    under ``src/training`` for readers, then is copied to that original runtime
    location. Existing source is retained only when identical to this checkout.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for source_folder, target_folder in [
        (REPOSITORY / "src/training", root / "code"),
        (REPOSITORY / "src/preparation", root / "preparation/code"),
    ]:
        target_folder.mkdir(parents=True, exist_ok=True)
        for source in source_folder.glob("*.py"):
            # The A100 wrappers run outside the hashed original code directory.
            if source.name.startswith("a100_"):
                continue
            target = target_folder / source.name
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise ValueError(
                    "Use a fresh working directory; source differs: " + str(target)
                )
            shutil.copyfile(source, target)
    extract_checked(REPOSITORY / "data/preparation_sources.zip", root / "preparation")
    (root / "reports").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    return root


def restore_training_source(archive, root, condition):
    """Restore verified original code bytes and checkpoints for historical resume.

    The original run identity hashes source bytes, including comments. The
    readable formatting in this repository changes those hashes. A historical
    resume therefore uses the source stored in its own recovery archive rather
    than bypassing the identity check or rewriting the recorded identity.
    """
    root = Path(root)
    with zipfile.ZipFile(archive) as bundle:
        hashes = json.loads(bundle.read("recovery_manifest.json"))["files"]
        settings = json.loads(bundle.read("settings.json"))
        original_code = {name for name in hashes if name.startswith("code/")}
        for path in (root / "code").glob("*.py"):
            if "code/" + path.name not in original_code:
                raise ValueError(
                    "Extra source would change the historical run identity: "
                    + str(path)
                )
        for name, expected in hashes.items():
            if (
                name.startswith("code/")
                or name.startswith("runs/" + condition + "/")
                or name in ("settings.json", "environment_lock.json")
            ):
                data = bundle.read(name)
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError("Recovery checksum differs: " + name)
                target = (root / name).resolve()
                if not target.is_relative_to(root.resolve()):
                    raise ValueError("Unsafe recovery path: " + name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    return settings


if __name__ == "__main__":
    ensure_analysis_inputs()
    print("Frozen CPU analysis inputs are ready.")
