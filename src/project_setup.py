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
    """Mount Google Drive at the standard Colab location when google.colab is
    available. Outside Colab the helper returns immediately, allowing CPU
    analysis to run locally.
    """
    try:
        from google.colab import drive
    except ImportError:
        return
    drive.mount("/content/drive")


def extract_checked(archive, destination):
    """Unpack a data archive into a specified directory after checking each
    destination path. Existing identical files are reused, while different
    contents raise an error to avoid mixing frozen inputs with another run.
    """
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
    """Extract the bundled frozen predictions and references into the repository
    layout expected by the CPU scripts. The checked extractor makes repeated
    calls safe when the same inputs are already present.
    """
    extract_checked(REPOSITORY / "data/frozen_analysis_inputs.zip", REPOSITORY)


def prepare_project(root):
    """Copy readable training and preparation modules into the working paths used
    by the original notebooks, then unpack the preparation data. Return the
    working root after creating log and report folders; conflicting source
    files require a fresh directory.
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
    """Restore checksum-verified historical source, settings and condition
    checkpoints from a recovery archive. Source-byte hashes are part of the
    original run identity, so resuming that run uses its archived code rather
    than bypassing identity checks after documentation changes.
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
