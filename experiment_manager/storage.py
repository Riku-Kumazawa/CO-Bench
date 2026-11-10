"""Artifact storage and checks shared by experiment commands."""

import contextlib
import fcntl
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import re
import sys
from uuid import uuid4

import main as single

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def hashes(folder):
    folder = Path(folder)
    paths = sorted(folder.rglob("*"))
    if any(p.is_symlink() for p in paths):
        raise ValueError(f"Symlinks are not allowed in snapshots: {folder}")
    return {p.relative_to(folder).as_posix(): digest(p) for p in paths if p.is_file()}


def verify_hashes(folder, expected):
    if hashes(folder) != expected:
        raise RuntimeError(f"Saved artifacts changed: {folder}")


def environment():
    return {"python": sys.version, "packages": {n: version(n) for n in (*single.PACKAGES, "pyyaml")}}


def implementation():
    paths = [ROOT / "experiment.py", ROOT / "main.py", ROOT / "uv.lock", ROOT / "scripts/serve_qwen.sh", ROOT / "agents/utils.py",
             ROOT / "evaluation/controller.py", ROOT / "evaluation/utils.py", ROOT / "evaluation/evaluate.py"]
    paths += sorted((ROOT / "experiment_manager").glob("*.py"))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


@contextlib.contextmanager
def lock(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Already in use: {folder}") from None
        yield
