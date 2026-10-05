"""Shared archive boundaries; standard library only."""
from pathlib import Path, PurePosixPath
import hashlib
import json


METADATA = Path(__file__).resolve().parents[1] / "metadata"
RECORD = "04_research_archive/record"


def paper_root(value):
    if value is None:
        raise ValueError("--paper-root is required: supply the companion study package, not this repository")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Companion package directory does not exist: {root}")
    if not (root / "04_research_archive").is_dir():
        raise ValueError(f"Not a companion study package (04_research_archive is missing): {root}")
    return root


def inside(root, locator, *, must_exist=True):
    root = Path(root).resolve()
    rel = PurePosixPath(str(locator))
    if rel.is_absolute() or ".." in rel.parts or not rel.parts or "\\" in str(locator):
        raise ValueError(f"Unsafe package-relative locator: {locator}")
    path = root.joinpath(*rel.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Locator escapes package through a symlink: {locator}")
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Missing companion file: {locator}")
    return path


def external_output(root, value):
    output = Path(value).expanduser().resolve()
    if output.is_relative_to(Path(root).resolve()):
        raise ValueError("Outputs must be outside the companion package")
    return output


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def emit(report, output=None, root=None):
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if output:
        destination = external_output(root, output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    print(text, end="")

