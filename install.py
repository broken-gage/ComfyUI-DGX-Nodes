from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


LEGACY_FOLDER_NAME = "dgx-gb10-nodes"
CURRENT_PACKAGE_NAME = "comfyui-dgx-nodes"
LEGACY_PACKAGE_NAME = "dgx-gb10-nodes"
EXPECTED_REPOSITORY_FRAGMENT = "broken-gage/ComfyUI-DGX-Nodes"
PREFIX = "[ComfyUI-DGX-Nodes install]"


def _read_project_metadata(pyproject_path: Path) -> tuple[str, str]:
    if not pyproject_path.is_file():
        return "", ""

    try:
        import tomllib

        with pyproject_path.open("rb") as handle:
            data = tomllib.load(handle)

        project = data.get("project", {})
        urls = project.get("urls", {})
        name = str(project.get("name", "")).strip()
        repository = str(urls.get("Repository", "")).strip()
        return name, repository
    except Exception:
        pass

    name = ""
    repository = ""
    in_project = False
    in_urls = False

    try:
        lines = pyproject_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "", ""

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_project = line == "[project]"
            in_urls = line == "[project.urls]"
            continue
        if in_project and line.startswith("name"):
            name = _parse_toml_string_value(line)
        elif in_urls and line.startswith("Repository"):
            repository = _parse_toml_string_value(line)

    return name, repository


def _parse_toml_string_value(line: str) -> str:
    if "=" not in line:
        return ""
    value = line.split("=", 1)[1].strip()
    if "#" in value:
        value = value.split("#", 1)[0].strip()
    return value.strip("\"'")


def _is_dirty_git_repo(path: Path) -> bool:
    git_dir = path / ".git"
    if not git_dir.exists():
        return False

    try:
        result = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return True

    return result.returncode != 0 or bool(result.stdout.strip())


def _looks_like_legacy_dgx_folder(path: Path) -> bool:
    name, repository = _read_project_metadata(path / "pyproject.toml")
    normalized_name = name.strip().lower()
    if normalized_name not in {LEGACY_PACKAGE_NAME, CURRENT_PACKAGE_NAME}:
        return False
    return EXPECTED_REPOSITORY_FRAGMENT in repository


def migrate_legacy_folder(current_dir: Path | None = None) -> bool:
    current = (current_dir or Path(__file__).resolve().parent).resolve()
    legacy = current.parent / LEGACY_FOLDER_NAME

    if not legacy.exists():
        print(f"{PREFIX} no legacy {LEGACY_FOLDER_NAME} folder found")
        return False

    legacy_resolved = legacy.resolve()
    if legacy_resolved == current:
        print(f"{PREFIX} current folder is still {LEGACY_FOLDER_NAME}; skipping self-removal")
        return False

    if not legacy.is_dir():
        print(f"{PREFIX} {legacy} exists but is not a directory; skipping")
        return False

    if not _looks_like_legacy_dgx_folder(legacy):
        print(f"{PREFIX} {legacy} does not look like the legacy DGX node; skipping")
        return False

    if _is_dirty_git_repo(legacy):
        print(f"{PREFIX} {legacy} has local git changes; skipping cleanup")
        return False

    try:
        shutil.rmtree(legacy)
    except OSError as exc:
        print(f"{PREFIX} failed to remove {legacy}: {exc}")
        return False

    print(f"{PREFIX} removed legacy folder: {legacy}")
    return True


if __name__ == "__main__":
    migrate_legacy_folder()
