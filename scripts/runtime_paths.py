"""Load local runtime paths and provide validated path lookups."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"


def load_dotenv(path: Path = ENV_FILE) -> None:
    """Load KEY=VALUE entries without overriding variables already in the environment."""
    if not path.is_file():
        return

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env entry at {path}:{line_number}; expected KEY=VALUE")

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key.isidentifier():
            raise ValueError(f"Invalid environment variable name at {path}:{line_number}: {key!r}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def configured_path(name: str, default: str | Path | None = None) -> Path:
    """Return a configured path, resolving relative values from the repository root."""
    value = os.environ.get(name)
    if not value:
        if default is not None:
            value = str(default)
    if not value:
        raise RuntimeError(
            f"{name} is not configured. Copy .env.example to .env and set {name}, "
            "or define it in the process environment."
        )

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def artifact_directory() -> Path:
    """Return the configured artifact output directory, creating it if needed."""
    path = configured_path("ARTIFACT_DIR", PROJECT_ROOT / "output" / "artifacts")
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_sumo_environment() -> Path:
    """Expose the configured SUMO installation and executable to the CARLA bridge."""
    sumo_home = configured_path("SUMO_HOME")
    binary = sumo_binary()
    os.environ["SUMO_HOME"] = str(sumo_home)
    os.environ["SUMO_BINARY"] = str(binary)
    os.environ["PATH"] = str(binary.parent) + os.pathsep + os.environ.get("PATH", "")
    return sumo_home


def sumo_binary() -> Path:
    """Return SUMO_BINARY when set, otherwise the platform's standard SUMO executable."""
    if os.environ.get("SUMO_BINARY"):
        return configured_path("SUMO_BINARY")
    return configured_path("SUMO_HOME") / "bin" / ("sumo.exe" if os.name == "nt" else "sumo")


load_dotenv()
