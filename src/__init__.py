"""Basketball in-class teaching system — production package."""

from pathlib import Path

_VERSION_FILE = Path(__file__).resolve().parents[1] / "VERSION"
try:
    __version__ = _VERSION_FILE.read_text(encoding="utf-8").strip()
except OSError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
