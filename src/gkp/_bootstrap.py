import sys
from pathlib import Path


def ensure_project_root_on_path() -> list[str]:
    """Add repository root to sys.path based on this file location."""
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return sys.path


def add_root_to_path() -> list[str]:
    """Ensure root path is available, then delegate to src-level helper when available."""
    ensure_project_root_on_path()

    try:
        from src import add_root_to_path as _src_add_root_to_path
    except ModuleNotFoundError:
        return sys.path

    return _src_add_root_to_path()


__all__ = [
    "ensure_project_root_on_path",
    "add_root_to_path",
]
