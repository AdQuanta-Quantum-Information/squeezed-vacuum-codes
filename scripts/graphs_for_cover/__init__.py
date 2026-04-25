from pathlib import Path
import sys


root_path = Path(__file__).parents[2]


def _add_path_to_sys_path(path: Path) -> list[str]:
    as_str = str(path.resolve())
    if as_str not in sys.path:
        sys.path.append(as_str)
    return sys.path


def add_root_to_path():
    return _add_path_to_sys_path(root_path)
