from pathlib import Path
import sys

def assert_path_in_sys(path:Path) -> None:
    s = path.__str__()
    if s not in sys.path:
        sys.path.append(s)


src : Path = Path(__file__).parent
base : Path = src.parent
scripts : Path = base/"scripts"
outputs : Path = base/"outputs"
dump : Path = outputs/"dump"


assert_path_in_sys(src)