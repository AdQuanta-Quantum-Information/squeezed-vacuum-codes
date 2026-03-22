from typing import Final
from dataclasses import dataclass


@dataclass(frozen=True)
class Globals:
    PRECISE: Final[bool] = False
    DEBUG: Final[bool] = True
    CACHE_ON_DISK: Final[bool] = True
    LaTeX_RENDERING: Final[bool] = True
