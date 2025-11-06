__all__ = [
    "print_traceback",
    "get_traceback",
]
# ============================================================================ #
#|                                Imports                                     |#
# ============================================================================ #

# For error handling:
import traceback
from typing import Any, Optional

from . import strings

# ============================================================================ #
#|                                Exports                                     |#
# ============================================================================ #


# ============================================================================ #
#|                           Custom Error Classes                             |#
# ============================================================================ #





# ============================================================================ #
#|                           Declared Functions                               |#
# ============================================================================ #


def print_traceback(e: Exception) -> None:
    s = get_traceback(e)
    s = strings.add_color(s, strings.PrintColors.RED)
    print(s)


def get_traceback(e: Exception) -> str:
    lines = traceback.format_exception(type(e), e, e.__traceback__)
    return ''.join(lines)


