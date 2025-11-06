from collections import deque   # used as a stack

import numpy as np
from numpy import inf

# For defining print std_out or other:
import sys

import time
import re

import functools


if __name__ == "__main__":
    from __init__ import add_root_to_path
    add_root_to_path()

from src.utils.strings import StrEnum, SpecialChars, num_out_of_num, str_len_until_first_newline
from src.utils.visuals.terminal_window import get_crnt_terminal_width, TerminalWidthTracker
from src.utils.metas import SingletonMeta
from src.utils import decorators, lists

from typing import Any, Literal, Optional, TextIO, List, Generic, TypeVar, Iterator, Iterable, cast, Collection, Final, Callable, Iterator
_T = TypeVar('_T')


BAR_SAFETY_MARGIN : Final[int] = 15
DEFAULT_PROG_BAR_FRAME_RATE : Final[float] = 15.0  # frames per second

# Compile ANSI escape pattern once at module level
_ANSI_ESCAPE_PATTERN : Final[re.Pattern[str]] = re.compile(r'\033\[[0-9;]*m')



class _FrameRateLimiter:
    def __init__(self, fps: float|Literal[False] = 30.0, always_allow_first: bool = True):
        if fps is False:
            interval = False
        elif isinstance(fps, (int, float)):
            assert fps >= 0.0, "Frame rate must be non-negative"
            interval = 1.0 / fps
        else:
            raise TypeError(f"Frame rate value of {fps!r} is not valid")

        self.interval : float|Literal[False] = interval
        self.last_time = time.time()
        self.always_allow_first : bool = always_allow_first
        self._is_first : bool = True

    def check(self) -> bool:
        now = time.time()

        if self.interval is False:
            return True
    
        elif self.always_allow_first and self._is_first:
            self._is_first = False
            self.last_time = now
            return True
        
        elif (now - self.last_time) >= self.interval:
            self.last_time = now
            return True
        
        return False
    

class StaticPrinter():

    def __init__(self, print_out:TextIO|Literal[False]=sys.stdout, in_place:bool=True) -> None:
        self.print_out : TextIO|Literal[False] = print_out
        self.in_place : bool = in_place
        self._last_printed_str : str = ""
        self._last_printed_printed_lines_lengths : List[int] = [0]
        self._method_when_number_of_lines_printed_changed = None

    def set_method_to_invoke_on_change_in_number_of_lines_printed(self, method:Callable[[], Any])->None:
        self._method_when_number_of_lines_printed_changed = method

    def _report_change_in_number_of_lines_printed(self)->None:
        if self._method_when_number_of_lines_printed_changed is not None:
            self._method_when_number_of_lines_printed_changed()

    @property
    def end_char(self)->str:
        if self.in_place:
            return ''
        else:
            return '\n'

    def _flush_now(self)->None:
        print_out = self.print_out
        try:
            if hasattr(print_out, 'flush'):
                print_out.flush()
        except Exception:
            pass

    def _print(self, s:str, end:Optional[str]=None)->None:
        """Print s to the configured output. Use self.end_char if end is None.
        No-op if self.print_out is the boolean False.
        """
        if end is None:
            end = self.end_char
        print_out = self.print_out
        if isinstance(print_out, bool) and print_out==False:
            return
        print(s, end=end, file=print_out)

    def _update_printed_lines_according_to_current_terminal_width(self) -> None:
        updated_printer_lines_lengths = _get_actual_printed_lines_lengths(self._last_printed_str)
        if len(updated_printer_lines_lengths) != len(self._last_printed_printed_lines_lengths):
            self._report_change_in_number_of_lines_printed()
        self._last_printed_printed_lines_lengths = updated_printer_lines_lengths
    
    def print(self, s:str, _skip_clear:bool=False, _flush_now:bool=True) -> None:
        """Print with bookkeeping for animated output.
        Tracks printed line lengths, clears previous output, prints s, flushes,
        and updates last-printed state.

        _skip_clear: If True, skips clearing previous output before printing.
                        Use only if you know what you're doing. Used by `ProgressBar.reprint_now()`.
        """
        if self.print_out is None:
            return
        # printing happens after data analyzed for smoother animation:
        if not _skip_clear:
            self.clear(force_flush_now=False)
        self._print(s)        
        if _flush_now:
            self._flush_now()  # Ensure sequences take effect immediately
        # Refresh memory:
        self._last_printed_str = s


    @decorators.ignore_first_method_call
    def clear(self, extra_lines:int=0, force_flush_now:bool=True) -> None:
        # If nothing was printed, nothing to clear
        if not self._last_printed_printed_lines_lengths:
            return

        self._update_printed_lines_according_to_current_terminal_width()
        reversed_printed_lengths = list(reversed(self._last_printed_printed_lines_lengths))

        print_out = self.print_out
        if isinstance(print_out, bool) and print_out is False:
            return
        
        if not self.in_place:
            # clear last line caused by `\n`
            self._print(SpecialChars.LineUp, end='')

        # Clear from the last printed logical line upwards
        is_first : bool = True
        for line_width in reversed_printed_lengths:
            num_rows = 1 + extra_lines
            # Clear any wrapped rows above
            for _ in range(num_rows):

                if is_first:
                    is_first = False
                else:
                    self._print(SpecialChars.LineUp, end='')

                self._print(SpecialChars.LineClear, end='')
                self._print(SpecialChars.CarriageReturn, end='')

        # Ensure sequences take effect immediately
        if force_flush_now:
            self._flush_now()
    



class StaticNumOutOfNum(Generic[_T]):
    def __init__(self, items:Iterable[_T]|int, print_prefix:str="", print_suffix:str="", print_out:TextIO|Literal[False]=sys.stdout, in_place:bool=False, _expected_length:int|None=None) -> None:    
        ## Fix different way of calling this object:
        if isinstance(items, int):
            items_ = range(items)
        else:
            items_ = items
        items_ = cast(Iterable[_T], items_,)

        ## Save basic data:
        self.static_printer : StaticPrinter = StaticPrinter(print_out=print_out, in_place=in_place)
        self.print_prefix :str = print_prefix
        self.print_suffix :str = print_suffix
        self.counter : int = -1
        self._is_iterated : bool = False
        self._items_iter : Iterator[_T] = iter(items_)

        # Get expected length:
        try: 
            _expected_length = len(items_)  #type: ignore   we're doing type ducking here
        except Exception:
            if _expected_length is None:
                _expected_length = int(inf)
        self.expected_length : int = _expected_length

        # First print:
        if self.expected_length>0:
            self._show()

    def __next__(self) -> _T:
        try:
            val = self.next()
        except StopIteration:
            self.clear()
            raise StopIteration
        return val

    def __iter__(self) -> "StaticNumOutOfNum":
        self._is_iterated = True
        return self

    def _check_end_iterations(self)->bool:
        return self._is_iterated and self.iteration_num > self.expected_length

    def next(self, increment:int=1, extra_str:Optional[str]=None, every:int=1) -> _T:
        self.counter += increment        
        self._show(extra_str)
        if self._check_end_iterations():
            raise StopIteration
        
        ## Chose return values:
        res = next(self._items_iter)
        return res

    def append_extra_str(self, extra_str:str)->None:
        self._show(extra_str)

    def clear(self):
        self.static_printer.clear()

    def _print(self, s:str):
        self.static_printer.print( s + self.print_suffix )

    @property
    def iteration_num(self) -> int:
        """Iteration-Number starting from 1"""
        return self.counter+1

    def _show(self, extra_str:Optional[str]=None):
        i = self.iteration_num
        expected_end = int( self.expected_length )
        s = num_out_of_num(i, expected_end)
        self._print( s )


class _ActiveBarsStack(Generic[_T], deque["ProgressBar[_T]"], metaclass=SingletonMeta(strict=True)):

    def __init__(self) -> None:
        super().__init__()
        self._terminal_width_tracker : TerminalWidthTracker = TerminalWidthTracker()
        self._frame_rate_limiter : _FrameRateLimiter = _FrameRateLimiter(fps=30.0)

    def __repr__(self) -> str:
        s = super().__repr__()
        for i, bar in enumerate(self):
            s += f"\n[{i}] {bar.print_prefix!r}"
        return s
    
    def iter_newest_to_oldest(self) -> Iterator["ProgressBar[_T]"]:
        """Iterate through progress bars from newest (top of stack) to oldest (bottom)."""
        return reversed(self)
    
    def iter_oldest_to_newest(self) -> Iterator["ProgressBar[_T]"]:
        """Iterate through progress bars from oldest (bottom of stack) to newest (top)."""
        return iter(self)   
    
    def on_wake_up(self)->None:
        if not self._frame_rate_limiter.check():
            return
        if self._terminal_width_tracker.width_changed():
            self.reprint_all_bars()

    def reprint_all_bars(self)->None:
        for bar in self.iter_newest_to_oldest():
            bar.static_printer.clear(force_flush_now=True)

        for bar in self.iter_oldest_to_newest():
            bar.reprint_now(_skip_clear=True, _flush_now=True)
        




class ProgressBar(Generic[_T]):
    _active_bars : _ActiveBarsStack[_T] = _ActiveBarsStack[_T]()

    @classmethod
    def _add_to_stack(cls, bar:"ProgressBar[_T]") -> None:
        cls._active_bars.append(bar)

    @classmethod    
    def _remove_from_stack(cls, bar:"ProgressBar[_T]") -> None:
        cls._active_bars.remove(bar)

    # A method to get the top active bar:
    @classmethod
    def newest(cls) -> "ProgressBar[_T]":
        if len(cls._active_bars)==0:
            raise ValueError("There are no active progress bars")
        return cls._active_bars[-1]
    
    @staticmethod
    def _wake_up_stack_after_call(method):
        """ A decorator to wake up the active bars stack after a method call. """
        @functools.wraps(method)
        def wrapper(self: "ProgressBar[_T]", *args, **kwargs):
            result = method(self, *args, **kwargs)
            self.active_bars_stack().on_wake_up()
            return result
        return wrapper

    def __new__(cls, items:int|Collection[_T]|Iterable[_T], **kwargs):
        if isinstance(items, int):
            assert "expected_end" not in kwargs
            kwargs["expected_end"] = items
            items = range(items)
        instance = super().__new__(cls)
        instance.__init__(items, **kwargs)

        # add to active bars stack:
        cls._add_to_stack(instance)
        return instance

    def __init__(self, 
        items:Collection[_T]|Iterable[_T]|int,
        /, *,
        expected_end:int|Literal['infinity']|None=None,
        prefix: str = "", 
        suffix: str = "", 
        print_length: int | None = None,   # If None then resort to terminal width
        out: TextIO | Literal[False] = sys.stdout, 
        _bar_empty_char: str = '.',
        _bar_full_char: str = u'█',
        _frame_rate:float|Literal[False] = DEFAULT_PROG_BAR_FRAME_RATE  # frames per second
    ) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            return

        ## The main object to track what should be printed and deleted:
        self.static_printer : StaticPrinter = StaticPrinter(print_out=out, in_place=False)
        ## Save basic data:
        self.print_prefix :str = prefix
        self.print_suffix :str = suffix
        self.print_length :int|None = print_length
        ## Iteration variables:
        self._main_iter : Iterator[tuple[int, _T]] = enumerate(items)
        self.expected_end : int|Literal['unknown'] = _derive_expected_end(items, expected_end)
        self._crnt_run_index : int = 0
        ## Printing characters:
        self._bar_empty_char : str = _bar_empty_char
        self._bar_full_char : str = _bar_full_char
        ## Prevents double calling when using __new__
        self._initialized : bool = True
        self._been_cleared : bool = False
        ## Frame rate limiter:
        self._frame_rate_limiter : _FrameRateLimiter = _FrameRateLimiter(fps=_frame_rate)
        ## Terminal width watcher flags:
        self._freeze_showing : bool = False    
        # Additional strings the user can append:
        self._extra_str : str = ""

    @classmethod
    def range(
        cls,
        num:int,
        *,
        prefix:str="", 
        suffix:str="", 
        print_length:int|None=None,   # If None then resort to terminal width
        out:TextIO|Literal[False]=sys.stdout, 
        _bar_empty_char:str='.',
        _bar_full_char:str=u'█',
        _frame_rate:float|Literal[False]=15.0  # frames per second
    ) -> "ProgressBar[int]":
        prog_bar = cls(    #type: ignore
            range(num),
            expected_end = num,
            prefix = prefix,
            suffix = suffix,
            print_length = print_length,
            out = out,
            _bar_empty_char = _bar_empty_char,
            _bar_full_char = _bar_full_char,
            _frame_rate = _frame_rate
        )
        prog_bar = cast(ProgressBar[int], prog_bar)
        return prog_bar  

    def active_bars_stack(self) -> _ActiveBarsStack[_T]:
        return self.__class__._active_bars

    def __iter__(self) -> "_ProgressBarIterator[_T]":
        return _ProgressBarIterator(self)

    def _derive_bar_str(self)->str:     
        """ Derives the full progress bar string + prefix and suffixes.
        """
        # Unpack properties:
        prefix_str = self.print_prefix
        suffix_str = self.print_suffix
        expected_end = self.expected_end
        e_c = self._bar_empty_char  #usually '.'
        f_c = self._bar_full_char  #usually  u'█'
        crnt_run_index = self._crnt_run_index

        ## Get the expected end: 
        print_length = self.print_length
        adjusted_terminal_width = get_crnt_terminal_width() - 5  # 5 for good measure
        if print_length is None:
            print_length = adjusted_terminal_width
        else:
            print_length = min(print_length, adjusted_terminal_width)

        if expected_end == 'unknown':
            expected_end = np.inf
            _num_out_of_num_str = f"{crnt_run_index:4}"
        elif isinstance(expected_end, int):
            _num_out_of_num_str = num_out_of_num(crnt_run_index, expected_end)
        else:
            raise TypeError(f"Expected end value of {expected_end} is not valid")

        ## Lengths of the full and empty bars:
        # Adding two for brackets and a few more for good measure = BAR_SAFETY_MARGIN:
        entire_bar_length = print_length - max(len(_num_out_of_num_str), 10) - len(prefix_str) - len(suffix_str) - BAR_SAFETY_MARGIN
        if crnt_run_index > expected_end:
            full_bar_length = entire_bar_length
        else:
            full_bar_length = int(entire_bar_length*crnt_run_index/expected_end)
        empty_bar_length = entire_bar_length-full_bar_length
        
        ## The actual string:
        s = f"{prefix_str}[{f_c*full_bar_length}{(e_c*empty_bar_length)}] "+_num_out_of_num_str+suffix_str

        return s
    
    def full_str(self) -> str:
        s = self._derive_bar_str()
        s += " "+self.get_extra_str()
        return s

    def reprint_now(self, _skip_clear:bool=False, _flush_now:bool=True) -> None:
        """ Forces immediate print regardless of frame rate limiter. 
        Also does not wake up the stack after call (the stack itself is usually the one calling this method). """
        s = self.full_str()
        self.static_printer.print(s, _skip_clear=_skip_clear, _flush_now=_flush_now)

    @_wake_up_stack_after_call
    def _show(self, print_now:bool=True) -> None:
        ## Don't show if frozen or if frame rate limiter says no:
        if not print_now:
            if self._freeze_showing: 
                return
            
            if not self._frame_rate_limiter.check():
                return

        ## Get full string to print and print using our static printer object:
        self.reprint_now()

    def get_extra_str(self) -> str:
        return self._extra_str

    def reset_extra_str(self) -> None:
        self._extra_str = ""

    def append_extra_str(self, s:str, print_now:bool=True) -> None:
        self._extra_str += s
        self._show(print_now=print_now)

    def next(self, skip:int=1, extra_str:str='') -> _T:
        self._extra_str = extra_str  # Reset extra string and set to new value

        assert skip>0, "Increment must be greater than 0"
        item : _T = None  #type: ignore
        i: int = -1  # just for type checking

        for _ in range(skip):
            i, item = next(self._main_iter)

        self._crnt_run_index = i + 1

        self._show()

        return item
    
    def clear(self) -> None:
        self._been_cleared = True
        # Remove related text:
        self.static_printer.clear()
        # Remove from active bars stack:
        self.__class__._remove_from_stack(self)

    def __del__(self) -> None:
        if not self._been_cleared:
            self.clear()
    

class _ProgressBarIterator(Generic[_T]):
    """
    An iterator for the ProgressBar class.
    This class enables iteration over the elements of a ProgressBar instance,
    updating the progress bar as items are consumed. When the underlying
    ProgressBar is exhausted, the progress bar is cleared and StopIteration is raised.
    Typical usage:
        for item in ProgressBar(...):
            # process item
    Attributes:
        prog_bar_obj (ProgressBar[_T]): The ProgressBar instance being iterated.
    Raises:
        StopIteration: When the ProgressBar has no more items to iterate.
    """

    def __init__(self, prog_bar_obj:ProgressBar[_T]) -> None:
        self.prog_bar_obj : ProgressBar[_T] = prog_bar_obj

    def __next__(self) -> _T:
        try:
            val = self.prog_bar_obj.next()
        except StopIteration:
            self.prog_bar_obj.clear()
            raise StopIteration
        return val
    
    def __del__(self) -> None:
        if not self.prog_bar_obj._been_cleared:
            self.prog_bar_obj.clear()


def _strip_ansi_codes(s: str) -> str:
    """Remove ANSI escape sequences from a string."""
    return _ANSI_ESCAPE_PATTERN.sub('', s)


def _get_actual_printed_lines_lengths(s:str) -> List[int]:
        lines = s.split(SpecialChars.NewLine)
        # Strip ANSI codes before calculating lengths
        intended_lines_lengths = [len(_strip_ansi_codes(line)) for line in lines]

        terminal_width = get_crnt_terminal_width()
        actual_lines_lengths : List[int] = []
        for line_length in intended_lines_lengths:
            if line_length==0:
                actual_lines_lengths.append(0)
            elif line_length <= terminal_width:
                actual_lines_lengths.append(line_length)
            else:
                num_wrapped_lines = (line_length-1)//terminal_width + 1
                for _ in range(num_wrapped_lines-1):
                    actual_lines_lengths.append(terminal_width)
                last_line_length = line_length - (num_wrapped_lines-1)*terminal_width
                actual_lines_lengths.append(last_line_length)
        
        return actual_lines_lengths



def _derive_expected_end(items:Collection[_T]|Iterable[_T], _expected_end:int|Literal['unknown']|None)->int|Literal['unknown']:
    try:
        if isinstance(items, np.nditer):
            items_len = items.itersize
        else:
            items_len = len(items)  #type: ignore
    except Exception:
        items_len = 'unknown'
    
    if _expected_end is None:
        return items_len
    elif _expected_end == 'unknown':
        return items_len
    elif isinstance(_expected_end, int):
        if isinstance(items_len, int) and _expected_end != items_len:
            raise ValueError(f"Expected end value of {_expected_end} is not equal to the length of the items {items_len}")
        return _expected_end
    else:
        raise TypeError(f"Expected end value of {_expected_end} is not valid")


class PrintColors(StrEnum):
    DEFAULT = '\033[0m'
    # Styles:
    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    UNDERLINE_THICK = '\033[21m'
    STRIKE_THROUGH = '\033[9m'
    ## Highlighted:
    HIGHLIGHTED = '\033[7m'
    HIGHLIGHTED_BLACK = '\033[40m'
    HIGHLIGHTED_RED = '\033[41m'
    HIGHLIGHTED_GREEN = '\033[42m'
    HIGHLIGHTED_YELLOW = '\033[43m'
    HIGHLIGHTED_BLUE = '\033[44m'
    HIGHLIGHTED_PURPLE = '\033[45m'
    HIGHLIGHTED_CYAN = '\033[46m'
    HIGHLIGHTED_GREY = '\033[47m'
    #
    HIGHLIGHTED_GREY_LIGHT = '\033[100m'
    HIGHLIGHTED_RED_LIGHT = '\033[101m'
    HIGHLIGHTED_GREEN_LIGHT = '\033[102m'
    HIGHLIGHTED_YELLOW_LIGHT = '\033[103m'
    HIGHLIGHTED_BLUE_LIGHT = '\033[104m'
    HIGHLIGHTED_PURPLE_LIGHT = '\033[105m'
    HIGHLIGHTED_CYAN_LIGHT = '\033[106m'
    HIGHLIGHTED_WHITE_LIGHT = '\033[107m'

    MARGIN_1 = '\033[51m'
    MARGIN_2 = '\033[52m' # seems equal to MARGIN_1
    
    ## colors
    BLACK = '\033[30m'
    RED_DARK = '\033[31m'
    GREEN_DARK = '\033[32m'
    YELLOW_DARK = '\033[33m'
    BLUE_DARK = '\033[34m'
    PURPLE_DARK = '\033[35m'
    CYAN_DARK = '\033[36m'
    GREY_DARK = '\033[37m'

    BLACK_LIGHT = '\033[90m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[96m'



def add_color(s:str, color:PrintColors, active:bool=True)->str:
    if not active:
        return s
    return color+s+PrintColors.DEFAULT

def print_warning(s:str)->None:
    warn1color = PrintColors.HIGHLIGHTED_YELLOW
    warn2color = PrintColors.YELLOW_DARK
    s = add_color("Warning: ", warn1color)+add_color(s, warn2color)
    print(s)


def fix_numpy_print_length(linewidth:int=10000, precision:int=2):
    np.set_printoptions(linewidth=linewidth)
    np.set_printoptions(precision=precision)


def numpy_array_shortened(arr:np.ndarray, precision:int=3)->None:
     with np.printoptions(precision=precision, suppress=True):
        print(arr)








def _run_prog_bar_test(
    num_items:int = 100,
    sleep_time:float = 0.01
)->None:

    print("Begin progress bar test...")
    for i in ProgressBar.range(num_items, prefix="Progress i: "):
        for j in ProgressBar.range(num_items, prefix="Progress j: "):
            time.sleep(sleep_time)  # Simulate work being done

if __name__ == "__main__":
    _run_prog_bar_test()
