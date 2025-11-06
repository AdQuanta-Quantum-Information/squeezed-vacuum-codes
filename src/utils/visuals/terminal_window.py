from typing import Callable, Literal, Final

# For signalling 
import signal
import threading

# For getting the terminal width:
import shutil

import time


DEFAULT_TERMINAL_WIDTH : Final[int] = 80  # Fallback terminal width if it cannot be determined


class TerminalWidthWatcher:
    """
    Call callbacks when the terminal width changes.

    - POSIX: uses SIGWINCH
    - Otherwise: polls at poll_interval seconds
    
    Calls `on_resize_start` immediately when resizing begins, then debounces
    and calls `on_resize_done` only after the user stops resizing.
    """
    def __init__(
        self, 
        on_resize_start: Callable,
        on_resize_done: Callable,
        poll_freq: float|Literal[False] = 4,
        debounce_delay: float = 0.5
    ):
        """
        Args:
            on_resize_start: method to call when resize starts (called immediately)
            on_resize_done: method to call when resize is done (debounced)
            poll_freq: Polling frequency in Hz (or False to use default interval)
            debounce_delay: Seconds to wait after last resize before calling on_resize_done
        """
        if poll_freq is False:
            poll_interval = 0.01  # [seconds]
        else:
            poll_interval = 1 / poll_freq

        self._on_resize_start_method : Callable = on_resize_start
        self._on_resize_done_method  : Callable = on_resize_done
        self._poll_interval : float = poll_interval
        self._debounce_delay : float = debounce_delay
        self._stop = False
        self._width = self._get_width()
        self._is_resizing = False

        self._using_sigwinch = False
        self._prev_handler = None
        self._poll_thread = None
        
        # Debouncing state
        self._debounce_timer: threading.Timer | None = None
        self._debounce_lock = threading.Lock()

        # Try to use SIGWINCH (POSIX, main thread only)
        if hasattr(signal, "SIGWINCH"):
            try:
                self._prev_handler = signal.getsignal(signal.SIGWINCH)
                signal.signal(signal.SIGWINCH, self._on_sigwinch)
                self._using_sigwinch = True
            except (ValueError, OSError):
                # Not in main thread or unsupported -> fall back to polling
                self._using_sigwinch = False

        # If SIGWINCH not available or failed, start polling
        if not self._using_sigwinch:
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._poll_thread.start()

    def close(self):
        """Stop polling and restore previous handlers (if any)."""
        self._stop = True
        
        # Cancel any pending debounce timer and trigger final done notification
        with self._debounce_lock:
            if self._debounce_timer is not None:
                self._debounce_timer.cancel()
                self._debounce_timer = None
            # If we were resizing, call the done callback one last time
            if self._is_resizing:
                self._notify_done(self._width)
                self._is_resizing = False
        
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=self._poll_interval * 2)
        if self._using_sigwinch and hasattr(signal, "SIGWINCH"):
            try:
                signal.signal(signal.SIGWINCH, self._prev_handler)
            except Exception:
                pass

    # ---------- internals ----------

    def _get_width(self) -> int:
        return get_crnt_terminal_width()

    def _notify_start(self, width: int):
        """Call the on_resize_start callback."""
        cb = self._on_resize_start_method
        if callable(cb):
            try:
                cb(width)
            except Exception:
                # swallow to keep handler lightweight
                pass

    def _notify_done(self, width: int):
        """Call the on_resize_done callback."""
        cb = self._on_resize_done_method
        if callable(cb):
            try:
                cb(width)
            except Exception:
                # swallow to keep handler lightweight
                pass

    def _handle_width_change(self, width: int):
        """Handle a detected width change."""
        # If this is the first change, mark as resizing and call start callback
        if not self._is_resizing:
            self._is_resizing = True
            self._notify_start(width)
        
        # Schedule debounced done callback
        self._schedule_debounced_done(width)

    def _schedule_debounced_done(self, width: int):
        """Schedule a done notification after debounce delay, canceling any previous timer."""
        with self._debounce_lock:
            # Cancel previous timer if it exists
            if self._debounce_timer is not None:
                self._debounce_timer.cancel()
            
            # Create new timer
            self._debounce_timer = threading.Timer(
                self._debounce_delay,
                self._execute_done_notify,
                args=(width,)
            )
            self._debounce_timer.daemon = True
            self._debounce_timer.start()

    def _execute_done_notify(self, width: int):
        """Execute the done notification after debounce delay has elapsed."""
        with self._debounce_lock:
            self._debounce_timer = None
            self._is_resizing = False
        self._notify_done(width)

    def _on_sigwinch(self, signum, frame):
        w = self._get_width()
        if w != self._width:
            self._width = w
            self._handle_width_change(w)

    def _poll_loop(self):
        while not self._stop:
            w = self._get_width()
            if w != self._width:
                self._width = w
                self._handle_width_change(w)
            time.sleep(self._poll_interval)


def get_crnt_terminal_width()->int:
    try:
        terminal_size = shutil.get_terminal_size()
        width = terminal_size.columns
    except Exception:
        width = DEFAULT_TERMINAL_WIDTH
    return width



class TerminalWidthTracker():
    def __init__(self):
        self._prev_width : int = get_crnt_terminal_width()
    
    def width_changed(self) -> bool:
        """ Check if the terminal width has changed since the last check.

        Returns:
            True if the width has changed, False otherwise.
        """

        curr_width = get_crnt_terminal_width()
        if curr_width != self._prev_width:
            self._prev_width = curr_width
            return True
        return False
