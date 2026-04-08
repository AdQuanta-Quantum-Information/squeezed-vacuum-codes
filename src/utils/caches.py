from typing import Any, Callable, ParamSpec, TypeVar
P = ParamSpec("P")
T = TypeVar("T")


import functools
import joblib
from pathlib import Path

this_folder = Path(__file__).parent
root_folder = this_folder.parents[1]

# Configure joblib with better error handling for Windows
memory = joblib.Memory(
    location=root_folder / "joblib_cache", 
    verbose=0,
    backend='local'     # Use local backend (more reliable on Windows)
)


def clear_memory_cache():
    memory.clear()


def clear_disk_cache_entry(cached_func: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
    """
    Remove one cached entry for a joblib-cached function call.

    This helper is for functions decorated with this module's disk cache wrapper,
    which returns a joblib MemorizedFunc. It computes the exact cache key from
    the provided positional and keyword arguments and deletes only that entry.

    Args:
        cached_func: A joblib-cached function (for example from cache(disk=True)).
        *args: Positional arguments of the cached call to delete.
        **kwargs: Keyword arguments of the cached call to delete.

    Returns:
        True if an entry existed and was removed, False if no matching entry exists.

    Raises:
        TypeError: If cached_func is not a joblib-cached function.
    """

    required_attrs = ("_get_args_id", "func_id", "store_backend")
    if not all(hasattr(cached_func, attr) for attr in required_attrs):
        raise TypeError("cached_func must be a joblib-cached function (MemorizedFunc).")

    args_id = cached_func._get_args_id(*args, **kwargs)  # type: ignore[attr-defined]
    call_id = (cached_func.func_id, args_id)             # type: ignore[attr-defined]
    store_backend = cached_func.store_backend            # type: ignore[attr-defined]

    if store_backend.contains_item(call_id):
        store_backend.clear_item(call_id)
        return True

    return False


def _normalize_module_name(func: Callable) -> str | None:
    """
    Convert __main__ module names to their actual module paths.
    Returns the normalized module name, or None if normalization fails.
    """
    original_module = func.__module__
    
    if original_module != '__main__':
        return None  # No normalization needed
    
    # Try to extract actual module path from __file__
    if not (hasattr(func, '__globals__') and '__file__' in func.__globals__):
        return None
    
    file_path = Path(func.__globals__['__file__'])
    
    try:
        rel_path = file_path.relative_to(root_folder)
        # Convert file path to module notation
        module_path = str(rel_path.with_suffix('')).replace('\\', '.').replace('/', '.')
        return module_path
    except ValueError:
        # File is not under root_folder
        return None


def disk_cache(func: Callable[P, T]) -> Callable[P, T]:
    original_module = func.__module__
    normalized_module = _normalize_module_name(func)
    
    # Temporarily set normalized module name if available
    if normalized_module:
        func.__module__ = normalized_module
    
    cached_func : Callable[P, T] = memory.cache(func)   #type: ignore
    
    # Restore original module name for introspection
    func.__module__ = original_module
    
    return cached_func


def ram_cache(func: Callable[P, T]) -> Callable[P, T]:
    return functools.cache(func)  # type: ignore


def cache(ram: bool = False, disk: bool = True) -> Callable[[Callable[P, T]], Callable[P, T]]:

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        if disk:
            func = disk_cache(func)  
            
        if ram:
            func = ram_cache(func)  

        return func
    
    return decorator
