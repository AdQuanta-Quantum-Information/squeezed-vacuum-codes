from typing import Callable, ParamSpec, TypeVar
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
