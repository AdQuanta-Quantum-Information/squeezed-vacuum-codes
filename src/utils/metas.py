from typing import Literal


class _SingletonMetaStrictOrConventional(type):
    _instances = {}
    _policy : Literal["strict", "conventional"] = "conventional"

    def __call__(cls, *args, **kwargs):
        if cls in cls._instances:
            if cls._policy == "strict":
                raise Exception(f"Class {cls.__name__!r} is a strict-singleton and cannot be instantiated more than once.")
            elif cls._policy == "conventional":
                new_class = cls._instances[cls]
            else:
                raise ValueError(f"Unexpected policy {cls._policy!r}")
            
        else:
            new_class = super().__call__(*args, **kwargs)
            cls._instances[cls] = new_class
        return new_class
    

class SingletonMeta(_SingletonMetaStrictOrConventional):
    """
    Metaclass for a configurable per-class singleton.

    Policies:
    - "conventional" (default): cache and return the first instance.
    - "strict": raise Exception on any subsequent instantiation.

    Instances are stored per concrete class in `_instances`. Not thread-/process-safe.
    """
    def __new__(cls, *args, **kwargs):
        if "strict" in kwargs:
            strict = kwargs.pop("strict")
            assert isinstance(strict, bool), f"Expected bool, got {strict!r}"
            if strict:
                cls._policy = "strict"
            else:
                cls._policy = "conventional"
            assert not kwargs, f"Unexpected kwargs {kwargs!r}"
            return cls
        return super().__new__(cls, *args, **kwargs)