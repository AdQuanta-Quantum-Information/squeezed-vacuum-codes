from typing import Callable, Any, ParamSpec, TypeVar, List, Tuple, Final, cast
from functools import wraps
from . import errors



# ============================================================================ #
#|                             Helper Types                                   |#
# ============================================================================ #
_In = ParamSpec("_In")
_Out = TypeVar("_Out")
P = ParamSpec('P')
R = TypeVar('R')

def when_fails_do(func_secondary:Callable[_In, _Out])->Callable[[Callable[_In, _Out]],Callable[_In, _Out]]:
    from .prints import print_warning
    def decorator(func_primary:Callable[_In, _Out])->Callable[_In, _Out]:
        def wrapper(*args:_In.args, **kwargs:_In.kwargs)->_Out:
            try:
                results = func_primary(*args, **kwargs)
            except Exception as e:
                print_warning(
                    f"Function {func_primary.__name__!r} failed because of error:"+
                    f"\n{errors.get_traceback(e)}"+
                    f"\nRunning {func_secondary.__name__!r} in its stead."
                )
                results = func_secondary(*args, **kwargs)
            return results
        return wrapper
    return decorator



def ignore_first_method_call(func:Callable)->Callable: # decorator that returns a wrapper:
    objects_that_already_called : List[Tuple[object, Callable]] = []

    def wrapper(self, *args, **kwargs)->Any: # wrapeer that cals the function            
        nonlocal objects_that_already_called
        if (self, func) in objects_that_already_called:
            results = func(self, *args, **kwargs)
        else:
            objects_that_already_called.append((self, func))
            results = None
        return results
    return wrapper



def limited_number_of_runs(num:int)->Callable[[Callable[_In, None]], Callable[_In, None]]: # function that returns a decorator
    """ 
    This decorator is used for limiting the number of runs of a function.
    Note that this function must return `None`.
    """
    counter : int = 0
    # Return decorator:
    def decorator(func:Callable[_In, None])->Callable[_In, None]: # decorator that returns a wrapper:
        # Return a wrapper to func:
        @wraps(func)
        def wrapper(*args:_In.args, **kwargs:_In.kwargs)->None:  # wrapeer that cals the function            
            nonlocal counter
            if counter >= num:
                return None            
            counter += 1
            return func(*args, **kwargs)
        return wrapper
    return decorator


def run_only_once(func:Callable)->Callable:
    return limited_number_of_runs(1)(func)


def multiple_tries(num:int)->Callable[[Callable[_In, _Out]], Callable[_In, _Out]]: # function that returns a decorator
    # Return decorator:
    def decorator(func:Callable[_In, _Out])->Callable[_In, _Out]: # decorator that returns a wrapper:
        # Return a wrapper to func:
        @wraps(func)
        def wrapper(*args:_In.args, **kwargs:_In.kwargs)->_Out: # wrapeer that cals the function            
            last_error = Exception("Temp Exception")
            for i in range(num):
                try:
                    results = func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                else:
                    return results
            raise last_error
        return wrapper
    return decorator


def list_tries(list_:list)->Callable[[Callable], Callable]: # function that returns a decorator
    # Return decorator:
    def decorator(func:Callable)->Callable: # decorator that returns a wrapper:
        @wraps(func)
        def wrapper(*args, **kwargs)->Any: # wrapper that calls the function            
            last_error = Exception("Temp Exception")
            for val in list_:
                try:
                    results = func(val)
                    return results
                except Exception as e:
                    last_error = e
            raise last_error
        return wrapper
    return decorator


def only_if_true_field(field_name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """ Wraps a method so that it only runs if the field `field_name` is `True`. """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> Any:
            field_value = getattr(self, field_name)
            if field_value:
                return func(self, *args, **kwargs)
            else:
                return None
        return wrapper
    return decorator




def sparse_execution(skip_num:int, default_results:R) -> Callable[[Callable[P, R]], Callable[P, R]]:
    assert isinstance(skip_num, int)
    assert skip_num > 0

    def decorator(func:Callable[P, R]) -> Callable[P, R]:
        counter : int = 0

        @wraps(func)
        def wrapper(*args:P.args, **kwargs:P.kwargs) -> R:
            nonlocal counter
            
            if counter >= skip_num:
                results = func(*args, **kwargs)
                counter = 0
            else:
                results = default_results
                counter += 1
            
            return results
        return wrapper
    return decorator




class _MethodsCaches:
    """ This class is used to store the caches of methods. """
    _property_name : Final[str] = "_methods_caches"

    def __init__(self) -> None:
        self.caches : dict[str, dict] = {}

    def get_cache(self, method_name:str) -> dict:
        if method_name not in self.caches:
            self.caches[method_name] = {}
        return self.caches[method_name]
    


def method_cache(method:Callable[P, R]) -> Callable[P, R]:
    """ This creates a new container in an object that caches the results of a methods.
    If the method is called again with the same arguments, the cached results are returned.
    """
    @wraps(method)
    def wrapper(self, *args:P.args, **kwargs:P.kwargs) -> R:       

        key, this_method_cache = _get_cache(self, method, *args, **kwargs)
        try:
            results = this_method_cache[key]
        except KeyError:
            results = method(self, *args, **kwargs)
            this_method_cache[key] = results
        return results
    
    def _get_cache(obj:Any, method:Callable[P, R], *args:P.args, **kwargs:P.kwargs) -> Tuple[tuple, dict]:
        ## Get all methods caches:
        methods_caches : _MethodsCaches|None = getattr(obj, _MethodsCaches._property_name, None)
        if methods_caches is None:
            setattr(obj, _MethodsCaches._property_name, _MethodsCaches())            
            methods_caches = getattr(obj, _MethodsCaches._property_name)
        methods_caches = cast(_MethodsCaches, methods_caches)  # just for type checking

        ## Get cache and key for this specific method and args
        this_method_cache : dict = methods_caches.get_cache(method.__name__)
        key : tuple = (args, tuple(kwargs.items()))

        return key, this_method_cache
    
    return wrapper