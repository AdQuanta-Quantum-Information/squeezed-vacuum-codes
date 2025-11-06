from typing import TypeVar, Generic, Optional, Union
import pickle

K = TypeVar('K')  # Type variable for keys
V = TypeVar('V')  # Type variable for values

class TwoWayDict(dict, Generic[K, V]):
    """ 
    A dictionary that allows bidirectional lookup between keys and values.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reverse: dict[V, K] = {}
        for key, value in self.items():
            self._reverse[value] = key

    def __setitem__(self, key: K, value: V) -> None:
        if key in self:
            # Remove the old reverse mapping
            raise KeyError(f"Key {key} already exists in the dictionary")
            old_value = self[key]
            del self._reverse[old_value]
        if value in self._reverse:
            # Remove the old forward mapping
            raise ValueError(f"Value {value} already exists in the dictionary")
            old_key = self._reverse[value]
            del self[old_key]


        super().__setitem__(key, value)
        self._reverse[value] = key

    def __delitem__(self, key: K) -> None:
        value = self[key]
        super().__delitem__(key)
        del self._reverse[value]

    def get_value(self, key: K) -> V:
        return self[key]

    def get_key(self, value: V) -> K:
        return self._reverse[value]


def is_empty(d: dict) -> bool:
    """
    Check if a dictionary is empty
    """
    return len(d) == 0


def _is_object_serializable(obj) -> bool:
    try:
        pickle.dumps(obj)
    except Exception as e:
        return False
    return True


def deep_to_dict(obj, enforce_serializables:bool=False) -> dict:
    """
    For a given object, return a dict with the same keys and values.
    For every value that can be converted to a dict, convert it to a dict.
    """
    ## Unpack the initial dict:
    if isinstance(obj, dict):
        d = obj
    else:
        if not hasattr(obj, '__dict__'):
            raise ValueError(f"Object {obj} has no __dict__ attribute")
        d = obj.__dict__
    
    ## Keep track of the ids of the objects we have seen in order to avoid infinite recursion:
    ids = set()
    ids.add(id(d))

    options = dict(
        enforce_serializables=enforce_serializables
    )

    ## Call the recursive function:
    res = _deep_to_dict_recursive_call(d, ids, **options)
    return res


def _deep_to_dict_recursive_call(d:dict[K, V], ids:set[int], **options) -> dict[K, V|dict|str]:

    def _get_single_object(value:V) -> V|str:         
        if options['enforce_serializables']:
            if not _is_object_serializable(value):
                return f"Object of type {type(value).__name__!r} @[{id(value)}] that is not serializable:\n"+repr(value)
        return value
            
    def _what_to_keep(value:V) -> V|dict|str:

        if not hasattr(value, '__dict__'):
            return _get_single_object(value)
            
        val_d = value.__dict__
        if is_empty(val_d):
            return _get_single_object(value)
        
        ## Before the recursive call, try to avoid infinite recursion by checking if we have seen this id before
        if id(val_d) in ids:
            return f"Object of type {type(value).__name__!r} @[{id(val_d)}] that already exists in the dictionary:\n"+repr(value)

        ids.add(id(val_d))
        val_d = _deep_to_dict_recursive_call(val_d, ids, **options)
        return val_d


    res_dict = dict()
    for key, value in d.items():
        to_keep = _what_to_keep(value)
        res_dict[key] = to_keep

    return res_dict

    