"""Specific operations for lists that are sorted
"""

from typing import TypeAlias, TypeVar, List, Callable
_T = TypeVar("_T")

from .main import (
    iterate_with_edge_indicators, 
    iterate_with_periodic_prev_next_items, 
    iterate_with_prev_next_items_with_edge_indicators
)


SortedList : TypeAlias = List

def unique_values(l:SortedList) -> bool:
    for _, crnt, next in iterate_with_periodic_prev_next_items(l):
        if crnt == next:
            return False
    return True


def add_item_to_sorted_list(l:SortedList[_T], item:_T, *, key:Callable[[_T], int|float]|None=None) -> None:
    ## Helper function to get the value of an item:
    def get_value(item_:_T)->int|float:
        return key(item_) if key is not None else item_
    target_val = get_value(item)
    
    ## Check if the list is empty:
    if len(l) == 0:
        l.append(item)
        return
    
    ## Check if should be first value:
    if target_val <= get_value(l[0]):
        l.insert(0, item)
        return

    ## Go value by value and find the appropriate place to insert the item:
    for i, (_, is_last, _, crnt, next) in enumerate(iterate_with_prev_next_items_with_edge_indicators(l)):
        crnt_val = get_value(crnt)

        if is_last:
            break
        else:
            assert next is not None

        next_val = get_value(next)            
        if crnt_val <= target_val <= next_val:
            l.insert(i+1, item)
            return
        
    ## If the item is larger than all values in the list, append it:    
    l.append(item)