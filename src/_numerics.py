from projects.controlled_squeezing.globals import Globals


import numpy as np
from mpmath import mp

if Globals.PRECISE:
    mp.dps = 80
    π = mp.pi
else:
    π = np.pi



def exp(x: complex) -> complex:
    """Wrapper for exp to handle high-precision if needed."""
    if Globals.PRECISE:
        return mp.exp(x)
    else:
        return np.exp(x)
    
    
def log(x: complex) -> complex:
    """Wrapper for log to handle high-precision if needed."""
    if Globals.PRECISE:
        return mp.log(x)
    else:
        return np.log(x)
    

def sqrt(x: complex) -> complex:
    """Wrapper for sqrt to handle high-precision if needed."""
    if Globals.PRECISE:
        return mp.sqrt(x)
    else:
        return np.sqrt(x)