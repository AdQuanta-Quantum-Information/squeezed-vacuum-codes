import numpy as np 


def recommended_N_from_nbar(nbar: float, safety_sigma: float = 4.0):
    """
    Heuristic Fock cutoff N based on nbar for highly squeezed / GKP states.
    
    Unlike coherent states (where spread ~ sqrt(nbar)), squeezed states 
    have super-Poissonian tails where the spread scales linearly with nbar.
    """
    if nbar <= 0:
        raise ValueError("nbar must be > 0.")
    
    # 1. Calculate the standard deviation for a squeezed vacuum
    # Formula: sigma_n = sqrt( 2 * nbar * (nbar + 1) )
    sigma_n = np.sqrt(2.0 * nbar * (nbar + 1.0))
    
    # 2. Add the mean, a safety multiplier for the tails, and a flat buffer for very low nbar
    N_cutoff = nbar + (safety_sigma * sigma_n) + 15
    
    return int(np.ceil(N_cutoff))