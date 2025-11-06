
## This section is designed to suppress the `SparseEfficiencyWarning` 
# that is raised by qutip for large operators on many degrees (for example for a large fock basis) 
# using the warning defined in the `scipy.sparse` module.
import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter("ignore", SparseEfficiencyWarning)   # for a user-friendly experience, should be either "once" or "ignore"


import qutip