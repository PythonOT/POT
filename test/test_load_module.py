

import ot
import doctest

# test lp solver
doctest.testmod(ot.lp,verbose=True)

# test bregman solver
doctest.testmod(ot.bregman,verbose=True)
