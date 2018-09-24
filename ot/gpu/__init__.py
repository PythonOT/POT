# -*- coding: utf-8 -*-

from . import bregman
from . import da
from .bregman import sinkhorn

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Leo Gautheron <https://github.com/aje>
#
# License: MIT License

import warnings

warnings.warn("the ot.gpu module is deprecated because cudamat in no longer maintained", DeprecationWarning,
              stacklevel=2)

__all__ = ["bregman", "da", "sinkhorn"]
