# -*- coding: utf-8 -*-

from . import bregman
from . import da
from .bregman import sinkhorn

from . import utils
from .utils import dist, to_gpu, to_np


# Author: Remi Flamary <remi.flamary@unice.fr>
#         Leo Gautheron <https://github.com/aje>
#
# License: MIT License

__all__ = ["utils", "dist", "sinkhorn"]
