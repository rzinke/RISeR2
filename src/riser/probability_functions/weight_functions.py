# -*- coding: utf-8 -*-
#
# Copyright (c) 2025, 2026 Robert Zinke. Licensed under the MIT License.


# Public API
__all__ = [
    "WeightFunction",
]


# Import modules
import numpy as np

from .. import integration


#################### WEIGHT FUNCTION CLASS ####################
class WeightFunction:
    def __init__(
        self,
        x: np.ndarray,
        px: np.ndarray,
    ):
        """
        Parameters
        ----------

        Returns
        -------
        """
        self.x = x
        self.px = px

    def area(self):
        return integration.integrate(x=self.x, px=self.px)


# end of file
