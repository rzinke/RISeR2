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

    def area(self) -> float:
        """Compute the area under the curve of the weight function.

        The meaning of the area is context-dependent.

        Returns
        -------
        area : float
            Area under the curve of the weight function.
        """
        return integration.integrate(x=self.x, px=self.px)

    def normalize(self, name: str, variable_type: str, unit: str) -> PDF:
        """
        """
        return


# end of file
