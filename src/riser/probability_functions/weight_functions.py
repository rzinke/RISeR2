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
from .probability_density_function import ProbabilityDensityFunction as PDF


#################### WEIGHT FUNCTION CLASS ####################
class WeightFunction:
    """A density-like array of weight values used to update prior likelihoods.

    Unlike a PDF, the area under the. urve does not need to equal 1.0.
    """
    def __init__(
        self,
        x: np.ndarray,
        wx: np.ndarray,
    ):
        """
        Parameters
        ----------
        x : np.ndarray
            Domain values corresponding to the values of a random variable.
        wx : np.ndarray
            Weight values.

        Returns
        -------
        """
        self.x = x
        self.wx = wx

    def area(self) -> float:
        """Compute the area under the curve of the weight function.

        The meaning of the area is context-dependent.

        Returns
        -------
        area : float
            Area under the curve of the weight function.
        """
        return integration.integrate(x=self.x, px=self.wx)

    def normalize(self, **metadata) -> PDF:
        """Normalize the area of the weight function to 1.0 and format as a PDF.

        An area that is negative or infinite raises.
    
        Returns
        -------
        pdf : PDF
            PDF with same shape as the original weighting function,
            with unit area guaranteed.
        """
        # Compute area
        area = self.area()

        # Check that area is non-negative and finite.
        if not (0 < area < float("inf")):
            raise ValueError(
                f"Cannot normalize weight function to PDF. "
                f"Total area is {area}"
            )
        
        # Normalize area to 1.0
        px = self.wx / self.area()

        # Create PDF
        pdf = PDF(x=self.x, px=px, **metadata)

        return pdf


# end of file
