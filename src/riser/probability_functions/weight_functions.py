# -*- coding: utf-8 -*-
#
# Copyright (c) 2025, 2026 Robert Zinke. Licensed under the MIT License.


# Public API
__all__ = [
    "WeightFunction",
    "flat_weight",
    "pass_above",
    "pass_below",
]


# Import modules
import numpy as np

from .. import integration
from .probability_density_function import ProbabilityDensityFunction as PDF


#################### WEIGHT FUNCTION CLASS ####################
class WeightFunction:
    """A density-like array of weight values used to update prior likelihoods.

    Unlike a PDF, the area under the curve does not need to equal 1.0.
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
        # Ensure domain values are numpy array
        x = np.array(x, dtype=float)

        # Check number of domain values
        nx = len(x)
        if nx < 2:
            raise ValueError(
                f"A weighting function must consist of at least 2 values, got {nx}"
            )

        # Record domain values
        self._x = x
        
        # Ensure weight values are numpy array
        wx = np.array(wx, dtype=float)

        # Check number of weight values
        nwx = len(wx)
        if nwx != nx:
            raise ValueError(
                f"The number of weight values `wx` ({nwx}) "
                f"must equal the number of domain values `x` ({nx})"
            )

        # Record probability density values
        self._wx = wx

    @classmethod
    def from_pdf(cls, pdf: PDF):
        """Build a WeightFunction from a PDF.
        """
        return cls(x=pdf.x, wx=pdf.px)

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def wx(self) -> np.ndarray:
        return self._wx

    def __len__(self) -> int:
        return len(self.x)

    @property
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
        area = self.area

        # Check that area is non-negative and finite.
        if not (0 < area < float("inf")):
            raise ValueError(
                f"Cannot normalize weight function to PDF. "
                f"Total area is {area}"
            )

        # Create PDF - guarantees unit area
        pdf = PDF(x=self.x, px=self.wx, **metadata)

        return pdf


#################### WEIGHTING FUNCTIONS ####################
def flat_weight(x: np.ndarray) -> WeightFunction:
    """Create a weight function with unit-value weights over the specified domain.

    Parameters
    ----------
    x : np.ndarray
        Domain values of the random variable.

    Returns
    -------
    WeightFunction
        All-ones weight function.
    """
    return WeightFunction(x=x, wx=np.ones_like(x))


# end of file
