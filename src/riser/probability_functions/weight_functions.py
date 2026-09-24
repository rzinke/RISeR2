# -*- coding: utf-8 -*-
#
# Copyright (c) 2025, 2026 Robert Zinke. Licensed under the MIT License.


# Public API
__all__ = [
    "WeightFunction",
    "flat_weight",
    "zero_where",
]


# Import modules
import numpy as np

from .. import integration
from .probability_density_function import ProbabilityDensityFunction as PDF


#################### WEIGHT FUNCTION CLASS ####################
class WeightFunction:
    """A density-like array of weight values used to update prior likelihoods.

    Like a PDF, a weight function is continuous (monotonic) and everywhere
    non-negative.

    Unlike a PDF, the area under the curve does not necessarily equal 1.0.

    A weight function does not carry metadata.
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
        """
        # Ensure domain values are numpy array
        x = np.array(x, dtype=float)

        # Check number of domain values
        nx = len(x)
        if nx < 2:
            raise ValueError(
                f"A weight function must consist of at least 2 values, "
                f"got {nx}"
            )

        # Record domain values
        self._x = x

        # Check monotonic
        self._check_monotonic_()
        
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

        # Check non-negative
        self._check_nonnegative_()

    def _check_monotonic_(self) -> None:
        """Check condition 1: Domain values increase monotonically.
        """
        diff_x = np.diff(self._x)
        if np.any(diff_x <= 0):
            raise ValueError("Domain values must strictly increase")

    def _check_nonnegative_(self) -> None:
        """Check no negative weight values.
        """
        if -1 in np.sign(self._wx):
            raise ValueError("All weight values must be non-negative")


    @classmethod
    def from_pdf(cls, pdf: PDF):
        """Build a WeightFunction from a PDF.

        Parameters
        ----------
        pdf : PDF
            PDF to convert to a WeightFunction.

        Returns
        -------
        WeightFunction
            Weight function with domain of the input PDF, and weight values
            matching those of the input PDF.
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
    """Create a weight function with unit-value weights over the specified 
    domain.

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


def zero_where(x: np.ndarray, condition: np.ndarray) -> WeightFunction:
    """Create a weight function that is 1.0 everywhere, except 0.0 wherever
    condition is True.

    Parameters
    ----------
    x : np.ndarray
        Domain values of the random variable.
    condition : np.ndarray
        Boolean array, same length as x. Wherever True, the weight is set
        to zero.

    Returns
    -------
    WeightFunction
        All-ones weight function, zeroed where condition holds.
    """
    wx = np.ones_like(x)
    wx[condition] = 0

    return WeightFunction(x=x, wx=wx)


# end of file
