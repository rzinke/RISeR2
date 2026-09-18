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

    def normalize(
        self,
        name: str | None = None,
        variable_type: str | None = None,
        unit: str | None = None,
    ) -> PDF:
        """Normalize the area of the weight function to 1.0 and format as a PDF.
    
        Returns
        -------
        pdf : PDF
            PDF with same shape as the original weighting function,
            with unit area guaranteed.
        """
        # Normalize area to 1.0
        px = self.px / self.area()

        # Create PDF
        pdf = PDF(
            x=self.x,
            px=px,
            name=name,
            variable_type=variable_type,
            unit=unit,
        )

        return pdf



# end of file
