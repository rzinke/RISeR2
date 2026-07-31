# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

# Public API
__all__ = [
    "compute_sample_confidence",
]


# Import modules
from dataclasses import dataclass

import numpy as np

from .. import constants


#################### SAMPLE STATISTICS ####################
@dataclass
class SampleStatistics:
    """Class to store discrete sample statistics.
    """

    # Confidence values
    confidence: float
    range_values: tuple[float, ...]

    # PDF metadata
    name: str
    variable_type: str
    unit: str

    def __str__(self):
        print_str = "Sample statistics:"
        if self.name is not None:
            print_str += f" {self.name}"

        if self.unit is not None:
            print_str += f" ({self.unit})"

        print_str += f"\n{100 * self.confidence:.2f} % : "
        print_str += (
            f"({self.range_values[0]:.3f} "
            f"- {self.range_values[1]:.3f})"
        )

        return print_str


def compute_sample_confidence(
    samples: np.ndarray,
    confidence: float = constants.Psigma["1"],
    *,
    name: str | None = None,
    variable_type: str | None = None,
    unit: str | None = None,
    verbose: bool = False,
) -> SampleStatistics:
    """Compute the percent of values within a range and at the 50% percentile
    (median).

    Parameters
    ----------
    samples : np.ndarray
        Discrete samples on which to compute statistics.
    confidence : float
        Confidence level.
    name : str
        Brief descriptive identifier.
    variable_type : str
        Sampled quantity, e.g., age, displacement, slip rate.
    unit : str
        Value unit.

    Returns
    -------
    SampleStatistics
        Statistics of the given sample set.
    """
    # Determine the lower and upper confidence levels
    half_confidence = confidence / 2
    lower = 0.5 - half_confidence
    upper = 0.5 + half_confidence

    # Determine percentiles
    range_values = np.percentile(samples, (100*lower, 100*upper))

    # Format values into SampleStatistics object
    conf_range = SampleStatistics(
        confidence=confidence,
        range_values=range_values,
        name=name,
        variable_type=variable_type,
        unit=unit,
    )

    return conf_range


# end of file
