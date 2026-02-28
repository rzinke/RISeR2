# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

# Constants
from riser.constants import Psigma


# Import modules
import numpy as np


#################### SAMPLE STATISTICS ####################
class SampleStatistics:
    def __init__(self, 
        confidence: float,
        range_values: tuple[float],
        name: str | None=None,
        unit: str | None=None
    ):
        """Store sample statistics.

        Args    range_values - sets of confidence values
        """
        # Record confidence values
        self.confidence = confidence
        self.range_values = range_values

        # Record metadata
        self.name = name
        self.unit = unit

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
    picks: np.ndarray,
    confidence: float=Psigma["1"],
    name: str | None=None,
    unit: str | None=None,
    verbose: bool=False
) -> "SampleStatistics":
    """Compute the percent of values within a range and at the 50% percentile
    (median).

    Args    picks - np.ndarray, discrete sample picks
            confidence - float, confidence level
    Returns SampleStatistics
    """
    # Determine the lower and upper confidence levels
    half_confidence = confidence / 2
    lower = 0.5 - half_confidence
    upper = 0.5 + half_confidence

    # Determine percentiles
    range_values = np.percentile(picks, (100*lower, 100*upper))

    # Format values into SampleStatistics object
    conf_range = SampleStatistics(
        confidence=confidence,
        range_values=range_values,
        name=name,
        unit=unit
    )

    return conf_range


# end of file