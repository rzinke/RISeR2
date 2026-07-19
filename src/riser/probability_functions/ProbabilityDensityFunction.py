# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

# Public API
__all__ = [
    "ProbabilityDensityFunction",
]


# Import modules
import numpy as np

from .. import precision
from .. import integration


#################### PDF BASE CLASS ####################
class ProbabilityDensityFunction:
    """Create a probability density function (PDF).
    Values are set once on instantiation and are effectively immutable.

    A probability density function (PDF) expresses the relative likelihood of
    a random variable taking a specific value. It meets the criteria:
    1. Is a continuous random variable
    2. Is non-negative for all values
    3. The integral over the entire domain (probability) is 1.0

    Probabilities of the random varible falling within a range of values is
    determined by the area under the curve between those two values. The
    probability of any single, discrete value is 0.0.

    Note that this class provides a discrete representation of a PDF, and is
    not explicitly defined everywhere between negative and positive infinity.
    It can be assumed that all probability density values outside the defined
    domain are 0.0.

    Every PDF has a corresponding Cumulative Distribution Function (CDF),
    which is the probability that the random variable will take a value less
    than or equal to value x.
    """

    metadata_items = (
            "name",
            "variable_type",
            "unit",
        )

    def __init__(
        self,
        x: np.ndarray,
        px: np.ndarray,
        normalize_area: bool = True,
        name: str | None = None,
        variable_type: str | None = None,
        unit: str | None = None,
    ):
        """Initialize a PDF.
        Automatically validate the PDF by ensuring that it meets the criteria
        of a PDF, as defined above.

        Args    x - np.ndarray, domain values of the random variable
                px - np.ndarray, probability density values
                normalize_area - bool, scale px value to so the area = 1.0

                name - str, brief descriptive identifier
                variable_type - str, sampled quantity, e.g., age, displacement
                unit - str, value unit
        """
        # Ensure domain values are numpy array
        x = np.array(x, dtype=float)

        # Check number of domain values
        nx = len(x)
        if nx < 2:
            raise ValueError(
                f"A PDF must consist of at least 2 values, "
                f"got {nx}"
            )

        # Record domain values
        self._x = x

        # Check condition 1
        self._check_monotonic_()

        # Ensure probability density values are numpy array
        px = np.array(px, dtype=float)

        # Check number of probability density values
        npx = len(px)
        if npx != nx:
            raise ValueError(
                f"The number of probability density values `px` ({npx}) "
                f"must equal the number of domain values `x` ({nx})"
            )

        # Record probability density values
        self._px = px

        # Condition 2: Check non-negative
        self._check_nonnegative_()

        # Normalize area under the curve
        if normalize_area:
            self._normalize_mass_()

        # Condition 3: Check probability mass
        self._check_unit_mass_()

        # Compute CDF
        self._Px = self._compute_cdf_()

        # Enforce immutability
        self._x.setflags(write=False)
        self._px.setflags(write=False)
        self._Px.setflags(write=False)

        # Record metadata
        self.name = name
        self.variable_type = variable_type
        self.unit = unit

    def _compute_mass_(self) -> float:
        return integration.integrate(x=self._x, px=self._px)

    def _normalize_mass_(self) -> None:
        pmass = self._compute_mass_()
        self._px /= pmass

    def _compute_cdf_(self) -> None:
        """Compute the cumulative distribution function.
        """
        # Cumulative integration
        Px = integration.integrate_cumulative(x=self._x, px=self._px)

        # Normalize final value to 1.0
        Px /= Px[-1]

        return Px

    def _check_monotonic_(self) -> None:
        """Check condition 1: Domain values increase monotonically.
        """
        diff_x = np.diff(self._x)
        if np.any(diff_x <= 0):
            raise ValueError("Domain values must strictly increase")

    def _check_nonnegative_(self) -> None:
        """Check condition 2: No negative probability density values.
        """
        if -1 in np.sign(self._px):
            raise ValueError("All probability values must be non-negative")

    def _check_unit_mass_(self) -> None:
        """Check that the area under the curve is 1.0.
        """
        pmass = self._compute_mass_()
        if np.abs(1.0 - pmass) > precision.RISER_PRECISION:
            raise ValueError(
                f"Probability mass should be 1.0, got {pmass}. "
                f"Suggest setting `normalize_area` to True."
            )


    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def px(self) -> np.ndarray:
        return self._px

    @property
    def Px(self) -> np.ndarray:
        return self._Px


    def pdf_at_value(self, x: float) -> float:
        """Compute the value of the PDF at x.
        """
        return np.interp(x, self.x, self.px, left=0.0, right=0.0)


    def cdf_at_value(self, x: float) -> float:
        """Compute the value of the CDF at x.
        """
        return np.interp(x, self.x, self.Px, left=0.0, right=1.0)


    def compute_probability_less_than(self, x: float) -> float:
        """Compute the probability that the vlaue is less than x.
        """
        return self.cdf_at_value(x)


    def compute_probability_greater_than(self, x: float) -> float:
        """Compute the probability that the value is greater than x.
        """
        return 1.0 - self.cdf_at_value(x)


    def compute_probability_between(self, x1: float, x2: float) -> float:
        """Compute the probability that the value is between x1 and x2.
        """
        return self.cdf_at_value(x2) - self.cdf_at_value(x1)


    def pit(self, y: float | np.ndarray) -> float | np.ndarray:
        """Compute probability inverse transform (PIT).
        Note: PIT interpolation requires a strictly increasing function.
        """
        return np.interp(y, self.Px, self.x)


    def __len__(self):
        """Return the length of the PDF array.
        """
        return len(self._x)


    def __str__(self):
        print_str = "PDF"

        # Add metadata to print string
        if self.name is not None:
            print_str += f": {self.name}"
        if self.variable_type is not None:
            print_str += f" - {self.variable_type}"
        if self.unit is not None:
            print_str += f" ({self.unit})"

        return print_str


# end of file