# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Import modules
import numpy as np
import scipy as sp


#################### PDF BASE CLASS ####################
class ProbabilityDensityFunction:
    """A probability density function (PDF) expresses probability or a
    random variable. It meets the criteria:
    1. Is a continuous random variable
    2. Is non-negative for all values
    3. The integral over the entire domain (probability) is 1.0

    Probabilities of the random varible falling within a range of values is
    determined by the area under the curve between those two values. The
    probability of any single, discrete value is 0.0.

    Note that this class provides a discrete representation of a PDF, and is
    not explicitly defined everywhere between negative and positive infinity.
    It is assumed that all probability density values outside the defined
    domain are zero.

    Every PDF has a corresponding Cumulative Distribution Function (CDF),
    which is the probability that the random variable will take a value less
    than or equal to domain value x.
    """
    def __init__(self, x:np.ndarray, px:np.ndarray, normalize_area:bool=False):
        """Initialize a PDF.
        Automatically validate the PDF by ensuring that it meets the criteria
        of a PDF, as defined above.

        Args    x - np.ndarray, domain values of the random variable
                px - np.ndarray, probability density values
                normalize_area - bool, normalize the area to 1.0
        """
        # Record domain and probability density values
        self.x = x
        self.px = px

        # Check that array sizes are equal
        self.__check_array_lengths__()

        # Normalize area under curve
        if normalize_area == True:
            self.__normalize_area__()

        # Validate
        self.validate()

        # Compute CDF
        self.__compute_cdf__()


    def __check_array_lengths__(self):
        """Check that the number of points is x is the same as the number of
        points in px.
        """
        # Number of points in arrays
        nx = len(self.x)
        npx = len(self.px)

        # Check that array lengths are equal
        if nx != npx:
            raise Exception(f"Number of points in x ({nx}) must equal the "
                            f"number of points in px ({npx})")


    def __compute_area__(self):
        """Compute the area over the defined domain.
        """
        return sp.integrate.trapezoid(self.px, self.x)


    def __normalize_area__(self):
        """Scale probability density values so that the area under the curve
        is 1.0. Operates in-place.
        """
        self.px /= self.__compute_area__()


    def validate(self) -> bool:
        """Ensure that the object meets the criteria of a PDF.
        """
        # Check monotonic
        diff_x = np.diff(self.x)
        if -1 in np.sign(diff_x):
            raise Exception("Domain values must only increase")

        # Check non-negative
        if -1 in np.sign(self.px):
            raise Exception("All probability values must be non-negative")

        # Total probability is 1.0
        area = self.__compute_area__()
        if np.abs(1.0 - area) > 1E-12:
            raise Exception(f"Area is {area}; should be 1.0. "
                            f"Suggest setting normalize_area to True.")

        return True


    def __compute_cdf__(self):
        """Compute the cumulative distribution function.
        """
        # Cumulative integration
        self.Px = sp.integrate.cumulative_trapezoid(self.px, self.x, initial=0)

        # Normalize final value to 1.0
        self.Px /= self.Px[-1]


    def compute_cdf_value(self, x:float) -> float:
        """Compute the value of the CDF at x.
        """
        return np.interp(x, self.x, self.Px, left=0.0, right=1.0)


    def compute_probability(self, x1:float, x2:float) -> float:
        """Compute the probability that the value is between x1 and x2.
        """
        return self.compute_cdf_value(x2) - self.compute_cdf_value(x1)


    def pit(self, y:float|np.ndarray) -> float|np.ndarray:
        """Compute probability inverse transform (PIT).
        Note: PIT interpolation requires a strictly increasing function.
        """
        return np.interp(y, self.Px, self.x)


    def __len__(self):
        """Return the length of the PDF array.
        """
        # Check that array sizes are equal
        self.__check_array_lengths__()

        return len(self.x)


#################### MEASUREMENT CLASSES ####################
class Age(ProbabilityDensityFunction):
    """Represent a sample age as a PDF.
    """
    sampled_quantity = "age"
    unit = "ky"

    def __init__(self, x:np.ndarray, px:np.ndarray, normalize_area:bool=False,
            name:str=None):
        """Initialize object.

        Args    name - str, sample name
        """
        # Sample name
        self.name = name

        # Pass domain and probability density values to base class
        super().__init__(x, px, normalize_area=normalize_area)


class Displacement(ProbabilityDensityFunction):
    """Represent a displacement measurement as a PDF.
    """
    sampled_quantity = "disp"
    unit = "m"

    def __init__(self, x:np.ndarray, px:np.ndarray, normalize_area:bool=False,
            name:str=None):
        """Initialize object.

        Args    name - str, measurement name
        """
        # Sample name
        self.name = name

        # Pass domain and probability density values to base class
        super().__init__(x, px, normalize_area=normalize_area)


class SlipRate(ProbabilityDensityFunction):
    """Represent a slip rate measurement as a PDF.
    """
    sampled_quantity = "slip rate"
    unit = "m/ky"

    def __init__(self, x:np.ndarray, px:np.ndarray, normalize_area:bool=False,
            name:str=None):
        """Initialize object.

        Args    name - str, measurement name
        """
        # Sample name
        self.name = name

        # Pass domain and probability density values to base class
        super().__init__(x, px, normalize_area=normalize_area)


# end of file