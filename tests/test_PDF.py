#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Import modules
import numpy as np


def test():
    """Sanity check: attempt to access the package.
    """
    # Import module
    from riser.probability_functions.ProbabilityDensityFunction import \
        ProbabilityDensityFunction as PDF

    # Instantiate object
    n = 10
    args = {
        'x': np.arange(n),
        'px': np.ones(n),
        'normalize_area': True,
    }

    return 0


if __name__ == "__main__":
    # Invoke the driver
    status = test()

    # Share the status with the shell
    raise SystemExit(status)


# end of file