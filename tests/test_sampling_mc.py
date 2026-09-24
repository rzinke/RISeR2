# -*- coding: utf-8 -*-
#
# Copyright (c) 2025-2026 Robert Zinke. Licensed under the MIT License.

# Import modules
import numpy as np
import pytest

from riser import (
    probability_functions as PDFs,
    sampling,
)


# Seed random number generator
np.random.seed(0)


# Tests
class TestSampleMonteCarlo:
    def test_all_pass(self):
        """Trivial case in which no samples are rejected.
        """
        ...


# end of file
