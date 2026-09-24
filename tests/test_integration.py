# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

# Import modules
import numpy as np
import pytest

from riser import integration


# Tests
class TestIntegrate:
    def test_boxcar_integration(self):
        x = np.linspace(0, 1, 10)
        px = np.ones(10)
        integral = integration.integrate(x=x, px=px)
        assert isinstance(integral, float)
        assert integral == pytest.approx(1.0)

    def test_triangular_integration(self):
        x = np.array([0.0, 1.0, 2.0])
        px = np.array([0.0, 1.0, 0.0])
        integral = integration.integrate(x=x, px=px)
        assert isinstance(integral, float)
        assert integral == pytest.approx(1.0)


class TestIntegrateCumulative:
    def test_boxcar_cumulative_integration(self):
        x = np.linspace(0, 1, 10)
        px = np.ones(10)
        cum_integral = integration.integrate_cumulative(x=x, px=px)
        assert isinstance(cum_integral, np.ndarray)
        assert cum_integral[0] == 0.0
        assert cum_integral[-1] == 1.0

    def test_triangular_cumulative_integration(self):
        x = np.array([0.0, 1.0, 2.0])
        px = np.array([0.0, 1.0, 0.0])
        cum_integral = integration.integrate_cumulative(x=x, px=px)
        assert isinstance(cum_integral, np.ndarray)
        assert cum_integral[0] == 0.0
        assert cum_integral[1] == 0.5
        assert cum_integral[2] == 1.0


# end of file
