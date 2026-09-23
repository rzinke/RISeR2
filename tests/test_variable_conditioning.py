# -*- coding: utf-8 -*-
#
# Copyright (c) 2025-2026 Robert Zinke. Licensed under the MIT License.

# Import modules
import numpy as np
import pytest

from riser import (
    probability_functions as PDFs,
    variable_functions as var_fcns,
)


# Tests
class TestCondition:
    def test_known_prior_weighting(self):
        x = PDFs.value_arrays.precise_array(0.0, 1.0, 0.1)
        px = PDFs.parametric_functions.uniform(x=x, a=0.0, b=1.0)
        prior = PDFs.PDF(x=x, px=px)

        weight = 1 - x

        posterior, area = var_fcns.condition.core.condition(prior, weight)

        assert posterior.px[0] == pytest.approx(2.0)
        assert posterior.px[-1] == pytest.approx(0.0)
        assert area == pytest.approx(0.5)

    def test_different_sized_arrays_raise(self):
        x = PDFs.value_arrays.precise_array(0.0, 1.0, 0.1)
        px = PDFs.parametric_functions.uniform(x=x, a=0.0, b=1.0)
        prior = PDFs.PDF(x=x, px=px)

        weight = np.ones(1)

        with pytest.raises(ValueError, match="The size of the weighting array"):
            var_fcns.condition.core.condition(prior, weight)

    def test_metadata_consistent(self):
        x = PDFs.value_arrays.precise_array(0.0, 1.0, 0.1)
        px = PDFs.parametric_functions.uniform(x=x, a=0.0, b=1.0)

        prior = PDFs.PDF(
            x=x, px=px, name="X1", variable_type="age", unit="y"
        )

        weight = 1 - x

        posterior, _ = var_fcns.condition.core.condition(prior, weight)

        assert posterior.name is None
        assert posterior.variable_type is None
        assert posterior.unit is None


# end of file
