# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

# Import modules
import numpy as np
import pytest

from riser import precision


# Tests
class TestCheckPrecision:
    @pytest.mark.parametrize("x", [-1.0, 0.5, 1000.0])
    def test_silent_if_above_threshold(self, x, recwarn):
        precision.check_precision(x)
        assert len(recwarn) == 0

    @pytest.mark.parametrize("x", [1e-15, 1e-12, 1e-10, -1e-12])
    def test_warn_if_below_threshold(self, x):
        with pytest.warns(UserWarning):
            precision.check_precision(x)


class TestFixPrecision:
    def test_scalar_type_preserved(self):
        result = precision.fix_precision(1.0)
        assert isinstance(result, float)

    def test_array_type_preserved(self):
        result = precision.fix_precision(np.array([1.0, 2.0]))
        assert isinstance(result, np.ndarray)

    def test_scalar_precision_fixed(self):
        noisy_scalar = 1.0 + 1e-13
        assert noisy_scalar != 1.0
        assert precision.fix_precision(noisy_scalar) == 1.0

    def test_array_precision_fixed(self):
        clean_array = np.array([1.0, 2.0])
        noisy_array = np.array([1.0 + 1e-13, 2.0 - 1e-13])
        assert not np.all(noisy_array == clean_array)
        np.testing.assert_allclose(
            precision.fix_precision(noisy_array), clean_array
        )


# end of file
