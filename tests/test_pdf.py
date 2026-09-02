# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

# Import modules
import numpy as np
import pytest

from riser.probability_functions import PDF


# Tests
class TestProbabilityDensityFunction:
    """
    Based on the definition of a PDF, test:

    1. Is a continuous random variable
        x has two or more values
        x increases monotonically
        px has same number of values as x

    2. Is non-negative

    3. Has integrated area of 1.0
    """

    def test_construction_with_minimum_valid_inputs(self):
        x = np.array([0.0, 1.0, 2.0])
        px = np.array([0.0, 1.0, 0.0])
        pdf = PDF(x, px)
        np.testing.assert_allclose(pdf.x, x)
        np.testing.assert_allclose(pdf.px, px)

    def test_too_few_values_raises(self):
        x = np.array([1.0])
        px = np.array([1.0])
        with pytest.raises(
            ValueError,
            match="A PDF must consist of at least 2 values",
        ):
            PDF(x, px)

    def test_non_monotonic_values_raises(self):
        x = np.array([0.0, 2.0, 1.0])
        px = np.array([0.0, 0.0, 1.0])
        with pytest.raises(
            ValueError,
            match="Domain values must strictly increase",
        ):
            PDF(x, px)

    def test_different_number_x_px_raises(self):
        x = np.array([0.0, 1.0, 2.0])
        px = np.array([1.0])
        with pytest.raises(
            ValueError,
            match="The number of probability density values",
        ):
            PDF(x, px)

    def test_negative_px_raises(self):
        x = np.array([0.0, 1.0, 2.0])
        px = np.array([0.0, 1.0, -0.1])
        with pytest.raises(
            ValueError,
            match="All probability values must be non-negative",
        ):
            PDF(x, px)

    def test_area_zero_raises(self):
        x = np.array([0.0, 1.0, 2.0])
        px = np.array([0.0, 0.0, 0.0])
        with pytest.raises(
            ValueError,
            match="Total probability is too close to 0.0"
        ):
            PDF(x, px)

    def test_unit_area(self):
        x = np.array([0.0, 1.0, 2.0])
        px = np.array([0.0, 2.0, 0.0])
        pdf = PDF(x, px)
        area = np.trapezoid(pdf.px, pdf.x)
        assert area == pytest.approx(1.0)


x = np.array([0.0, 1.0, 2.0])
px = np.array([0.0, 1.0, 0.0])


class TestMetadataProperties:

    x = np.array([0.0, 1.0, 2.0])
    px = np.array([0.0, 1.0, 0.0])

    def test_metadata_properties_return_set_values(self):
        pdf = PDF(
            self.x,
            self.px, name="x",
            variable_type="age",
            unit="y",
        )
        assert pdf.name == "x"
        assert pdf.variable_type == "age"
        assert pdf.unit == "y"

    def test_metadata_properties_default_to_none(self):
        pdf = PDF(
            self.x,
            self.px,
        )
        assert pdf.name is None
        assert pdf.variable_type is None
        assert pdf.unit is None


class TestStr:
    def test_str_does_not_crash_and_returns_a_string(self):
        x = np.array([0.0, 1.0, 2.0])
        px = np.array([0.0, 1.0, 0.0])
        pdf = PDF(x, px, name="x", variable_type="age", unit="y")
        assert isinstance(str(pdf), str)


class TestCDFfunctions:

    x = np.array([0.0, 0.5, 1.0])
    px = np.array([1.0, 1.0, 1.0])
    pdf = PDF(x, px)

    def test_len(self):
        assert len(self.pdf) == 3

    def test_probability_less_than(self):
        P = self.pdf.compute_probability_less_than(0.25)
        assert P == pytest.approx(0.25)

    def test_probability_greater_than(self):
        P = self.pdf.compute_probability_greater_than(0.25)
        assert P == pytest.approx(0.75)

    def test_compute_probability_between(self):
        P = self.pdf.compute_probability_between(0.4, 0.6)
        assert P == pytest.approx(0.2)

    def test_pit(self):
        value = self.pdf.pit(0.25)
        assert value == pytest.approx(0.25)

    def test_pit_array_input_preserves_type(self):
        y = np.array([0.25, 0.5, 0.75])
        result = self.pdf.pit(y)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [0.25, 0.5, 0.75])


# end of file
