# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

# Import modules
import numpy as np
import pytest

from riser import (
    integration,
    probability_functions as PDFs,
)


# Tests
class TestWeightFunction:
    """
    Based on the definition of a WeightFunction, test:

    1. Is continuous
        `x` has two or more values
        `x` increases monotonically
        `wx` has same number of values as `x`

    2. Is non-negative

    3. Does not necessarily have unit area.
    """

    def test_construction_with_valid_inputs(self):
        x = np.array([0.0, 1.0])
        wx = np.array([1.0, 1.0])
        weights = PDFs.weight_functions.WeightFunction(x, wx)
        np.testing.assert_allclose(weights.x, x)
        np.testing.assert_allclose(weights.wx, wx)

    def test_too_few_values_raises(self):
        x = np.array([1.0])
        wx = np.array([1.0])
        with pytest.raises(
            ValueError,
            match="A weight function must consist of at least 2 values",
        ):
            PDFs.weight_functions.WeightFunction(x, wx)

    def test_non_monotonic_values_raises(self):
        x = np.array([0.0, 2.0, 1.0])
        wx = np.array([1.0, 1.0, 1.0])
        with pytest.raises(
            ValueError,
            match="Domain values must strictly increase",
        ):
            PDFs.weight_functions.WeightFunction(x, wx)

    def test_different_number_x_wx_raises(self):
        x = np.array([0.0, 1.0, 2.0])
        wx = np.array([1.0])
        with pytest.raises(
            ValueError,
            match="The number of weight values",
        ):
            PDFs.weight_functions.WeightFunction(x, wx)

    def test_negative_wx_raises(self):
        x = np.array([0.0, 1.0, 2.0])
        wx = np.array([1.0, 1.0, -0.1])
        with pytest.raises(
            ValueError,
            match="All weight values must be non-negative",
        ):
            PDFs.weight_functions.WeightFunction(x, wx)

    @pytest.mark.parametrize(
        "wx, expected_area",
        [
            (np.array([1.0, 1.0, 1.0]), 2.0),
            (np.array([0.25, 0.25, 0.25]), 0.5),
        ],
    )
    def test_non_unit_area(self, wx, expected_area):
        """
        By definition, the area under the curve of a weight function does not
        need to be 1.0, unlike a PDF.
        """
        x = np.array([0.0, 1.0, 2.0])
        weights = PDFs.weight_functions.WeightFunction(x, wx)

        assert weights.area == pytest.approx(expected_area)


class TestWeightFunctionToPdf:
    def test_pdf_unit_area(self):
        """
        Test that a weight function with non-unit area converts to a PDF with
        unit area.
        """
        x = np.array([0.0, 1.0, 2.0])
        wx = np.array([1.0, 1.0, 1.0])

        weights = PDFs.weight_functions.WeightFunction(x, wx)

        pdf = weights.normalize()
        pdf_area = integration.integrate(x=pdf.x, px=pdf.px)

        assert isinstance(pdf, PDFs.PDF)
        np.testing.assert_allclose(pdf.x, x)
        assert pdf_area == pytest.approx(1.0)

    def test_pdf_None_metadata(self):
        x = np.array([0.0, 1.0, 2.0])
        wx = np.array([1.0, 1.0, 1.0])

        weights = PDFs.weight_functions.WeightFunction(x, wx)

        pdf = weights.normalize()

        assert pdf.name is None
        assert pdf.variable_type is None
        assert pdf.unit is None

    @pytest.mark.parametrize(
        "metadata",
        [
            {"name": None, "variable_type": None, "unit": None},
            {"name": "X1", "variable_type": "age", "unit": "y"},
        ],
    )
    def test_pdf_metadata(self, metadata):
        x = np.array([0.0, 1.0, 2.0])
        wx = np.array([1.0, 1.0, 1.0])

        weights = PDFs.weight_functions.WeightFunction(x, wx)

        pdf = weights.normalize(**metadata)

        assert pdf.name == metadata["name"]
        assert pdf.variable_type == metadata["variable_type"]
        assert pdf.unit == metadata["unit"]

    def test_all_zero_weights_raise(self):
        """
        A PDF cannot be created from all-absolute zero values.
        """
        x = np.array([0.0, 1.0, 2.0])
        wx = np.array([0.0, 0.0, 0.0])

        weights = PDFs.weight_functions.WeightFunction(x, wx)

        with pytest.raises(ValueError, match="Total area is"):
            weights.normalize()


class TestWeightFunctionFromPdf:
    def test_weight_function_from_pdf(self):
        x = np.array([0.0, 1.0, 2.0])
        px = np.array([1/3, 1/3, 1/3])

        pdf = PDFs.PDF(x, px)

        weights = PDFs.weight_functions.WeightFunction.from_pdf(pdf)

        np.testing.assert_allclose(weights.x, pdf.x)
        np.testing.assert_allclose(weights.wx, pdf.px)        


class TestFlatWeight:
    def test_flat_weight(self):
        x = np.array([0.0, 1.0, 2.0])

        weights = PDFs.weight_functions.flat_weight(x)

        np.testing.assert_allclose(weights.x, x)
        np.testing.assert_allclose(weights.wx, 1.0)


class TestZeroWhere:
    def test_zero_above(self):
        x = np.linspace(-1.0, 1.0, 5)

        condition = x <= 0

        weights = PDFs.weight_functions.zero_where(x, condition)

        np.testing.assert_allclose(
            weights.wx,
            np.array([0.0, 0.0, 0.0, 1.0, 1.0]),
        )

    def test_zero_below(self):
        x = np.linspace(-1.0, 1.0, 5)

        condition = x >= 0

        weights = PDFs.weight_functions.zero_where(x, condition)

        np.testing.assert_allclose(
            weights.wx,
            np.array([1.0, 1.0, 0.0, 0.0, 0.0]),
        )


# end of file
