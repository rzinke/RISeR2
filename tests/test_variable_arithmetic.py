# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

# Import modules
import numpy as np
import scipy as sp
import pytest

from riser import (
    probability_functions as PDFs,
    variable_operations as var_ops,
)


# Tests
class TestConvolveInputSide:
    def test_matches_numpy_reference(self):
        x = np.array([1.0, 2.0, 3.0])
        h = np.array([0.5, 0.5])
        result = var_ops.arithmetic.convolve_input_side(x, h)
        expected = np.convolve(x, h, mode="full")
        np.testing.assert_allclose(result, expected)

class TestConvolveOutputSide:
    def test_matches_numpy_reference(self):
        x = np.array([1.0, 2.0, 3.0])
        h = np.array([0.5, 0.5])
        result = var_ops.arithmetic.convolve_output_side(x, h)
        expected = np.convolve(x, h, mode="full")
        np.testing.assert_allclose(result, expected)

    def test_matches_input_side_formulation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        h = np.array([0.5, 0.25, 0.25])
        result_output = var_ops.arithmetic.convolve_output_side(x, h)
        result_input = var_ops.arithmetic.convolve_input_side(x, h)
        np.testing.assert_allclose(result_output, result_input)


class TestNegateVariable:
    def test_negation_reflects_shape_correctly(self):
        x = PDFs.value_arrays.precise_array(0.0, 5.0, 0.01)
        px = PDFs.parametric_functions.triangular(x, a=0.0, c=1.0, b=5.0)
        pdf = PDFs.PDF(x, px)

        neg_pdf = var_ops.arithmetic.negate_variable(pdf)

        assert PDFs.analytics.pdf_mean(neg_pdf) == pytest.approx(-2.0)
        assert PDFs.analytics.pdf_mode(neg_pdf) == pytest.approx(-1.0)
        assert neg_pdf.x.min() == pytest.approx(-5.0)
        assert neg_pdf.x.max() == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "original_name, expected_name",
        [
            (None, None),
            ("X", "(negative) X"),
        ],
    )
    def test_name_handling(self, original_name, expected_name):
        x = PDFs.value_arrays.precise_array(0.0, 5.0, 0.5)
        px = PDFs.parametric_functions.triangular(x, a=0.0, c=1.0, b=5.0)
        pdf = PDFs.PDF(
            x=x,
            px=px,
            name=original_name,
            variable_type="age",
            unit="y",
        )

        neg_pdf = var_ops.arithmetic.negate_variable(pdf)

        assert neg_pdf.name == expected_name
        assert neg_pdf.variable_type == "age"
        assert neg_pdf.unit == "y"


class TestAddVariables:
    def test_gaussian_sum_closed_form(self):
        x = PDFs.value_arrays.precise_array(-20.0, 20.0, 0.01)

        sigma1 = 1.5
        px1 = PDFs.parametric_functions.gaussian(x, mu=2.0, sigma=sigma1)
        pdf1 = PDFs.PDF(x=x, px=px1)

        sigma2 = 2.0
        px2 = PDFs.parametric_functions.gaussian(x, mu=-1.0, sigma=sigma2)
        pdf2 = PDFs.PDF(x=x, px=px2)

        pdf_sum = var_ops.arithmetic.add_variables(pdf1, pdf2)

        assert PDFs.analytics.pdf_mean(pdf_sum) == pytest.approx(1.0)
        assert PDFs.analytics.pdf_variance(pdf_sum) == pytest.approx(
            sigma1**2 + sigma2**2
        )

    def test_rejects_mismatched_sampling(self):
        dx = 0.01

        x1 = PDFs.value_arrays.precise_array(-4.0, 8.0, dx)
        px1 = PDFs.parametric_functions.gaussian(x1, mu=2.0, sigma=1.5)
        pdf1 = PDFs.PDF(x=x1, px=px1)

        x2 = PDFs.value_arrays.precise_array(-9.0, 7.0, dx)
        px2 = PDFs.parametric_functions.gaussian(x2, mu=-1.0, sigma=2.0)
        pdf2 = PDFs.PDF(x=x2, px=px2)

        with pytest.raises(
            ValueError, match="Not all PDFs are sampled over same values"
        ):
            var_ops.arithmetic.add_variables(pdf1, pdf2)

    @pytest.mark.parametrize(
        "vartype1, unit1, vartype2, unit2",
        [
            ("age", "y", "age", "m"),
            ("age", "y", "displacement", "y"),
        ],
    )
    def test_different_metadata_warn(self, vartype1, unit1, vartype2, unit2):
        x = PDFs.value_arrays.precise_array(-20.0, 20.0, 0.01)

        px1 = PDFs.parametric_functions.gaussian(x, mu=2.0, sigma=1.5)
        pdf1 = PDFs.PDF(x=x, px=px1, variable_type=vartype1, unit=unit1)

        px2 = PDFs.parametric_functions.gaussian(x, mu=-1.0, sigma=2.0)
        pdf2 = PDFs.PDF(x=x, px=px2, variable_type=vartype2, unit=unit2)

        with pytest.warns(UserWarning, match=""):
            var_ops.arithmetic.add_variables(pdf1, pdf2)

    def test_name(self, recwarn):
        x = PDFs.value_arrays.precise_array(-20.0, 20.0, 0.01)
        vartype = "age"
        unit = "y"

        px1 = PDFs.parametric_functions.gaussian(x, mu=2.0, sigma=1.5)
        pdf1 = PDFs.PDF(
            x=x,
            px=px1,
            name="X1",
            variable_type=vartype,
            unit=unit,
        )

        px2 = PDFs.parametric_functions.gaussian(x, mu=-1.0, sigma=2.0)
        pdf2 = PDFs.PDF(
            x=x,
            px=px2,
            name="X2",
            variable_type=vartype,
            unit=unit,
        )

        pdf_sum = var_ops.arithmetic.add_variables(pdf1, pdf2, name="X12")

        assert len(recwarn) == 0
        assert pdf_sum.name == "X12"


class TestSubtractVariables:
    def test_gaussian_sum_closed_form(self):
        x = PDFs.value_arrays.precise_array(-20.0, 20.0, 0.01)

        sigma1 = 1.5
        px1 = PDFs.parametric_functions.gaussian(x, mu=2.0, sigma=sigma1)
        pdf1 = PDFs.PDF(x=x, px=px1)

        sigma2 = 2.0
        px2 = PDFs.parametric_functions.gaussian(x, mu=-1.0, sigma=sigma2)
        pdf2 = PDFs.PDF(x=x, px=px2)

        pdf_diff = var_ops.arithmetic.subtract_variables(pdf1, pdf2)

        assert PDFs.analytics.pdf_mean(pdf_diff) == pytest.approx(3.0)
        assert PDFs.analytics.pdf_variance(pdf_diff) == pytest.approx(
            sigma1**2 + sigma2**2
        )

    def test_limit_positive(self):
        dx = 0.001
        x = PDFs.value_arrays.precise_array(-20.0, 20.0, dx)

        mu1 = 2.0
        sigma1 = 1.5
        px1 = PDFs.parametric_functions.gaussian(x, mu=2.0, sigma=sigma1)
        pdf1 = PDFs.PDF(x=x, px=px1)

        mu2 = 1.0
        sigma2 = 2.0
        px2 = PDFs.parametric_functions.gaussian(x, mu=1.0, sigma=sigma2)
        pdf2 = PDFs.PDF(x=x, px=px2)

        pdf_diff = var_ops.arithmetic.subtract_variables(
            pdf1, pdf2, limit_positive=True
        )

        diff_mu = mu1 - mu2
        diff_sigma = np.sqrt(sigma1**2 + sigma2**2)

        px_expected = sp.stats.truncnorm.pdf(
            x=pdf_diff.x,
            a=(0.0 - diff_mu) / diff_sigma,
            b=np.inf,
            loc=diff_mu,
            scale=diff_sigma,
        )

        np.testing.assert_allclose(pdf_diff.px, px_expected, atol=1e-4)

    def test_rejects_mismatched_sampling(self):
        dx = 0.01

        x1 = PDFs.value_arrays.precise_array(-4.0, 8.0, dx)
        px1 = PDFs.parametric_functions.gaussian(x1, mu=2.0, sigma=1.5)
        pdf1 = PDFs.PDF(x=x1, px=px1)

        x2 = PDFs.value_arrays.precise_array(-9.0, 7.0, dx)
        px2 = PDFs.parametric_functions.gaussian(x2, mu=-1.0, sigma=2.0)
        pdf2 = PDFs.PDF(x=x2, px=px2)

        with pytest.raises(
            ValueError, match="Not all PDFs are sampled over same values"
        ):
            var_ops.arithmetic.subtract_variables(pdf1, pdf2)

    @pytest.mark.parametrize(
        "vartype1, unit1, vartype2, unit2",
        [
            ("age", "y", "age", "m"),
            ("age", "y", "displacement", "y"),
        ],
    )
    def test_different_metadata_warn(self, vartype1, unit1, vartype2, unit2):
        x = PDFs.value_arrays.precise_array(-20.0, 20.0, 0.01)

        px1 = PDFs.parametric_functions.gaussian(x, mu=2.0, sigma=1.5)
        pdf1 = PDFs.PDF(x=x, px=px1, variable_type=vartype1, unit=unit1)

        px2 = PDFs.parametric_functions.gaussian(x, mu=-1.0, sigma=2.0)
        pdf2 = PDFs.PDF(x=x, px=px2, variable_type=vartype2, unit=unit2)

        with pytest.warns(UserWarning, match=""):
            var_ops.arithmetic.subtract_variables(pdf1, pdf2)

    def test_name(self, recwarn):
        x = PDFs.value_arrays.precise_array(-20.0, 20.0, 0.01)
        vartype = "age"
        unit = "y"

        px1 = PDFs.parametric_functions.gaussian(x, mu=2.0, sigma=1.5)
        pdf1 = PDFs.PDF(
            x=x,
            px=px1,
            name="X1",
            variable_type=vartype,
            unit=unit,
        )

        px2 = PDFs.parametric_functions.gaussian(x, mu=-1.0, sigma=2.0)
        pdf2 = PDFs.PDF(
            x=x,
            px=px2,
            name="X2",
            variable_type=vartype,
            unit=unit,
        )

        pdf_diff = var_ops.arithmetic.subtract_variables(pdf1, pdf2, name="X12")

        assert len(recwarn) == 0
        assert pdf_diff.name == "X12"


# end of file
