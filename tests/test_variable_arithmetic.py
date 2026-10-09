# -*- coding: utf-8 -*-
#
# Copyright (c) 2025, 2026 Robert Zinke. Licensed under the MIT License.

# Import modules
import numpy as np
import pytest
import scipy as sp

from riser import (
    probability_functions as PDFs,
    variable_functions as var_fcns,
)


# Test bespoke convolution functions
class TestConvolveInputSide:
    def test_matches_numpy_reference(self):
        """Check that the output of this library's bespoke input-side
        convolution function matches that of the standard numpy convolution
        function.
        """
        x = np.array([1.0, 2.0, 3.0])
        h = np.array([0.5, 0.5])
        result = var_fcns.transform.arithmetic.convolve_input_side(x, h)
        expected = np.convolve(x, h, mode="full")
        np.testing.assert_allclose(result, expected)


class TestConvolveOutputSide:
    def test_matches_numpy_reference(self):
        """Check that the output of this library's bespoke output-side
        convolution function matches that of the standard numpy convolution
        function.
        """
        x = np.array([1.0, 2.0, 3.0])
        h = np.array([0.5, 0.5])
        result = var_fcns.transform.arithmetic.convolve_output_side(x, h)
        expected = np.convolve(x, h, mode="full")
        np.testing.assert_allclose(result, expected)

    def test_matches_input_side_formulation(self):
        """Check that the bespoke convolution functions of this library
        produce consistent outputs.
        """
        x = np.array([1.0, 2.0, 3.0, 4.0])
        h = np.array([0.5, 0.25, 0.25])
        result_output = var_fcns.transform.arithmetic.convolve_output_side(x, h)
        result_input = var_fcns.transform.arithmetic.convolve_input_side(x, h)
        np.testing.assert_allclose(result_output, result_input)


# Test variable negation
class TestNegateVariable:
    def test_negation_reflects_shape_correctly(self):
        """The shape of a PDF should be reflected across 0.0.
        This will be especially apparent for an asymmetric shape.
        """
        x = PDFs.value_arrays.precise_array(0.0, 5.0, 0.01)
        px = PDFs.parametric_functions.triangular(x, a=0.0, c=1.0, b=5.0)
        pdf = PDFs.PDF(x, px)

        neg_pdf = var_fcns.transform.arithmetic.negate_variable(pdf)

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
        """
        Check that if no name is passed, no name is returned.
        Or is a name is passed, the name is signified to be the negated version
        of the input.
        """
        x = PDFs.value_arrays.precise_array(0.0, 5.0, 0.5)
        px = PDFs.parametric_functions.triangular(x, a=0.0, c=1.0, b=5.0)
        pdf = PDFs.PDF(
            x=x,
            px=px,
            name=original_name,
            variable_type="age",
            unit="y",
        )

        neg_pdf = var_fcns.transform.arithmetic.negate_variable(pdf)

        assert neg_pdf.name == expected_name
        assert neg_pdf.variable_type == "age"
        assert neg_pdf.unit == "y"


# Test variable arithmetic
class TestAddVariables:
    def test_gaussian_sum_closed_form(self):
        """The sum of two Gaussians will be Gaussian in shape.
        The mean of the sum will be the sum of the input means.
        The standard deviation will be the root sum of squares of the
        input standard deviations.
        """
        x = PDFs.value_arrays.precise_array(-20.0, 20.0, 0.01)

        sigma1 = 1.5
        px1 = PDFs.parametric_functions.gaussian(x, mu=2.0, sigma=sigma1)
        pdf1 = PDFs.PDF(x=x, px=px1)

        sigma2 = 2.0
        px2 = PDFs.parametric_functions.gaussian(x, mu=-1.0, sigma=sigma2)
        pdf2 = PDFs.PDF(x=x, px=px2)

        pdf_sum = var_fcns.transform.arithmetic.add_variables(pdf1, pdf2)

        assert PDFs.analytics.pdf_mean(pdf_sum) == pytest.approx(1.0)
        assert PDFs.analytics.pdf_variance(pdf_sum) == pytest.approx(
            sigma1**2 + sigma2**2
        )

    def test_rejects_mismatched_sampling(self):
        """Passing functions that are sampled on different value arrays
        should raise an error.
        """
        dx = 0.01

        x1 = PDFs.value_arrays.precise_array(-4.0, 8.0, dx)
        px1 = PDFs.parametric_functions.gaussian(x1, mu=2.0, sigma=1.5)
        pdf1 = PDFs.PDF(x=x1, px=px1)

        x2 = PDFs.value_arrays.precise_array(-9.0, 7.0, dx)
        px2 = PDFs.parametric_functions.gaussian(x2, mu=-1.0, sigma=2.0)
        pdf2 = PDFs.PDF(x=x2, px=px2)

        with pytest.raises(ValueError, match="Not all value arrays"):
            var_fcns.transform.arithmetic.add_variables(pdf1, pdf2)

    @pytest.mark.parametrize(
        "vartype1, unit1, vartype2, unit2",
        [
            ("age", "y", "age", "m"),
            ("age", "y", "displacement", "y"),
        ],
    )
    def test_different_metadata_warn(self, vartype1, unit1, vartype2, unit2):
        """Adding variables of fundamentally different types or units is not
        consistent with physics.
        However, variable types are not strictly enforced in this library, so
        only a warning should be raised.
        """
        x = PDFs.value_arrays.precise_array(-20.0, 20.0, 0.01)

        px1 = PDFs.parametric_functions.gaussian(x, mu=2.0, sigma=1.5)
        pdf1 = PDFs.PDF(x=x, px=px1, variable_type=vartype1, unit=unit1)

        px2 = PDFs.parametric_functions.gaussian(x, mu=-1.0, sigma=2.0)
        pdf2 = PDFs.PDF(x=x, px=px2, variable_type=vartype2, unit=unit2)

        with pytest.warns(UserWarning, match=""):
            var_fcns.transform.arithmetic.add_variables(pdf1, pdf2)

    def test_name(self, recwarn):
        """Different variables are expected to have different names, so no
        warning should be raised when different names are passed.
        Explicitly passing a name to `add_variables` should asribe that name
        to the resulting PDF.
        """
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

        pdf_sum = var_fcns.transform.arithmetic.add_variables(
            pdf1, pdf2, name="X12"
        )

        assert len(recwarn) == 0
        assert pdf_sum.name == "X12"


class TestSubtractVariables:
    def test_gaussian_sum_closed_form(self):
        """Similar to addition, the subtraction of two Gaussian variables
        should produce a Gaussian with a mean that is the difference of the
        input means, and a standard deviation that is the root sum of squares
        of the inputs.
        """
        x = PDFs.value_arrays.precise_array(-20.0, 20.0, 0.01)

        sigma1 = 1.5
        px1 = PDFs.parametric_functions.gaussian(x, mu=2.0, sigma=sigma1)
        pdf1 = PDFs.PDF(x=x, px=px1)

        sigma2 = 2.0
        px2 = PDFs.parametric_functions.gaussian(x, mu=-1.0, sigma=sigma2)
        pdf2 = PDFs.PDF(x=x, px=px2)

        pdf_diff = var_fcns.transform.arithmetic.subtract_variables(pdf1, pdf2)

        assert PDFs.analytics.pdf_mean(pdf_diff) == pytest.approx(3.0)
        assert PDFs.analytics.pdf_variance(pdf_diff) == pytest.approx(
            sigma1**2 + sigma2**2
        )

    def test_rejects_mismatched_sampling(self):
        """Passing functions that are sampled on different value arrays
        should raise an error.
        """
        dx = 0.01

        x1 = PDFs.value_arrays.precise_array(-4.0, 8.0, dx)
        px1 = PDFs.parametric_functions.gaussian(x1, mu=2.0, sigma=1.5)
        pdf1 = PDFs.PDF(x=x1, px=px1)

        x2 = PDFs.value_arrays.precise_array(-9.0, 7.0, dx)
        px2 = PDFs.parametric_functions.gaussian(x2, mu=-1.0, sigma=2.0)
        pdf2 = PDFs.PDF(x=x2, px=px2)

        with pytest.raises(ValueError, match="Not all value arrays"):
            var_fcns.transform.arithmetic.subtract_variables(pdf1, pdf2)

    @pytest.mark.parametrize(
        "vartype1, unit1, vartype2, unit2",
        [
            ("age", "y", "age", "m"),
            ("age", "y", "displacement", "y"),
        ],
    )
    def test_different_metadata_warn(self, vartype1, unit1, vartype2, unit2):
        """Warn if variables of fundamentally different types are passed."""
        x = PDFs.value_arrays.precise_array(-20.0, 20.0, 0.01)

        px1 = PDFs.parametric_functions.gaussian(x, mu=2.0, sigma=1.5)
        pdf1 = PDFs.PDF(x=x, px=px1, variable_type=vartype1, unit=unit1)

        px2 = PDFs.parametric_functions.gaussian(x, mu=-1.0, sigma=2.0)
        pdf2 = PDFs.PDF(x=x, px=px2, variable_type=vartype2, unit=unit2)

        with pytest.warns(UserWarning, match=""):
            var_fcns.transform.arithmetic.subtract_variables(pdf1, pdf2)

    def test_name(self, recwarn):
        """Passing variables with different names should not raise a warning.
        An explicitly passed named should be ascribed to the resulting PDF.
        """
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

        pdf_diff = var_fcns.transform.arithmetic.subtract_variables(
            pdf1, pdf2, name="X12"
        )

        assert len(recwarn) == 0
        assert pdf_diff.name == "X12"

    @staticmethod
    def _two_gaussians_():
        """Two Gaussian displacement PDFs, 12 +/- 2 m and 10 +/- 2 m, whose
        difference is Gaussian with a mean of 2.0 and a standard deviation of
        sqrt(8), so that about 24 % of the difference lies below zero.
        """
        x = PDFs.value_arrays.precise_array(-20.0, 20.0, 0.01)

        pdf1 = PDFs.PDF(
            x=x,
            px=PDFs.parametric_functions.gaussian(x, mu=12.0, sigma=2.0),
            name="X1",
            variable_type="displacement",
            unit="m",
        )
        pdf2 = PDFs.PDF(
            x=x,
            px=PDFs.parametric_functions.gaussian(x, mu=10.0, sigma=2.0),
            name="X2",
            variable_type="displacement",
            unit="m",
        )

        return pdf1, pdf2

    def test_limit_positive_crops_axis_and_normalizes(self):
        """Limiting to positive values should drop all non-positive values
        from the output axis and renormalize the remaining density to an area
        of 1.0.
        """
        pdf1, pdf2 = self._two_gaussians_()

        pdf_diff = var_fcns.transform.arithmetic.subtract_variables(
            pdf1, pdf2, limit_positive=True
        )

        assert np.all(pdf_diff.x > 0.0)
        assert np.all(pdf_diff.px >= 0.0)
        assert np.trapezoid(pdf_diff.px, pdf_diff.x) == pytest.approx(1.0)

    def test_limit_positive_matches_truncated_gaussian(self):
        """The difference of two Gaussians limited to positive values should
        follow a Gaussian truncated at zero.
        """
        pdf1, pdf2 = self._two_gaussians_()
        mu = 2.0
        sigma = np.sqrt(8.0)
        truncated = sp.stats.truncnorm(-mu / sigma, np.inf, loc=mu, scale=sigma)

        pdf_diff = var_fcns.transform.arithmetic.subtract_variables(
            pdf1, pdf2, limit_positive=True
        )

        assert PDFs.analytics.pdf_mean(pdf_diff) == pytest.approx(
            truncated.mean(), rel=5e-3
        )
        assert np.sqrt(PDFs.analytics.pdf_variance(pdf_diff)) == pytest.approx(
            truncated.std(), rel=5e-3
        )

    def test_limit_positive_is_cropped_and_rescaled_unconstrained(self):
        """Limiting to positive values should be equivalent to keeping the
        positive part of the unconstrained difference and rescaling it by the
        probability that the difference is positive.
        """
        pdf1, pdf2 = self._two_gaussians_()

        pdf_free = var_fcns.transform.arithmetic.subtract_variables(pdf1, pdf2)
        pdf_pos = var_fcns.transform.arithmetic.subtract_variables(
            pdf1, pdf2, limit_positive=True
        )

        positive = pdf_free.x > 0.0
        np.testing.assert_allclose(pdf_pos.x, pdf_free.x[positive])
        area_positive = np.trapezoid(
            pdf_free.px[positive], pdf_free.x[positive]
        )
        np.testing.assert_allclose(
            pdf_pos.px, pdf_free.px[positive] / area_positive, rtol=1e-3
        )

    def test_default_keeps_full_axis(self):
        """Without `limit_positive`, the output should retain its full
        (including negative) range.
        """
        pdf1, pdf2 = self._two_gaussians_()

        pdf_diff = var_fcns.transform.arithmetic.subtract_variables(pdf1, pdf2)

        assert pdf_diff.x[0] == pytest.approx(-40.0)
        assert pdf_diff.x[-1] == pytest.approx(40.0)

    def test_limit_positive_metadata(self):
        """Limiting to positive values should not alter the metadata, and an
        explicitly passed name should still be used.
        """
        pdf1, pdf2 = self._two_gaussians_()

        pdf_diff = var_fcns.transform.arithmetic.subtract_variables(
            pdf1, pdf2, limit_positive=True, name="X12"
        )

        assert pdf_diff.name == "X12"
        assert pdf_diff.variable_type == "displacement"
        assert pdf_diff.unit == "m"

    def test_limit_positive_all_negative_raises(self):
        """If every possible difference is negative, there is nothing to
        retain and the function should raise.
        """
        x = PDFs.value_arrays.precise_array(-20.0, 20.0, 0.01)
        pdf1 = PDFs.PDF(
            x=x, px=PDFs.parametric_functions.gaussian(x, mu=-12.0, sigma=1.0)
        )
        pdf2 = PDFs.PDF(
            x=x, px=PDFs.parametric_functions.gaussian(x, mu=10.0, sigma=1.0)
        )

        with pytest.raises(ValueError, match="too close to 0.0"):
            var_fcns.transform.arithmetic.subtract_variables(
                pdf1, pdf2, limit_positive=True
            )


class TestMultiplyVariables:
    def test_uniform_product_closed_form(self):
        """The product of two uniform distributions defined over the interval
        [0.0, 1.0] should result in a distribution of the form -ln(z).

        The accuracy of the result is highly dependent on the spacing of the
        input grids.
        """
        dx = 1e-4
        x = PDFs.value_arrays.precise_array(0.0, 1.0, dx)
        px = PDFs.parametric_functions.uniform(x, 0.0, 1.0)
        pdf1 = PDFs.PDF(x=x, px=px)
        pdf2 = PDFs.PDF(x=x, px=px)

        pdf_prod, _ = var_fcns.transform.arithmetic.multiply_variables(
            pdf1, pdf2, dz=dx
        )

        check_ndx = pdf_prod.x > 1e-2

        px_expected = -np.log(pdf_prod.x[check_ndx])

        np.testing.assert_allclose(
            pdf_prod.px[check_ndx], px_expected, atol=5e-3
        )


class TestDivideVariables:
    def test_ratio_of_normals_is_cauchy(self):
        """The quotient of two standard normal distributions should follow
        a Cauchy distribution.
        """
        dx = 0.001
        x = PDFs.value_arrays.precise_array(-8.0, 8.0, dx)
        px = PDFs.parametric_functions.gaussian(x, mu=0.0, sigma=1.0)
        numerator = PDFs.PDF(x=x, px=px)
        denominator = PDFs.PDF(x=x, px=px)

        min_q, max_q = -20.0, 20.0
        pdf_quot, _ = var_fcns.transform.arithmetic.divide_variables(
            numerator,
            denominator,
            dz=0.01,
            min_quotient=min_q,
            max_quotient=max_q,
        )

        # divide_variables correctly returns the CONDITIONAL density given
        # Z is within [min_q, max_q] -- the raw Cauchy pdf must be rescaled
        # to account for the (non-negligible) tail mass outside that range
        p_inside = sp.stats.cauchy.cdf(max_q) - sp.stats.cauchy.cdf(min_q)
        px_expected = sp.stats.cauchy.pdf(pdf_quot.x) / p_inside

        # Avoid the extreme edges, where truncation itself distorts the shape
        interior = (pdf_quot.x > -10) & (pdf_quot.x < 10)

        np.testing.assert_allclose(
            pdf_quot.px[interior], px_expected[interior], atol=1e-3
        )

    def test_too_coarse_dz_raises(self):
        """For small quotient ranges (large denominator relative to numerator),
        the default `dz` value may be too coarse and the resulting `pz` values
        - though correct - will be too scarce to produce a valid PDF.

        Test that a too-coarse `dz` value, whether user-specified or default,
        raises.
        """
        # Age in yBP
        age_axis = PDFs.value_arrays.precise_array(15_000, 25_000, 10)
        age_density = PDFs.parametric_functions.gaussian(
            age_axis, mu=20_000, sigma=1_000
        )
        age_pdf = PDFs.PDF(age_axis, age_density)

        # Displacement in meters
        disp_axis = PDFs.value_arrays.precise_array(15.0, 25.0, 0.01)
        disp_density = PDFs.parametric_functions.triangular(
            disp_axis, a=15.0, c=20.0, b=25.0
        )
        disp_pdf = PDFs.PDF(disp_axis, disp_density)

        with pytest.raises(ValueError, match="Output array spacing"):
            var_fcns.transform.arithmetic.divide_variables(
                disp_pdf, age_pdf, dz=0.1
            )

    def test_dz_None_default_succeeds(self):
        """Related to the previous pitfall, setting `dz` to None should
        automatically determine the sample spacing in a way that will not
        fail.
        """
        # Age in yBP
        age_axis = PDFs.value_arrays.precise_array(15_000, 25_000, 10)
        age_density = PDFs.parametric_functions.gaussian(
            age_axis, mu=20_000, sigma=1_000
        )
        age_pdf = PDFs.PDF(age_axis, age_density)

        # Displacement in meters
        disp_axis = PDFs.value_arrays.precise_array(15.0, 25.0, 0.01)
        disp_density = PDFs.parametric_functions.triangular(
            disp_axis, a=15.0, c=20.0, b=25.0
        )
        disp_pdf = PDFs.PDF(disp_axis, disp_density)

        var_fcns.transform.arithmetic.divide_variables(
            disp_pdf, age_pdf, dz=None
        )

    @pytest.mark.parametrize(
        "limits",
        [
            pytest.param({"min_quotient": 1.0}, id="min above natural max"),
            pytest.param({"max_quotient": 1e-5}, id="max below natural min"),
            pytest.param(
                {"min_quotient": 1e-3, "max_quotient": 1e-3},
                id="min equals max",
            ),
            pytest.param(
                {"min_quotient": 2e-3, "max_quotient": 1e-3}, id="min above max"
            ),
        ],
    )
    def test_empty_quotient_range_raises(self, limits):
        """User-specified limits can only narrow the natural quotient range.
        If they leave no values (zero or negative range), raise a clear error
        rather than failing deep inside the array construction.
        """
        # Age in yBP
        age_axis = PDFs.value_arrays.precise_array(15_000, 25_000, 10)
        age_density = PDFs.parametric_functions.gaussian(
            age_axis, mu=20_000, sigma=1_000
        )
        age_pdf = PDFs.PDF(age_axis, age_density)

        # Displacement in meters
        disp_axis = PDFs.value_arrays.precise_array(15.0, 25.0, 0.01)
        disp_density = PDFs.parametric_functions.triangular(
            disp_axis, a=15.0, c=20.0, b=25.0
        )
        disp_pdf = PDFs.PDF(disp_axis, disp_density)

        with pytest.raises(ValueError, match="No quotient values remain"):
            var_fcns.transform.arithmetic.divide_variables(
                disp_pdf, age_pdf, dz=None, **limits
            )

    def test_empty_range_with_denominator_straddling_zero_raises(self):
        """When the denominator range includes zero the user must provide the
        limits. Equal limits leave an empty range and should raise too.
        """
        # Denominator spans zero
        denom_axis = PDFs.value_arrays.precise_array(-1.0, 1.0, 0.01)
        denom_density = PDFs.parametric_functions.gaussian(
            denom_axis, mu=0.0, sigma=0.2
        )
        denom_pdf = PDFs.PDF(denom_axis, denom_density)

        numer_axis = PDFs.value_arrays.precise_array(1.0, 3.0, 0.01)
        numer_density = PDFs.parametric_functions.gaussian(
            numer_axis, mu=2.0, sigma=0.2
        )
        numer_pdf = PDFs.PDF(numer_axis, numer_density)

        with pytest.raises(ValueError, match="No quotient values remain"):
            var_fcns.transform.arithmetic.divide_variables(
                numer_pdf,
                denom_pdf,
                min_quotient=5.0,
                max_quotient=5.0,
                dz=None,
            )


# end of file
