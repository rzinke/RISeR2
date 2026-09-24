# -*- coding: utf-8 -*-
#
# Copyright (c) 2025-2026 Robert Zinke. Licensed under the MIT License.

# Import modules
import numpy as np
import pytest
import scipy as sp

from riser import (
    probability_functions as PDFs,
    variable_functions as var_fcns,
)


# Tests
class TestVariableCombination:
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_gaussian_product_stdev(self, n):
        """The product of two or more Gaussian PDFs will reduce the standard
        deviation by sqrt(n) / n.
        """
        x = PDFs.value_arrays.precise_array(-20.0, 20.0, 1E-3)
        px = PDFs.parametric_functions.gaussian(x, mu=0, sigma=1.0)
        pdf = PDFs.PDF(x, px)

        pdf_combo, _ = var_fcns.condition.combination.combine_variables(n*[pdf])

        # Test combined standard deviation
        stdev = PDFs.analytics.pdf_std(pdf_combo)

        assert stdev == pytest.approx(np.sqrt(n) / n)

    def test_gaussian_product_area(self):
        """Combining two Gaussian PDFs has a known, closed-form area:

            area = N(mu1 - mu2; 0, sigma1^2 + sigma2^2)

        This is the classical result for the integral of a product of two
        Gaussian densities.
        """
        x = PDFs.value_arrays.precise_array(-20.0, 20.0, 1E-3)
        mu1, sigma1 = 2.0, 1.5
        mu2, sigma2 = -1.0, 2.0
        px1 = PDFs.parametric_functions.gaussian(x, mu=mu1, sigma=sigma1)
        px2 = PDFs.parametric_functions.gaussian(x, mu=mu2, sigma=sigma2)
        pdf1 = PDFs.PDF(x, px1)
        pdf2 = PDFs.PDF(x, px2)

        _, area = var_fcns.condition.combination.combine_variables([pdf1, pdf2])

        # Test combined area
        combined_sigma = np.sqrt(sigma1**2 + sigma2**2)
        expected_area = sp.stats.norm.pdf(
            mu1 - mu2, loc=0.0, scale=combined_sigma
        )

        assert area == pytest.approx(expected_area, rel=1e-4)

class TestCombineVariables:
    @pytest.mark.parametrize(
        "name, vartype, unit, name_expected",
        [
            (None, None, None, None),
            ("combined", "age", "y", "combined"),
        ],
    )
    def test_metadata_consistent(self, name, vartype, unit, name_expected):
        x = PDFs.value_arrays.precise_array(-10.0, 10.0, 0.01)
        px = PDFs.parametric_functions.gaussian(x, mu=0.0, sigma=1.0)
        pdf1 = PDFs.PDF(x, px, variable_type=vartype, unit=unit)
        pdf2 = PDFs.PDF(x, px, variable_type=vartype, unit=unit)

        result, _ = var_fcns.condition.combination.combine_variables(
            [pdf1, pdf2], name=name
        )

        assert result.name == name_expected
        assert result.variable_type == vartype
        assert result.unit == unit

    def test_differing_names_stay_silent(self, recwarn):
        """Two independent measurements of the same event naturally carry
        different sample names, which should not raise a warning.
        """
        x = PDFs.value_arrays.precise_array(-10.0, 10.0, 0.01)
        px = PDFs.parametric_functions.gaussian(x, mu=0.0, sigma=1.0)
        pdf_a = PDFs.PDF(x, px, name="Sample_A14C", variable_type="age", unit="y")
        pdf_b = PDFs.PDF(x, px, name="Sample_B14C", variable_type="age", unit="y")

        result, _ = var_fcns.condition.combination.combine_variables(
            [pdf_a, pdf_b]
        )

        assert len(recwarn) == 0
        assert result.name is None

    def test_differing_physical_properties_warn(self):
        """Unlike names, mismatched variable_type or unit genuinely
        indicates a physically nonsensical combination and should raise a
        warning.
        """
        x = PDFs.value_arrays.precise_array(-10.0, 10.0, 0.01)
        px = PDFs.parametric_functions.gaussian(x, mu=0.0, sigma=1.0)
        pdf_age = PDFs.PDF(x, px, variable_type="age", unit="y")
        pdf_disp = PDFs.PDF(x, px, variable_type="displacement", unit="y")

        with pytest.warns(UserWarning, match="differs"):
            var_fcns.condition.combination.combine_variables(
                [pdf_age, pdf_disp]
            )

    def test_three_pdfs_metadata(self):
        """Confirm metadata handling generalizes beyond a pair, since
        combine_variables (unlike trim/bracket) accepts an arbitrary list.
        """
        x = PDFs.value_arrays.precise_array(-10.0, 10.0, 0.01)
        px = PDFs.parametric_functions.gaussian(x, mu=0.0, sigma=1.0)
        pdfs = [
            PDFs.PDF(x, px, name=f"Sample_{i}", variable_type="age", unit="y")
            for i in "ABC"
        ]

        result, _ = var_fcns.condition.combination.combine_variables(
            pdfs, name="combined_age"
        )

        assert result.name == "combined_age"
        assert result.variable_type == "age"
        assert result.unit == "y"


# end of file
