# -*- coding: utf-8 -*-
#
# Copyright (c) 2025, 2026 Robert Zinke. Licensed under the MIT License.

# Import modules
import math

import numpy as np
import pytest
import scipy as sp

from riser import (
    probability_functions as PDFs,
    variable_functions as var_fcns,
)


# Helper functions
def _gaussian_pdfs_(means, sigma, xmin, xmax, dx, **metadata):
    """Create Gaussian PDFs with a common value array."""
    x = PDFs.value_arrays.precise_array(xmin, xmax, dx)

    return [
        PDFs.PDF(
            x,
            PDFs.parametric_functions.gaussian(x, mean, sigma),
            name=f"X{k + 1}",
            **metadata,
        )
        for k, mean in enumerate(means)
    ]


# Test
class TestTrimVariables:
    def test_known_solution(self):
        """Test that two uniform distributions trim each other.

        Leverage the fact that if two PDFs are identically distributed,
        symmetry dictates that P(pdf1 < pdf2) = P(pdf2 > pdf1) = 0.5.

        Test the application of the definition of the trimming routine.
        """
        x = PDFs.value_arrays.precise_array(0.0, 1.0, 1e-5)
        px1 = PDFs.parametric_functions.uniform(x=x, a=0.0, b=1.0)
        pdf1 = PDFs.PDF(x, px1)
        px2 = PDFs.parametric_functions.uniform(x=x, a=0.0, b=1.0)
        pdf2 = PDFs.PDF(x, px2)

        (
            pdf1_trimmed,
            pdf2_trimmed,
            area,
        ) = var_fcns.condition.trimming.trim_variables(pdf1, pdf2)

        assert area == pytest.approx(0.5)

        np.testing.assert_allclose(pdf1_trimmed.px, (1 - pdf2.Px) / area)
        np.testing.assert_allclose(pdf2_trimmed.px, pdf1.Px / area)

    @pytest.mark.parametrize(
        "name1, name2, vartype, unit, name1_expected, name2_expected",
        [
            (None, None, None, None, None, None),
            ("X1", "X2", "age", "y", "X1", "X2"),
        ],
    )
    def test_metadata_consistent(
        self, name1, name2, vartype, unit, name1_expected, name2_expected
    ):
        x = PDFs.value_arrays.precise_array(0.0, 1.0, 1e-5)
        px1 = PDFs.parametric_functions.uniform(x=x, a=0.0, b=1.0)
        pdf1 = PDFs.PDF(x, px1, name=name1, variable_type=vartype, unit=unit)
        px2 = PDFs.parametric_functions.uniform(x=x, a=0.0, b=1.0)
        pdf2 = PDFs.PDF(x, px2, name=name2, variable_type=vartype, unit=unit)

        (
            pdf1_trimmed,
            pdf2_trimmed,
            _,
        ) = var_fcns.condition.trimming.trim_variables(pdf1, pdf2)

        assert pdf1_trimmed.name == name1_expected
        assert pdf1_trimmed.variable_type == vartype
        assert pdf1_trimmed.unit == unit

        assert pdf2_trimmed.name == name2_expected
        assert pdf2_trimmed.variable_type == vartype
        assert pdf2_trimmed.unit == unit

    def test_name_override(self):
        vartype = "age"
        unit = "y"

        x = PDFs.value_arrays.precise_array(0.0, 1.0, 1e-5)
        px1 = PDFs.parametric_functions.uniform(x=x, a=0.0, b=1.0)
        pdf1 = PDFs.PDF(x, px1, name=None, variable_type=vartype, unit=unit)
        px2 = PDFs.parametric_functions.uniform(x=x, a=0.0, b=1.0)
        pdf2 = PDFs.PDF(x, px2, name=None, variable_type=vartype, unit=unit)

        (
            pdf1_trimmed,
            pdf2_trimmed,
            _,
        ) = var_fcns.condition.trimming.trim_variables(
            pdf1, pdf2, name1="X1 trimmed", name2="X2 trimmed"
        )

        assert pdf1_trimmed.name == "X1 trimmed"
        assert pdf1_trimmed.variable_type == vartype
        assert pdf1_trimmed.unit == unit

        assert pdf2_trimmed.name == "X2 trimmed"
        assert pdf2_trimmed.variable_type == vartype
        assert pdf2_trimmed.unit == unit


class TestTrimVariableSequence:
    def test_single_pdf_unchanged(self):
        """A lone PDF has no ordering constraints."""
        pdfs = _gaussian_pdfs_([5.0], 1.0, 0.0, 10.0, 0.01)

        (trimmed,) = var_fcns.condition.trimming.trim_variable_sequence(pdfs)

        np.testing.assert_allclose(trimmed.px, pdfs[0].px)

    def test_two_pdfs_match_pairwise(self):
        """The sequence function reduces to `trim_variables` for two PDFs."""
        pdfs = _gaussian_pdfs_([4.0, 5.0], 1.0, 0.0, 10.0, 0.01)

        trimmed = var_fcns.condition.trimming.trim_variable_sequence(pdfs)
        (
            pdf1_trimmed,
            pdf2_trimmed,
            _,
        ) = var_fcns.condition.trimming.trim_variables(*pdfs)

        np.testing.assert_allclose(trimmed[0].px, pdf1_trimmed.px, atol=1e-12)
        np.testing.assert_allclose(trimmed[1].px, pdf2_trimmed.px, atol=1e-12)

    @pytest.mark.parametrize("n_pdfs", [3, 5])
    def test_known_solution(self, n_pdfs):
        """Test that identical uniform distributions give order statistics.

        If n independent variables are identically distributed on [0, 1] and
        are known to be ordered X_1 < ... < X_n, then X_k is the k-th order
        statistic, with the Beta(k, n - k + 1) density

            n! / ((k - 1)! (n - k)!) · x^(k - 1) · (1 - x)^(n - k)
        """
        x = PDFs.value_arrays.precise_array(0.0, 1.0, 1e-4)
        px = PDFs.parametric_functions.uniform(x=x, a=0.0, b=1.0)
        pdfs = [PDFs.PDF(x, px) for _ in range(n_pdfs)]

        trimmed = var_fcns.condition.trimming.trim_variable_sequence(pdfs)

        for k, pdf in enumerate(trimmed, start=1):
            coefficient = math.factorial(n_pdfs) / (
                math.factorial(k - 1) * math.factorial(n_pdfs - k)
            )
            expected = coefficient * x ** (k - 1) * (1 - x) ** (n_pdfs - k)

            np.testing.assert_allclose(pdf.px, expected, rtol=1e-5, atol=1e-6)

    def test_separated_pdfs_unchanged(self):
        """PDFs that are already certainly ordered are not altered."""
        pdfs = _gaussian_pdfs_([10.0, 30.0, 50.0], 1.0, 0.0, 60.0, 0.01)

        trimmed = var_fcns.condition.trimming.trim_variable_sequence(pdfs)

        for pdf, pdf_trimmed in zip(pdfs, trimmed):
            np.testing.assert_allclose(pdf_trimmed.px, pdf.px, atol=1e-9)

    def test_matches_rejection_sampling(self):
        """Compare overlapping Gaussians against brute-force sampling.

        Draw all variables independently, and keep only the draws that
        satisfy the ordering. The CDF of each kept variable must match the
        CDF of the corresponding trimmed PDF.
        """
        x = PDFs.value_arrays.precise_array(0.0, 20.0, 0.01)
        pdfs = _gaussian_pdfs_([9.0, 10.0, 11.0, 12.0], 2.0, 0.0, 20.0, 0.01)

        trimmed = var_fcns.condition.trimming.trim_variable_sequence(pdfs)

        rng = np.random.default_rng(0)
        draws = np.column_stack(
            [np.interp(rng.random(3_000_000), pdf.Px, x) for pdf in pdfs]
        )
        accepted = draws[np.all(np.diff(draws, axis=1) > 0, axis=1)]
        assert len(accepted) > 300_000

        for k, pdf in enumerate(trimmed):
            for value in (6.0, 9.0, 10.0, 11.0, 12.0, 15.0):
                assert pdf.compute_probability_less_than(value) == (
                    pytest.approx(np.mean(accepted[:, k] < value), abs=4e-3)
                )

    def test_tail_precision(self):
        """Small tail values are preserved in both sweep directions.

        Two identical, symmetric PDFs trim each other symmetrically, so the
        upper tail of the first must mirror the lower tail of the second.
        Both tails must also match the exact value, even where the
        probability is as small as 1e-29.
        """
        pdfs = _gaussian_pdfs_([0.0, 0.0], 1.0, -10.0, 10.0, 0.01)
        x = pdfs[0].x

        trimmed = var_fcns.condition.trimming.trim_variable_sequence(pdfs)

        # Mirror symmetry (the area of each trimmed PDF is 0.5 before scaling)
        np.testing.assert_allclose(
            trimmed[0].px, trimmed[1].px[::-1], rtol=1e-9, atol=0.0
        )

        # Exact solution: pdf1(x) · (1 - CDF2(x)) / P(X1 < X2)
        survival = 0.5 * sp.special.erfc(x / np.sqrt(2.0))
        expected = 2.0 * pdfs[0].px * survival
        upper_tail = (x >= 5.0) & (x <= 8.0)

        np.testing.assert_allclose(
            trimmed[0].px[upper_tail], expected[upper_tail], rtol=1e-2
        )

    @pytest.mark.parametrize(
        "vartype, unit, vartype_expected, unit_expected",
        [
            (None, None, "age", "y"),
            ("depth", "m", "depth", "m"),
        ],
    )
    def test_metadata(self, vartype, unit, vartype_expected, unit_expected):
        pdfs = _gaussian_pdfs_(
            [4.0, 5.0, 6.0],
            1.0,
            0.0,
            10.0,
            0.01,
            variable_type="age",
            unit="y",
        )

        trimmed = var_fcns.condition.trimming.trim_variable_sequence(
            pdfs, variable_type=vartype, unit=unit
        )

        assert [pdf.name for pdf in trimmed] == ["X1", "X2", "X3"]
        assert all(pdf.variable_type == vartype_expected for pdf in trimmed)
        assert all(pdf.unit == unit_expected for pdf in trimmed)

    def test_mismatched_sampling_raises(self):
        pdfs = _gaussian_pdfs_([4.0], 1.0, 0.0, 10.0, 0.01)
        pdfs += _gaussian_pdfs_([5.0], 1.0, 0.0, 10.0, 0.02)

        with pytest.raises(ValueError):
            var_fcns.condition.trimming.trim_variable_sequence(pdfs)


# end of file
