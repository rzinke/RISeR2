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


def reject_mismatched_sampling_test(fcn):
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

    with pytest.raises(
        ValueError, match="Not all PDFs are sampled over same values"
    ):
        fcn(pdf1, pdf2)


# Tests
class TestComputeCosineSimilarity:
    def test_same_pdfs_returns_unit_similarity(self):
        """If two PDFs with the same values are passed, the cosine similarity
        should be 1.0.
        """
        x = PDFs.value_arrays.precise_array(-4.0, 4.0, 0.01)
        px = PDFs.parametric_functions.gaussian(x=x, mu=0.0, sigma=1.0)
        pdf = PDFs.PDF(x=x, px=px)

        r = var_fcns.compare.cosine_similarity(pdf, pdf)
        
        assert r == pytest.approx(1.0)

    def test_rejects_mismatched_sampling(self):
        reject_mismatched_sampling_test(var_fcns.compare.cosine_similarity)


class TestCrossCorrelateVariables:
    @pytest.mark.parametrize(
        "mu1, mu2, lag_expected",
        [
            (0.0, 0.0, 0),
            (0.0, 1.0, -100),
            (2.0, 1.0, 100),
        ],
    )
    def test_cross_correlation(self, mu1, mu2, lag_expected):
        x = PDFs.value_arrays.precise_array(-10.0, 10.0, 0.01)
        px1 = PDFs.parametric_functions.gaussian(x=x, mu=mu1, sigma=1.0)
        pdf1 = PDFs.PDF(x=x, px=px1)
        px2 = PDFs.parametric_functions.gaussian(x=x, mu=mu2, sigma=1.0)
        pdf2 = PDFs.PDF(x=x, px=px2)

        lags, corr_vals = var_fcns.compare.cross_correlate_variables(pdf1, pdf2)

        assert corr_vals.max() == pytest.approx(1.0)
        assert lags[np.argmax(corr_vals)] == lag_expected

    def test_rejects_mismatched_sampling(self):
        reject_mismatched_sampling_test(
            var_fcns.compare.cross_correlate_variables
        )


class TestOverlapIndex:
    def test_perfect_overlap(self):
        a = 0.0
        c = 1.0
        b = 2.0
        x = PDFs.value_arrays.precise_array(-2.0, 2.0, 0.1)
        px1 = PDFs.parametric_functions.triangular(x=x, a=a, c=c, b=b)
        pdf1 = PDFs.PDF(x=x, px=px1)
        px2 = PDFs.parametric_functions.triangular(x=x, a=a, c=c, b=b)
        pdf2 = PDFs.PDF(x=x, px=px2)

        px_min, eta = var_fcns.compare.overlap_index([pdf1, pdf2])

        np.testing.assert_allclose(px_min, px1)
        assert eta == pytest.approx(1.0)


    def test_no_overlap(self):
        x = PDFs.value_arrays.precise_array(-2.0, 2.0, 0.1)
        px1 = PDFs.parametric_functions.triangular(x=x, a=0.0, c=1.0, b=2.0)
        pdf1 = PDFs.PDF(x=x, px=px1)
        px2 = PDFs.parametric_functions.triangular(x=x, a=-2.0, c=-1.0, b=0.0)
        pdf2 = PDFs.PDF(x=x, px=px2)

        px_min, eta = var_fcns.compare.overlap_index([pdf1, pdf2])

        np.testing.assert_allclose(px_min, np.zeros(len(x)))
        assert eta == pytest.approx(0.0)

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
            var_fcns.compare.overlap_index([pdf1, pdf2])


class TestKSstatistic:
    def test_perfect_overlap(self):
        a = 0.0
        c = 1.0
        b = 2.0
        x = PDFs.value_arrays.precise_array(-2.0, 2.0, 0.1)
        px1 = PDFs.parametric_functions.triangular(x=x, a=a, c=c, b=b)
        pdf1 = PDFs.PDF(x=x, px=px1)
        px2 = PDFs.parametric_functions.triangular(x=x, a=a, c=c, b=b)
        pdf2 = PDFs.PDF(x=x, px=px2)

        ks_stat, _ = var_fcns.compare.ks_statistic(pdf1, pdf2)

        assert ks_stat == pytest.approx(0.0)

    def test_imperfect_overlap(self):
        x = PDFs.value_arrays.precise_array(0.0, 3.0, 0.001)
        px1 = PDFs.parametric_functions.triangular(x=x, a=1.0, c=1.0, b=3.0)
        pdf1 = PDFs.PDF(x=x, px=px1)
        px2 = PDFs.parametric_functions.uniform(x=x, a=1.0, b=3.0)
        pdf2 = PDFs.PDF(x=x, px=px2)

        ks_stat, ks_ndx = var_fcns.compare.ks_statistic(pdf1, pdf2)

        assert ks_stat == pytest.approx(0.25)
        assert ks_ndx == 2000

    def test_rejects_mismatched_sampling(self):
        reject_mismatched_sampling_test(var_fcns.compare.ks_statistic)


# end of file
