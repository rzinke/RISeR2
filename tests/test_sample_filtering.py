# -*- coding: utf-8 -*-
#
# Copyright (c) 2025-2026 Robert Zinke. Licensed under the MIT License.

# Import modules
import numpy as np
import pytest

from riser import (
    probability_functions as PDFs,
    sampling,
)


# Tests
class TestMeanFilter:
    @pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 21])
    def test_mean_filter_properties(self, width):
        filt = sampling.filtering.MeanFilter(width=width)

        # Area = 1.0
        assert np.sum(filt.h) == pytest.approx(1.0)

        # All values equal
        np.testing.assert_allclose(filt.h, filt.h[0])


class TestGaussFilter:
    @pytest.mark.parametrize("width", [3, 5, 7])
    def test_gauss_filter_properties(self, width):
        filt = sampling.filtering.GaussFilter(width)

        # Area = 1.0
        assert np.sum(filt.h) == pytest.approx(1.0)

        # Half-width
        w2 = width // 2

        # Symmetric
        np.testing.assert_allclose(filt.h[:w2], filt.h[w2+1:][::-1])

        # Peaked
        assert filt.h[w2] == np.max(filt.h)


class TestFilterPdf:
    def test_smooth_pdf_filtering(self):
        # Create smooth PDF
        x = PDFs.value_arrays.precise_array(0.0, 1.0, 0.01)
        px = PDFs.parametric_functions.uniform(x, a=0.0, b=1.0)
        pdf = PDFs.PDF(x, px)

        # Run smoothing filter
        pdf_filt = sampling.filtering.filter_pdf(
            pdf=pdf,
            filter_type="mean",
            filter_width=3,
            preserve_edges=True,
        )

        # Confirm nothing disrupted
        np.testing.assert_allclose(pdf_filt.px, pdf.px)


# end of file
