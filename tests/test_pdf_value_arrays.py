# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

# Import modules
import numpy as np
import pytest

from riser import probability_functions as PDFs


# Tests
class TestSampleSpacingFromPDF:
    def test_regular_sampling_silent(self, recwarn):
        x = np.linspace(0, 2, 3)
        px=np.array([0.0, 1.0, 0.0])
        pdf = PDFs.PDF(x, px)
        dx = PDFs.value_arrays.sample_spacing_from_pdf(pdf)
        assert len(recwarn) == 0
        assert dx == pytest.approx(1.0)

    def test_irregular_sampling_warns(self):
        x=np.array([90.0, 95.0, 100.0, 105.25])
        px=np.array([0.0, 1.0, 1.0, 0.0])
        pdf = PDFs.PDF(x, px)
        with pytest.warns(UserWarning, match="Sample spacing varies"):
            dx = PDFs.value_arrays.sample_spacing_from_pdf(pdf)
        assert dx == pytest.approx(5.0)

    def test_returns_median_not_mean(self):
        x=np.array([90.0, 95.0, 100.0, 105.25])
        px=np.array([0.0, 1.0, 1.0, 0.0])
        pdf = PDFs.PDF(x, px)
        with pytest.warns(UserWarning):
            dx = PDFs.value_arrays.sample_spacing_from_pdf(pdf)
        assert dx == pytest.approx(np.median(np.diff(x)))
        assert dx != pytest.approx(np.mean(np.diff(x)), abs=0.01)


class TestSampleSpacingArrayFromPDF:
    def test_sample_spacing_irregular(self):
        x=np.array([90.0, 95.0, 100.0, 105.25])
        px=np.array([0.0, 1.0, 1.0, 0.0])
        pdf = PDFs.PDF(x, px)
        dx = PDFs.value_arrays.sample_spacing_array_from_pdf(pdf)
        np.testing.assert_allclose(dx, [5.0, 5.0, 5.25, 0.0])

    def test_sample_spacing_regular(self):
        x = np.linspace(0.0, 2.0, 3)
        px = np.array([0.0, 1.0, 0.0])
        pdf = PDFs.PDF(x, px)
        dx = PDFs.value_arrays.sample_spacing_array_from_pdf(pdf)
        np.testing.assert_allclose(dx, [1.0, 1.0, 1.0])


class TestValueArrayParamsFromPDFs:
    def test_array_params(self):
        pdf1 = PDFs.PDF(
            x=np.linspace(3.0, 5.0, 3),
            px=np.array([0.0, 1.0, 0.0]),
        )
        pdf2 = PDFs.PDF(
            x=np.linspace(-5.0, -3.0, 5),
            px=np.array([0.0, 0.5, 1.0, 0.5, 0.0]),
        )
        xmin, xmax, dx = PDFs.value_arrays.value_array_params_from_pdfs(
            [pdf1, pdf2]
        )
        assert xmin == pytest.approx(-5.0)
        assert xmax == pytest.approx(5.0)
        assert dx == pytest.approx(0.5)


class TestCreatePreciseValueArray:
    def test_precise_array(self):
        x = PDFs.value_arrays.precise_array(xmin=0, xmax=2.0, dx=0.5)
        assert np.all(x == np.array([0.0, 0.5, 1.0, 1.5, 2.0]))


class TestCheckPdfsSampling:
    x3 = np.array([0.0, 1.0, 2.0])
    px3 = np.array([0.0, 1.0, 0.0])

    x5 = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    px5 = np.array([0.0, 0.5, 1.0, 0.5, 0.0])

    def test_different_sampling_raise(self):
        pdf3 = PDFs.PDF(x=self.x3, px=self.px3)
        pdf5 = PDFs.PDF(x=self.x5, px=self.px5)
        with pytest.raises(
            ValueError, match="Not all PDFs are sampled over same values"
        ):
            PDFs.value_arrays.check_pdfs_sampling([pdf3, pdf5])
    
    def test_same_sampling_silent(self):
        pdf3 = PDFs.PDF(x=self.x3, px=self.px3)
        PDFs.value_arrays.check_pdfs_sampling([pdf3, pdf3])


# end of file
