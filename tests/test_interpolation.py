# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

# Import modules
import numpy as np
import pytest

from riser import probability_functions as PDFs


"""
Because `interpolate_pdfs` depends on the `value_arrays` module,
`test_value_arrays.py` should be run first.
"""


# Tests
class TestInterpolatePdf:
    """
    Resample onto the same domain:
    px values should come back unchanged.

    Resample onto a wider domain:
    the new region should be 0.0 and area still 1.0.

    Resample onto a narrower domain:
    area renormalizes.

    Zero overlap:
    raises ValueError via the constructor's zero-area guard.

    Metadata is preserved:
    all metadata items preserved.
    """

    xmin = 0.0
    xmax = 2.0
    x = np.linspace(xmin, xmax, 3)
    px = np.array([0.0, 1.0, 0.0])
    pdf = PDFs.PDF(x=x, px=px, name="x", variable_type="age", unit="y")

    def test_sample_onto_same_domain(self):
        pdf_intp = PDFs.interpolation.interpolate_pdf(self.pdf, self.x)
        assert pdf_intp is not self.pdf
        np.testing.assert_allclose(pdf_intp.x, self.pdf.x)
        np.testing.assert_allclose(pdf_intp.px, self.pdf.px)

    def test_resample_onto_same_domain_finer(self):
        x_intp = np.linspace(self.xmin, self.xmax, 5)
        pdf_intp = PDFs.interpolation.interpolate_pdf(self.pdf, x_intp)
        np.testing.assert_allclose(pdf_intp.x, x_intp)
        np.testing.assert_allclose(
            pdf_intp.px, np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        )

    def test_resample_wider_domain(self):
        x_wide = np.linspace(-0.5, 3.0, 8)
        pdf_intp = PDFs.interpolation.interpolate_pdf(self.pdf, x_wide)
        np.testing.assert_allclose(pdf_intp.x, x_wide)
        np.testing.assert_allclose(
            pdf_intp.px, np.array([0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0])
        )

    def test_resample_narrower_domain(self):
        x_narr = np.linspace(0.0, 1.0, 3)
        pdf_intp = PDFs.interpolation.interpolate_pdf(self.pdf, x_narr)
        np.testing.assert_allclose(pdf_intp.x, x_narr)
        np.testing.assert_allclose(
            pdf_intp.px, np.array([0.0, 1.0, 2.0])
        )

    def test_zero_overlap(self):
        x_zero = np.linspace(-2.5, -0.5, 3)
        with pytest.raises(
            ValueError, match="Total probability is too close to 0.0."
        ):
            PDFs.interpolation.interpolate_pdf(self.pdf, x_zero)

    def test_metadata_preserved(self):
        x_intp = np.linspace(self.xmin, self.xmax, 5)
        pdf_intp = PDFs.interpolation.interpolate_pdf(self.pdf, x_intp)
        assert pdf_intp.name == self.pdf.name
        assert pdf_intp.variable_type == self.pdf.variable_type
        assert pdf_intp.unit == self.pdf.unit


class TestInterpolatePDFs:
    def test_interpolation(self):
        pdf1 = PDFs.PDF(
            x=np.linspace(0.0, 2.0, 3),
            px=np.array([0.0, 1.0, 0.0]),
        )
        pdf2 = PDFs.PDF(
            x=np.linspace(-2.0, .0, 5),
            px=np.array([0.0, 0.5, 1.0, 0.5, 0.0]),
        )

        pdf1_intp, pdf2_intp = PDFs.interpolation.interpolate_pdfs(
            [pdf1, pdf2]
        )

        np.testing.assert_allclose(
            pdf1_intp.x,
            np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]),
        )
        np.testing.assert_allclose(
            pdf1_intp.px,
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0]),
        )
        np.testing.assert_allclose(
            pdf2_intp.x,
            np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]),
        )
        np.testing.assert_allclose(
            pdf2_intp.px,
            np.array([0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )


# end of file
