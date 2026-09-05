# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

# Import modules
import numpy as np
import pytest

from riser import probability_functions as PDFs


# PDF values and probability densities
x = np.array([0.0, 1.0, 2.0])
px = np.array([0.0, 1.0, 0.0])


# Tests
class TestDetermineIfScalingAppropriate:
    """
    pdf.unit is None -> False

    unit_out is None -> False

    base units differ (e.g. pdf.unit="y", unit_out="m") -> False

    same scale, same base (e.g. pdf.unit="m", unit_out="m") -> False

    genuine scale change (pdf.unit="m", unit_out="km") -> True
    """
    @pytest.mark.parametrize(
        "pdf_unit, unit_out, expected_bool",
        [
            (None, "y", False),
            ("ky", None, False),
            ("ky", "m", False),
            ("ky", "ky", False),
            ("ky", "y", True),
        ]
    )
    def test_scaling_appropriateness_decision_table(
        self, pdf_unit, unit_out, expected_bool
    ):
        pdf = PDFs.PDF(x=x, px=px, unit=pdf_unit)
        assert PDFs.scaling.determine_if_scaling_appropriate(
            pdf=pdf, unit_out=unit_out
        ) == expected_bool


class TestScalePdfByUnits:
    """
    inappropriate scaling returns a distinct-but-equal-content copy

    appropriate scaling produces correctly-scaled
    """
    def test_inappropriate_scaling(self):
        pdf = PDFs.PDF(x=x, px=px, unit="ky")
        scaled_pdf = PDFs.scaling.scale_pdf_by_units(pdf=pdf, unit_out="ky")
        assert scaled_pdf is not pdf
        assert scaled_pdf.unit == pdf.unit
        np.testing.assert_allclose(scaled_pdf.x, pdf.x)
        np.testing.assert_allclose(scaled_pdf.px, pdf.px)

    def test_appropriate_scaling(self):
        pdf = PDFs.PDF(x=x, px=px, unit="ky")
        scaled_pdf = PDFs.scaling.scale_pdf_by_units(pdf=pdf, unit_out="y")
        assert scaled_pdf.unit == "y"
        np.testing.assert_allclose(
            scaled_pdf.x, np.array([0_000.0, 1_000.0, 2_000.0])
        )


# end of file
