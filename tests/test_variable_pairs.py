# -*- coding: utf-8 -*-
#
# Copyright (c) 2025-2026 Robert Zinke. Licensed under the MIT License.

# Import modules
import numpy as np
import pytest

from riser import (
    probability_functions as PDFs,
    variable_pairs,
)


# Tests
class TestVariablePair:
    def test_construction_with_minimum_valid_inputs(self):
        x1 = PDFs.value_arrays.precise_array(0.0, 16.0, 0.01)
        px1 = PDFs.parametric_functions.gaussian(x1, mu=8.0, sigma=1.0)
        pdf1 = PDFs.PDF(x=x1, px=px1)

        x2 = PDFs.value_arrays.precise_array(0.0, 50.0, 0.01)
        px2 = PDFs.parametric_functions.triangular(x2, a=35.0, c=41.0, b=45.0)
        pdf2 = PDFs.PDF(x=x2, px=px2)

        var_pair = variable_pairs.VariablePair(pdf1=pdf1, pdf2=pdf2)

    def test_invalid_inputs_raise(self):
        x1 = PDFs.value_arrays.precise_array(0.0, 16.0, 0.01)
        px1 = PDFs.parametric_functions.gaussian(x1, mu=8.0, sigma=1.0)
        pdf1 = PDFs.PDF(x=x1, px=px1)

        x2 = PDFs.value_arrays.precise_array(0.0, 50.0, 0.01)
        px2 = PDFs.parametric_functions.triangular(x2, a=35.0, c=41.0, b=45.0)
        pdf2 = PDFs.PDF(x=x2, px=px2)

        with pytest.raises(TypeError, match="Variable `pdf1`"):
            variable_pairs.VariablePair(pdf1=np.arange(3), pdf2=pdf2)

        with pytest.raises(TypeError, match="Variable `pdf2`"):
            variable_pairs.VariablePair(pdf1=pdf1, pdf2="x2")

    def test_variable_pair_attributes_different_from_originals(self):
        x1 = PDFs.value_arrays.precise_array(0.0, 16.0, 0.01)
        px1 = PDFs.parametric_functions.gaussian(x1, mu=8.0, sigma=1.0)
        pdf1 = PDFs.PDF(x=x1, px=px1)

        x2 = PDFs.value_arrays.precise_array(0.0, 50.0, 0.01)
        px2 = PDFs.parametric_functions.triangular(x2, a=35.0, c=41.0, b=45.0)
        pdf2 = PDFs.PDF(x=x2, px=px2)

        var_pair = variable_pairs.VariablePair(pdf1=pdf1, pdf2=pdf2)

        assert var_pair.pdf1 is not pdf1
        assert var_pair.pdf1.x is not pdf1.x
        assert var_pair.pdf1.px is not pdf1.px

        assert var_pair.pdf2 is not pdf2
        assert var_pair.pdf2.x is not pdf2.x
        assert var_pair.pdf2.px is not pdf2.px


class TestDatedMarker:
    def test_valid_age_displacement_inputs_silent(self):
        age_pdf = PDFs.PDF(
            x=np.array([0.0, 1.0, 2.0]), px=np.array([0.0, 1.0, 0.0]),
            name="x_age", variable_type="age", unit="y",
        )

        disp_pdf = PDFs.PDF(
            x=np.array([9.0, 11.0, 12.0]), px=np.array([0.0, 1.0, 0.0]),
            name="x_disp", variable_type="displacement", unit="m"
        )

        var_pair = variable_pairs.DatedMarker(
            age=age_pdf, displacement=disp_pdf
        )

    def test_age_unit_present_but_wrong_base_raises(self):
        age_pdf = PDFs.PDF(
            x=np.array([0.0, 1.0, 2.0]), px=np.array([0.0, 1.0, 0.0]),
            name="x_age", variable_type="age", unit="m",
        )

        disp_pdf = PDFs.PDF(
            x=np.array([9.0, 11.0, 12.0]), px=np.array([0.0, 1.0, 0.0]),
            name="x_disp", variable_type="displacement", unit="m"
        )

        with pytest.raises(ValueError, match="Age base unit must be"):
            var_pair = variable_pairs.DatedMarker(
                age=age_pdf, displacement=disp_pdf
            )

    def test_displacement_unit_present_but_wrong_base_raises(self):
        age_pdf = PDFs.PDF(
            x=np.array([0.0, 1.0, 2.0]), px=np.array([0.0, 1.0, 0.0]),
            name="x_age", variable_type="age", unit="y",
        )

        disp_pdf = PDFs.PDF(
            x=np.array([9.0, 11.0, 12.0]), px=np.array([0.0, 1.0, 0.0]),
            name="x_disp", variable_type="displacement", unit="y"
        )

        with pytest.raises(ValueError, match="Displacement base unit must be"):
            var_pair = variable_pairs.DatedMarker(
                age=age_pdf, displacement=disp_pdf
            )

    def test_None_unit_warns_but_does_not_raise(self, recwarn):
        age_pdf = PDFs.PDF(
            x=np.array([0.0, 1.0, 2.0]), px=np.array([0.0, 1.0, 0.0])
        )

        disp_pdf = PDFs.PDF(
            x=np.array([9.0, 11.0, 12.0]), px=np.array([0.0, 1.0, 0.0])
        )

        marker = variable_pairs.DatedMarker(age=age_pdf, displacement=disp_pdf)

        assert len(recwarn) > 0

    def test_properties_alias_properly(self):
        age_pdf = PDFs.PDF(
            x=np.array([0.0, 1.0, 2.0]), px=np.array([0.0, 1.0, 0.0]),
            variable_type="age", unit="y",
        )

        disp_pdf = PDFs.PDF(
            x=np.array([9.0, 11.0, 12.0]), px=np.array([0.0, 1.0, 0.0]),
            variable_type="displacement", unit="m",
        )

        marker = variable_pairs.DatedMarker(age=age_pdf, displacement=disp_pdf)

        age_pdf2 = PDFs.PDF(
            x=np.array([3.0, 4.0, 5.0]), px=np.array([0.0, 1.0, 0.0]),
            variable_type="age", unit="y",
        )

        marker.age = age_pdf2

        np.testing.assert_allclose(marker.pdf1.x, age_pdf2.x)
        np.testing.assert_allclose(marker.pdf1.px, age_pdf2.px)

        disp_pdf2 = PDFs.PDF(
            x=np.array([3.0, 4.0, 5.0]), px=np.array([0.0, 1.0, 0.0]),
            variable_type="displacement", unit="m",
        )

        marker.displacement = disp_pdf2

        np.testing.assert_allclose(marker.pdf2.x, disp_pdf2.x)
        np.testing.assert_allclose(marker.pdf2.px, disp_pdf2.px)


# end of file
