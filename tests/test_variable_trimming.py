# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

# Import modules
import numpy as np
import pytest
import scipy as sp

from riser import (
    probability_functions as PDFs,
    variable_functions as var_fcns,
)


# Test
class TestTrimVariables:
    def test_known_solution(self):
        """Test that two uniform distributions trim each other.

        Leverage the fact that if two PDFs are identically distributed,
        symmetry dictates that P(pdf1 < pdf2) = P(pdf2 > pdf1) = 0.5.

        Test the application of the definition of the trimming routine.
        """
        x = PDFs.value_arrays.precise_array(0.0, 1.0, 1E-5)
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
            ("X1", "X2", "age", "y", "X1 trimmed", "X2 trimmed"),
        ],
    )
    def test_metadata_consistent(
        self, name1, name2, vartype, unit, name1_expected, name2_expected
    ):
        x = PDFs.value_arrays.precise_array(0.0, 1.0, 1E-5)
        px1 = PDFs.parametric_functions.uniform(x=x, a=0.0, b=1.0)
        pdf1 = PDFs.PDF(x, px1, name=name1, variable_type=vartype, unit=unit)
        px2 = PDFs.parametric_functions.uniform(x=x, a=0.0, b=1.0)
        pdf2 = PDFs.PDF(x, px2, name=name2, variable_type=vartype, unit=unit)

        (
            pdf1_trimmed,
            pdf2_trimmed,
            area,
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

        x = PDFs.value_arrays.precise_array(0.0, 1.0, 1E-5)
        px1 = PDFs.parametric_functions.uniform(x=x, a=0.0, b=1.0)
        pdf1 = PDFs.PDF(x, px1, name=None, variable_type=vartype, unit=unit)
        px2 = PDFs.parametric_functions.uniform(x=x, a=0.0, b=1.0)
        pdf2 = PDFs.PDF(x, px2, name=None, variable_type=vartype, unit=unit)

        (
            pdf1_trimmed,
            pdf2_trimmed,
            area,
        ) = var_fcns.condition.trimming.trim_variables(
            pdf1, pdf2, name1="X1 trimmed", name2="X2 trimmed"
        )
        
        assert pdf1_trimmed.name == "X1 trimmed"
        assert pdf1_trimmed.variable_type == vartype
        assert pdf1_trimmed.unit == unit

        assert pdf2_trimmed.name == "X2 trimmed"
        assert pdf2_trimmed.variable_type == vartype
        assert pdf2_trimmed.unit == unit


# end of file
