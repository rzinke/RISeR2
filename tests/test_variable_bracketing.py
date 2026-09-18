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


# Tests
class TestInferBracketed:
def test_known_bracketed(self):
    """Test the case in which two very tight uniform distributions define
    the edges of an unknown variable to be inferred by bracketing it
    between the two known distributions.

    The inferred distribution should have a mean between the two ends.
    Since pdf1 and pdf2 are each essentially point masses at -10 and 10,
    the expected bracket size E[(X2 - X1)+] is known: approximately
    10 - (-10) = 20, independent of the width of the sampling domain.
    """
    dx = 1E-5
    x = PDFs.value_arrays.precise_array(-20.0, 20.0, dx)
    px1 = PDFs.parametric_functions.uniform(x=x, a=-10.0 - dx, b=-10.0)
    pdf1 = PDFs.PDF(x, px1)
    px2 = PDFs.parametric_functions.uniform(x=x, a=10.0, b=10.0 + dx)
    pdf2 = PDFs.PDF(x, px2)

    bracketed, area = var_fcns.condition.bracketing.infer_bracketed(
        pdf1=pdf1, pdf2=pdf2
    )

    assert PDFs.analytics.pdf_mean(bracketed) == pytest.approx(0.0)
    assert (
        np.min(bracketed.x[bracketed.px > 0.0])
        == pytest.approx(-10.0 - dx)
    )
    assert (
        np.max(bracketed.x[bracketed.px > 0.0])
        == pytest.approx(10.0 + dx)
    )
    assert area == pytest.approx(20.0, abs=1e-3)

def test_area_is_domain_independent(self):
    """area represents E[(X2 - X1)+], a property of the two
    distributions themselves. It must not depend on the width of the
    (arbitrary) sampling domain used to represent them.
    """
    dx = 1E-3
    x_narrow = PDFs.value_arrays.precise_array(-10.0, 10.0, dx)
    x_wide = PDFs.value_arrays.precise_array(-50.0, 50.0, dx)

    px1_narrow = PDFs.parametric_functions.gaussian(
        x_narrow, mu=-2.0, sigma=1.0
    )
    px2_narrow = PDFs.parametric_functions.gaussian(
        x_narrow, mu=2.0, sigma=1.0
    )
    pdf1_narrow = PDFs.PDF(x_narrow, px1_narrow)
    pdf2_narrow = PDFs.PDF(x_narrow, px2_narrow)

    px1_wide = PDFs.parametric_functions.gaussian(
        x_wide, mu=-2.0, sigma=1.0
    )
    px2_wide = PDFs.parametric_functions.gaussian(
        x_wide, mu=2.0, sigma=1.0
    )
    pdf1_wide = PDFs.PDF(x_wide, px1_wide)
    pdf2_wide = PDFs.PDF(x_wide, px2_wide)

    _, area_narrow = var_fcns.condition.bracketing.infer_bracketed(
        pdf1=pdf1_narrow, pdf2=pdf2_narrow
    )
    _, area_wide = var_fcns.condition.bracketing.infer_bracketed(
        pdf1=pdf1_wide, pdf2=pdf2_wide
    )

    assert area_narrow == pytest.approx(area_wide, rel=1e-3)


    @pytest.mark.parametrize(
        "name, vartype, unit, name_expected",
        [
            (None, None, None, None),
            ("X", "age", "y", "X"),
        ],
    )
    def test_metadata_consistent(self, name, vartype, unit, name_expected):
        dx = 1E-5
        x = PDFs.value_arrays.precise_array(-20.0, 20.0, dx)
        px1 = PDFs.parametric_functions.uniform(x=x, a=-10.0 - dx, b=-10.0)
        pdf1 = PDFs.PDF(x, px1, variable_type=vartype, unit=unit)
        px2 = PDFs.parametric_functions.uniform(x=x, a=10.0, b=10.0 + dx)
        pdf2 = PDFs.PDF(x, px2, variable_type=vartype, unit=unit)

        bracketed, area = var_fcns.condition.bracketing.infer_bracketed(
            pdf1=pdf1, pdf2=pdf2, name=name
        )

        assert bracketed.name == name_expected
        assert bracketed.variable_type == vartype
        assert bracketed.unit == unit


# end of file
