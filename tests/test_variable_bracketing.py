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
        """Test the chase in which two very tight uniform distributions define
        the edges of an unknown variable to be inferred by bracketing it 
        between the two known distributions.

        The inferred distribution should have a mean between the two ends.
        The pre-normalization area should be known: a raw weight of 1.0 over
        a width of 20 within a domain of 40 is 0.5.
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
        assert area == pytest.approx(0.5)

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
