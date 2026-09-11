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
class TestConstrainAbove:
    def test_known_result(self):
        x = PDFs.value_arrays.precise_array(0.0, 1.0, 1E-7)
        px = PDFs.parametric_functions.uniform(x=x, a=0.0, b=1.0)
        pdf = PDFs.PDF(x=x, px=px)

        pdf_constr, area = var_fcns.condition.self_constraint.constrain_above(
            pdf, 0.5
        )

        np.testing.assert_allclose(pdf_constr.px[pdf_constr.x < 0.5], 0.0)
        assert area == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "name, vartype, unit, name_expected",
        [
            (None, None, None, None),
            ("X", "age", "y", "X constr"),
        ],
    )
    def test_metadata_consistent(self, name, vartype, unit, name_expected):
        x = PDFs.value_arrays.precise_array(0.0, 1.0, 0.01)
        px = PDFs.parametric_functions.uniform(x=x, a=0.0, b=1.0)
        pdf = PDFs.PDF(x, px, name=name, variable_type=vartype, unit=unit)

        pdf_constr, _ = var_fcns.condition.self_constraint.constrain_above(
            pdf, 0.5
        )

        assert pdf_constr.name == name_expected
        assert pdf_constr.variable_type == vartype
        assert pdf_constr.unit == unit

    def test_name_override(self):
        x = PDFs.value_arrays.precise_array(0.0, 1.0, 0.01)
        px = PDFs.parametric_functions.uniform(x=x, a=0.0, b=1.0)
        pdf = PDFs.PDF(x, px, name="X", variable_type="age", unit="y")

        pdf_constr, _ = var_fcns.condition.self_constraint.constrain_above(
            pdf, 0.5, name="custom"
        )

        assert pdf_constr.name == "custom"
        assert pdf_constr.variable_type == "age"
        assert pdf_constr.unit == "y"


class TestConstrainBelow:
    def test_known_result(self):
        x = PDFs.value_arrays.precise_array(0.0, 1.0, 1E-7)
        px = PDFs.parametric_functions.uniform(x=x, a=0.0, b=1.0)
        pdf = PDFs.PDF(x=x, px=px)

        pdf_constr, area = var_fcns.condition.self_constraint.constrain_below(
            pdf, 0.5
        )

        np.testing.assert_allclose(pdf_constr.px[pdf_constr.x > 0.5], 0.0)
        assert area == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "name, vartype, unit, name_expected",
        [
            (None, None, None, None),
            ("X", "age", "y", "X constr"),
        ],
    )
    def test_metadata_consistent(self, name, vartype, unit, name_expected):
        x = PDFs.value_arrays.precise_array(0.0, 1.0, 0.01)
        px = PDFs.parametric_functions.uniform(x=x, a=0.0, b=1.0)
        pdf = PDFs.PDF(x, px, name=name, variable_type=vartype, unit=unit)

        pdf_constr, _ = var_fcns.condition.self_constraint.constrain_above(
            pdf, 0.5
        )

        assert pdf_constr.name == name_expected
        assert pdf_constr.variable_type == vartype
        assert pdf_constr.unit == unit

    def test_name_override(self):
        x = PDFs.value_arrays.precise_array(0.0, 1.0, 0.01)
        px = PDFs.parametric_functions.uniform(x=x, a=0.0, b=1.0)
        pdf = PDFs.PDF(x, px, name="X", variable_type="age", unit="y")

        pdf_constr, _ = var_fcns.condition.self_constraint.constrain_above(
            pdf, 0.5, name="custom"
        )

        assert pdf_constr.name == "custom"
        assert pdf_constr.variable_type == "age"
        assert pdf_constr.unit == "y"


# end of file
