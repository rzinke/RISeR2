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
class TestPoolVariables:
    def test_pool_known_variables(self):
        """Test that the expected distribution is produced when known
        distributions are provided.
        """
        x = PDFs.value_arrays.precise_array(0.0, 8.0, 1.0)
        px1 = PDFs.parametric_functions.triangular(x=x, a=0.0, c=1.0, b=2.0)
        pdf1 = PDFs.PDF(x=x, px=px1)
        px2 = PDFs.parametric_functions.triangular(x=x, a=3.0, c=4.0, b=5.0)
        pdf2 = PDFs.PDF(x=x, px=px2)
        px3 = PDFs.parametric_functions.triangular(x=x, a=6.0, c=7.0, b=8.0)
        pdf3 = PDFs.PDF(x=x, px=px3)

        pdf_pooled = var_fcns.pool.pool_variables([pdf1, pdf2, pdf3])

        np.testing.assert_allclose(
            pdf_pooled.px,
            np.array([0.0, 1/3, 0.0, 0.0, 1/3, 0.0, 0.0, 1/3, 0.0])
        )

    def test_rejects_mismatched_sampling(self):
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
            var_fcns.pool.pool_variables([pdf1, pdf2])

    @pytest.mark.parametrize(
        "vartype1, unit1, vartype2, unit2",
        [
            ("age", "y", "age", "m"),
            ("age", "y", "displacement", "y"),
        ],
    )
    def test_different_metadata_warn(self, vartype1, unit1, vartype2, unit2):
        """Adding variables of fundamentally different types or units is not
        consistent with physics.
        However, variable types are not strictly enforced in this library, so
        only a warning should be raised.
        """
        x = PDFs.value_arrays.precise_array(-20.0, 20.0, 0.01)

        px1 = PDFs.parametric_functions.gaussian(x, mu=2.0, sigma=1.5)
        pdf1 = PDFs.PDF(x=x, px=px1, variable_type=vartype1, unit=unit1)

        px2 = PDFs.parametric_functions.gaussian(x, mu=-1.0, sigma=2.0)
        pdf2 = PDFs.PDF(x=x, px=px2, variable_type=vartype2, unit=unit2)

        with pytest.warns(UserWarning):
            var_fcns.pool.pool_variables([pdf1, pdf2])

    def test_name(self, recwarn):
        """Different variables are expected to have different names, so no
        warning should be raised when different names are passed.
        Explicitly passing a name to `add_variables` should asribe that name
        to the resulting PDF.
        """
        x = PDFs.value_arrays.precise_array(-20.0, 20.0, 0.01)
        vartype = "age"
        unit = "y"

        px1 = PDFs.parametric_functions.gaussian(x, mu=2.0, sigma=1.5)
        pdf1 = PDFs.PDF(
            x=x,
            px=px1,
            name="X1",
            variable_type=vartype,
            unit=unit,
        )

        px2 = PDFs.parametric_functions.gaussian(x, mu=-1.0, sigma=2.0)
        pdf2 = PDFs.PDF(
            x=x,
            px=px2,
            name="X2",
            variable_type=vartype,
            unit=unit,
        )

        pdf_sum = var_fcns.pool.pool_variables([pdf1, pdf2], name="X12")

        assert len(recwarn) == 0
        assert pdf_sum.name == "X12"


# end of file
