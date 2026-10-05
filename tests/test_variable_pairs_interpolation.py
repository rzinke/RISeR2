# Import modules
import numpy as np
import pytest

from riser import (
    probability_functions as PDFs,
    variable_pairs,
)


# Variable pairs
def _differently_sampled_pairs_():
    """
    Create two variable pairs whose PDFs differ in domain and sampling.
    """
    A_pdf1 = PDFs.PDF(
        x=np.array([9.5, 10.5, 11.5]),
        px=np.array([0.0, 1.0, 0.0]),
    )

    x = np.arange(12.0, 14.65, 0.1)
    A_pdf2 = PDFs.PDF(
        x=x,
        px=PDFs.parametric_functions.triangular(x, 12.0, 13.5, 14.6),
    )

    x = PDFs.value_arrays.precise_array(15.0, 65.0, 1.0)
    B_pdf1 = PDFs.PDF(
        x=x,
        px=PDFs.parametric_functions.gaussian(x, 20.0, 1.0),
    )

    x = PDFs.value_arrays.precise_array(15.0, 65.0, 0.1)
    B_pdf2 = PDFs.PDF(
        x=x,
        px=PDFs.parametric_functions.gaussian(x, 45.0, 2.0),
    )

    return {
        "A": variable_pairs.VariablePair(pdf1=A_pdf1, pdf2=A_pdf2),
        "B": variable_pairs.VariablePair(pdf1=B_pdf1, pdf2=B_pdf2),
    }


# Tests
class TestVariablePairsInterpolation:
    def test_pdfs_share_axes(self):
        """
        Variable pairs with differently sampled PDFs should come back with
        all `pdf1`s on one axis and all `pdf2`s on another.
        """
        pairs = _differently_sampled_pairs_()

        interp_pairs = variable_pairs.interpolation.interpolate_variable_pairs(
            pairs
        )

        for attr in ("pdf1", "pdf2"):
            axes = [getattr(pair, attr).x for pair in interp_pairs.values()]
            for x in axes[1:]:
                np.testing.assert_allclose(x, axes[0])

    def test_axes_span_all_inputs(self):
        """
        The common axes should cover the union of the input domains, and keep
        the sampling of the finest input where that sampling is uniform.
        """
        pairs = _differently_sampled_pairs_()

        interp_pairs = variable_pairs.interpolation.interpolate_variable_pairs(
            pairs
        )

        # pdf1: 9.5 (A) to 65.0 (B)
        x1 = interp_pairs["A"].pdf1.x
        assert x1[0] == pytest.approx(9.5)
        assert x1[-1] == pytest.approx(65.0)

        # pdf2: 12.0 (A) to 65.0 (B), both sampled at 0.1
        x2 = interp_pairs["A"].pdf2.x
        assert x2[0] == pytest.approx(12.0)
        assert x2[-1] == pytest.approx(65.0)
        np.testing.assert_allclose(np.diff(x2), 0.1)

    def test_well_sampled_pdfs_are_preserved(self):
        """
        Interpolation should not move well-sampled PDFs.
        """
        pairs = _differently_sampled_pairs_()

        interp_pairs = variable_pairs.interpolation.interpolate_variable_pairs(
            pairs
        )

        for name, attr in (("A", "pdf2"), ("B", "pdf1"), ("B", "pdf2")):
            original = getattr(pairs[name], attr)
            interpolated = getattr(interp_pairs[name], attr)

            assert PDFs.analytics.pdf_mean(interpolated) == pytest.approx(
                PDFs.analytics.pdf_mean(original), abs=0.01
            )

    def test_inputs_not_modified(self):
        """
        Interpolation returns new pairs and leaves the originals alone.
        """
        pairs = _differently_sampled_pairs_()
        original_x = {name: pair.pdf1.x.copy() for name, pair in pairs.items()}

        interp_pairs = variable_pairs.interpolation.interpolate_variable_pairs(
            pairs
        )

        assert interp_pairs is not pairs
        for name, pair in pairs.items():
            assert interp_pairs[name] is not pair
            np.testing.assert_array_equal(pair.pdf1.x, original_x[name])


# end of file
