# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

# Import modules
import numpy as np
import scipy.stats as st
import pytest

from riser import probability_functions as PDFs


# Tests
class TestCheckMassAgainstValueRange:
    def test_mass_inside_silent(self, recwarn):
        x = np.linspace(0.0, 1.0, 11)
        PDFs.parametric_functions.check_mass_against_value_range(
            x, 0.2, 0.8
        )
        assert len(recwarn) == 0

    def test_mass_outside_warns(self):
        x = np.linspace(0.0, 1.0, 11)
        with pytest.warns(
            UserWarning, match="Significant probability lies below"
        ):
            PDFs.parametric_functions.check_mass_against_value_range(
                x, -0.5, 0.5
            )
        with pytest.warns(
            UserWarning, match="Significant probability lies above"
        ):
            PDFs.parametric_functions.check_mass_against_value_range(
                x, 0.5, 1.5
            )
        with pytest.warns(
            UserWarning, match="Significant probability lies outside"
        ):
            PDFs.parametric_functions.check_mass_against_value_range(
                x, -0.5, 1.5
            )


class TestUniform:
    def test_uniform_shape(self):
        x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        px = PDFs.parametric_functions.uniform(x, a=0.0, b=4.0)
        np.testing.assert_allclose(
            px, [0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0]
        )

    @pytest.mark.parametrize("n_points", [5, 9, 13])
    def test_area_resolution_independent(self, n_points):
        x = np.linspace(0.0, 4.0, n_points)
        px = PDFs.parametric_functions.uniform(x, a=0.0, b=4.0)
        area = np.trapezoid(px, x)
        assert area == pytest.approx(1.0)


class TestTriangular:
    def test_triangular_shape(self):
        x = np.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
        px = PDFs.parametric_functions.triangular(x, a=0.0, c=1.0, b=2.0)
        np.testing.assert_allclose(
            px, np.array([0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0])
        )

    @pytest.mark.parametrize("n_points", [5, 9, 17])
    def test_area_exact_when_grid_includes_peak(self, n_points):
        x = np.linspace(0.0, 2.0, n_points)
        px = PDFs.parametric_functions.triangular(x, a=0.0, c=1.0, b=2.0)
        area = np.trapezoid(px, x)
        assert area == pytest.approx(1.0)

    def test_area_approximate_when_peak_not_on_grid(self):
        x = np.linspace(0.0, 2.0, 10)
        px = PDFs.parametric_functions.triangular(x, a=0.0, c=1.0, b=2.0)
        area = np.trapezoid(px, x)
        assert area == pytest.approx(1.0, abs=0.02)


class TestCumulativeUniform:
    def test_cumulative_uniform_shape(self):
        x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        Px = PDFs.parametric_functions.cumulative_uniform(x, a=0.0, b=4.0)
        np.testing.assert_allclose(
            Px, [0.0, 0.0, 0.25, 0.50, 0.75, 1.00, 1.00]
        )

    def test_cumulative_uniform_is_monotonic(self):
        x = np.linspace(-2.0, 6.0, 41)
        Px = PDFs.parametric_functions.cumulative_uniform(x, a=0.0, b=4.0)
        assert np.all(np.diff(Px) >= 0)


class TestCumulativeTriangular:
    def test_cumulative_triangular_shape(self):
        x = np.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
        Px = PDFs.parametric_functions.cumulative_triangular(
            x, a=0.0, c=1.0, b=2.0
        )
        np.testing.assert_allclose(
            Px, np.array([0.0, 0.0, 0.125, 0.5, 0.875, 1.0, 1.0])
        )

    def test_cumulative_triangular_is_monotonic(self):
        x = np.linspace(-2.0, 6.0, 41)
        Px = PDFs.parametric_functions.cumulative_triangular(
            x, a=0.0, c=1.0, b=2.0
        )
        assert np.all(np.diff(Px) >= 0)


class TestTrapezoidal:
    def test_trapezoidal_shape(self):
        x = np.array([-1.0, 0.0, 1.0, 2.0, 3.5, 5.0, 6.0, 7.0, 8.0])
        px = PDFs.parametric_functions.trapezoidal(
            x, a=0.0, b=2.0, c=5.0, d=7.0
        )
        np.testing.assert_allclose(
            px, [0.0, 0.0, 0.1, 0.2, 0.2, 0.2, 0.1, 0.0, 0.0]
        )

    @pytest.mark.parametrize("n_points", [8, 15, 29])
    def test_area_exact_when_grid_includes_corners(self, n_points):
        x = np.linspace(0.0, 7.0, n_points)
        px = PDFs.parametric_functions.trapezoidal(
            x, a=0.0, b=2.0, c=5.0, d=7.0
        )
        area = np.trapezoid(px, x)
        assert area == pytest.approx(1.0)

    def test_invalid_ordering_raises(self):
        x = np.linspace(0.0, 7.0, 8)
        with pytest.raises(ValueError, match="must be <="):
            PDFs.parametric_functions.trapezoidal(
                x, a=2.0, b=1.0, c=5.0, d=7.0
            )


class TestCumulativeTrapezoidal:
    def test_cumulative_trapezoidal_shape(self):
        x = np.array([-1.0, 0.0, 1.0, 2.0, 3.5, 5.0, 6.0, 7.0, 8.0])
        Px = PDFs.parametric_functions.cumulative_trapezoidal(
            x, a=0.0, b=2.0, c=5.0, d=7.0
        )
        np.testing.assert_allclose(
            Px, [0.0, 0.0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0, 1.0]
        )

    def test_cumulative_trapezoidal_is_monotonic(self):
        x = np.linspace(-2.0, 9.0, 41)
        Px = PDFs.parametric_functions.cumulative_trapezoidal(
            x, a=0.0, b=2.0, c=5.0, d=7.0
        )
        assert np.all(np.diff(Px) >= 0)

    def test_stays_at_one_beyond_d(self):
        # the specific regression case: without the far_ndx fix, this
        # silently dropped back to 0.0 instead of staying at 1.0
        x = np.array([8.0, 10.0, 100.0])
        Px = PDFs.parametric_functions.cumulative_trapezoidal(
            x, a=0.0, b=2.0, c=5.0, d=7.0
        )
        np.testing.assert_allclose(Px, [1.0, 1.0, 1.0])


class TestGaussian:
    def test_peak_height_matches_analytical_formula(self):
        mu, sigma = 5.0, 2.0
        xmin, xmax = PDFs.parametric_functions._gaussian_limits_(mu, sigma)
        x = np.linspace(xmin, xmax, 9)
        mu_ndx = np.argmin(np.abs(x - mu))
        px = PDFs.parametric_functions.gaussian(x, mu=mu, sigma=sigma)
        expected = 1 / (sigma * np.sqrt(2 * np.pi))
        assert px[mu_ndx] == pytest.approx(expected)

    def test_symmetric_about_mu(self):
        mu, sigma = 5.0, 2.0
        xmin, xmax = PDFs.parametric_functions._gaussian_limits_(mu, sigma)
        x = np.linspace(xmin, xmax, 17)
        ndx_minus = np.argmin(np.abs(x - (mu - 3.0)))
        ndx_plus = np.argmin(np.abs(x - (mu + 3.0)))
        px = PDFs.parametric_functions.gaussian(x, mu=mu, sigma=sigma)
        assert px[ndx_minus] == pytest.approx(px[ndx_plus])


class TestCumulativeGaussian:
    def test_matches_scipy_reference(self):
        mu, sigma = 5.0, 2.0
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 9)
        Px = PDFs.parametric_functions.cumulative_gaussian(
            x, mu=mu, sigma=sigma
        )
        Px_scipy = st.norm.cdf(x, loc=mu, scale=sigma)
        np.testing.assert_allclose(Px, Px_scipy)

    def test_is_monotonic(self):
        mu, sigma = 5.0, 2.0
        x = np.linspace(mu - 6 * sigma, mu + 6 * sigma, 41)
        Px = PDFs.parametric_functions.cumulative_gaussian(
            x, mu=mu, sigma=sigma
        )
        assert np.all(np.diff(Px) >= 0)


class TestExponential:
    def test_exponential_shape(self):
        scale = 2.0
        xmin, xmax = PDFs.parametric_functions._exponential_limits_(scale)
        x = np.unique(np.concatenate([
            [-1.0, 0.0, scale, 2 * scale],
            np.linspace(xmin, xmax, 20),
        ]))
        ndx_neg1 = np.argmin(np.abs(x - (-1.0)))
        ndx_0 = np.argmin(np.abs(x - 0.0))
        ndx_s = np.argmin(np.abs(x - scale))
        ndx_2s = np.argmin(np.abs(x - 2 * scale))
        px = PDFs.parametric_functions.exponential(x, scale=scale)
        np.testing.assert_allclose(
            [px[ndx_neg1], px[ndx_0], px[ndx_s], px[ndx_2s]],
            [
                0.0,
                1 / scale,
                (1 / scale) * np.exp(-1),
                (1 / scale) * np.exp(-2),
            ],
        )


class TestLognormal:
    def test_lognormal_shape(self):
        mu, sigma = 1.0, 0.5
        xmin, xmax = PDFs.parametric_functions._lognormal_limits_(mu, sigma)
        x = np.unique(np.concatenate([
            [-1.0, 0.0, np.exp(mu)],
            np.linspace(xmin, xmax, 9),
        ]))
        px = PDFs.parametric_functions.lognormal(x, mu=mu, sigma=sigma)

        ndx_neg = np.argmin(np.abs(x - (-1.0)))
        ndx_zero = np.argmin(np.abs(x - 0.0))
        assert px[ndx_neg] == 0.0
        assert px[ndx_zero] == 0.0

    def test_matches_scipy_reference(self):
        mu, sigma = 1.0, 0.5
        xmin, xmax = PDFs.parametric_functions._lognormal_limits_(mu, sigma)
        x = np.linspace(xmin, xmax, 9)  # was: max(xmin, 1e-6)
        px = PDFs.parametric_functions.lognormal(x, mu=mu, sigma=sigma)
        px_scipy = st.lognorm.pdf(x, s=sigma, scale=np.exp(mu))
        np.testing.assert_allclose(px, px_scipy)

    def test_area_approximately_one(self):
        mu, sigma = 1.0, 0.5
        xmin, xmax = PDFs.parametric_functions._lognormal_limits_(mu, sigma)
        x = np.linspace(xmin, xmax, 5000)
        px = PDFs.parametric_functions.lognormal(x, mu=mu, sigma=sigma)
        area = np.trapezoid(px, x)
        assert area == pytest.approx(1.0, abs=1e-3)


class TestCumulativeLognormal:
    def test_matches_scipy_reference(self):
        mu, sigma = 1.0, 0.5
        x = np.linspace(0.01, np.exp(mu + 4 * sigma), 9)
        Px = PDFs.parametric_functions.cumulative_lognormal(x, mu=mu, sigma=sigma)
        Px_scipy = st.lognorm.cdf(x, s=sigma, scale=np.exp(mu))
        np.testing.assert_allclose(Px, Px_scipy, atol=1e-10)

    def test_zero_for_nonpositive_x(self):
        # regression test: negative x previously produced NaN with a
        # raw RuntimeWarning, instead of the correct 0.0
        x = np.array([-1.0, 0.0, 1.0])
        Px = PDFs.parametric_functions.cumulative_lognormal(
            x, mu=1.0, sigma=0.5
        )
        np.testing.assert_allclose(Px[:2], [0.0, 0.0])
        assert Px[2] > 0.0


class TestStudentsT:
    def test_symmetric_and_mode_at_mu(self):
        dof, mu, scale = 10.0, 5.0, 2.0
        xmin, xmax = PDFs.parametric_functions._students_t_limits_(
           dof, mu, scale
        )
        x = np.linspace(xmin, xmax, 9)
        px = PDFs.parametric_functions.students_t(
            x, dof=dof, mu=mu, scale=scale
        )

        mode_x = x[np.argmax(px)]
        assert mode_x == pytest.approx(mu)

        ndx_minus = np.argmin(np.abs(x - (mu - 3.0)))
        ndx_plus = np.argmin(np.abs(x - (mu + 3.0)))
        assert px[ndx_minus] == pytest.approx(px[ndx_plus])

    def test_area_approximately_one(self):
        dof, mu, scale = 10.0, 5.0, 2.0
        xmin, xmax = PDFs.parametric_functions._students_t_limits_(
            dof, mu, scale
        )
        x = np.linspace(xmin, xmax, 2001)
        px = PDFs.parametric_functions.students_t(
            x, dof=dof, mu=mu, scale=scale
        )
        area = np.trapezoid(px, x)
        assert area == pytest.approx(1.0, abs=1e-3)

    def test_invalid_dof_raises(self):
        x = np.linspace(-10, 10, 9)
        with pytest.raises(ValueError, match="must be positive"):
            PDFs.parametric_functions.students_t(x, dof=-1.0, mu=0.0, scale=1.0)


class TestGetFunctionByName:
    def test_returns_correct_function(self):
        fcn = PDFs.parametric_functions.get_function_by_name("uniform")
        assert fcn is PDFs.parametric_functions.uniform

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="is not defined"):
            PDFs.parametric_functions.get_function_by_name("not_real")


class TestCheckNumberInputs:
    def test_correct_count_silent(self, recwarn):
        PDFs.parametric_functions.check_number_inputs("uniform", [0.0, 4.0])
        assert len(recwarn) == 0

    def test_wrong_count_raises(self):
        with pytest.raises(ValueError, match="must be specified"):
            PDFs.parametric_functions.check_number_inputs("uniform", [0.0])


class TestDetermineMinMaxLimits:
    def test_uniform(self):
        assert PDFs.parametric_functions.determine_min_max_limits(
            "uniform", [0.0, 4.0]
        ) == (0.0, 4.0)

    def test_gaussian(self):
        xmin, xmax = PDFs.parametric_functions.determine_min_max_limits(
            "gaussian", [5.0, 2.0]
        )
        assert (xmin, xmax) == pytest.approx((-3.0, 13.0))

    def test_gaussian_limit_positive(self):
        xmin, xmax = PDFs.parametric_functions.determine_min_max_limits(
            "gaussian", [1.0, 2.0], limit_positive=True
        )
        assert (xmin, xmax) == pytest.approx((0.0, 9.0))

    def test_unsupported_distribution_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            PDFs.parametric_functions.determine_min_max_limits(
                "students_t", [10.0, 0.0, 1.0]
            )


# end of file
