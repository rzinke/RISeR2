# -*- coding: utf-8 -*-
#
# Copyright (c) 2025, 2026 Robert Zinke. Licensed under the MIT License.

# Import modules
import numpy as np
import pytest
import scipy.stats as st

from riser import integration, probability_functions as PDFs


# Tests
class TestCheckMassAgainstValueRange:
    def test_mass_inside_silent(self, recwarn):
        x = np.linspace(0.0, 1.0, 11)
        PDFs.parametric_functions.check_mass_against_value_range(x, 0.2, 0.8)
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
        np.testing.assert_allclose(px, [0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0])

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
        np.testing.assert_allclose(Px, [0.0, 0.0, 0.25, 0.50, 0.75, 1.00, 1.00])

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
            PDFs.parametric_functions.trapezoidal(x, a=2.0, b=1.0, c=5.0, d=7.0)


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
        x = np.unique(
            np.concatenate(
                [
                    [-1.0, 0.0, scale, 2 * scale],
                    np.linspace(xmin, xmax, 20),
                ]
            )
        )
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
        x = np.unique(
            np.concatenate(
                [
                    [-1.0, 0.0, np.exp(mu)],
                    np.linspace(xmin, xmax, 9),
                ]
            )
        )
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
        Px = PDFs.parametric_functions.cumulative_lognormal(
            x, mu=mu, sigma=sigma
        )
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


class TestDelta:
    def test_on_grid_puts_all_density_on_one_sample(self):
        x = PDFs.value_arrays.precise_array(4.0, 6.0, 0.5)
        px = PDFs.parametric_functions.delta(x, 5.0)

        assert np.count_nonzero(px) == 1
        assert px[x == 5.0] == pytest.approx(1 / 0.5)

    def test_off_grid_splits_density_by_proximity(self):
        """The closer sample receives the larger share of the probability."""
        x = np.linspace(0.0, 4.0, 5)
        px = PDFs.parametric_functions.delta(x, 2.25)

        np.testing.assert_allclose(px, [0.0, 0.0, 0.75, 0.25, 0.0])

    @pytest.mark.parametrize("x0", [5.0, 5.1, 5.25, 5.4999, 7.3])
    def test_unit_area_and_mean(self, x0):
        """The mean is `x0` whether or not `x0` lies on the grid."""
        x = PDFs.value_arrays.precise_array(2.0, 10.0, 0.5)
        px = PDFs.parametric_functions.delta(x, x0)

        assert np.count_nonzero(px) <= 2
        mean = integration.integrate(x=x, px=x * px)
        assert integration.integrate(x=x, px=px) == pytest.approx(1.0)
        assert mean == pytest.approx(x0, abs=1e-9)

        pdf = PDFs.PDF(x, px)
        assert PDFs.analytics.pdf_mean(pdf) == pytest.approx(x0, abs=1e-9)

    def test_spread_within_one_sample_spacing(self):
        x = PDFs.value_arrays.precise_array(2.0, 10.0, 0.5)
        pdf = PDFs.PDF(x, PDFs.parametric_functions.delta(x, 5.2))

        assert PDFs.analytics.pdf_std(pdf) < 0.5

    def test_resampled_pdf_keeps_mean(self):
        """Resampling onto a finer common grid does not shift the spike."""
        x = PDFs.value_arrays.precise_array(119.0, 121.0, 0.05)
        pdf = PDFs.PDF(x, PDFs.parametric_functions.delta(x, 119.63))

        x_fine = PDFs.value_arrays.precise_array(100.0, 140.0, 0.01)
        broad_pdf = PDFs.PDF(
            x_fine, PDFs.parametric_functions.gaussian(x_fine, 120.0, 5.0)
        )

        resampled, _ = PDFs.interpolation.interpolate_pdfs([pdf, broad_pdf])

        assert PDFs.analytics.pdf_mean(resampled) == pytest.approx(
            119.63, abs=1e-3
        )

    @pytest.mark.parametrize(
        "x0, dx", [(0.3, 0.1), (2.3, 0.1), (2.3, 1.0), (119.6, 0.01)]
    )
    def test_suggested_limits_grid_accepted(self, x0, dx):
        """Rounding of the value array must not fail the domain check."""
        xmin, xmax = PDFs.parametric_functions.determine_min_max_limits(
            "delta", [x0], dx=dx
        )
        x = PDFs.value_arrays.precise_array(xmin, xmax, dx)

        px = PDFs.parametric_functions.delta(x, x0)

        assert len(x) == 3
        assert np.count_nonzero(px) == 1

    @pytest.mark.parametrize("x0", [-1.0, 0.0, 0.25, 10.0, 11.0])
    def test_too_close_to_domain_edge_raises(self, x0):
        x = PDFs.value_arrays.precise_array(0.0, 10.0, 0.5)

        with pytest.raises(ValueError, match="inside the domain"):
            PDFs.parametric_functions.delta(x, x0)

    @pytest.mark.parametrize("x0", [0.5, 9.5])
    def test_one_sample_inside_domain_allowed(self, x0):
        x = PDFs.value_arrays.precise_array(0.0, 10.0, 0.5)

        px = PDFs.parametric_functions.delta(x, x0)

        assert integration.integrate(x=x, px=x * px) == pytest.approx(x0)


class TestCumulativeDelta:
    def test_step_at_x0(self):
        x = np.linspace(0.0, 10.0, 11)
        Px = PDFs.parametric_functions.cumulative_delta(x, 4.0)

        np.testing.assert_array_equal(Px[x < 4.0], 0.0)
        np.testing.assert_array_equal(Px[x >= 4.0], 1.0)

    def test_off_grid_step(self):
        x = np.linspace(0.0, 10.0, 11)
        Px = PDFs.parametric_functions.cumulative_delta(x, 4.5)

        np.testing.assert_array_equal(Px, [0.0] * 5 + [1.0] * 6)


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

    def test_delta(self):
        xmin, xmax = PDFs.parametric_functions.determine_min_max_limits(
            "delta", [5.0], dx=0.01
        )
        assert (xmin, xmax) == pytest.approx((4.99, 5.01))

    def test_delta_requires_dx(self):
        with pytest.raises(ValueError, match="must be specified"):
            PDFs.parametric_functions.determine_min_max_limits("delta", [5.0])


    def test_unsupported_function_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            PDFs.parametric_functions.determine_min_max_limits(
                "students_t", [10.0, 0.0, 1.0]
            )


# end of file
