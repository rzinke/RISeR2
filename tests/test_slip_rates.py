# -*- coding: utf-8 -*-
#
# Copyright (c) 2025-2026 Robert Zinke. Licensed under the MIT License.


"""
These functions focus on orchestration of more primitive functions that are
already tested.
"""

# Import modules
import numpy as np
import pytest

from riser import (
    probability_functions as PDFs,
    sampling,
    slip_rates,
    variable_pairs,
)


# Slip Rate Tail Cap Tests
def _gaussian_pdf_(mu, sigma, xmin, xmax, dx, variable_type, unit):
    """
    Gaussian PDF on a regular axis.
    """
    x = PDFs.value_arrays.precise_array(xmin, xmax, dx)

    return PDFs.PDF(
        x=x,
        px=PDFs.parametric_functions.gaussian(x, mu=mu, sigma=sigma),
        variable_type=variable_type,
        unit=unit,
    )


def _example_disp_age_():
    """
    Displacement of 10 +/- 1 m and age of 4 +/- 1 ky, both on 0.01 grids.
    The age axis is limited to non-negative values, as required of the
    denominator.
    """
    displacement = _gaussian_pdf_(
        10.0, 1.0, 4.0, 16.0, 0.01, "displacement", "m"
    )
    age = _gaussian_pdf_(4.0, 1.0, 0.0, 8.0, 0.01, "age", "ky")

    return displacement, age


class TestFindSlipRateTailCap:
    @pytest.mark.parametrize(
        "epsilon, expected",
        [(1e-2, 6.084), (1e-3, 11.030)],
    )
    def test_known_answer(self, epsilon, expected):
        """
        Values for the example case, confirmed independently by Monte Carlo
        sampling (see `test_matches_monte_carlo`).
        """
        displacement, age = _example_disp_age_()

        v_max = slip_rates.rate_computation.find_slip_rate_tail_cap(
            displacement, age, epsilon=epsilon
        )

        assert v_max == pytest.approx(expected, rel=1e-3)

    @pytest.mark.parametrize("epsilon", [1e-2, 1e-3])
    def test_matches_monte_carlo(self, epsilon):
        """
        Independent check: the fraction of sampled positive slip rates above
        the cap should be about `epsilon`.
        """
        displacement, age = _example_disp_age_()

        v_max = slip_rates.rate_computation.find_slip_rate_tail_cap(
            displacement, age, epsilon=epsilon
        )

        rng = np.random.default_rng(0)
        n_samples = 2_000_000
        disp_samples = rng.normal(10.0, 1.0, n_samples)
        age_samples = rng.normal(4.0, 1.0, n_samples)
        valid = age_samples > 0.0
        rates = disp_samples[valid] / age_samples[valid]

        assert np.mean(rates > v_max) == pytest.approx(epsilon, rel=0.1)

    def test_smaller_epsilon_gives_larger_cap(self):
        """
        Excluding less of the tail requires a larger maximum slip rate.
        """
        displacement, age = _example_disp_age_()

        caps = [
            slip_rates.rate_computation.find_slip_rate_tail_cap(
                displacement, age, epsilon=epsilon
            )
            for epsilon in (1e-1, 1e-2, 1e-3, 1e-4)
        ]

        assert np.all(np.diff(caps) > 0.0)

    def test_scales_with_units(self):
        """
        Expressing displacement in mm instead of m should scale the cap by
        1000.
        """
        displacement, age = _example_disp_age_()
        displacement_mm = _gaussian_pdf_(
            10_000.0, 1_000.0, 4_000.0, 16_000.0, 10.0, "displacement", "mm"
        )

        v_max = slip_rates.rate_computation.find_slip_rate_tail_cap(
            displacement, age
        )
        v_max_mm = slip_rates.rate_computation.find_slip_rate_tail_cap(
            displacement_mm, age
        )

        assert v_max_mm == pytest.approx(1000.0 * v_max, rel=1e-3)

    def test_different_grid_lengths(self):
        """
        Displacement and age PDFs need not share a grid or even a length.
        Regression test for masking the age array with the displacement axis.
        """
        displacement, age = _example_disp_age_()
        age_coarse = _gaussian_pdf_(4.0, 1.0, 0.0, 8.0, 0.02, "age", "ky")
        assert len(displacement.x) != len(age_coarse.x)

        v_max = slip_rates.rate_computation.find_slip_rate_tail_cap(
            displacement, age
        )
        v_max_coarse = slip_rates.rate_computation.find_slip_rate_tail_cap(
            displacement, age_coarse
        )

        assert v_max_coarse == pytest.approx(v_max, rel=1e-3)

    def test_exact_zeros_in_displacement_density(self):
        """
        A displacement PDF padded with exact zero densities should give the
        same cap as the same PDF without the padding. Regression test for
        deriving the lower bound from the density values instead of the axis,
        which never terminated.
        """
        displacement, age = _example_disp_age_()

        padded_axis = PDFs.value_arrays.precise_array(0.0, 40.0, 0.01)
        padded_density = np.zeros_like(padded_axis)
        in_range = (padded_axis >= 4.0) & (padded_axis <= 16.0)
        padded_density[in_range] = PDFs.parametric_functions.gaussian(
            padded_axis[in_range], mu=10.0, sigma=1.0
        )
        displacement_padded = PDFs.PDF(
            x=padded_axis,
            px=padded_density,
            variable_type="displacement",
            unit="m",
        )
        assert np.sum(displacement_padded.px == 0.0) > 0

        v_max = slip_rates.rate_computation.find_slip_rate_tail_cap(
            displacement, age
        )
        v_max_padded = slip_rates.rate_computation.find_slip_rate_tail_cap(
            displacement_padded, age
        )

        assert v_max_padded == pytest.approx(v_max, rel=1e-3)

    @pytest.mark.parametrize("epsilon", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_epsilon_raises(self, epsilon):
        displacement, age = _example_disp_age_()

        with pytest.raises(ValueError, match="epsilon"):
            slip_rates.rate_computation.find_slip_rate_tail_cap(
                displacement, age, epsilon=epsilon
            )

    def test_negative_ages_raise(self):
        displacement, _ = _example_disp_age_()
        age = _gaussian_pdf_(4.0, 1.0, -2.0, 8.0, 0.01, "age", "ky")

        with pytest.raises(ValueError, match="non-negative"):
            slip_rates.rate_computation.find_slip_rate_tail_cap(
                displacement, age
            )

    def test_no_positive_displacement_raises(self):
        _, age = _example_disp_age_()
        displacement = _gaussian_pdf_(
            -10.0, 1.0, -16.0, -4.0, 0.01, "displacement", "m"
        )

        with pytest.raises(ValueError, match="no possible positive"):
            slip_rates.rate_computation.find_slip_rate_tail_cap(
                displacement, age
            )


# Markers
def _two_markers_():
    """
    Create two widely spaced displacement-age markers.
    """
    return {
        "young": variable_pairs.DatedMarker(
            age=PDFs.PDF(
                x=np.array([4.0, 5.0, 6.0]),
                px=np.array([0.0, 1.0, 0.0]),
                variable_type="age",
                unit="y",
            ),
            displacement=PDFs.PDF(
                x=np.array([9.0, 10.0, 11.0]),
                px=np.array([0.0, 1.0, 0.0]),
                variable_type="displacement",
                unit="m",
            ),
            name="young",
        ),
        "old": variable_pairs.DatedMarker(
            age=PDFs.PDF(
                x=np.array([14.0, 15.0, 16.0]),
                px=np.array([0.0, 1.0, 0.0]),
                variable_type="age",
                unit="y",
            ),
            displacement=PDFs.PDF(
                x=np.array([29.0, 30.0, 31.0]),
                px=np.array([0.0, 1.0, 0.0]),
                variable_type="displacement",
                unit="m",
            ),
            name="old",
        ),
    }


def _three_markers_():
    """
    Create three widely spaced displacement-age markers.
    """
    return {
        "young": variable_pairs.DatedMarker(
            age=PDFs.PDF(
                x=np.array([4.0, 5.0, 6.0]),
                px=np.array([0.0, 1.0, 0.0]),
                variable_type="age",
                unit="y",
            ),
            displacement=PDFs.PDF(
                x=np.array([9.0, 10.0, 11.0]),
                px=np.array([0.0, 1.0, 0.0]),
                variable_type="displacement",
                unit="m",
            ),
            name="young",
        ),
        "middle": variable_pairs.DatedMarker(
            age=PDFs.PDF(
                x=np.array([9.0, 10.0, 11.0]),
                px=np.array([0.0, 1.0, 0.0]),
                variable_type="age",
                unit="y",
            ),
            displacement=PDFs.PDF(
                x=np.array([19.0, 20.0, 21.0]),
                px=np.array([0.0, 1.0, 0.0]),
                variable_type="displacement",
                unit="m",
            ),
            name="middle",
        ),
        "old": variable_pairs.DatedMarker(
            age=PDFs.PDF(
                x=np.array([14.0, 15.0, 16.0]),
                px=np.array([0.0, 1.0, 0.0]),
                variable_type="age",
                unit="y",
            ),
            displacement=PDFs.PDF(
                x=np.array([29.0, 30.0, 31.0]),
                px=np.array([0.0, 1.0, 0.0]),
                variable_type="displacement",
                unit="m",
            ),
            name="old",
        ),
    }


def _overlapping_markers_():
    """
    Create three displacement-age markers that overlap in age and displacement.
    """
    ages = PDFs.value_arrays.precise_array(0.0, 25.0, 0.01)
    displacements = PDFs.value_arrays.precise_array(0.0, 40.0, 0.01)

    return {
        "young": variable_pairs.DatedMarker(
            age=PDFs.PDF(
                x=ages,
                px=PDFs.parametric_functions.gaussian(ages, 10.0, 2.0),
                variable_type="age",
                unit="y",
            ),
            displacement=PDFs.PDF(
                x=displacements,
                px=PDFs.parametric_functions.gaussian(displacements, 15.0, 3.0),
                variable_type="displacement",
                unit="m",
            ),
            name="young",
        ),
        "middle": variable_pairs.DatedMarker(
            age=PDFs.PDF(
                x=ages,
                px=PDFs.parametric_functions.gaussian(ages, 12.0, 2.0),
                variable_type="age",
                unit="y",
            ),
            displacement=PDFs.PDF(
                x=displacements,
                px=PDFs.parametric_functions.gaussian(displacements, 20.0, 3.0),
                variable_type="displacement",
                unit="m",
            ),
            name="middle",
        ),
        "old": variable_pairs.DatedMarker(
            age=PDFs.PDF(
                x=ages,
                px=PDFs.parametric_functions.gaussian(ages, 14.0, 2.0),
                variable_type="age",
                unit="y",
            ),
            displacement=PDFs.PDF(
                x=displacements,
                px=PDFs.parametric_functions.gaussian(displacements, 25.0, 3.0),
                variable_type="displacement",
                unit="m",
            ),
            name="old",
        ),
    }


# Helpers
def _max_cdf_gap_(pdf1, pdf2, n=5000):
    """
    Largest absolute difference between the CDFs of two PDFs, evaluated on a
    common axis. Unlike comparing `px` arrays, this does not require the PDFs
    to share a grid, and it is insensitive to histogram bin noise.
    """
    x = np.linspace(min(pdf1.x[0], pdf2.x[0]), max(pdf1.x[-1], pdf2.x[-1]), n)

    return np.max(np.abs(pdf1.cdf_at_value(x) - pdf2.cdf_at_value(x)))


# Tests
class TestComputeSlipRate:
    def test_known_answer(self):
        """
        Known-answer sanity check with tight, near-deterministic age and
        displacement PDFs.
        """
        age_mu = 10.0
        age_sigma = 0.001
        age_axis = PDFs.value_arrays.precise_array(9.9, 10.1, 0.0001)
        age_density = PDFs.parametric_functions.gaussian(
            age_axis, mu=age_mu, sigma=age_sigma
        )
        age_pdf = PDFs.PDF(
            x=age_axis,
            px=age_density,
            variable_type="age",
            unit="y",
        )

        disp_mu = 50.0
        disp_sigma = 0.001
        disp_axis = PDFs.value_arrays.precise_array(49.9, 50.1, 0.0001)
        disp_density = PDFs.parametric_functions.gaussian(
            disp_axis, mu=disp_mu, sigma=disp_sigma
        )
        disp_pdf = PDFs.PDF(
            x=disp_axis,
            px=disp_density,
            variable_type="displacement",
            unit="m",
        )

        marker = variable_pairs.DatedMarker(
            age=age_pdf, displacement=disp_pdf, name="X"
        )

        rate_pdf = slip_rates.rate_computation.compute_slip_rate(
            marker=marker,
        )

        se = 5 * np.sqrt(
            (age_sigma / age_mu) ** 2 + (disp_sigma / disp_mu) ** 2
        )
        assert PDFs.analytics.pdf_mean(rate_pdf) == pytest.approx(5, abs=2 * se)


class TestComputeSlipRatesAnalytical:
    def test_known_answer(self):
        """
        Known-answer sanity check with tight, near-deterministic age and
        displacement PDFs.
        """
        markers = _three_markers_()

        incr_rates = slip_rates.rate_computation.compute_slip_rates_analytical(
            markers=markers,
        )

        assert len(incr_rates) == 2

    def test_ordering_slip_rates_positive(self):
        """
        Enforcing ordering should result in no negative slip rates.
        """
        markers = _overlapping_markers_()

        # Ordering enforced
        incr_rates_ordering = (
            slip_rates.rate_computation.compute_slip_rates_analytical(
                markers=markers,
                enforce_ordering=True,
                limit_positive=True,
            )
        )

        # Check that all incremental slip rates have only positive values
        for incr_rate in incr_rates_ordering.values():
            assert np.min(incr_rate.x) > 0

    def test_enforce_ordering_trims_markers(self):
        """
        Enforcing ordering pushes each marker in the direction the ordering
        dictates: forward trimming can only make a marker later/more displaced
        and backward trimming only earlier/less displaced. The first marker
        is unconstrained going forward, and the last going backward.
        """
        markers = list(_overlapping_markers_().values())

        for variable in ("age", "displacement"):
            pdfs = [getattr(marker, variable) for marker in markers]
            means = [PDFs.analytics.pdf_mean(pdf) for pdf in pdfs]

            forward = slip_rates.rate_computation._forward_trim_pdfs_(pdfs)
            backward = slip_rates.rate_computation._backward_trim_pdfs_(pdfs)
            forward_means = [PDFs.analytics.pdf_mean(pdf) for pdf in forward]
            backward_means = [PDFs.analytics.pdf_mean(pdf) for pdf in backward]

            # Ends of the stack are unchanged by their own pass
            assert forward_means[0] == pytest.approx(means[0])
            assert backward_means[-1] == pytest.approx(means[-1])

            # All other markers move in the direction of the ordering
            for mean, forward_mean in zip(means[1:], forward_means[1:]):
                assert forward_mean > mean
            for mean, backward_mean in zip(means[:-1], backward_means[:-1]):
                assert backward_mean < mean

    def test_default_metadata_is_derived(self):
        """
        With no metadata overrides, variable_type defaults to "slip rate"
        and unit is derived as displacement/age from the marker metadata.
        """
        markers = _two_markers_()

        rates = slip_rates.rate_computation.compute_slip_rates_analytical(
            markers=markers
        )
        rate = next(iter(rates.values()))

        assert rate.variable_type == "slip rate"
        assert rate.unit == "m/y"

    def test_explicit_metadata_overrides_defaults(self):
        """
        Explicit variable_type/unit must reach the output PDF unchanged,
        rather than being silently replaced by the derived defaults.
        """
        markers = _two_markers_()

        rates = slip_rates.rate_computation.compute_slip_rates_analytical(
            markers=markers,
            variable_type="custom_vt",
            unit="custom_unit",
        )
        rate = next(iter(rates.values()))

        assert rate.variable_type == "custom_vt"
        assert rate.unit == "custom_unit"


class TestComputeSlipRatesMc:
    def test_known_answer(self):
        """
        Known-answer sanity check with tight, near-deterministic age and
        displacement PDFs.
        """
        markers = _three_markers_()

        criterion = sampling.mc_sampling.get_sample_criterion(
            "PassNonnegative"
        )()

        (
            incr_rates,
            age_picks,
            disp_picks,
            rate_picks,
        ) = slip_rates.rate_computation.compute_slip_rates_mc(
            markers=markers,
            criterion=criterion,
            n_samples=10_000,
        )

        assert len(incr_rates) == 2
        np.testing.assert_allclose(
            rate_picks,
            np.diff(disp_picks, axis=0) / np.diff(age_picks, axis=0),
        )

    def test_default_metadata_is_derived(self):
        """
        With no metadata overrides, variable_type defaults to "slip rate"
        and unit is derived as displacement/age from the marker metadata.
        """
        markers = _two_markers_()
        criterion = sampling.mc_sampling.get_sample_criterion(
            "PassNonnegative"
        )()

        rates, *_ = slip_rates.rate_computation.compute_slip_rates_mc(
            markers=markers,
            criterion=criterion,
            n_samples=500,
        )
        rate = next(iter(rates.values()))

        assert rate.variable_type == "slip rate"
        assert rate.unit == "m/y"

    def test_explicit_metadata_overrides_defaults(self):
        """
        Explicit variable_type/unit must reach the output PDF unchanged,
        rather than being silently replaced by the derived defaults.
        """
        markers = _two_markers_()
        criterion = sampling.mc_sampling.get_sample_criterion(
            "PassNonnegative"
        )()

        rates, *_ = slip_rates.rate_computation.compute_slip_rates_mc(
            markers=markers,
            criterion=criterion,
            n_samples=500,
            variable_type="custom_vt",
            unit="custom_unit",
        )
        rate = next(iter(rates.values()))

        assert rate.variable_type == "custom_vt"
        assert rate.unit == "custom_unit"


class TestAnalyticalMonteCarlo:
    def test_analytical_trimmed_matches_monte_carlo(self):
        """
        With ordering enforced, the analytical slip rates should match Monte
        Carlo sampling that rejects samples violating the order of the whole
        stack, and match it more closely than pairwise treatment does.
        """
        markers = _overlapping_markers_()
        max_rate = 150.0

        mc_criterion = sampling.mc_sampling.get_sample_criterion(
            "PassNonnegativeBounded"
        )(max_sample_rate=max_rate)
        incr_rates_mc, *_ = slip_rates.rate_computation.compute_slip_rates_mc(
            markers=markers,
            criterion=mc_criterion,
            n_samples=100_000,
            max_rate=max_rate,
        )

        # The analytical maximum rate must match the Monte Carlo criterion
        incr_rates_ordering = (
            slip_rates.rate_computation.compute_slip_rates_analytical(
                markers=markers,
                enforce_ordering=True,
                max_rate=max_rate,
            )
        )
        incr_rates_pairwise = (
            slip_rates.rate_computation.compute_slip_rates_analytical(
                markers=markers,
                max_rate=max_rate,
            )
        )

        for rate_name, rate_mc in incr_rates_mc.items():
            gap_ordering = _max_cdf_gap_(
                rate_mc, incr_rates_ordering[rate_name]
            )
            gap_pairwise = _max_cdf_gap_(
                rate_mc, incr_rates_pairwise[rate_name]
            )

            # Order-enforced analytical result agrees with sampling...
            assert gap_ordering < 0.02

            # ...and better than the pairwise result does
            assert gap_ordering < gap_pairwise

    def test_automatic_cap_sets_rate_axis_maximum(self):
        """
        With `max_rate=None` the slip rate PDF should end at the tail cap.
        """
        displacement, age = _example_disp_age_()
        marker = variable_pairs.DatedMarker(
            age=age, displacement=displacement, name="X"
        )

        v_max = slip_rates.rate_computation.find_slip_rate_tail_cap(
            displacement, age
        )
        rate_pdf = slip_rates.rate_computation.compute_slip_rate(marker=marker)

        assert rate_pdf.x[-1] == pytest.approx(v_max, rel=1e-3)

    def test_user_max_rate_overrides_cap(self):
        """
        A user-specified `max_rate` should be used in place of the
        automatically determined cap.
        """
        displacement, age = _example_disp_age_()
        marker = variable_pairs.DatedMarker(
            age=age, displacement=displacement, name="X"
        )

        rate_pdf = slip_rates.rate_computation.compute_slip_rate(
            marker=marker, max_rate=4.0
        )

        assert rate_pdf.x[-1] == pytest.approx(4.0, rel=1e-3)


def _reversed_separated_markers_():
    """Create markers listed from oldest to youngest with little overlap."""
    ages = PDFs.value_arrays.precise_array(0.0, 40.0, 0.01)
    disps = PDFs.value_arrays.precise_array(0.0, 80.0, 0.01)

    def marker(name, age, disp):
        return variable_pairs.DatedMarker(
            age=PDFs.PDF(
                x=ages,
                px=PDFs.parametric_functions.gaussian(ages, age, 1.0),
                variable_type="age",
                unit="y",
            ),
            displacement=PDFs.PDF(
                x=disps,
                px=PDFs.parametric_functions.gaussian(disps, disp, 1.0),
                variable_type="displacement",
                unit="m",
            ),
            name=name,
        )

    return {
        "old": marker("old", 25.0, 40.0),
        "middle": marker("middle", 15.0, 20.0),
        "young": marker("young", 5.0, 10.0),
    }


class TestMarkerOrderCheck:
    """Reversed markers are rejected before any computation."""

    def test_analytical_raises_for_reversed_markers(self):
        markers = _reversed_separated_markers_()

        with pytest.raises(ValueError, match="youngest"):
            slip_rates.rate_computation.compute_slip_rates_analytical(
                markers=markers,
            )

    def test_monte_carlo_raises_before_sampling(self):
        markers = _reversed_separated_markers_()
        criterion = sampling.mc_sampling.get_sample_criterion(
            "PassNonnegative"
        )()

        with pytest.raises(ValueError, match="youngest"):
            slip_rates.rate_computation.compute_slip_rates_mc(
                markers=markers,
                criterion=criterion,
                n_samples=100,
            )


# end of file
