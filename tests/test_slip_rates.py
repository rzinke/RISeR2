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
            n_samples=10_000,
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


# end of file
