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
    return {
        "young": variable_pairs.DatedMarker(
            age=PDFs.PDF(
                x=np.array([4.0, 5.0, 6.0]),
                px=np.array([0.0, 1.0, 0.0]),
                variable_type="age", unit="y",
            ),
            displacement=PDFs.PDF(
                x=np.array([9.0, 10.0, 11.0]),
                px=np.array([0.0, 1.0, 0.0]),
                variable_type="displacement", unit="m",
            ),
            name="young",
        ),
        "old": variable_pairs.DatedMarker(
            age=PDFs.PDF(
                x=np.array([14.0, 15.0, 16.0]),
                px=np.array([0.0, 1.0, 0.0]),
                variable_type="age", unit="y",
            ),
            displacement=PDFs.PDF(
                x=np.array([29.0, 30.0, 31.0]),
                px=np.array([0.0, 1.0, 0.0]),
                variable_type="displacement", unit="m",
            ),
            name="old",
        ),
    }


def _three_markers_():
    return {
        "young": variable_pairs.DatedMarker(
            age=PDFs.PDF(
                x=np.array([4.0, 5.0, 6.0]),
                px=np.array([0.0, 1.0, 0.0]),
                variable_type="age", unit="y",
            ),
            displacement=PDFs.PDF(
                x=np.array([9.0, 10.0, 11.0]),
                px=np.array([0.0, 1.0, 0.0]),
                variable_type="displacement", unit="m",
            ),
            name="young",
        ),
        "middle": variable_pairs.DatedMarker(
            age=PDFs.PDF(
                x=np.array([9.0, 10.0, 11.0]),
                px=np.array([0.0, 1.0, 0.0]),
                variable_type="age", unit="y",
            ),
            displacement=PDFs.PDF(
                x=np.array([19.0, 20.0, 21.0]),
                px=np.array([0.0, 1.0, 0.0]),
                variable_type="displacement", unit="m",
            ),
            name="middle",
        ),
        "old": variable_pairs.DatedMarker(
            age=PDFs.PDF(
                x=np.array([14.0, 15.0, 16.0]),
                px=np.array([0.0, 1.0, 0.0]),
                variable_type="age", unit="y",
            ),
            displacement=PDFs.PDF(
                x=np.array([29.0, 30.0, 31.0]),
                px=np.array([0.0, 1.0, 0.0]),
                variable_type="displacement", unit="m",
            ),
            name="old",
        ),
    }


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
            x=age_axis, px=age_density,
            variable_type="age", unit="y",
        )

        disp_mu = 50.0
        disp_sigma = 0.001
        disp_axis = PDFs.value_arrays.precise_array(49.9, 50.1, 0.0001)
        disp_density = PDFs.parametric_functions.gaussian(
            disp_axis, mu=disp_mu, sigma=disp_sigma
        )
        disp_pdf = PDFs.PDF(
            x=disp_axis, px=disp_density,
            variable_type="displacement", unit="m",
        )

        marker = variable_pairs.DatedMarker(
            age=age_pdf, displacement=disp_pdf, name="X"
        )

        rate_pdf = slip_rates.rate_computation.compute_slip_rate(
            marker=marker,
        )

        se = 5 * np.sqrt((age_sigma/age_mu)**2 + (disp_sigma/disp_mu)**2)
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
            markers=markers, criterion=criterion, n_samples=500,
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


# end of file
