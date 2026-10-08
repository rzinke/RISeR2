# -*- coding: utf-8 -*-
#
# Copyright (c) 2025-2026 Robert Zinke. Licensed under the MIT License.

# Import modules
import numpy as np
import pytest

from riser import (
    probability_functions as PDFs,
    sampling,
)


# Seed random number generator
np.random.seed(0)


# Tests
class TestSamplesToPdfHistogram:
    def test_minimal(self):
        mu = 1.0
        sigma = 2.0
        n_samples = 1_000_000

        samples = np.random.normal(mu, sigma, n_samples)

        pdf = sampling.pdf_formation.samples_to_pdf_histogram(
            samples=samples,
        )

        se_mean = sigma / np.sqrt(n_samples)
        assert PDFs.analytics.pdf_mean(pdf) == pytest.approx(
            mu, abs=2 * se_mean
        )

        se_std = sigma / np.sqrt(2 * n_samples)
        assert PDFs.analytics.pdf_std(pdf) == pytest.approx(
            sigma, abs=2 * se_std
        )


class TestSamplesToPdfKde:
    def test_minimal(self):
        """
        The standard deviation increases with KDE.
        See Scott's rule for kernel width.
        """
        mu = 1.0
        sigma = 2.0
        n_samples = 1_000

        samples = np.random.normal(mu, sigma, n_samples)

        pdf = sampling.pdf_formation.samples_to_pdf_kde(
            samples=samples,
        )

        se_mean = sigma / np.sqrt(n_samples)
        assert PDFs.analytics.pdf_mean(pdf) == pytest.approx(
            mu, abs=2 * se_mean
        )

    def test_default_step_is_unit_independent(self):
        """The default value array has the same size in mm/y and m/y."""
        samples = np.random.normal(5.0, 1.0, 5_000)

        pdf_mm = sampling.pdf_formation.samples_to_pdf_kde(samples=samples)
        pdf_m = sampling.pdf_formation.samples_to_pdf_kde(
            samples=samples * 1e-3
        )

        assert len(pdf_mm.x) == len(pdf_m.x)
        assert PDFs.analytics.pdf_mean(pdf_m) == pytest.approx(
            PDFs.analytics.pdf_mean(pdf_mm) * 1e-3, rel=1e-6
        )

    def test_samples_outside_limits_do_not_widen_kernel(self):
        """The kernel is fit to samples within the limits only."""
        samples = np.concatenate([np.random.normal(5.0, 1.0, 5_000), [1e4]])

        pdf_all = sampling.pdf_formation.samples_to_pdf_kde(
            samples=samples, xmin=0.0, xmax=10.0
        )
        pdf_in = sampling.pdf_formation.samples_to_pdf_kde(
            samples=samples[:-1], xmin=0.0, xmax=10.0
        )

        assert len(pdf_all.x) == len(pdf_in.x)
        assert PDFs.analytics.pdf_std(pdf_all) == pytest.approx(
            PDFs.analytics.pdf_std(pdf_in)
        )


# end of file
