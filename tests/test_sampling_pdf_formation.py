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
        assert (
            PDFs.analytics.pdf_mean(pdf) == pytest.approx(mu, abs=2 * se_mean)
        )

        se_std = sigma / np.sqrt(2 * n_samples)
        assert (
            PDFs.analytics.pdf_std(pdf) == pytest.approx(sigma, abs=2 * se_std)
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
        assert (
            PDFs.analytics.pdf_mean(pdf) == pytest.approx(mu, abs=2 * se_mean)
        )


# end of file
