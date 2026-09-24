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
class TestSampleMonteCarlo:
    def test_all_pass(self):
        """Trivial case in which no samples are rejected.
        """


class TestSamplesToPdfHistogram:
    def test_minimal(self):
        mu = 1.0
        sigma = 2.0

        samples = np.random.normal(mu, sigma, 1_000_000)

        pdf = sampling.pdf_formation.samples_to_pdf_histogram(
            samples=samples,
        )

        assert PDFs.analytics.pdf_mean(pdf) == pytest.approx(mu)
        assert PDFs.analytics.pdf_std(pdf) == pytest.approx(sigma)


class TestSamplesToPdfKde:
    def test_minimal(self):
        mu = 1.0
        sigma = 2.0

        samples = np.random.normal(mu, sigma, 1_000)

        pdf = sampling.pdf_formation.samples_to_pdf_kde(
            samples=samples,
        )

        assert PDFs.analytics.pdf_mean(pdf) == pytest.approx(mu)
        assert PDFs.analytics.pdf_std(pdf) == pytest.approx(sigma)


# end of file
