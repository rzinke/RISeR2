# -*- coding: utf-8 -*-
#
# Copyright (c) 2025-2026 Robert Zinke. Licensed under the MIT License.

# Import modules
import numpy as np
import pytest
import scipy as sp

from riser import (
    constants,
    sampling,
)


np.random.seed(0)


# Tests
class TestComputeSampleConfidence:
    @pytest.mark.parametrize(
        "mu, sigma, confidence",
        [
            (0.0, 1.0, constants.Psigma["1"]),
            (0.0, 1.0, constants.Psigma["2"]),
            (1.0, 2.0, constants.Psigma["2"]),
        ],
    )
    def test_sample_confidence_computation(self, mu, sigma, confidence):
        n_samples = 100_000

        conf = sampling.sample_statistics.compute_sample_confidence(
            samples=np.random.normal(mu, sigma, n_samples),
            confidence=confidence,
        )

        p_lo = (1 - confidence) / 2
        p_hi = (1 + confidence) / 2

        z_lo = sp.stats.norm.ppf(p_lo)
        z_hi = sp.stats.norm.ppf(p_hi)

        expected_lo = mu + sigma * z_lo
        expected_hi = mu + sigma * z_hi

        # Standard error of an order-statistic (quantile) estimator:
        #   SE(x_p) = sqrt(p(1-p)/n) / f(x_p)
        f_lo = sp.stats.norm.pdf(z_lo) / sigma
        f_hi = sp.stats.norm.pdf(z_hi) / sigma

        se_lo = np.sqrt(p_lo * (1 - p_lo) / n_samples) / f_lo
        se_hi = np.sqrt(p_hi * (1 - p_hi) / n_samples) / f_hi

        assert conf.range_values[0] == pytest.approx(expected_lo, abs=2 * se_lo)
        assert conf.range_values[1] == pytest.approx(expected_hi, abs=2 * se_hi)


# end of file
