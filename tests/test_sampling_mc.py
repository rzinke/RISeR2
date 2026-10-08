# -*- coding: utf-8 -*-
#
# Copyright (c) 2025-2026 Robert Zinke. Licensed under the MIT License.

# Import modules
import numpy as np
import pytest

from riser import (
    probability_functions as PDFs,
    sampling,
    variable_pairs,
)


# Seed random number generator
np.random.seed(0)


# Dated markers
age_axis = PDFs.value_arrays.precise_array(0.0, 20.0, 0.01)
disp_axis = PDFs.value_arrays.precise_array(0.0, 50.0, 0.01)

distant_markers = {
    "younger": variable_pairs.DatedMarker(
        age=PDFs.PDF(
            x=age_axis,
            px=PDFs.parametric_functions.gaussian(
                x=age_axis, mu=12.0, sigma=1.0
            ),
            name="younger_age",
            variable_type="age",
            unit="y",
        ),
        displacement=PDFs.PDF(
            x=disp_axis,
            px=PDFs.parametric_functions.triangular(
                x=disp_axis, a=13.0, c=15.0, b=17.0
            ),
            name="younger_disp",
            variable_type="displacement",
            unit="mm",
        ),
        name="younger",
    ),
    "older": variable_pairs.DatedMarker(
        age=PDFs.PDF(
            x=age_axis,
            px=PDFs.parametric_functions.gaussian(
                x=age_axis, mu=15.0, sigma=1.0
            ),
            name="older_age",
            variable_type="age",
            unit="y",
        ),
        displacement=PDFs.PDF(
            x=disp_axis,
            px=PDFs.parametric_functions.triangular(
                x=disp_axis, a=43.0, c=45.0, b=47.0
            ),
            name="older_disp",
            variable_type="displacement",
            unit="mm",
        ),
        name="older",
    ),
}


close_markers = {
    "younger": variable_pairs.DatedMarker(
        age=PDFs.PDF(
            x=age_axis,
            px=PDFs.parametric_functions.gaussian(
                x=age_axis, mu=10.0, sigma=1.0
            ),
            name="younger_age",
            variable_type="age",
            unit="y",
        ),
        displacement=PDFs.PDF(
            x=disp_axis,
            px=PDFs.parametric_functions.triangular(
                x=disp_axis, a=25.0, c=30.0, b=35.0
            ),
            name="younger_disp",
            variable_type="displacement",
            unit="mm",
        ),
        name="younger",
    ),
    "older": variable_pairs.DatedMarker(
        age=PDFs.PDF(
            x=age_axis,
            px=PDFs.parametric_functions.gaussian(
                x=age_axis, mu=12.0, sigma=1.0
            ),
            name="older_age",
            variable_type="age",
            unit="y",
        ),
        displacement=PDFs.PDF(
            x=disp_axis,
            px=PDFs.parametric_functions.triangular(
                x=disp_axis, a=30.0, c=35.0, b=40.0
            ),
            name="older_disp",
            variable_type="displacement",
            unit="mm",
        ),
        name="older",
    ),
}


# Tests
class TestSampleMonteCarlo:
    def test_pass_all(self):
        """
        Trivial case in which no samples are rejected.
        """
        criterion = sampling.mc_sampling.get_sample_criterion("PassAll")()

        n_samples = 1_000

        (age_picks, disp_picks, success_rate) = (
            sampling.mc_sampling.sample_monte_carlo(
                markers=close_markers,
                criterion=criterion,
                n_samples=n_samples,
            )
        )

        assert age_picks.shape[1] == disp_picks.shape[1] == n_samples
        assert success_rate == 1.0

    def test_partial_success(self, recwarn):
        criterion = sampling.mc_sampling.get_sample_criterion(
            "PassNonnegative"
        )()

        n_samples = 1_000

        (_, _, success_rate) = sampling.mc_sampling.sample_monte_carlo(
            markers=distant_markers,
            criterion=criterion,
            n_samples=n_samples,
        )

        assert len(recwarn) == 0
        assert 0.0 < success_rate < 1.0

    def test_less_than_desired_samples_warns(self, recwarn):
        criterion = sampling.mc_sampling.get_sample_criterion(
            "PassNonnegative"
        )()

        n_samples = 1_000

        (age_picks, disp_picks, _) = sampling.mc_sampling.sample_monte_carlo(
            markers=close_markers,
            criterion=criterion,
            n_samples=n_samples,
            hard_stop=n_samples,
        )

        assert age_picks.shape == disp_picks.shape
        assert age_picks.shape[1] < n_samples
        assert len(recwarn) > 0

    def test_no_successes(self):
        criterion = sampling.mc_sampling.get_sample_criterion(
            "PassNonnegativeBounded"
        )(max_sample_rate=0.0)

        n_samples = 1_000

        with pytest.raises(RuntimeError, match="No samples meet"):
            sampling.mc_sampling.sample_monte_carlo(
                markers=close_markers,
                criterion=criterion,
                n_samples=n_samples,
                hard_stop=n_samples,
            )


class TestSampleCriterionStr:
    @pytest.mark.parametrize(
        "criterion_name, kwargs, expected",
        [
            ("PassAll", {}, "pass all"),
            ("PassNonnegative", {}, "pass non-negative"),
            (
                "PassNonnegativeBounded",
                {"max_sample_rate": 10.0},
                "pass non-negative (bounded 10.0)",
            ),
        ],
    )
    def test_description_used_in_reports(
        self, criterion_name, kwargs, expected
    ):
        """
        The text description of a criterion is written to the slip rate
        report, so it should state the condition that is applied.
        """
        criterion = sampling.mc_sampling.get_sample_criterion(criterion_name)(
            **kwargs
        )

        assert str(criterion) == expected


# end of file
