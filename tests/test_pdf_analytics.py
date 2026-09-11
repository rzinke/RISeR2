# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

# Import modules
import numpy as np
import pytest

from riser import (
    constants,
    probability_functions as PDFs,
)


# Uniform distribution with mean 0.5
_dx_unif = 0.01
_x_unif = PDFs.value_arrays.precise_array(-1.0, 2.0, _dx_unif)
_px_unif = PDFs.parametric_functions.uniform(_x_unif, a=0.0, b=1.0)
UNIFORM = {
    "x": _x_unif,
    "px": _px_unif,
    "dx": _dx_unif,
}

# Asymmetric triangular distribution with mean 7 / 3
_dx_asym = 0.01
_x_asym = PDFs.value_arrays.precise_array(-3.0, 10.0, _dx_asym)
_px_asym = PDFs.parametric_functions.triangular(_x_asym, a=-1.0, c=0.0, b=8.0)
TRI_ASYM = {
    "x": _x_asym,
    "px": _px_asym,
    "dx": _dx_asym,
}

# Irregularly and very coarsely sampled triangular distribution
# Numerically determined mean (0.875) is not expected to match analytically
# determined mean (1.0)
_x_irreg = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
_px_irreg = PDFs.parametric_functions.triangular(_x_irreg, a=0.0, c=1.0, b=2.0)
_dx_irreg = np.diff(_x_irreg, append=_x_irreg[-1])
TRI_IRREG = {
    "x": _x_irreg,
    "px": _px_irreg,
    "dx": _dx_irreg,
}

# Standard normal distribution with fine point spacing and broad domain
_dx_std = 0.001
_x_std = PDFs.value_arrays.precise_array(-10.0, 10.0, _dx_std)
_px_std = PDFs.parametric_functions.gaussian(_x_std, mu=0.0, sigma=1.0)
STD_NORM = {
    "x": _x_std,
    "px": _px_std,
    "dx": _dx_std,
}

# Shifted and wide Gaussian distribution with mean 1.0 and standard dev 2.0
_dx_shift = 0.001
_x_shift = PDFs.value_arrays.precise_array(-16.0, 24.0, _dx_shift)
_px_shift = PDFs.parametric_functions.gaussian(_x_shift, mu=1.0, sigma=2.0)
SHIFT_NORM = {
    "x": _x_shift,
    "px": _px_shift,
    "dx": _dx_shift,
}

# Bimodal triangular distribution with asymmetric triangles.
# Distribution is symmetric about 0.
_dx_bimodal = 0.001
_x_bimodal = PDFs.value_arrays.precise_array(-10.0, 10.0, _dx_bimodal)
_px_bimodal = (
    PDFs.parametric_functions.triangular(_x_bimodal, a=-2.0, c=-1.0, b=0.0)
    + PDFs.parametric_functions.triangular(_x_bimodal, a=0.0, c=1.0, b=2.0)
)
BIMODAL = {
    "x": _x_bimodal,
    "px": _px_bimodal,
    "dx": _dx_bimodal,
}


# Tests
class TestExpectedValue:
    @pytest.mark.parametrize(
        "var_dict, expected",
        [
            (TRI_ASYM, 7/3),
            (TRI_IRREG, 0.875),
            (STD_NORM, 0.0),
        ],
    )
    def test_known_expected_values(self, var_dict, expected):
        ev = PDFs.analytics.expected_value(x=var_dict["x"], px=var_dict["px"])
        assert ev == pytest.approx(expected)


class TestComputeRawMoment:
    @pytest.mark.parametrize(
        "var_dict, n, expected",
        [
            (TRI_ASYM, 1, 7/3),
            (TRI_IRREG, 1, 0.875),
            (STD_NORM, 1, 0.0),
            (STD_NORM, 2, 1.0),
            (SHIFT_NORM, 1, 1.0),
        ],
    )
    def test_known_raw_moments(self, var_dict, n, expected):
        theta_n = PDFs.analytics.compute_raw_moment(
            x=var_dict["x"], px=var_dict["px"], n=n
        )
        assert theta_n == pytest.approx(expected)


class TestCentralMoment:
    @pytest.mark.parametrize(
        "var_dict, n, expected",
        [
            (TRI_ASYM, 1, 0.0),
            (STD_NORM, 1, 0.0),
            (STD_NORM, 2, 1.0),
            (SHIFT_NORM, 1, 0.0),
            (SHIFT_NORM, 2, 4.0),
        ],
    )
    def test_known_central_moments(self, var_dict, n, expected):
        mu_n = PDFs.analytics.compute_central_moment(
            x=var_dict["x"], px=var_dict["px"], n=n
        )
        assert mu_n == pytest.approx(expected)


class TestStandardizedMoment:
    @pytest.mark.parametrize(
        "var_dict, n, expected",
        [
            (TRI_ASYM, 1, 0.0),
            (STD_NORM, 1, 0.0),
            (STD_NORM, 2, 1.0),
            (STD_NORM, 3, 0.0),
            (STD_NORM, 4, 3.0),
            (SHIFT_NORM, 1, 0.0),
            (SHIFT_NORM, 2, 1.0),
            (SHIFT_NORM, 3, 0.0),
        ],
    )
    def test_known_standardized_moments(self, var_dict, n, expected):
        mu_n = PDFs.analytics.compute_standardized_moment(
            x=var_dict["x"], px=var_dict["px"], n=n
        )
        assert mu_n == pytest.approx(expected)


class TestPdfMean:
    @pytest.mark.parametrize(
        "var_dict, expected",
        [
            (TRI_ASYM, 7/3),
            (TRI_IRREG, 0.875),
            (STD_NORM, 0.0),
            (SHIFT_NORM, 1.0),
        ],
    )
    def test_known_pdf_mean(self, var_dict, expected):
        pdf = PDFs.PDF(x=var_dict["x"], px=var_dict["px"])
        mu_n = PDFs.analytics.pdf_mean(pdf)
        assert mu_n == pytest.approx(expected)


class TestPdfVariance:
    @pytest.mark.parametrize(
        "var_dict, expected",
        [
            (STD_NORM, 1.0),
            (SHIFT_NORM, 4.0),
        ],
    )
    def test_known_pdf_variance(self, var_dict, expected):
        pdf = PDFs.PDF(x=var_dict["x"], px=var_dict["px"])
        sigma2 = PDFs.analytics.pdf_variance(pdf)
        assert sigma2 == pytest.approx(expected)


class TestPdfStandardDeviation:
    @pytest.mark.parametrize(
        "var_dict, expected",
        [
            (STD_NORM, 1.0),
            (SHIFT_NORM, 2.0),
        ],
    )
    def test_known_pdf_std(self, var_dict, expected):
        pdf = PDFs.PDF(x=var_dict["x"], px=var_dict["px"])
        sigma2 = PDFs.analytics.pdf_std(pdf)
        assert sigma2 == pytest.approx(expected)


class TestPdfSkewness:
    @pytest.mark.parametrize(
        "var_dict, expected",
        [
            (STD_NORM, 0.0),
            (SHIFT_NORM, 0.0),
        ],
    )
    def test_known_pdf_skewness(self, var_dict, expected):
        """Zero skew is expected for symmetric functions.
        """
        pdf = PDFs.PDF(x=var_dict["x"], px=var_dict["px"])
        skewness = PDFs.analytics.pdf_skewness(pdf)
        assert skewness == pytest.approx(expected)


class TestPdfKurtosis:
    @pytest.mark.parametrize(
        "var_dict, expected",
        [
            (STD_NORM, 3.0),
            (SHIFT_NORM, 3.0),
        ],
    )
    def test_known_pdf_kurtosis(self, var_dict, expected):
        """Unlike the other moments, a zero-centered, symmetric distribution
        should have a non-zero kurtosis (e.g., 3.0).
        """
        pdf = PDFs.PDF(x=var_dict["x"], px=var_dict["px"])
        kurtosis = PDFs.analytics.pdf_kurtosis(pdf)
        assert kurtosis == pytest.approx(expected)


class TestPdfMode:
    @pytest.mark.parametrize(
        "var_dict, expected",
        [
            (TRI_ASYM, 0.0),
            (TRI_IRREG, 1.0),
            (STD_NORM, 0.0),
            (SHIFT_NORM, 1.0),
        ],
    )
    def test_known_mode(self, var_dict, expected):
        pdf = PDFs.PDF(x=var_dict["x"], px=var_dict["px"])
        mode = PDFs.analytics.pdf_mode(pdf)
        assert mode == pytest.approx(expected)


class TestPdfMedian:
    @pytest.mark.parametrize(
        "var_dict, expected",
        [
            (UNIFORM, 0.5),
            (TRI_ASYM, 2.0),
            (STD_NORM, 0.0),
            (SHIFT_NORM, 1.0),
        ],
    )
    def test_known_median(self, var_dict, expected):
        pdf = PDFs.PDF(x=var_dict["x"], px=var_dict["px"])
        mode = PDFs.analytics.pdf_median(pdf)
        assert mode == pytest.approx(expected)


class TestPdfStatistics:
    def test_returns_a_string(self):
        stats = PDFs.analytics.PDFstatistics(
            mode=1.0, median=2.0, mean=3.0, std=4.0,
            variance=16.0, skewness=0.5, kurtosis=3.0,
            name="x", variable_type="age", unit="y",
        )
        assert isinstance(str(stats), str)


class TestComputePdfStatistics:
    @pytest.mark.parametrize(
        "var_dict",
        [TRI_ASYM],
    )
    def test_returns_pdf_statistics(self, var_dict):
        """Test each parameter in PDF statistics returns the value that would
        be computed from the raw functions.
        """
        pdf = PDFs.PDF(x=var_dict["x"], px=var_dict["px"])
        pdf_stats = PDFs.analytics.compute_pdf_statistics(pdf)
        assert isinstance(pdf_stats, PDFs.analytics.PDFstatistics)
        assert pdf_stats.mean == pytest.approx(PDFs.analytics.pdf_mean(pdf))
        assert pdf_stats.mode == pytest.approx(PDFs.analytics.pdf_mode(pdf))
        assert pdf_stats.median == pytest.approx(PDFs.analytics.pdf_median(pdf))
        assert pdf_stats.std == pytest.approx(PDFs.analytics.pdf_std(pdf))
        assert pdf_stats.variance == pytest.approx(
            PDFs.analytics.pdf_variance(pdf)
        )
        assert pdf_stats.skewness == pytest.approx(
            PDFs.analytics.pdf_skewness(pdf)
        )
        assert pdf_stats.kurtosis == pytest.approx(
            PDFs.analytics.pdf_kurtosis(pdf)
        )


class TestConfidenceRange:
    def test_iterates_single_range(self):
        conf_range = PDFs.analytics.ConfidenceRange(
            metric="CI", confidence=0.68, range_values=((0.1, 0.9),)
        )
        assert list(conf_range) == [(0.1, 0.9)]

    def test_iterates_multiple_ranges(self):
        conf_range = PDFs.analytics.ConfidenceRange(
            metric="HPD",
            confidence=0.95,
            range_values=((0.1, 0.3), (0.6, 0.9)),
        )
        assert list(conf_range) == [(0.1, 0.3), (0.6, 0.9)]

    def test_returns_a_str(self):
        conf_range = PDFs.analytics.ConfidenceRange(
            metric="CI", confidence=0.68, range_values=((0.1, 0.9),),
            pdf_name="x", variable_type="age", unit="y",
        )
        assert isinstance(str(conf_range), str)


class TestComputeInterquantileRange:
    @pytest.mark.parametrize(
        "confidence, expected",
        [
            (constants.Psigma["1"], [(-1.0, 3.0)]),
            (constants.Psigma["2"], [(-3.0, 5.0)]),
        ],
    )
    def test_iterates_single_range(self, confidence, expected):
        pdf = PDFs.PDF(x=SHIFT_NORM["x"], px=SHIFT_NORM["px"])
        conf_range = PDFs.analytics.compute_interquantile_range(
            pdf=pdf, confidence=confidence,
        )
        conf_list = list(conf_range)
        assert len(conf_list) == 1
        for conf_result, expected_result in zip(conf_list, expected):
            assert conf_result == pytest.approx(expected_result)


class TestComputeHighestPosteriorDensity:
    @pytest.mark.parametrize(
        "confidence, expected",
        [
            (constants.Psigma["1"], [(-1.0, 3.0)]),
            (constants.Psigma["2"], [(-3.0, 5.0)]),
        ],
    )
    def test_single_peak(self, confidence, expected):
        pdf = PDFs.PDF(x=SHIFT_NORM["x"], px=SHIFT_NORM["px"])
        conf_range = PDFs.analytics.compute_highest_posterior_density(
            pdf=pdf, confidence=confidence,
        )
        conf_list = list(conf_range)
        assert len(conf_list) == 1
        tolerance = 4 * SHIFT_NORM["dx"]
        for conf_result, expected_result in zip(conf_list, expected):
            assert conf_result == pytest.approx(expected_result, abs=tolerance)

    @pytest.mark.parametrize(
        "confidence, expected",
        [
            (constants.Psigma["1"],
             [(-1.436697, -0.563303), (0.563303, 1.436697)]
            ),
            (constants.Psigma["2"],
             [(-1.786692, -0.213308), (0.213308, 1.786692)]
            ),
        ],
    )
    def test_multi_peak(self, confidence, expected):
        pdf = PDFs.PDF(x=BIMODAL["x"], px=BIMODAL["px"])
        conf_range = PDFs.analytics.compute_highest_posterior_density(
            pdf=pdf, confidence=confidence,
        )
        conf_list = list(conf_range)
        assert len(conf_list) == 2
        tolerance = 4 * BIMODAL["dx"]
        for conf_result, expected_result in zip(conf_list, expected):
            assert conf_result == pytest.approx(expected_result, abs=tolerance)


class TestGetPdfConfidenceFunction:
    def test_case_insensitive_dispatch(self):
        fn_upper = PDFs.analytics.get_pdf_confidence_function("HPD")
        fn_lower = PDFs.analytics.get_pdf_confidence_function("hpd")
        assert (
            fn_upper
            is fn_lower
            is PDFs.analytics.compute_highest_posterior_density
        )

    def test_iqr_maps_correctly(self):
        fn = PDFs.analytics.get_pdf_confidence_function("IQR")
        assert fn is PDFs.analytics.compute_interquantile_range

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            PDFs.analytics.get_pdf_confidence_function("xyz")


class TestComputePdfConfidenceRange:
    @pytest.mark.parametrize(
        "metric, direct_fcn",
        [
            ("IQR", PDFs.analytics.compute_interquantile_range),
            ("HPD", PDFs.analytics.compute_highest_posterior_density),
        ],
    )
    def test_dispatches_to_correct_function(self, metric, direct_fcn):
        pdf = PDFs.PDF(x=SHIFT_NORM["x"], px=SHIFT_NORM["px"])
        confidence = constants.Psigma["1"]

        direct_result = list(direct_fcn(pdf=pdf, confidence=confidence))
        result_from_dispatch = list(
            PDFs.analytics.compute_pdf_confidence_range(
                pdf, metric=metric, confidence=confidence
            )
        )
        assert direct_result == result_from_dispatch


# end of file
