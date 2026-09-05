# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

# Import modules
from dataclasses import dataclass

import numpy as np
import pytest

from riser import (
    constants,
    probability_functions as PDFs,
)


# Function parameters
_dx_unif = 0.01
_x_unif = PDFs.value_arrays.precise_array(-1.0, 2.0, _dx_unif)
_px_unif = PDFs.parametric_functions.uniform(_x_unif, a=0.0, b=1.0)
UNIFORM = {
    "x": _x_unif,
    "px": _px_unif,
    "dx": _dx_unif,
}

_dx_asym = 0.01
_x_asym = PDFs.value_arrays.precise_array(-3.0, 10.0, _dx_asym)
_px_asym = PDFs.parametric_functions.triangular(_x_asym, a=-1.0, c=0.0, b=8.0)
TRI_ASYM = {
    "x": _x_asym,
    "px": _px_asym,
    "dx": _dx_asym,
}

_x_irreg = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
_px_irreg = PDFs.parametric_functions.triangular(_x_irreg, a=0.0, c=1.0, b=2.0)
_dx_irreg = np.diff(_x_irreg, append=_x_irreg[-1])
TRI_IRREG = {
    "x": _x_irreg,
    "px": _px_irreg,
    "dx": _dx_irreg,
}

_dx_std = 0.001
_x_std = PDFs.value_arrays.precise_array(-10.0, 10.0, _dx_std)
_px_std = PDFs.parametric_functions.gaussian(_x_std, mu=0.0, sigma=1.0)
STD_NORM = {
    "x": _x_std,
    "px": _px_std,
    "dx": _dx_std,
}

_dx_shift = 0.001
_x_shift = PDFs.value_arrays.precise_array(-16.0, 24.0, _dx_shift)
_px_shift = PDFs.parametric_functions.gaussian(_x_shift, mu=1.0, sigma=2.0)
SHIFT_NORM = {
    "x": _x_shift,
    "px": _px_shift,
    "dx": _dx_shift,
}


# Tests
class TestExpectedValue:
    @pytest.mark.parametrize(
        "var_dict, expected",
        [
            (TRI_ASYM, 7/3),
            (TRI_IRREG, 1.125),
            (STD_NORM, 0.0),
        ],
    )
    def test_known_expected_values(self, var_dict, expected):
        ev = PDFs.analytics.expected_value(**var_dict)
        assert ev == pytest.approx(expected)


class TestComputeRawMoment:
    @pytest.mark.parametrize(
        "var_dict, n, expected",
        [
            (TRI_ASYM, 1, 7/3),
            (TRI_IRREG, 1, 1.125),
            (STD_NORM, 1, 0.0),
            (STD_NORM, 2, 1.0),
            (SHIFT_NORM, 1, 1.0),
        ],
    )
    def test_known_raw_moments(self, var_dict, n, expected):
        theta_n = PDFs.analytics.compute_raw_moment(**var_dict, n=n)
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
        mu_n = PDFs.analytics.compute_central_moment(**var_dict, n=n)
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
        mu_n = PDFs.analytics.compute_standardized_moment(**var_dict, n=n)
        assert mu_n == pytest.approx(expected)


class TestPdfMean:
    @pytest.mark.parametrize(
        "var_dict, expected",
        [
            (TRI_ASYM, 7/3),
            (TRI_IRREG, 1.125),
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


# end of file
