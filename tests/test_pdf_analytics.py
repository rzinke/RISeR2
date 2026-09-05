# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

# Import modules
from dataclasses import dataclass

import numpy as np
import pytest

from riser import probability_functions as PDFs


# Function parameters
_dx = 0.01
_x = PDFs.value_arrays.precise_array(-3.0, 5.0, _dx)
_px = PDFs.parametric_functions.triangular(_x, a=-2.0, c=2.0, b=3.0)
TRI_ASYM = {
    "x": _x,
    "px": _px,
    "dx": _dx,
}

_x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
_px = PDFs.parametric_functions.triangular(_x, a=0.0, c=1.0, b=2.0)
_dx = np.diff(_x, append=_x[-1])
TRI_IRREG = {
    "x": _x,
    "px": _px,
    "dx": _dx,
}

_dx = 0.001
_x = PDFs.value_arrays.precise_array(-10.0, 10.0, _dx)
_px = PDFs.parametric_functions.gaussian(_x, mu=0.0, sigma=1.0)
STD_NORM = {
    "x": _x,
    "px": _px,
    "dx": _dx,
}


_dx = 0.001
_x = PDFs.value_arrays.precise_array(-16.0, 24.0, _dx)
_px = PDFs.parametric_functions.gaussian(_x, mu=1.0, sigma=2.0)
SHIFT_NORM = {
    "x": _x,
    "px": _px,
    "dx": _dx,
}


# Tests
class TestExpectedValue:
    @pytest.mark.parametrize(
        "var_dict, expected",
        [
            (TRI_ASYM, 1.0),
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
            (TRI_ASYM, 1, 1.0),
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
            (TRI_ASYM, 1.0),
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


# end of file
