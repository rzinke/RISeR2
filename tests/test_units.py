# -*- coding: utf-8 -*-
#
# Copyright (c) 2025-2026 Robert Zinke. Licensed under the MIT License.

# Import modules
import numpy as np
import pytest

from riser import units


# Tests
class TestCheckBaseUnitSupported:
    def test_warn_if_none(self):
        """Check that a warning is raised is no base unit is supported."""
        with pytest.warns(UserWarning):
            units.check_base_unit_supported(None)

    @pytest.mark.parametrize("base_unit", ["y", "m"])
    def test_silent_if_valid(self, base_unit, recwarn):
        """Check that no warning is raised if supported base units are passed."""
        units.check_base_unit_supported(base_unit)
        assert len(recwarn) == 0

    def test_raise_if_not_supported(self):
        """Check that an error is raised if an unsupported base unit is passed."""
        with pytest.raises(ValueError):
            units.check_base_unit_supported("q")


class TestParseUnit:
    @pytest.mark.parametrize(
        "unit, expected_scale, expected_base",
        [
            ("y", 1.0, "y"),
            ("m", 1.0, "m"),
        ],
    )
    def test_bare_base_unit_has_scale_one(
        self, unit, expected_scale, expected_base
    ):
        scale, base = units.parse_unit(unit)
        assert scale == expected_scale
        assert base == expected_base

    @pytest.mark.parametrize("compound_unit", ["m2", "m.m"])
    def test_compound_unit_raises(self, compound_unit):
        with pytest.raises(ValueError):
            units.parse_unit(compound_unit)

    def test_unsupported_base_raises(self):
        with pytest.raises(ValueError, match="Base unit"):
            units.parse_unit("kq")

    def test_unsupported_prefix_raises(self):
        with pytest.raises(ValueError, match="Unit prefix"):
            units.parse_unit("qy")

    @pytest.mark.parametrize(
        "unit, expected_scale",
        [
            ("my", 0.001),
            ("ky", 1_000.0),
            ("My", 1_000_000.0),
        ],
    )
    def test_unit_scales(self, unit, expected_scale):
        scale, _ = units.parse_unit(unit)
        assert scale == expected_scale


class TestParseRateUnit:
    @pytest.mark.parametrize(
        "unit, expected_scale",
        [
            ("m/y", 1.0),
            ("mm/y", 1e-3),
            ("cm/y", 1e-2),
            ("m/ky", 1e-3),
            ("km/My", 1e-3),
        ],
    )
    def test_rate_unit_scales(self, unit, expected_scale):
        scale, base = units.parse_unit(unit)
        assert scale == pytest.approx(expected_scale)
        assert base == "m/y"

    def test_equivalent_rate_units_are_identical(self):
        """1 mm/y is exactly 1 m/ky."""
        assert units.parse_unit("mm/y") == units.parse_unit("m/ky")

    @pytest.mark.parametrize(
        "rate_unit, message",
        [
            ("y/m", "length/time"),
            ("m/m", "length/time"),
            ("mm/s", "Base unit"),
            ("mm/qy", "Unit prefix"),
            ("m/y/y", "form"),
            ("mm/", "form"),
            ("/y", "form"),
            ("m^2/y", "Compound"),
            ("m.y/y", "Compound"),
            ("m2/y", "Compound"),
        ],
    )
    def test_invalid_rate_unit_raises(self, rate_unit, message):
        with pytest.raises(ValueError, match=message):
            units.parse_unit(rate_unit)


class TestScaleValuesByUnits:
    @pytest.mark.parametrize(
        "unit_in, unit_out",
        [
            ("y", None),
            (None, "y"),
        ],
    )
    def test_None_units_raise(self, unit_in, unit_out):
        with pytest.raises(ValueError, match="Neither input unit"):
            units.scale_values_by_units(1.0, unit_in, unit_out)

    @pytest.mark.parametrize(
        "unit_in, unit_out",
        [
            ("y", "m"),
            ("mm/y", "mm"),
            ("m", "m/y"),
        ],
    )
    def test_different_bases_raise(self, unit_in, unit_out):
        with pytest.raises(ValueError, match="Units do not match"):
            units.scale_values_by_units(1.0, unit_in, unit_out)

    @pytest.mark.parametrize(
        "values, unit_in, unit_out, expected_values",
        [
            (1.0, "y", "y", 1.0),
            (2.0, "ky", "y", 2_000.0),
            (np.arange(4), "km", "m", 1000 * np.arange(4)),
            (2.0, "mm/y", "m/ky", 2.0),
            (2.0, "mm/y", "m/y", 0.002),
            (2.0, "m/ky", "cm/y", 0.2),
            (np.arange(4), "m/y", "mm/y", 1000 * np.arange(4)),
        ],
    )
    def test_unit_scaling(self, values, unit_in, unit_out, expected_values):
        scaled_values = units.scale_values_by_units(values, unit_in, unit_out)
        assert type(values) is type(scaled_values)
        np.testing.assert_allclose(scaled_values, expected_values)


# end of file
