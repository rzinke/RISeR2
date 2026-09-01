# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

# Import modules
import numpy as np
import pytest

from riser import units


# Tests
class TestCheckBaseUnitSupported:
    def test_warn_if_none(self):
        with pytest.warns(UserWarning):
            units.check_base_unit_supported(None)

    @pytest.mark.parametrize("base_unit", ["y", "m"])
    def test_silent_if_valid(self, base_unit, recwarn):
        units.check_base_unit_supported(base_unit)
        assert len(recwarn) == 0

    def test_raise_if_not_supported(self):
        with pytest.raises(ValueError):
            units.check_base_unit_supported("q")


class TestParseUnit:
    @pytest.mark.parametrize(
        "unit, expected_scale, expected_base",
        [
            ("y", 1.0, "y"),
            ("m", 1.0, "m"),
        ]
    )
    def test_bare_base_unit_has_scale_one(
        self, unit, expected_scale, expected_base
    ):
        scale, base = units.parse_unit(unit)
        assert scale == expected_scale
        assert base == expected_base

    @pytest.mark.parametrize(
        "compound_unit", ["m/y", "m2", "m.m"]
    )
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
            ("ky", 1_000.),
            ("My", 1_000_000.),
        ]
    )
    def test_unit_scales(self, unit, expected_scale):
        scale, _ = units.parse_unit(unit)
        assert scale == expected_scale


class TestScaleValuesByUnits:
    @pytest.mark.parametrize(
        "unit_in, unit_out",
        [
            ("y", None),
            (None, "y"),
        ]
    )
    def test_None_units_raise(self, unit_in, unit_out):
        with pytest.raises(ValueError, match="Neither input unit"):
            units.scale_values_by_units(1., unit_in, unit_out)

    def test_different_bases_raise(self):
        with pytest.raises(ValueError, match="Units do not match"):
            units.scale_values_by_units(1., "y", "m")

    @pytest.mark.parametrize(
        "values, unit_in, unit_out, expected_values",
        [
            (1., "y", "y", 1.),
            (2., "ky", "y", 2_000.),
            (np.arange(4), "km", "m", 1000 * np.arange(4)),
        ]
    )
    def test_unit_scaling(self, values, unit_in, unit_out, expected_values):
        scaled_values = units.scale_values_by_units(values, unit_in, unit_out)
        assert type(values) is type(scaled_values)
        np.testing.assert_allclose(scaled_values, expected_values)


# end of file
