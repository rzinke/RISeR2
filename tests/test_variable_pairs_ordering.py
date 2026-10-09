# -*- coding: utf-8 -*-
#
# Copyright (c) 2025, 2026 Robert Zinke. Licensed under the MIT License.

# Import modules
import pytest

from riser import probability_functions as PDFs, variable_pairs


# Helper functions
def _marker_(name, age, disp):
    """Create a marker with Gaussian age and displacement PDFs."""
    ages = PDFs.value_arrays.precise_array(0.0, 40.0, 0.05)
    disps = PDFs.value_arrays.precise_array(0.0, 80.0, 0.05)

    return variable_pairs.DatedMarker(
        age=PDFs.PDF(
            x=ages,
            px=PDFs.parametric_functions.gaussian(ages, age, 1.0),
            variable_type="age",
            unit="y",
        ),
        displacement=PDFs.PDF(
            x=disps,
            px=PDFs.parametric_functions.gaussian(disps, disp, 1.0),
            variable_type="displacement",
            unit="m",
        ),
        name=name,
    )


def _ordered_markers_():
    return {
        "young": _marker_("young", 5.0, 10.0),
        "middle": _marker_("middle", 15.0, 20.0),
        "old": _marker_("old", 25.0, 40.0),
    }


# Tests
class TestFindOrderingViolations:
    def test_ordered_markers_no_violations(self):
        markers = _ordered_markers_()
        assert variable_pairs.ordering.find_ordering_violations(markers) == []

    def test_reversed_markers_report_age_and_displacement(self):
        markers = dict(reversed(list(_ordered_markers_().items())))

        violations = variable_pairs.ordering.find_ordering_violations(markers)

        assert len(violations) == 4
        assert any(
            v.startswith("Marker 'middle' appears to be younger than 'old'")
            for v in violations
        )
        assert any(
            v.startswith(
                "Marker 'young' appears to be less displaced than 'middle'"
            )
            for v in violations
        )

    def test_only_age_out_of_order(self):
        markers = {
            "a": _marker_("a", 10.0, 10.0),
            "b": _marker_("b", 5.0, 20.0),
        }

        violations = variable_pairs.ordering.find_ordering_violations(markers)

        assert len(violations) == 1
        assert violations[0].startswith(
            "Marker 'b' appears to be younger than 'a'"
        )

    def test_overlapping_markers_are_not_violations(self):
        """Markers with overlapping PDFs are handled by conditioning."""
        markers = {
            "a": _marker_("a", 10.0, 10.0),
            "b": _marker_("b", 9.0, 9.0),
        }
        assert variable_pairs.ordering.find_ordering_violations(markers) == []

        # ...but the reader's more sensitive limit still flags them
        violations = variable_pairs.ordering.find_ordering_violations(
            markers, limit=0.5
        )
        assert len(violations) == 2

    def test_equal_means_are_not_violations(self):
        markers = {
            "a": _marker_("a", 10.0, 10.0),
            "b": _marker_("b", 10.0, 10.0),
        }
        assert variable_pairs.ordering.find_ordering_violations(markers) == []


class TestComputeReversalProbability:
    def test_ordered_markers_near_zero(self):
        young, old = _marker_("y", 5.0, 10.0), _marker_("o", 25.0, 40.0)
        p = variable_pairs.ordering.compute_reversal_probability(
            old.age, young.age
        )
        assert p == pytest.approx(0.0, abs=1e-6)

    def test_identical_markers_one_half(self):
        marker = _marker_("m", 10.0, 10.0)
        p = variable_pairs.ordering.compute_reversal_probability(
            marker.age, marker.age
        )
        assert p == pytest.approx(0.5, abs=1e-3)

    def test_reversed_markers_near_one(self):
        young, old = _marker_("y", 5.0, 10.0), _marker_("o", 25.0, 40.0)
        p = variable_pairs.ordering.compute_reversal_probability(
            young.age, old.age
        )
        assert p == pytest.approx(1.0, abs=1e-6)


class TestCheckMarkerOrder:
    def test_ordered_markers_pass(self):
        variable_pairs.ordering.check_marker_order(_ordered_markers_())

    def test_reversed_markers_raise_naming_markers(self):
        markers = dict(reversed(list(_ordered_markers_().items())))

        with pytest.raises(ValueError, match="youngest") as exc:
            variable_pairs.ordering.check_marker_order(markers)

        assert "'middle'" in str(exc.value)
        assert "'old'" in str(exc.value)

    def test_overlapping_reversed_markers_pass(self):
        markers = {
            "a": _marker_("a", 10.0, 10.0),
            "b": _marker_("b", 9.0, 9.0),
        }
        variable_pairs.ordering.check_marker_order(markers)


# end of file
