# -*- coding: utf-8 -*-
#
# Copyright (c) 2026 Robert Zinke. Licensed under the MIT License.

# Import modules
import pytest

from riser import (
    probability_functions as PDFs,
    slip_rates,
)


# Slip rates
def _slip_rates_():
    """
    One incremental slip rate PDF, keyed by marker pair name.
    """
    x = PDFs.value_arrays.precise_array(0.0, 10.0, 0.01)

    return {
        "B-A": PDFs.PDF(
            x=x,
            px=PDFs.parametric_functions.gaussian(x, mu=5.0, sigma=1.0),
            name="B-A",
            variable_type="slip rate",
            unit="mm/y",
        )
    }


def _read_report_lines_(output_prefix):
    """
    Lines of the slip rate report file written for the given prefix.
    """
    with open(f"{output_prefix}_slip_rate_report.txt") as report_file:
        return report_file.read().splitlines()


# Tests
class TestWriteSlipRatesReport:
    def test_conditions_written_below_header(self, tmp_path):
        """
        Provided conditions should be written on the line directly after the
        header.
        """
        prefix = str(tmp_path / "out")

        slip_rates.reporting.write_slip_rates_report(
            prefix,
            "analytical",
            _slip_rates_(),
            conditions="enforce ordering, limit positive",
        )

        lines = _read_report_lines_(prefix)
        assert lines[0].startswith("Incremental slip rates from")
        assert lines[1] == "Conditions: enforce ordering, limit positive"

    def test_no_conditions_no_conditions_line(self, tmp_path):
        """
        If no conditions are provided, no conditions line should be written.
        """
        prefix = str(tmp_path / "out")

        slip_rates.reporting.write_slip_rates_report(
            prefix, "analytical", _slip_rates_()
        )

        lines = _read_report_lines_(prefix)
        assert not any("Conditions" in line for line in lines)

    @pytest.mark.parametrize("formulation", ["analytical", "Monte Carlo"])
    def test_formulation_in_header(self, tmp_path, formulation):
        """
        The header should state the formulation that was passed.
        """
        prefix = str(tmp_path / "out")

        slip_rates.reporting.write_slip_rates_report(
            prefix, formulation, _slip_rates_()
        )

        header = _read_report_lines_(prefix)[0]
        assert f"from {formulation} formulation" in header

    def test_conditions_must_be_single_string(self, tmp_path):
        """
        A list of conditions should be rejected rather than written out in
        Python list notation.
        """
        prefix = str(tmp_path / "out")

        with pytest.raises(ValueError, match="single string"):
            slip_rates.reporting.write_slip_rates_report(
                prefix,
                "analytical",
                _slip_rates_(),
                conditions=["enforce ordering", "limit positive"],  # type: ignore[arg-type]
            )


# end of file
