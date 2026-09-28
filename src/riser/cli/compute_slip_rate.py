#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.


# Import modules
import argparse

import matplotlib.pyplot as plt

from riser import (
    constants,
    plotting,
    probability_functions as PDFs,
    variable_pairs,
)
from riser.slip_rates import rate_computation, reporting


#################### ARGUMENT PARSER ####################
description = (
    "Compute the slip rate for a single marker by dividing feature "
    "displacement by age, using the analytical formulation."
)

examples = """Examples:
riser-compute-slip-rate --age age_pdf.txt --displacement disp_pdf.txt -o rate
riser-compute-slip-rate marker_config.toml -o v1
riser-compute-slip-rate marker_config.toml --age-unit-out y --displacement-unit-out mm -o "v2/v2"
"""

def create_parser():
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=examples,
    )

    return parser

def cmd_parser(iargs=None):
    parser = create_parser()

    input_args = parser.add_argument_group("Inputs")
    input_args.add_argument(dest="marker_config",
        type=str,
        nargs="?",
        help="Dated displacement marker configuration file. "
             "Optional if age and displacement PDFs specified directly.")
    input_args.add_argument("--age", dest="age_fname",
        type=str,
        help="File name of age PDF. Optional if marker file specified.")
    input_args.add_argument("--displacement", dest="displacement_fname",
        type=str,
        help="File name of displacement PDF. "
             "Optional if marker file specified."
        )

    rate_args = parser.add_argument_group("Slip rates")
    rate_args.add_argument("--limit-positive", dest="limit_positive",
        action="store_true",
        help="Enforce the condition that values are >= to 0.")
    rate_args.add_argument("--max-rate", dest="max_rate",
        type=float,
        help="Maximum slip rate to consider.")
    rate_args.add_argument("--dv", dest="dv",
        type=float, default=0.01,
        help="Slip rate step. [0.01]")

    output_args = parser.add_argument_group("Outputs")
    output_args.add_argument("-o", "--output-prefix", dest="output_prefix",
        type=str, required=True,
        help="Output prefix as <prefix> or <folder>/<prefix>.")

    unit_args = parser.add_argument_group("Units")
    unit_args.add_argument("--age-unit-out", dest="age_unit_out",
        type=str,
        help="Output age units.")
    unit_args.add_argument(
        "--displacement-unit-out", dest="displacement_unit_out",
        type=str,
        help="Output displacement units.")

    metadata_args = parser.add_argument_group("Metadata")
    metadata_args.add_argument("--name", dest="name",
        type=str, default=None,
        help="Name of the output slip rate PDF.")

    reporting_args = parser.add_argument_group("Reporting")
    reporting_args.add_argument(
        "--confidence-metric", dest="confidence_metric",
        type=str, choices=PDFs.analytics.PDF_CONFIDENCE_METRICS,
        default=PDFs.analytics.DEFAULT_CONFIDENCE_METRIC,
        help=f"Function for computing function confidence. "
             f"[{PDFs.analytics.DEFAULT_CONFIDENCE_METRIC}]")
    reporting_args.add_argument(
        "--confidence-limits", dest="confidence_limits",
        type=float, default=constants.Psigma["1"],
        help=f"Confidence level. [{constants.Psigma['1']:.2f}]")

    diagnostic_args = parser.add_argument_group("Diagnostics")
    diagnostic_args.add_argument("-v", "--verbose", dest="verbose",
        action="store_true",
        help="Verbose mode.")
    diagnostic_args.add_argument("-p", "--plot", dest="plot",
        action="store_true",
        help="Show results plots. Figures will be generated and saved "
             "whether plot flag is raised.")

    return parser.parse_args(args=iargs)


#################### INPUT PARSING ####################
def parse_inputs(
    marker_config: str | None,
    age_fname: str | None,
    displacement_fname: str | None,
    verbose: bool = False,
) -> dict[str, variable_pairs.DatedMarker]:
    """Determine whether the age and displacement data used to calculate the
    slip rate are provided as a marker file or individual PDFs.

    Raise an error if it is ambiguous.
    """
    # Marker file provided, and age and/or displacement files provided
    if (
        marker_config is not None
        and (age_fname is not None or displacement_fname is not None)
    ):
        raise ValueError(
            "PDFs for determining slip rate should be provided either as a "
            "TOML-based marker file (see RISeR2/examples) "
            "or as individual PDF files via the --age and --displacement "
            "flags."
        )

    # Read dated displacement marker
    if marker_config is not None:
        # ... from marker config file
        markers = variable_pairs.readers.read_dated_markers_from_config(
            marker_config, verbose=verbose
        )

    else:
        # ... from direct specification of age and displacement PDFs
        if age_fname is None:
            raise ValueError(
                "A PDF representing the marker age must be passed via the "
                "--age option"
            )

        if displacement_fname is None:
            raise ValueError(
                "A PDF representing the marker displacement must be passed via "
                "the --displacement option"
            )

        # Read age and displacement PDFs as dated marker
        marker = variable_pairs.readers.initialize_dated_marker_from_files(
            age_fname=age_fname,
            displacement_fname=displacement_fname,
            marker_name="marker",
            verbose=verbose,
        )

        # Format dated marker as dict
        markers = {"marker": marker}

    return markers


#################### MAIN ####################
def main() -> None:
    # Parse arguments
    inps = cmd_parser()

    # Establish output directory
    reporting.establish_output_dir(inps.output_prefix, verbose=inps.verbose)

    # Read markers based on inputs
    markers = parse_inputs(
        marker_config=inps.marker_config,
        age_fname=inps.age_fname,
        displacement_fname=inps.displacement_fname,
        verbose=inps.verbose,
    )

    # Check that only one marker is specified
    if len(markers) > 1:
        raise ValueError("Only one marker can be specified")

    # Use only first marker
    marker = next(iter(markers.values()))

    # Scale input units to output units
    age_unit_out = (
        marker.age.unit if inps.age_unit_out is None else inps.age_unit_out
    )

    marker.age = PDFs.scaling.scale_pdf_by_units(
        marker.age, age_unit_out, verbose=inps.verbose
    )

    displacement_unit_out = (
        marker.displacement.unit if inps.displacement_unit_out is None
        else inps.displacement_unit_out
    )

    marker.displacement = PDFs.scaling.scale_pdf_by_units(
        marker.displacement, displacement_unit_out, verbose=inps.verbose
    )

    # Initialize figure and axis for input marker
    marker_fig, marker_ax = plt.subplots()

    # Plot marker
    plotting.variable_pair_plots.plot_variable_pair_whisker(
        marker_ax, marker, label=True
    )

    # Format marker fig
    plotting.variable_pair_plots.set_origin_zero(marker_ax)
    plotting.variable_pair_plots.format_marker_plot(marker_ax, marker)

    # Save marker fig
    reporting.save_marker_fig(
        inps.output_prefix, marker_fig, verbose=inps.verbose
    )

    # Format metadata
    name = inps.name
    variable_type = "slip rate"
    unit = f"{displacement_unit_out}/{age_unit_out}"

    # Compute slip rate
    slip_rate = rate_computation.compute_slip_rate(
        marker=marker,
        dv=inps.dv,
        limit_positive=inps.limit_positive,
        max_rate=inps.max_rate,
        name=name,
        variable_type=variable_type,
        unit=unit,
        verbose=inps.verbose,
    )

    # Save PDF to file
    rate_outname = f"{inps.output_prefix}_{slip_rate.name}_slip_rate.txt"
    PDFs.readers.save_pdf(rate_outname, slip_rate, verbose=inps.verbose)

    # Compute PDF statistics
    pdf_stats = PDFs.analytics.compute_pdf_statistics(
        slip_rate, verbose=inps.verbose
    )

    # Compute confidence range
    conf_range = PDFs.analytics.compute_pdf_confidence_range(
        pdf=slip_rate,
        metric=inps.confidence_metric,
        confidence=inps.confidence_limits,
        verbose=inps.verbose,
    )

    # Initialize figure and axis for slip rate PDF
    rate_fig, rate_ax = plt.subplots()

    # Plot slip rate PDF
    plotting.plot_pdf_labeled(rate_ax, slip_rate)

    # Plot confidence range
    plotting.plot_pdf_confidence_range(rate_ax, slip_rate, conf_range)

    # Save slip rate figure
    reporting.save_slip_rate_fig(
        inps.output_prefix, rate_fig, verbose=inps.verbose
    )

    # Save slip rate report to file
    if marker.name is None:
        raise AssertionError("Marker name is guaranteed non-None.")

    reporting.write_slip_rates_report(
        output_prefix=inps.output_prefix,
        formulation="analytical",
        slip_rates={marker.name: slip_rate},
        pdf_statistics={marker.name: pdf_stats},
        verbose=inps.verbose
    )

    # Plot if requested
    if inps.plot:
        plt.show()


if __name__ == "__main__":
    main()


# end of file
