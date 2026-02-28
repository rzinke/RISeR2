#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

# Constants
from riser.constants import Psigma
from riser.probability_functions.analytics import PDF_CONFIDENCE_METRICS


# Import modules
import argparse

import matplotlib.pyplot as plt

import riser.probability_functions as PDFs
from riser.markers import readers as marker_readers
from riser.slip_rates import rate_computation, reporting
from riser import units, plotting


#################### ARGUMENT PARSER ####################
description = (
    "Compute the slip rate for a single marker by dividing feature "
    "displacement by age, using the analytical formulation."
)

examples = """Examples:
compute_slip_rate.py marker_config.toml -o v1
compute_slip_rate.py marker_config.toml --age-unit-out y --displacement-unit-out mm -o v2/v2
"""

def create_parser():
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=examples
    )

    return parser

def cmd_parser(iargs=None):
    parser = create_parser()

    # Inputs
    input_args = parser.add_argument_group("Inputs")
    input_args.add_argument(dest="marker_config",
        type=str,
        help="Dated displacement marker configuration file.")

    # Units
    unit_args = parser.add_argument_group("Units")
    unit_args.add_argument("--age-unit-out", dest="age_unit_out",
        type=str,
        help="Output age units.")
    unit_args.add_argument(
        "--displacement-unit-out", dest="displacement_unit_out",
        type=str,
        help="Output displacement units.")

    # Slip rate
    rate_args = parser.add_argument_group("Slip rates")
    rate_args.add_argument("--max-rate", dest="max_rate",
        type=float, default=100,
        help="Maximum slip rate to consider. [100]")
    rate_args.add_argument("--dv", dest="dv",
        type=float, default=0.01,
        help="Slip rate step. [0.01]")

    # Reporting
    reporting_args = parser.add_argument_group("Reporting")
    reporting_args.add_argument(
        "--confidence-metric", dest="confidence_metric",
        type=str, choices=PDF_CONFIDENCE_METRICS, default="HPD",
        help="Function for computing function confidence. [HPD]")
    reporting_args.add_argument(
        "--confidence-limits", dest="confidence_limits",
        type=float, default=Psigma["1"],
        help="Confidence level. [0.682]")

    # Outputs
    output_args = parser.add_argument_group("Outputs")
    output_args.add_argument("-o", "--output-prefix", dest="output_prefix",
        type=str, required=True,
        help="Output prefix as <prefix> or <folder>/<prefix>.")
    output_args.add_argument("-v", "--verbose", dest="verbose",
        action="store_true",
        help="Verbose mode.")
    output_args.add_argument("-p", "--plot", dest="plot",
        action="store_true",
        help="Show results plots. Figures will be generated and saved "
             "whether plot flag is raised.")

    return parser.parse_args(args=iargs)


#################### MAIN ####################
def main():
    # Parse arguments
    inps = cmd_parser()

    # Establish output directory
    reporting.establish_output_dir(inps.output_prefix, verbose=inps.verbose)

    # Read markers
    markers = marker_readers.read_markers_from_config(
        inps.marker_config, verbose=inps.verbose
    )

    # Check that only one marker is specified
    if len(markers) > 1:
        raise Exception("Only one marker can be specified")

    # Use only first marker
    marker = [*markers.values()][0]

    # Initialize figure and axis for input marker
    marker_fig, marker_ax = plt.subplots()

    # Plot marker
    plotting.plot_marker_whisker(marker_ax, marker, label=True)

    # Format marker fig
    plotting.set_origin_zero(marker_ax)
    plotting.format_marker_plot(marker_ax, marker)

    # Save marker fig
    reporting.save_marker_fig(
        inps.output_prefix, marker_fig, verbose=inps.verbose
    )

    # Scale input units to output units
    marker.age = units.scale_pdf_by_units(marker.age, inps.age_unit_out)

    marker.displacement = units.scale_pdf_by_units(
        marker.displacement, inps.displacement_unit_out
    )

    # Compute slip rate
    slip_rate = rate_computation.compute_slip_rate(
        marker=marker,
        max_quotient=inps.max_rate,
        dq=inps.dv
    )

    # Save PDF to file
    rate_outname = f"{inps.output_prefix}_{slip_rate.name}_slip_rate.txt"
    PDFs.readers.save_pdf(rate_outname, slip_rate, verbose=inps.verbose)

    # Compute PDF statistics
    pdf_stats = PDFs.analytics.PDFstatistics(slip_rate)
    if inps.verbose == True:
        print(pdf_stats)

    # Retrieve confidence range function
    conf_fcn = PDFs.analytics.get_pdf_confidence_function(
        inps.confidence_metric, verbose=inps.verbose
    )

    # Compute confidence range
    conf_range = conf_fcn(slip_rate, inps.confidence_limits)
    if inps.verbose == True:
        print(conf_range)

    # Initialize figure and axis for slip rate PDF
    rate_fig, rate_ax = plt.subplots()

    # Plot slip rate PDF
    plotting.plot_pdf_labeled(rate_ax, slip_rate)

    # Plot confidence range
    plotting.plot_pdf_confidence_range(
        rate_ax, slip_rate, conf_range
    )

    # Save slip rate figure
    reporting.save_slip_rate_fig(
        inps.output_prefix, rate_fig, verbose=inps.verbose
    )

    # Save slip rate report to file
    reporting.write_slip_rates_report(
        output_prefix=inps.output_prefix,
        formulation="analytical",
        slip_rates={marker.name: slip_rate},
        pdf_statistics={marker.name: pdf_stats},
        verbose=inps.verbose
    )

    # Plot if requested
    if inps.plot == True:
        plt.show()


if __name__ == "__main__":
    main()


# end of file