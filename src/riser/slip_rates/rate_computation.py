# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


"""
These functions are used to compute incremental slip rates.
They implement the same basic concept:
That slip rate v is the averaged change in displacement over the change in time

    v = DeltaU / DeltaT

Incremental slip rates average over shorter subsets of time within the overall
record.
"""


# Import modules
from datetime import datetime

import numpy as np

import riser.probability_functions as PDFs
import riser.variable_operations as var_ops
from riser.markers import DatedMarker
from riser import variable_types, units
from riser.sampling import mc_sampling, pdf_formation, filtering


#################### ANALYTIC COMPUTATION ####################
def compute_slip_rate(
    marker: DatedMarker,
    max_rate: float=100.0,
    dq: float=0.01,
    verbose: bool=False
) -> PDFs.PDF:
    """Compute a single slip rate based on a dated displacement marker.

    Args    marker - DatedMarker, displacement-age pair used to calculate
                slip rate
            max_rate - float, maximum quotient value to consider
            dq - float, quotient step
    Returns slip_rate - PDF
    """
    if verbose == True:
        print("Computing slip rate")

    # Divide displacement by age
    slip_rate = var_ops.divide_variables(
        numerator=marker.displacement,
        denominator=marker.age,
        max_quotient=max_rate,
        dq=dq,
        name=marker.name,
        variable_type="slip rate"
    )

    return slip_rate


def compute_slip_rates_analytical(
    markers:dict,
    *,
    max_rate: float=100.0,
    dq: float=0.01,
    limit_positive: bool=False,
    verbose: bool=False,
) -> dict:
    """Compute the incremental slip rates between multiple dated displacement
    markers using analytical functions.
    First, compute the difference between each pair of adjacent displacements
    and corresponding pairs of ages to get DeltaD's and DeltaT's.
    Then, compute the slip rate over each increment by dividing the DeltaD by
    the corresponding DeltaT.

    Note: Per the divide_variables operator, denominator (age) values cannot
    be negative.

    Args    markers - dict, DatedMarkers bounding each interval
            max_rate - float, maximum quotient value to consider
            dq - float, quotient step
            limit_positive - bool, enforce condition that displacement values
                must be positive
    Return slip_rates - dict, incremental slip rates
    """
    # Marker parameters
    n_markers = len(markers)
    marker_names = [*markers.keys()]

    # Number of slip rates
    n_rates = n_markers - 1

    if verbose == True:
        print(f"Computing {n_rates} incremental slip rates")

    # Check age variable types
    variable_types.check_same_pdf_variable_types(
        [marker.age for marker in markers.values()])

    # Check displacement variable types
    variable_types.check_same_pdf_variable_types(
        [marker.displacement for marker in markers.values()])

    # Check age units
    units.check_same_pdf_units(
        [marker.age for marker in markers.values()])

    # Check displacement units
    units.check_same_pdf_units(
        [marker.displacement for marker in markers.values()])

    # Empty dictionary to store slip rates
    slip_rates = {}

    # Loop through marker pairs
    for i in range(n_rates):
        # Younger marker
        younger_name = marker_names[i]
        younger_marker = markers[younger_name]

        # Older marker
        older_name = marker_names[i+1]
        older_marker = markers[older_name]

        # Interpolate ages on same axis
        (younger_age,
         older_age) = PDFs.interpolation.interpolate_pdfs(
                [younger_marker.age, older_marker.age]
        )

        # Compute age difference - negative ages not supported
        DeltaT = var_ops.subtract_variables(
            pdf1=older_age,
            pdf2=younger_age,
            limit_positive=True
        )

        # Interpolate displacements on same axis
        (younger_displacement,
         older_displacement) = PDFs.interpolation.interpolate_pdfs(
                [younger_marker.displacement, older_marker.displacement]
        )

        # Compute displacement difference
        DeltaU = var_ops.subtract_variables(
            pdf1=older_displacement,
            pdf2=younger_displacement,
            limit_positive=limit_positive
        )

        # Formulate incremental slip rate name
        rate_name = f"{older_marker.name}-{younger_marker.name}"

        # Divide displacement by age
        slip_rate = var_ops.divide_variables(
            numerator=DeltaU,
            denominator=DeltaT,
            max_quotient=max_rate,
            dq=dq,
            name=rate_name,
            variable_type="slip rate"
        )

        # Report if requested
        if verbose == True:
            print(slip_rate)
            print(
                f"Mean: {PDFs.analytics.pdf_mean(slip_rate):.3f} "
                f"+- {PDFs.analytics.pdf_std(slip_rate):.3f}"
            )

        # Record to slip rate dictionary
        slip_rates[rate_name] = slip_rate

    return slip_rates


#################### MONTE CARLO COMPUTATION ####################
def compute_slip_rates_mc(
    markers: dict,
    criterion: mc_sampling.SampleCriterion,
    *,
    max_rate: float=100.0,
    dq: float=0.01,
    n_samples: int=1000000,
    hard_stop: int=1000000000,
    pdf_method: str="histogram",
    pdf_xmin: float | None=None,
    pdf_xmax: float | None=None,
    pdf_dx: float | None=None,
    smoothing_type: str | None=None,
    smoothing_width: int | None=None,
    verbose: bool=False,
) -> tuple:
    """Compute the incremental slip rates between multiple dated displacement
    markers using Monte Carlo sampling.

    Args    max
    """
    # Marker parameters
    n_markers = len(markers)
    marker_names = [*markers.keys()]

    # Number of slip rates
    n_rates = n_markers - 1

    if verbose == True:
        print(f"Computing {n_rates} incremental slip rates")

    # Check age variable types
    variable_types.check_same_pdf_variable_types(
        [marker.age for marker in markers.values()]
    )

    # Check displacement variable types
    variable_types.check_same_pdf_variable_types(
        [marker.displacement for marker in markers.values()]
    )

    # Check age units
    age_unit = units.check_same_pdf_units(
        [marker.age for marker in markers.values()]
    )

    # Check displacement units
    displacement_unit = units.check_same_pdf_units(
        [marker.displacement for marker in markers.values()]
    )

    # Determine slip rate unit
    if all([displacement_unit is not None, age_unit is not None]):
        unit = f"{displacement_unit}/{age_unit}"
    else:
        unit = None

    # Conduct Monte Carlo sampling - valid MC samples are called picks
    (age_picks,
     disp_picks) = mc_sampling.sample_monte_carlo(
        markers=markers,
        criterion=criterion,
        n_samples=n_samples,
        hard_stop=hard_stop,
        verbose=verbose
    )

    # Compute incremental differences between picks for rate = DeltaU / DeltaT
    age_diffs = np.diff(age_picks, axis=0)
    disp_diffs = np.diff(disp_picks, axis=0)

    # Determine incremental slip rates from DeltaU / DeltaT
    rate_picks = disp_diffs / age_diffs

    # Retrieve slip rate picks-to-PDF formation function
    pdf_fcn = pdf_formation.get_pdf_formation_function(pdf_method)

    # Empty dictionary to store slip rates
    slip_rates = {}

    # Loop through incremental slip rates
    for i in range(n_rates):
        # Formulate incremental slip rate name
        rate_name = f"{marker_names[i+1]}-{marker_names[i]}"

        # Form incremental slip rate samples into PDFs
        slip_rate = pdf_fcn(
            samples=rate_picks[i,:],
            xmin=pdf_xmin,
            xmax=pdf_xmax,
            dx=pdf_dx,
            name=rate_name,
            variable_type="slip rate",
            unit=unit,
            verbose=verbose
        )

        # Smooth sampled function
        if all([
            smoothing_type is not None,
            smoothing_width is not None
        ]):
            slip_rate = filtering.filter_pdf(
                pdf=slip_rate,
                filter_type=smoothing_type,
                filter_width=smoothing_width,
                preserve_edges=True,
                verbose=verbose
            )

        # Report if requested
        if verbose == True:
            print(slip_rate)
            print(
                f"Mean: {np.mean(rate_picks[i,:]):.3f} "
                f"+- {np.std(rate_picks[i,:]):.3f}"
            )

        # Record to slip rate dictionary
        slip_rates[rate_name] = slip_rate

    return slip_rates, age_picks, disp_picks, rate_picks


# end of file