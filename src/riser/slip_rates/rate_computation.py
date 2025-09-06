# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


"""
These functions are used to compute
They implement the same basic concept: That slip rate v is the averaged
change in displacement over the change in time

    v = DeltaU / DeltaT

Incremental slip rates average over shorter subsets of time within the overall
record.
"""


# Import modules
from datetime import datetime

import numpy as np

from riser.probability_functions import PDF, interpolation, analytics
from riser.markers import DatedMarker
from riser import variable_types, units, variable_operations
from riser.sampling import mc_sampling, pdf_formation, filtering


#################### ANALYTIC COMPUTATION ####################
def compute_slip_rate(marker:DatedMarker, verbose=False, **kwargs) -> PDF:
    """Compute a single slip rate based on a dated displacement marker.

    Args    marker - DatedMarker, displacement-age pair used to calculate
                slip rate
            kwargs - dict, keyword agruments for divide_variables
    Returns slip_rate - PDF
    """
    if verbose == True:
        print("Computing slip rate")

    # Divide displacement by age
    division_args = {
        'numerator': marker.displacement,
        'denominator': marker.age,
        'max_quotient': kwargs.get('max_quotient', 100.),
        'dq': kwargs.get('dq', 0.01),
        'name': marker.name,
        'variable_type': "slip rate",
    }
    slip_rate = variable_operations.divide_variables(**division_args)

    return slip_rate


def compute_slip_rates_analytical(
        markers:dict, verbose=False, **kwargs) -> dict:
    """Compute the incremental slip rates between multiple dated displacement
    markers using analytical functions.
    First, compute the difference between each pair of adjacent displacements
    and corresponding pairs of ages to get DeltaD's and DeltaT's.
    Then, compute the slip rate over each increment by dividing the DeltaD by
    the corresponding DeltaT.

    Note: Per the divide_variables operator, denominator (age) values cannot
    be negative.

    Args    markers - dict, DatedMarkers bounding each interval
            kwargs - dict, keyword arguments for:
                subtract_variables
                divide_variables
    Retrusn slip_rates - dict, incremental slip rates
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
         older_age) = interpolation.interpolate_pdfs(
                [younger_marker.age, older_marker.age])

        # Compute age difference - negative ages not supported
        age_args = {
            'pdf1': older_age,
            'pdf2': younger_age,
        }
        DeltaT = variable_operations.subtract_variables(**age_args,
                                                        limit_positive=True)

        # Interpolate displacements on same axis
        (younger_displacement,
         older_displacement) = interpolation.interpolate_pdfs(
                [younger_marker.displacement, older_marker.displacement])

        # Compute displacement difference
        disp_args = {
            'pdf1': older_displacement,
            'pdf2': younger_displacement,
            'limit_positive': kwargs.get('limit_positive', False)
        }
        DeltaU = variable_operations.subtract_variables(**disp_args)

        # Formulate incremental slip rate name
        rate_name = f"{older_marker.name}-{younger_marker.name}"

        # Divide displacement by age
        rate_args = {
            'numerator': DeltaU,
            'denominator': DeltaT,
            'max_quotient': kwargs.get('max_quotient', 100.),
            'dq': kwargs.get('dq', 0.01),
            'name': rate_name,
            'variable_type': "slip rate",
        }
        slip_rate = variable_operations.divide_variables(**rate_args)

        # Report if requested
        if verbose == True:
            print(slip_rate)
            print(f"Mean: {analytics.pdf_mean(slip_rate):.3f} "
                  f"+- {analytics.pdf_std(slip_rate):.3f}")

        # Record to slip rate dictionary
        slip_rates[rate_name] = slip_rate

    return slip_rates


#################### MONTE CARLO COMPUTATION ####################
def compute_slip_rates_mc(markers:dict, condition:str,
        n_samples:int=1000000, hard_stop:int=1000000000,
        pdf_method:str="histogram",
        pdf_xmin:float=None, pdf_xmax:float=None, pdf_dx:float=None,
        smoothing_type:str=None, smoothing_width:int=None,
        verbose=False, **kwargs):
    """Compute the incremental slip rates between multiple dated displacement
    markers using Monte Carlo sampling.
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
    age_unit = units.check_same_pdf_units(
        [marker.age for marker in markers.values()])

    # Check displacement units
    displacement_unit = units.check_same_pdf_units(
        [marker.displacement for marker in markers.values()])

    # Determine slip rate unit
    if all([displacement_unit is not None, age_unit is not None]):
        unit = f"{displacement_unit}/{age_unit}"
    else:
        unit = None

    # Retrieve function for sample criterion
    criterion = mc_sampling.get_sample_criterion(condition)

    # Conduct Monte Carlo sampling - valid MC samples are called picks
    mc_args = {
        'markers': markers,
        'criterion': criterion,
        'n_samples': n_samples,
        'hard_stop': hard_stop,
    }
    (age_picks,
     disp_picks) = mc_sampling.sample_monte_carlo(**mc_args, **kwargs,
                                                  verbose=verbose)

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
        pdf_args = {
            'samples': rate_picks[i,:],
            'xmin': pdf_xmin,
            'xmax': pdf_xmax,
            'dx': pdf_dx,
            'name': rate_name,
            'variable_type': "slip rate",
            'unit': unit,
        }
        slip_rate = pdf_fcn(**pdf_args, verbose=verbose)

        # Smooth sampled function
        if all([smoothing_type is not None, smoothing_width is not None]):
            filter_args = {
                'pdf': slip_rate,
                'filter_type': smoothing_type,
                'filter_width': smoothing_width,
            }
            slip_rate = filtering.filter_pdf(
                    **filter_args, preserve_edges=True, verbose=verbose)

        # Report if requested
        if verbose == True:
            print(slip_rate)
            print(f"Mean: {np.mean(rate_picks[i,:]):.3f} "
                  f"+- {np.std(rate_picks[i,:]):.3f}")

        # Record to slip rate dictionary
        slip_rates[rate_name] = slip_rate

    return slip_rates, age_picks, disp_picks, rate_picks


# end of file