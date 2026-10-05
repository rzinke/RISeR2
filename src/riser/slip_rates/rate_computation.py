# -*- coding: utf-8 -*-
#
# Copyright (c) 2025, 2026 Robert Zinke. Licensed under the MIT License.


"""
These functions are used to compute incremental slip rates.
They implement the same basic concept:
That slip rate v is the averaged change in displacement over the change in time

    v = delta_u / delta_t

Incremental slip rates average over shorter subsets of time within the overall
record.
"""

# Public API
__all__ = [
    "compute_slip_rate",
    "compute_slip_rates_analytical",
    "compute_slip_rates_mc",
]


# Import modules
import copy

import numpy as np

from .. import (
    probability_functions as PDFs,
    variable_functions as var_fcns,
    variable_pairs,
)
from ..sampling import filtering, mc_sampling, pdf_formation


#################### SINGLE SLIP RATE ANALYTIC COMPUTATION ####################
def compute_slip_rate(
    marker: variable_pairs.DatedMarker,
    *,
    # Slip rate
    dv: float = 0.01,
    limit_positive: bool = False,
    max_rate: float = 100.0,
    # PDF metadata
    name: str | None = None,
    variable_type: str | None = None,
    unit: str | None = None,
    # Misc
    verbose: bool = False,
) -> PDFs.PDF:
    """Compute a single slip rate based on a dated displacement marker.

    Parameters
    ----------
    marker : DatedMarker
        Displacement-age pair used to calculate slip rate.
    dv : float, optional
        Rate step.
    limit_positive : bool, optional
        Enforce condition that slip rate is >= 0.0.
    max_rate : float, optional
        Maximum quotient value to consider.
    name : str, optional
        Name of slip rate PDF.
    variable_type : str, optional
        Variable type of slip rate PDF.
    unit : str, optional
        Unit of slip rate PDF.

    Returns
    -------
    slip_rate : PDF
        Computed slip rate.
    """
    if verbose:
        print("Computing slip rate")

    # Set mimimum slip rate
    min_rate = 0.0 if limit_positive else None

    # Format metadata
    name = name if name is not None else marker.name
    variable_type = variable_type if variable_type is not None else "slip rate"

    # Divide displacement by age
    slip_rate, _ = var_fcns.transform.arithmetic.divide_variables(
        pdf1=marker.displacement,
        pdf2=marker.age,
        dz=dv,
        min_quotient=min_rate,
        max_quotient=max_rate,
        name=name,
        variable_type=variable_type,
        unit=unit,
    )

    return slip_rate


#################### MULTI SLIP RATE ANALYTIC COMPUTATION ####################
def _forward_trim_pdfs_(pdfs: list[PDFs.PDF], verbose: bool = False):
    if verbose:
        print("Forward-trimming PDFs")

    # Copy list of PDFs to avoid overwriting
    trimmed_pdfs = copy.deepcopy(pdfs)

    # Loop through PDFs
    for i in range(1, len(pdfs)):
        # Trim adjacent PDFs
        _, trimmed_pdf, _ = var_fcns.condition.trimming.trim_variables(
            pdf1=pdfs[i - 1],
            pdf2=pdfs[i],
            name1=pdfs[i - 1].name,
            name2=pdfs[i].name,
        )

        # Overwrite list value
        trimmed_pdfs[i] = trimmed_pdf

    return trimmed_pdfs

def _backward_trim_pdfs_(pdfs: list[PDFs.PDF], verbose: bool = False):
    if verbose:
        print("Backward-trimming PDFs")

    # Copy list of PDFs to avoid overwriting
    trimmed_pdfs = copy.deepcopy(pdfs)

    # Loop through PDFs
    for i in range(1, len(pdfs)):
        # Trim adjacent PDFs
        trimmed_pdf, _, _ = var_fcns.condition.trimming.trim_variables(
            pdf1=pdfs[-i - 1],
            pdf2=pdfs[-i],
            name1=pdfs[-i - 1].name,
            name2=pdfs[-i].name,
        )

        # Overwrite list value
        trimmed_pdfs[-i - 1] = trimmed_pdf

    return trimmed_pdfs

def compute_slip_rates_analytical(
    markers: dict[str, variable_pairs.DatedMarker],
    *,
    # Slip rate
    dv: float = 0.01,
    limit_positive: bool = False,
    max_rate: float = 100.0,
    # PDF metadata
    variable_type: str | None = None,
    unit: str | None = None,
    # Misc
    verbose: bool = False,
) -> dict[str, PDFs.PDF]:
    """Compute the incremental slip rates between multiple dated displacement
    markers using analytical functions.

    First, compute the difference between each pair of adjacent displacements
    and corresponding pairs of ages to get `delta_u`s and `delta_t`s.
    Then, compute the slip rate over each increment by dividing the `delta_u`
    by the corresponding `delta_t`.

    Note: Per the divide_variables operator, denominator (age) values cannot
    be negative, or zero. The "limit positive" condition is always applied to
    time values.

    Parameters
    ----------
    markers - dict[str, DatedMarker]
        Dated markers bounding each interval.
    max_rate : float
        Maximum quotient value to consider.
    dv : float, optional
        Rate step.
    limit_positive : bool, optional
        Enforce condition that displacement difference values must be positive.
        Time differences are always positive.
    variable_type : str, optional
        Variable type of slip rate PDF.
    unit : str, optional
        Unit of slip rate PDF.

    Returns
    -------
    slip_rates : dict[str, PDF]
        Incremental slip rates.
    """
    # Marker parameters
    n_markers = len(markers)
    marker_names = [*markers.keys()]

    # Check that multiple markers are specified
    if n_markers < 2:
        raise ValueError(
            f"Multiple markers must be specified for incremental slip rate "
            f"computation, got {n_markers}"
        )

    # Number of slip rates
    n_rates = n_markers - 1

    if verbose:
        print(f"Computing {n_rates} incremental slip rates")

    # Interpolate ages and displacements on same domains
    markers = variable_pairs.interpolation.interpolate_variable_pairs(
        markers, verbose=verbose
    )

    import matplotlib.pyplot as plt
    from riser import plotting

    # Forward-trim ages and displacements
    forw_trimmed_ages = _forward_trim_pdfs_(
        [marker.age for marker in markers.values()], verbose=verbose
    )

    forw_trimmed_displacements = _forward_trim_pdfs_(
        [marker.displacement for marker in markers.values()]
    )

    # Backward-trim ages and displacements
    back_trimmed_ages = _backward_trim_pdfs_(
        [marker.age for marker in markers.values()], verbose=verbose
    )

    back_trimmed_displacements = _backward_trim_pdfs_(
        [marker.displacement for marker in markers.values()]
    )

    # Warn of metadata mismatches for ages
    age_metadata = PDFs.metadata.get_common_metadata(
        [marker.age.metadata for marker in markers.values()]
    )

    # Warn of metadata mismatches for displacements
    displacement_metadata = PDFs.metadata.get_common_metadata(
        [marker.displacement.metadata for marker in markers.values()]
    )

    # Variable type
    variable_type = variable_type if variable_type is not None else "slip rate"

    # Slip rate unit
    if (
        unit is None
        and age_metadata.unit is not None
        and displacement_metadata.unit is not None
    ):
        unit = f"{displacement_metadata.unit}/{age_metadata.unit}"

    # Set mimimum slip rate
    min_rate = 0.0 if limit_positive else None

    # Empty dictionary to store slip rates
    slip_rates = {}

    # Loop through marker pairs
    for i in range(n_rates):
        # Formulate incremental slip rate name
        rate_name = f"{marker_names[i + 1]}-{marker_names[i]}"

        if verbose:
            print(f"Computing slip rate for {rate_name}")

        # Younger marker
        younger_name = marker_names[i]
        younger_marker = markers[younger_name]

        # Older marker
        older_name = marker_names[i + 1]
        older_marker = markers[older_name]

        # Compute age difference - negative ages not supported
        younger_age = forw_trimmed_ages[i]
        older_age = back_trimmed_ages[i + 1]
        delta_t = var_fcns.transform.arithmetic.subtract_variables(
            pdf1=older_age,
            pdf2=younger_age,
            limit_positive=True,
            verbose=verbose,
        )

        # Compute displacement difference
        younger_displacement = forw_trimmed_displacements[i]
        older_displacement = back_trimmed_displacements[i + 1]
        delta_u = var_fcns.transform.arithmetic.subtract_variables(
            pdf1=older_displacement,
            pdf2=younger_displacement,
            limit_positive=limit_positive,
            verbose=verbose,
        )

        # Divide displacement by age
        slip_rate, _ = var_fcns.transform.arithmetic.divide_variables(
            pdf1=delta_u,
            pdf2=delta_t,
            dz=dv,
            min_quotient=min_rate,
            max_quotient=max_rate,
            name=rate_name,
            variable_type=variable_type,
            unit=unit,
        )

        # Report if requested
        if verbose:
            print(slip_rate)
            print(
                f"Mean: {PDFs.analytics.pdf_mean(slip_rate):.3f} "
                f"+- {PDFs.analytics.pdf_std(slip_rate):.3f}"
            )

        # Record to slip rate dictionary
        slip_rates[rate_name] = slip_rate

    return slip_rates


#################### MULTI RATE MONTE CARLO COMPUTATION ####################
def compute_slip_rates_mc(
    markers: dict[str, variable_pairs.DatedMarker],
    criterion: mc_sampling.SampleCriterion,
    *,
    # Sampling
    n_samples: int = 1_000_000,
    hard_stop: int = 1_000_000_000,
    # Slip rate
    dv: float = 0.01,
    pdf_method: str = "histogram",
    pdf_xmin: float | None = None,
    pdf_xmax: float | None = None,
    pdf_dx: float | None = None,
    smoothing_type: str | None = None,
    smoothing_width: int | None = None,
    # PDF metadata
    variable_type: str | None = None,
    unit: str | None = None,
    # Misc
    verbose: bool = False,
) -> tuple[dict[str, PDFs.PDF], np.ndarray, np.ndarray, np.ndarray]:
    """Compute the incremental slip rates between multiple dated displacement
    markers using Monte Carlo sampling.

    Parameters
    ----------
    markers - dict[str, DatedMarker]
        Dated markers bounding each interval.
    criterion : SampleCriterion
        Criterion by which to evaluate validity of samples.
    n_samples : int, optional
        Number of valid samples to achieve.
    hard_stop : float, optional
        Maximum number of trials, regardless of success.
    dv : float, optional
        Rate step.
    pdf_method : str, optional
        PDF formation method.
    pdf_xmin : float, optional
        Minimum value to consider.
    pdf_xmax : float, optional
        Maximum value to consider.
    pdf_dx : float, optional
        Value array step.
    smoothing_type : str, optional
        Smoothing filter type.
    smoothing_width : int, optional
        Smoothing filter width in number of samples.
    variable_type : str, optional
        Variable type of slip rate PDF.
    unit : str, optional
        Unit of slip rate PDF.

    Returns
    -------
    slip_rates : dict[str, PDF]
        Incremental slip rate PDFs.
    age_picks : np.ndarray
        Age samples that meet the sample criterion.
    disp_picks : np.ndarray
        Displacement samples that meet the sample criterion.
    rate_picks : np.ndarray
        Slip rate samples that meet the sample criterion.
    """
    # Marker parameters
    n_markers = len(markers)
    marker_names = [*markers.keys()]

    # Number of slip rates
    n_rates = n_markers - 1

    if verbose:
        print(f"Computing {n_rates} incremental slip rates")

    # Warn of metadata mismatches for ages
    age_metadata = PDFs.metadata.get_common_metadata(
        [marker.age.metadata for marker in markers.values()]
    )

    # Warn of metadata mismatches for displacements
    displacement_metadata = PDFs.metadata.get_common_metadata(
        [marker.displacement.metadata for marker in markers.values()]
    )

    # Determine slip rate unit
    if (
        unit is None
        and age_metadata.unit is not None
        and displacement_metadata.unit is not None
    ):
        unit = f"{displacement_metadata.unit}/{age_metadata.unit}"

    # Variable type
    variable_type = variable_type if variable_type is not None else "slip rate"

    # Conduct Monte Carlo sampling - valid MC samples are called picks
    age_picks, disp_picks, _ = mc_sampling.sample_monte_carlo(
        markers=markers,
        criterion=criterion,
        n_samples=n_samples,
        hard_stop=hard_stop,
        verbose=verbose,
    )

    # Compute incremental differences between picks for rate = delta_u / delta_t
    age_diffs = np.diff(age_picks, axis=0)
    disp_diffs = np.diff(disp_picks, axis=0)

    # Determine incremental slip rates from delta_u / delta_t
    rate_picks = disp_diffs / age_diffs

    # Retrieve slip rate picks-to-PDF formation function
    pdf_fcn = pdf_formation.get_pdf_formation_function(pdf_method)

    # Empty dictionary to store slip rates
    slip_rates = {}

    # Loop through incremental slip rates
    for i in range(n_rates):
        # Formulate incremental slip rate name
        rate_name = f"{marker_names[i + 1]}-{marker_names[i]}"

        # Form incremental slip rate samples into PDFs
        slip_rate = pdf_fcn(
            samples=rate_picks[i, :],
            xmin=pdf_xmin,
            xmax=pdf_xmax,
            dx=pdf_dx,
            name=rate_name,
            variable_type=variable_type,
            unit=unit,
            verbose=verbose,
        )

        # Smooth sampled function
        if smoothing_type is not None and smoothing_width is not None:
            slip_rate = filtering.filter_pdf(
                pdf=slip_rate,
                filter_type=smoothing_type,
                filter_width=smoothing_width,
                preserve_edges=True,
                verbose=verbose,
            )

        # Report if requested
        if verbose:
            print(slip_rate)
            print(
                f"Mean: {np.mean(rate_picks[i, :]):.3f} "
                f"+- {np.std(rate_picks[i, :]):.3f}"
            )

        # Record to slip rate dictionary
        slip_rates[rate_name] = slip_rate

    return slip_rates, age_picks, disp_picks, rate_picks


# end of file
