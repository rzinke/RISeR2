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
    "find_slip_rate_tail_cap",
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


#################### TAIL MASS CAP ####################
def find_slip_rate_tail_cap(
    displacement: PDFs.PDF,
    age: PDFs.PDF,
    epsilon: float = 1e-2,
    verbose: bool = False,
) -> float:
    """Find the maximum slip rate value to consider based on the thickness
    of the slip rate PDF tail.

    Because slip rates can approach infinity as the age approaches zero,
    we can determine a "cap" or maximum slip rate that provides a finite
    maximum slip rate while capturing nearly all of the slip rate PDF mass.

    Note:
    This function is only valid for slip rates computed over non-negative age
    differences (Delta t >= 0.0).
    At least part of the displacement domain must be positive (> 0.0).

    Parameters
    ----------
    displacement : PDF
        Displacement PDF used in slip rate computation.
    age : PDF
        Age PDF used in slip rate computation.
    epsilon : float
        Fraction of the positive slip rate probability allowed to lie
        above v_max.

    Returns
    -------
    v_max : float
        Maximum slip rate value to consider.
    """
    if verbose:
        print("Determining maximum slip rate at which to cap slip rate PDF.")

    # Check epsilon within (0, 1)
    if not 0.0 < epsilon < 1.0:
        raise ValueError(
            f"`epsilon` must be between 0.0 and 1.0, got {epsilon}"
        )

    # Check smallest age is non-negative (>= 0.0)
    if np.min(age.x) < 0.0:
        raise ValueError(
            f"The denominator should only include non-negative values, "
            f"therefore the smallest age value should be >= 0.0, "
            f"got {np.min(age.x)}"
        )

    # Age step
    dt = PDFs.value_arrays.sample_spacing_from_pdf(age)

    # Array of positive ages
    t = age.x[age.x > 0]

    # Probabilities of positive ages
    ft = age.px[age.x > 0] * dt

    # Function to compute tail probability
    def tail_probability(v: float) -> float:
        """Compute the probability that the slip rate is greater than v,
        i.e.,

            P(V > v) = integral(f_T(t) · (1 - F_U(v · t)) · dt)
        """
        return np.sum(ft * (1 - displacement.cdf_at_value(v * t)))

    # Compute the probability that some slip rate values are positive:
    # P(V > 0.0) > 0.0
    positive_probability = tail_probability(0)

    # Ensure that at least some slip rate values are positive
    if positive_probability <= 0:
        raise ValueError("There are no possible positive slip rate values")

    # Area of tail to exclude based on threshold and area of positive
    # slip rate PDF
    area_to_exclude = epsilon * positive_probability

    # Establish lower bound:
    # minimum positive displacement / largest age
    lower_bound = np.min(displacement.x[displacement.x > 0]) / np.max(age.x)

    # Establish preliminary upper bound
    upper_bound = lower_bound

    # Double upper bound until it satisfies
    # P(V > upper_bound) <= epsilon · P(V > 0)
    while tail_probability(upper_bound) > area_to_exclude:
        upper_bound *= 2

    # Iteratively refine the lower and upper bounds
    # by bisecting them on a log scale
    for _ in range(30):
        # Use geometric mean as (log of) midpoint between lower and upper bounds
        midpoint = np.sqrt(lower_bound * upper_bound)

        # Check if midpoint satisfies probability that
        # slip rate meets target area
        if tail_probability(midpoint) > area_to_exclude:
            # Not enough area excluded - increase lower bound
            lower_bound = midpoint
        else:
            # Too much area excluded - decrease upper bound
            upper_bound = midpoint

    # Use refined upper bound as maximum slip rate to consider
    v_max = upper_bound

    if verbose:
        print(f"Maximum slip rate to consider: {v_max:.4f}")

    return v_max


#################### SINGLE SLIP RATE ANALYTIC COMPUTATION ####################
def compute_slip_rate(
    marker: variable_pairs.DatedMarker,
    *,
    # Slip rate
    limit_positive: bool = False,
    dv: float = 0.01,
    min_rate: float = 0.0,
    max_rate: float | None = None,
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
    limit_positive : bool, optional
        Enforce condition that slip rate is >= 0.0.
    dv : float, optional
        Rate step.
    min_rate : float, optional
        Minimum slip rate value to consider.
    max_rate : float, optional
        Maximum slip rate value to consider.
        If None, the maximum slip rate will be determined automatically
        based on the thickness of the slip rate tail.
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

    # Format metadata
    name = name if name is not None else marker.name
    variable_type = variable_type if variable_type is not None else "slip rate"

    # Limit to positive age values regardless of `limit_positive` flag
    marker.age, age_area = var_fcns.condition.self_constraint.constrain_above(
        pdf=marker.age,
        value=0.0,
        crop=True,
        name=marker.age.name,
        variable_type=marker.age.variable_type,
        unit=marker.age.unit,
        verbose=verbose,
    )

    # Report area of age PDF retained
    if verbose and age_area < 1.0:
        print(
            f"Limiting to positive ages only. "
            f"Fraction of age retained: {age_area:.3f}"
        )

    # Limit to positive displacement values
    if limit_positive:
        marker.displacement, displacement_area = (
            var_fcns.condition.self_constraint.constrain_above(
                pdf=marker.displacement,
                value=0.0,
                crop=True,
                name=marker.displacement.name,
                variable_type=marker.displacement.variable_type,
                unit=marker.displacement.unit,
                verbose=verbose,
            )
        )

        # Report area of age PDF retained
        if verbose and displacement_area < 1.0:
            print(
                f"Limiting to positive displacements only. "
                f"Fraction of displacement retained: {displacement_area:.3f}"
            )

    # Determine maximum slip rate to consider
    if max_rate is None:
        max_rate = find_slip_rate_tail_cap(
            displacement=marker.displacement,
            age=marker.age,
            verbose=verbose,
        )

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
            pdf1=trimmed_pdfs[i - 1],
            pdf2=trimmed_pdfs[i],
            name1=trimmed_pdfs[i - 1].name,
            name2=trimmed_pdfs[i].name,
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
            pdf1=trimmed_pdfs[-i - 1],
            pdf2=trimmed_pdfs[-i],
            name1=trimmed_pdfs[-i - 1].name,
            name2=trimmed_pdfs[-i].name,
        )

        # Overwrite list value
        trimmed_pdfs[-i - 1] = trimmed_pdf

    return trimmed_pdfs


def compute_slip_rates_analytical(
    markers: dict[str, variable_pairs.DatedMarker],
    *,
    # Slip rate
    enforce_ordering: bool = False,
    limit_positive: bool = False,
    dv: float = 0.01,
    min_rate: float = 0.0,
    max_rate: float | None = None,
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

    Sample ordering will be enforced through Bayesian conditioning when the
    `enforce_ordering` flag is passed. This trims the marker ages and
    displacements on the condition that the markers are provided in strict
    ordering (the first marker is younger/less displaceed than the second,
    etc.). Otherwise, each pair of adjacent markers will be treated
    independently.
    If ordering is enforced, displacements are limited to positive
    differences even if `limit_positive` is set to False.

    Note: Per the divide_variables operator, denominator (age) values cannot
    be negative, or zero. The "limit positive" condition is always applied to
    time values.

    Parameters
    ----------
    markers - dict[str, DatedMarker]
        Dated markers bounding each interval.
    enforce_ordering : bool, optional
        Trim the marker ages and displacements on the condition that the
        markers are provided in strict ordering.
    limit_positive : bool, optional
        Enforce condition that displacement difference values must be positive.
        Time differences are always positive.
    dv : float, optional
        Rate step.
    min_rate : float, optional
        Minimum slip rate value to consider.
    max_rate : float, optional
        Maximum slip rate value to consider.
        If None, the maximum slip rate will be determined automatically
        based on the thickness of the slip rate tail.
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

    # Pre-separate ages/displacements to be used as younger/older values in
    # each incremental slip rate:
    # rate = (older disp - younger disp) / (older age - younger age)
    ages = [marker.age for marker in markers.values()]
    displacements = [marker.displacement for marker in markers.values()]

    if enforce_ordering:
        # Trim the age and displacements based on order
        if verbose:
            print("Trimming age/displacement values based on marker order")

        # Forward-trim ages and displacements
        younger_ages = _forward_trim_pdfs_(ages, verbose=verbose)[:-1]
        younger_displacements = _forward_trim_pdfs_(displacements)[:-1]

        # Backward-trim ages and displacements
        older_ages = _backward_trim_pdfs_(ages, verbose=verbose)[1:]
        older_displacements = _backward_trim_pdfs_(displacements)[1:]

    else:
        # No order enforcement
        if verbose:
            print("Treating marker pairs independently")

        younger_ages = ages[:-1]
        younger_displacements = displacements[:-1]

        older_ages = ages[1:]
        older_displacements = displacements[1:]

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

    # Empty dictionary to store slip rates
    slip_rates = {}

    # Loop through marker pairs
    for i in range(n_rates):
        # Younger marker
        younger_name = marker_names[i]
        younger_marker = markers[younger_name]

        # Older marker
        older_name = marker_names[i + 1]
        older_marker = markers[older_name]

        # Slip rate marker pair
        rate_name = f"{older_marker.name}-{younger_marker.name}"
        if verbose:
            print(f"Computing slip rate for {rate_name}")

        # Compute age difference - negative ages not supported
        delta_t = var_fcns.transform.arithmetic.subtract_variables(
            pdf1=older_ages[i],
            pdf2=younger_ages[i],
            limit_positive=True,
            verbose=verbose,
        )

        # Compute displacement difference
        delta_u = var_fcns.transform.arithmetic.subtract_variables(
            pdf1=older_displacements[i],
            pdf2=younger_displacements[i],
            limit_positive=limit_positive,
            verbose=verbose,
        )

        # Determine maximum slip rate to consider
        if max_rate is None:
            max_incr_rate = find_slip_rate_tail_cap(
                displacement=delta_u,
                age=delta_t,
                verbose=verbose,
            )
        else:
            max_incr_rate = max_rate

        # Divide displacement by age
        slip_rate, _ = var_fcns.transform.arithmetic.divide_variables(
            pdf1=delta_u,
            pdf2=delta_t,
            dz=dv,
            min_quotient=min_rate,
            max_quotient=max_incr_rate,
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

    # Variable type
    variable_type = variable_type if variable_type is not None else "slip rate"

    # Determine slip rate unit
    if (
        unit is None
        and age_metadata.unit is not None
        and displacement_metadata.unit is not None
    ):
        unit = f"{displacement_metadata.unit}/{age_metadata.unit}"

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
