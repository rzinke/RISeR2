# -*- coding: utf-8 -*-
#
# Copyright (c) 2025, 2026 Robert Zinke. Licensed under the MIT License.

# Public API
__all__ = [
    "get_sample_criterion",
    "sample_monte_carlo",
]


# Import modules
import warnings

import numpy as np
from tqdm import tqdm

from .. import variable_pairs


#################### SAMPLING CRITERIA ####################
class SampleCriterion:
    def __init__(self, **kwargs) -> None:
        return

    def check_pass_fail(
        self,
        ages: np.ndarray,
        displacements: np.ndarray,
    ) -> bool:
        """Check whether a set of displacement-age pairs meets the sample
        criterion.

        Parameters
        ----------
        ages : np.ndarray
            Sampled age values.
        displacements : np.ndarray
            Sampled displacement values.

        Returns
        -------
        bool
            Pass (True) or fail (False).
        """
        raise NotImplementedError(
            "check_pass_fail not implemented. Override with child class."
        )

    def __str__(self) -> str:
        return "sample criterion"


class PassAll(SampleCriterion):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def check_pass_fail(
        self,
        ages: np.ndarray,
        displacements: np.ndarray,
    ) -> bool:
        """Trivial condition for which all random samples will be passed.
        (Allows negative slip rates).
        """
        return True

    def __str__(self) -> str:
        return "pass all"


class PassNonnegative(SampleCriterion):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def check_pass_fail(
        self,
        ages: np.ndarray,
        displacements: np.ndarray,
    ) -> bool:
        """Pass sample combinations that result in non-negative slip rates."""
        # Compute age, displacement deltas
        age_diffs = np.diff(ages)
        disp_diffs = np.diff(displacements)

        # Check condition
        return age_diffs.min() > 0 and disp_diffs.min() >= 0

    def __str__(self) -> str:
        return "pass non-negative"


class PassNonnegativeBounded(SampleCriterion):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Record maximum allowable sample rate
        self.max_sample_rate = kwargs.get("max_sample_rate", np.inf)

    def check_pass_fail(
        self,
        ages: np.ndarray,
        displacements: np.ndarray,
    ) -> bool:
        """Pass sample combinations that result in slip rates that are
        non-negativeand less than the specified maximum value.
        """
        # Compute age, displacement deltas
        age_diffs = np.diff(ages)
        disp_diffs = np.diff(displacements)

        # Compute incremental slip rates
        slip_rates = disp_diffs / age_diffs

        # Check condition
        return (
            age_diffs.min() > 0
            and disp_diffs.min() >= 0
            and slip_rates.max() <= self.max_sample_rate
        )

    def __str__(self) -> str:
        return f"pass non-negative (bounded {self.max_sample_rate})"


SAMPLE_CRITERIA = {
    "PassAll": PassAll,
    "PassNonnegative": PassNonnegative,
    "PassNonnegativeBounded": PassNonnegativeBounded,
}


def get_sample_criterion(
    criterion_name: str, verbose: bool = False
) -> type[SampleCriterion]:
    """Retrieve a sample criterion by name.

    Parameters
    ----------
    criterion_name : str
        Sample criterion name.

    Returns
    -------
    SampleCriterion
        Uninitialized sample criterion.
    """
    # Check criterion name
    if criterion_name not in SAMPLE_CRITERIA:
        raise ValueError(
            f"Sample criterion '{criterion_name}' not supported. "
            f"Use one of {', '.join(SAMPLE_CRITERIA.keys())}"
        )

    return SAMPLE_CRITERIA[criterion_name]


#################### MONTE CARLO SAMPLING ####################
def sample_monte_carlo(
    markers: dict[str, variable_pairs.DatedMarker],
    criterion: SampleCriterion,
    *,
    n_samples: int = 1_000_000,
    seed_val: int = 0,
    hard_stop: int = 100_000,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample valid possible slip rates using a Monte Carlo method.

    This function uses the inverse transform sampling method to randomly
    sample the displacement and age PDFs constraining a DatedMarker.
    The random samples are checked against a criterion, e.g., "no negative
    slip rates".

    Sampling continues until the desired number of valid picks is found.
    It stops early if `hard_stop` trials in a row fail the criterion.
    If no valid picks were found at that point, an error is raised, because
    the criterion is probably unsatisfiable, e.g., the markers are listed
    out of order.
    If some valid picks were found, a warning is raised and all valid picks
    are returned.

    The proportion of valid samples to total samples is stored and returned
    as a measure of how much area was rejected during the sampling process.

    Parameters
    ----------
    markers : dict[str, DatedMarker]
        Dated markers between which to compute incremental slip rates.
    criterion : SampleCriterion
        Criterion by which to evaluate validity of samples.
    n_samples : int
        Desired number of valid samples to achieve.
    seed_val : int
        Random number generator seed value.
    hard_stop : int
        Number of consecutive trials that may fail the criterion before
        sampling stops.

    Returns
    -------
    age_picks : np.ndarray
        Age samples that meet the sample criterion.
    disp_picks : np.ndarray
        Displacement samples that meet the sample criterion.
    success_rate : float
        Fraction of successful picks to total trials.
    """
    if verbose:
        print(f"Initializing MC sampling for {n_samples} samples")

    # Number of markers
    m_markers = len(markers)

    # Initialize sample arrays
    age_samps = np.empty(m_markers)
    disp_samps = np.empty(m_markers)

    # Initialize output arrays
    age_picks = np.empty((m_markers, n_samples))
    disp_picks = np.empty((m_markers, n_samples))

    # Seed random number generator
    np.random.seed(seed_val)

    # Initialize counters
    successes = 0
    tossed = 0
    consecutive_fails = 0

    # Initialize progress bar
    pbar = tqdm(total=n_samples)

    # Sample until enough successes collected or the criterion stops passing
    while successes < n_samples and consecutive_fails < hard_stop:
        # Generate random numbers over interval [0.0, 1.0)
        r_ages = np.random.rand(m_markers)
        r_disps = np.random.rand(m_markers)

        # Transform random numbers to age, displacement values
        for j, name in enumerate(markers.keys()):
            age_samps[j] = markers[name].age.pit(r_ages[j])
            disp_samps[j] = markers[name].displacement.pit(r_disps[j])

        # Check samples against condition
        if criterion.check_pass_fail(age_samps, disp_samps):
            # Condition met, record valid samples
            age_picks[:, successes] = age_samps
            disp_picks[:, successes] = disp_samps

            # Update counters
            successes += 1
            consecutive_fails = 0

            # Update progress bar
            pbar.update()

        else:
            # Condition not met
            tossed += 1
            consecutive_fails += 1

    # Close progress bar
    pbar.close()

    # Crop to successful picks
    age_picks = age_picks[:, :successes]
    disp_picks = disp_picks[:, :successes]

    # Report if requested
    if verbose:
        print(
            f"MC sampling finished with:"
            f"\n\t{successes} successes"
            f"\n\t{tossed} tossed"
        )

    # Raise error if no valid samples found
    if successes == 0:
        raise RuntimeError(
            f"No valid samples were found in {hard_stop:,} consecutive "
            f"trials (criterion: {criterion}). Check that the markers are "
            f"listed from youngest to oldest, and that their ages and "
            f"displacements are consistent with the criterion. If you "
            f"expect valid samples to be extremely rare, increase "
            f"`hard_stop` to keep sampling."
        )

    # Warn desired number of successful samples not found before hard stop
    if successes < n_samples:
        warnings.warn(
            f"Only {successes} valid samples found before {hard_stop:,} "
            f"consecutive trials failed the criterion. "
            f"Increase `hard_stop` to keep sampling.",
            stacklevel=2,
        )

    # Determine success rate
    success_rate = successes / (successes + tossed)

    return age_picks, disp_picks, success_rate


# end of file
