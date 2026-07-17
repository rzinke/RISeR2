# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

# Public API
__all__ = [
    "get_sample_criterion",
    "sample_monte_carlo",
]


# Import modules
import warnings

import numpy as np
from tqdm import tqdm


#################### SAMPLING CRITERIA ####################
class SampleCriterion:
    def __init__(self, **kwargs):
        return

    def check_pass_fail(
        self,
        ages: np.ndarray,
        displacements: np.ndarray,
    ) -> bool:
        """Check whether a set of displacement-age pairs meets the sample
        criterion.

        Args    ages - np.ndarray, sampled age values
                displacements - np.ndarray, sampled displacement values
        Returns bool, pass (True) or fail (False)
        """
        return NotImplementedError(
            "check_pass_fail not implemented. Override with child class."
        )

class PassAll(SampleCriterion):
    def __init__(self, **kwargs):
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

class PassNonnegative(SampleCriterion):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def check_pass_fail(
        self,
        ages: np.ndarray,
        displacements: np.ndarray,
    ) -> bool:
        """Pass sample combinations that result in non-negative slip rates.
        """
        # Compute age, displacement deltas
        age_diffs = np.diff(ages)
        disp_diffs = np.diff(displacements)

        # Check condition
        if all([age_diffs.min() > 0, disp_diffs.min() >= 0]):
            return True
        else:
            return False

class PassNonnegativeBounded(SampleCriterion):
    def __init__(self, **kwargs):
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
        if all([
            age_diffs.min() > 0,
            disp_diffs.min() >= 0,
            slip_rates.max() <= self.max_sample_rate,
        ]):
            return True
        else:
            return False


SAMPLE_CRITERIA = {
    "PassAll": PassAll,
    "PassNonnegative": PassNonnegative,
    "PassNonnegativeBounded": PassNonnegativeBounded,
}


def get_sample_criterion(
    criterion_name: str, verbose: bool = False
) -> SampleCriterion:
    """Retrieve a sample criterion by name.
    """
    # Check criterion name
    if criterion_name not in SAMPLE_CRITERIA:
        raise ValueError(
            f"Sample criterion '{criterion_name}' not supported. "
            f"Use one of {', '.join(SAMPLE_CRITERIA.keys())}"
        )

    return SAMPLE_CRITERIA.get(criterion_name)


#################### MONTE CARLO SAMPLING ####################
def sample_monte_carlo(
    markers: dict,
    criterion: SampleCriterion,
    *,
    n_samples: int = 10_000,
    seed_val: int = 0,
    hard_stop: int = 1_000_000_000,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """This method uses the inverse transform sampling method to randomly
    sample the displacement and age PDFs constraining a DatedMarker.
    The random samples are checked against a criterion, e.g., "no negative
    slip rates".

    Args    markers - dict, DatedMarkers between which to compute incremental
                slip rates
            criterion - function to evaluate validity of samples
            n_samples - int, number of valid samples to achieve
            max_rate_sample - float, maximum slip rate to consider
            seed_val - int, random number generator seed value
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

    # Initialize counter
    successes = 0
    tossed = 0

    # Initialize progress bar
    pbar = tqdm(total=n_samples)

    # Loop through samples until enough successess collected
    for i in range(hard_stop):
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
            age_picks[:,successes] = age_samps
            disp_picks[:,successes] = disp_samps

            # Update counter
            successes += 1

            # Update progress bar
            pbar.update()

            # Break loop if desired successes achieved
            if successes == n_samples:
                break
        else:
            # Condition not met
            tossed += 1

    # Close progress bar
    pbar.close()

    # Report if requested
    if verbose:
        print(
            f"MC sampling finished with:"
            f"\n\t{successes} successes"
            f"\n\t{tossed} tossed"
        )

    # Report if hard stop met
    if i == (hard_stop - 1):
        warnings.warn(
            f"Only {successes} valid samples found before reaching trial limit"
        )

    return age_picks, disp_picks


# end of file