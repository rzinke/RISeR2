# -*- coding: utf-8 -*-
#
# Copyright (c) 2025-2026 Robert Zinke. Licensed under the MIT License.

"""
These functions condition a random variable by multiplying by a weight and
then renormalizing.
"""


# Public API
__all__ = [
    "condition",
    "flat_weight",
]


# Import modules
import numpy as np

from riser import (
    integration,
    probability_functions as PDFs,
)


#################### CONDITIONING FUNCTIONS ####################
def condition(
    prior: PDFs.PDF, weight: np.ndarray, *, name: str | None = None
) -> tuple[PDFs.PDF, float]:
    """Weight a prior distribution, according to

        posterior(x) = prior(x) . weighting(x)

    The area-under-the-curve is recorded and the posterior distribution is
    then normalized to unit area and formed into a PDF.

    Metadata are preserved unless explicitly overridden with the `name`
    parameter.

    Parameters
    ----------
    prior : PDF
        Prior to weight.
    weight : np.ndarray
        Weighting array.
    name : str (optional)
        Name override for weighted PDF.

    Returns
    -------
    posterior : PDF
        Weighted prior.
    area : float
        Area of the conditioned distribution, before scaling.
    """
    # Check the size of the weighting array equals the size of the prior
    if len(weight) != len(prior):
        raise ValueError(
            f"The size of the weighting array ({len(weight)}) must equal the "
            f"size of the prior domain ({len(prior)})"
        )

    # Weight the probability densities of the prior
    px_weighted = prior.px * weight

    # Compute area of result
    area = integration.integrate(x=prior.x, px=px_weighted)

    # Formulate metadata of the posterior
    metadict = prior.metadata.as_dict()
    if name is not None:
        metadict["name"] = name

    # Format weighted prior as PDF (scaling carried out by PDF.__init__)
    posterior = PDFs.PDF(
        x=prior.x,
        px=px_weighted,
        **metadict,
    )

    return posterior, area


#################### WEIGHTING FUNCTIONS ####################
def flat_weight(x: np.ndarray) -> np.ndarray:
    """Create an array of unit-value weights based on the domain over which a
    prior is defined.

    This is a thin wrapper for `np.ones_like`.

    Parameters
    ----------
    x : np.ndarray
        Domain values of the random variable.
    """
    return np.ones_like(x)


# end of file
