# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

# Public API
__all__ = [
    "read_dated_markers_from_config",
]


# Import modules
import warnings

import toml

from .. import (
    units,
    probability_functions as PDFs,
)
from .dated_marker import DatedMarker


#################### MARKER READERS ####################
def initialize_dated_marker_from_files(
    age_fname: str,
    displacement_fname: str,
    *,
    marker_name: str | None = None,
    age_name: str | None = None,
    age_variable_type: str | None = None,
    age_unit: str | None = None,
    displacement_name: str | None = None,
    displacement_variable_type: str | None = None,
    displacement_unit: str | None = None,
    verbose: bool = False,
) -> DatedMarker:
    """Initialize a DatedMarker from a file.

    Metadata can be specified either in the PDF file itself, or in the config
    toml file. Precedence is given to metadata specified in the PDF file if
    there is a conflict, per the set_metadata_priority function.

    Parameters
    ----------
    age_fname : str
        Name of age PDF file.
    displacement_fname : str
        Name of displacement PDF file.
    kwargs : dict
        Metadata for marker and age, displacement PDFs
    marker_name : str, optional
        Marker name.
    age_name : str, optional
        Descriptive name of age PDF.
    age_variable_type : str, optional
        Age variable type (e.g., age).
    age_unit : str, optional
        Age physical unit.
    displacement_name : str, optional
        Descriptive name of displacement PDF.
    displacement_variable_type : str, optional
        Displacement variable type (e.g., displacement).
    displacement_unit : str, optional
        Displacement physical unit.

    Returns
    -------
    marker : DatedMarker
    """
    # Read age PDF
    age = PDFs.readers.read_pdf(
        fname=age_fname,
        name=age_name,
        variable_type=age_variable_type,
        unit=age_unit,
        verbose=verbose,
    )

    # Read displacement PDF
    displacement = PDFs.readers.read_pdf(
        fname=displacement_fname,
        name=displacement_name,
        variable_type=displacement_variable_type,
        unit=displacement_unit,
        verbose=verbose,
    )

    # Form age-displacement data into DatedMarker
    marker = DatedMarker(
        age=age,
        displacement=displacement,
        name=marker_name,
    )

    # Report marker attributes
    if verbose:
        print(marker)

    return marker


def read_dated_markers_from_config(
    fname: str, verbose: bool = False
) -> dict[str, DatedMarker]:
    """Read marker data from a TOML configuration file.

    The file should have one [marker_name] entry per marker, and each marker
    should have entries for "age file" and "displacement file".

    Optionally, age and displacement metadata can be specified in the config
    toml file, though any metadata encoded in the PDF file takes precedence
    over the config file, per the set_metadata_priority function.

    Parameters
    ----------
    fname : str
        Configuration file name.

    Returns
    -------
    markers : dict[str, DatedMarker]
        Dictionary with one DatedMarker per marker name entry.
    """
    with open(fname, "r") as age_disp_file:
        marker_specs = toml.load(age_disp_file)

    # Report number of markers read
    if verbose:
        n_markers = len(marker_specs)
        print(f"{n_markers} markers specified")

    # Empty dictionary of markers
    markers = {}

    # Loop through markers
    for marker_name, marker_spec in marker_specs.items():
        # Retrieve age PDF name
        age_fname = marker_spec.get("age file")
        if age_fname is None:
            raise ValueError(
                f"Age file must be specified "
                f"for marker '{marker_name}'"
            )

        # Retrieve displacement PDF name
        displacement_fname = marker_spec.get("displacement file")
        if displacement_fname is None:
            raise ValueError(
                f"Displacement file must be specified "
                f"for marker '{marker_name}'"
            )

        # Initialize marker
        marker = initialize_dated_marker_from_files(
            marker_name=marker_name,
            age_fname=age_fname,
            displacement_fname=displacement_fname,
            age_name=marker_spec.get("age name"),
            age_variable_type=marker_spec.get("age variable type"),
            age_unit=marker_spec.get("age unit"),
            displacement_name=marker_spec.get("displacement name"),
            displacement_variable_type=marker_spec.get(
                "displacement variable type"
            ),
            displacement_unit=marker_spec.get("displacement unit"),
            verbose=verbose,
        )

        # Write marker to dictionary
        markers[marker_name] = marker

    # Check that markers are ordered youngest/smallest to oldest/largest
    for i, marker in enumerate(markers.values()):
        if i > 0:
            # Compute reference age/displacement
            ref_age = PDFs.analytics.pdf_mean(
                ref_marker.age  # type: ignore[has-type]
            )
            ref_disp = PDFs.analytics.pdf_mean(
                ref_marker.displacement  # type: ignore[has-type]
            )

            # Compute marker age/displacement
            marker_age = PDFs.analytics.pdf_mean(
                marker.age  # type: ignore[has-type]
            )
            marker_disp = PDFs.analytics.pdf_mean(
                marker.displacement  # type: ignore[has-type]
            )

            # Check that marker is older/larger than previous
            if marker_age < ref_age:
                marker_display_name = marker.name
                ref_marker_display_name = ref_marker.name  # type: ignore[has-type]

                warnings.warn(
                    f"Marker '{marker_display_name}' appears to be younger "
                    f"than '{ref_marker_display_name}'. Confirm marker order.",
                    stacklevel=3,
                )

            if marker_disp < ref_disp:
                marker_display_name = marker.name
                ref_marker_display_name = ref_marker.name  # type: ignore[has-type]

                warnings.warn(
                    f"Marker '{marker_display_name}' appears to be less "
                    f"displaced than '{ref_marker_display_name}'. "
                    f"Confirm marker order.",
                    stacklevel=3,
                )

        # Update reference marker
        ref_marker = marker

    return markers


# end of file
