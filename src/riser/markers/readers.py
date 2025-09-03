# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

# Constants


# Import modules
import warnings

import toml

from riser.probability_functions import readers as pdf_readers, analytics
from riser.markers import DatedMarker
from riser import units


#################### MARKER READERS ####################
def set_metadata_priority(metadata_item:str, file_item:str, spec_item:str):
    """
    """
    if all([file_item is not None,
            spec_item is not None,
            file_item != spec_item]):
        # Warn user
        warnings.warn(f"{metadata_item} specified in file is different from "
                      f"user-specification.", stacklevel=2)

    if all([file_item is None, spec_item is not None]):
        return spec_item
    else:
        return file_item


def initialize_marker_from_files(age_fname:str, displacement_fname:str,
        verbose=False, **kwargs) -> DatedMarker:
    """
    Metadata can be specified either in the PDF file itself, or in the config
    toml file. Precedence is given to metadata specified in the PDF file if
    there is a conflict, per the set_metadata_priority function.

    Args    age_fname - str, name of age PDF file
            displacement_fname - str, name of displacement PDF file
            kwargs - dict, metadata for marker and age, displacement PDFs
                name - str, marker name
                age_name - str, descriptive name of age PDF
                age_variable_type - str, age variable type (e.g., age)
                age_unit - str, age physical unit
                displacement_name - str, descriptive name of displacement PDF
                displacement_variable_type - str, displacement variable type
                    (e.g., displacement)
                displacement_unit - str, displacement physical unit
    Returns marker - DatedMarker
    """
    # Read age PDF
    age = pdf_readers.read_pdf(age_fname)

    # Gather age metadata - precedence given to metadata in PDF file
    age.name = set_metadata_priority(
            "Age name", age.name,
            kwargs.get('age_name'))
    age.variable_type = set_metadata_priority(
            "Age variable type", age.variable_type,
            kwargs.get('age_variable_type'))
    age.unit = units.get_priority_unit(
            age.unit, kwargs.get('age_unit'))

    # Read displacement PDF
    displacement = pdf_readers.read_pdf(displacement_fname)

    # Gather displacement metadata - precedence given to metadata in PDF file
    displacement.name = set_metadata_priority(
            "Displacement name", displacement.name,
            kwargs.get('displacement_name'))
    displacement.variable_type = set_metadata_priority(
            "Displacement variable type", displacement.variable_type,
            kwargs.get('displacement_variable_type'))
    displacement.unit = units.get_priority_unit(
            displacement.unit, kwargs.get('displacement_unit'))

    # Form age-displacement data into DatedMarker
    marker = DatedMarker(age, displacement, name=kwargs.get('name'))

    # Report if requested
    if verbose == True:
        print(marker)

    return marker


def read_markers_from_config(fname:str, verbose=False) -> dict:
    """Read marker data from a TOML configuration file.
    The file should have one [marker_name] entry per marker, and each marker
    should have entries for "age file" and "displacement file".
    Optionally, age and displacement metadata can be specified in the config
    toml file, though any metadata encoded in the PDF file takes precedence
    over the config file, per the set_metadata_priority function.

    Args    fname - str, configuration file name.
    Returns markers - dict with one DatedMarker per marker name entry
    """
    with open(fname, 'r') as age_disp_file:
        marker_specs = toml.load(age_disp_file)

    # Report if requested
    if verbose == True:
        n_markers = len(marker_specs)
        print(f"{n_markers} markers specified")

    # Empty dictionary of markers
    markers = {}

    # Loop through markers
    for marker_name, marker_spec in marker_specs.items():
        # Retrieve age PDF name
        age_fname = marker_spec.get('age file')
        if age_fname is None:
            raise ValueError(f"Age file must be specified for marker "
                             f"{marker_name}")

        # Retrieve displacement PDF name
        displacement_fname = marker_spec.get('displacement file')
        if displacement_fname is None:
            raise ValueError(f"Displacement file must be specified for marker "
                             f"{marker_name}")

        # Initialize marker
        marker_args = {
            'name': marker_name,
            'age_fname': age_fname,
            'displacement_fname': displacement_fname,
            'age_name': marker_spec.get('age name'),
            'age_variable_type': marker_spec.get('age variable type'),
            'age_unit': marker_spec.get('age unit'),
            'displacement_name': marker_spec.get('displacement name'),
            'displacement_variable_type': marker_spec.get(
                                            'displacement variable type'),
            'displacement_unit': marker_spec.get('displacement unit'),
        }
        marker = initialize_marker_from_files(**marker_args, verbose=verbose)

        # Write marker to dictionary
        markers[marker_name] = marker

    # Check that markers are ordered youngest/smallest to oldest/largest
    for i, marker in enumerate(markers.values()):
        if i > 0:
            # Compute reference age/displacement
            ref_age = analytics.pdf_mean(ref_marker.age)
            ref_disp = analytics.pdf_mean(ref_marker.displacement)

            # Compute marker age/displacement
            marker_age = analytics.pdf_mean(marker.age)
            marker_disp = analytics.pdf_mean(marker.displacement)

            # Check that marker is older/larger than previous
            if marker_age < ref_age:
                warnings.warn(f"Marker {marker.name} appears to be younger "
                              f"than {ref_marker.name}. "
                              f"Confirm marker order.", stacklevel=3)

            if marker_disp < ref_disp:
                warnings.warn(f"Marker {marker.name} appears to be less "
                              f"displaced than {ref_marker.name}. "
                              f"Confirm marker order.", stacklevel=3)

        # Update reference marker
        ref_marker = marker

    return markers


# end of file