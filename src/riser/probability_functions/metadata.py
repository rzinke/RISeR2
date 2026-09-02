# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

"""
Class to organize PDF metadata items.
"""

# Public API
__all__ = [
    "PDFmetadata",
    "get_common_metadata",
]


# Import modules
import warnings
from dataclasses import dataclass, asdict, fields

from .. import (
    variable_types,
    units,
)


#################### PDF METADATA CLASS ####################
@dataclass(frozen=True)
class PDFmetadata:
    """Store PDF metadata items.

    Metadata are optional.
    That is, each piece of metadata can assume a value or be None.
    """

    name: str | None = None

    variable_type: variable_types.VariableType | None = None

    unit: units.Unit | None = None

    def __post_init__(self) -> None:
        # Check name
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError(
                f"`name` must be type str, got {type(self.name).__name__}"
            )

        # Check variable type type
        if (
            self.variable_type is not None
            and not isinstance(
                self.variable_type, variable_types.VariableType.__value__
            )
        ):
            raise TypeError(
                f"`variable_type` must be type "
                f"{variable_types.VariableType.__value__.__name__}, "
                f"got {type(self.variable_type).__name__}"
            )

        # Check unit type
        if (
            self.unit is not None
            and not isinstance(self.unit, units.Unit.__value__)
        ):
            raise TypeError(
                f"`unit` must be type {units.Unit.__value__.__name__}, "
                f"got {type(self.unit).__name__}"
            )

    def as_dict(self) -> dict[str, str]:
        """Format the PDF metadata items as a dictionary.

        Returns
        -------
        dict[str, str]
            Metadata keys and values.
        """
        return asdict(self)


METADATA_ITEMS = [field.name for field in fields(PDFmetadata)]


#################### METADATA FUNCTIONS ####################
def get_common_metadata(
    metadata_list: list[PDFmetadata],
    name: str | None = None,
    warn: bool = False,
    verbose: bool = False,
) -> PDFmetadata:
    """Find the common metadata values among a set of metadata objects.

    Fields that do not share common values across all metadata objects are set
    to None.

    Name can be explicitly overridden.

    Parameters
    ----------
    metadata_list : list[PDFmetadata]
        List of metadata objects from which to find common metadata values.
    name : str, optional
        Name of resulting PDF.
    warn : bool, optional
        Raise a warning if metadata values differ between PDFs.

    Returns
    -------
    PDFmetadata
        Metadata with common values.
    """
    if verbose:
        print(
            f"Determining common field values for {len(metadata_list)} "
            f"metadata objects"
        )

    # Initialize common metadata dict
    metadata_dict = {}

    # Loop through metadata fields
    for field in METADATA_ITEMS:
        # Get reference field value
        ref_value = getattr(metadata_list[0], field)

        # Check whether field is the same among all metadata objects
        if all(
            getattr(metadata, field) == ref_value for metadata in metadata_list
        ):
            # Use common field value
            metadata_dict[field] = ref_value

        else:
            if warn:
                warnings.warn(
                    f"`{field}` differs between PDF metadata, "
                    f"defaulting to 'None'",
                    stacklevel=2,
                )

            # Set non-common field values to None
            metadata_dict[field] = None

        # Report field value
        if verbose:
            print(f"\t{field}: {metadata_dict[field]}")

    # Override name
    if name is not None:
        metadata_dict["name"] = name

        if verbose:
            print(f"Name set to '{metadata_dict['name']}'")

    return PDFmetadata(**metadata_dict)


# end of file
