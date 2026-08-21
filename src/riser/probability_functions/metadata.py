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
    """

    name: str | None = None

    variable_type: variable_types.VariableType.__value__ | None = None

    unit: units.Unit.__value__ | None = None

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
) -> PDFmetadata:
    """Find the common metadata values among a set of metadata objects.

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
    # Check whether variable type is the same among all metadata
    variable_type = metadata_list[0].variable_type
    if not all(
        metadata.variable_type == variable_type for metadata in metadata_list
    ):
        # Warn of differences
        if warn:
            warnings.warn(
                "`variable_type` differs between PDFs",
                stacklevel=2,
            )

        # Reset variable type
        variable_type = None

    # Check whether unit is the same among all metadata
    unit = metadata_list[0].unit
    if not all(
        metadata.unit == unit for metadata in metadata_list
    ):
        # Warn of differences
        if warn:
            warnings.warn(
                "`unit` differs between PDFs",
                stacklevel=2,
            )

        # Reset unit
        unit = None

    return PDFmetadata(
        name=name,
        variable_type=variable_type,
        unit=unit,
    )


# end of file
