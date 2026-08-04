# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

# Public API
__all__ = [
    "DatedMarker",
]


# Import modules
import warnings
import copy

from .. import (
    units,
    probability_functions as PDFs,
)
from .variable_pair import VariablePair


#################### DATED MARKER ####################
class DatedMarker(VariablePair):
    """A DatedMarker stores the pair of displacement-age values that
    constrain a slip rate.

    Marker refers to some feature of the landscape or geologic record that
    indicates a measureable amount of fault slip (or lack thereof).
    A dated marker has some quantifiable age control.

    The age and displacement values are defined as PDFs.
    Each will have units of some multiple of years and meters, respectively.
    """

    def __init__(
        self,
        age: PDFs.PDF,
        displacement: PDFs.PDF,
        name: str | None = None,
    ):
        """Initialize a DatedMarker.

        Parameters
        ----------
        age : PDF
            PDF defining the age of the dated marker.
        displacement : PDF
            PDF defining the displacement of the dated marker.
        """
        # Initialize object
        super().__init__(
            x1=age,
            x2=displacement,
            name=name,
        )

        # Check units
        self._check_units_()

    @property
    def age(self) -> PDFs.PDF:
        return self.x1

    @age.setter
    def age(self, value):
        self.x1 = value
    
    @property
    def displacement(self) -> PDFs.PDF:
        return self.x2

    @displacement.setter
    def displacement(self, value):
        self.x2 = value
    
    def _check_units_(self):
        """Check that the age measurement is some multiple of years,
        and the displacement unit is some multiple of meters.
        """
        # Check age
        if self.age.unit is None:
            warnings.warn(
                "Age unit not specified. "
                "It is highly recommended to specify units for dated markers.",
                stacklevel=2,
            )
        else:
            _, base_unit = units.parse_unit(self.age.unit)
            if base_unit != 'y':
                raise ValueError(
                    f"Age base unit must be 'y' for dated marker, "
                    f"got '{base_unit}'"
                )

        # Check displacement
        if self.displacement.unit is None:
            warnings.warn(
                "Displacement unit not specified. "
                "It is highly recommended to specify units for dated markers.",
                stacklevel=2,
            )
        else:
            _, base_unit = units.parse_unit(self.displacement.unit)
            if base_unit != 'm':
                raise ValueError(
                    f"Displacement base unit must be 'm' for dated marker, "
                    f"got '{base_unit}'"
                )


    def __str__(self):
        print_str = f"DatedMarker {self.displacement.name}, comprising:"

        # Report age
        print_str += (
            f"\n\tage: {self.age.name} "
            f"{PDFs.analytics.pdf_mean(self.age)} "
            f"+- {PDFs.analytics.pdf_std(self.age):.2f} "
            f"{self.age.unit}"
        )

        # Report displacement
        print_str += (
            f"\n\tdisplacement: {self.displacement.name} "
            f"{PDFs.analytics.pdf_mean(self.displacement)} "
            f"+- {PDFs.analytics.pdf_std(self.displacement):.2f} "
            f"{self.displacement.unit}"
        )

        return print_str


# end of file
