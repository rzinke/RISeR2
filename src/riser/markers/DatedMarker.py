# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

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


#################### DATED MARKER ####################
class DatedMarker:
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

        Args    age - PDF defining the age of a dated marker
                displacement - PDF defining the displacement of a dated marker
        """
        # Check inputs
        if not isinstance(age, PDFs.PDF):
            raise ValueError(
                f"Age must be provided as a PDF, "
                f"got {type(age).__name__}"
            )

        if not isinstance(displacement, PDFs.PDF):
            raise ValueError(
                f"Displacement must be provided as a PDF, "
                f"got {type(displacement).__name__}"
            )

        # Record age and displacement data - copy PDFs just in case
        self.age = copy.deepcopy(age)
        self.displacement = copy.deepcopy(displacement)

        # Record metadata
        self.name = name

        # Check units
        self._check_units_()


    def _check_units_(self):
        """Check that the age measurement is some multiple of years,
        and the displacement unit is some multiple of meters.
        """
        # Check age
        if self.age.unit is None:
            warnings.warn(
                "Age unit not specified. "
                "It is highly recommended to specify units.",
                stacklevel=2,
            )
        else:
            if units.check_pdf_base_unit(self.age) != 'y':
                raise ValueError("Age base unit must be y")

        # Check displacement
        if self.displacement.unit is None:
            warnings.warn(
                "Displacement unit not specified. "
                "It is highly recommended to specify units.",
                stacklevel=2,
            )
        else:
            if units.check_pdf_base_unit(self.displacement) != 'm':
                raise ValueError("Displacement base unit must be m")


    def __str__(self):
        print_str = f"DatedMarker {self.displacement.name}, comprising:"

        # Report age
        print_str += (
            f"\n\tage: {self.age.name} "
            f"{PDFs.analytics.pdf_mode(self.age)} "
            f"+- {PDFs.analytics.pdf_std(self.age):.2f} "
            f"{self.age.unit}"
        )

        # Report displacement
        print_str += (
            f"\n\tdisplacement: {self.displacement.name} "
            f"{PDFs.analytics.pdf_mode(self.displacement)} "
            f"+- {PDFs.analytics.pdf_std(self.displacement):.2f} "
            f"{self.displacement.unit}"
        )

        return print_str


# end of file