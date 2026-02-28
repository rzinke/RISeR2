# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

# Constants


# Import modules
import warnings

import riser.probability_functions as PDFs
from riser import units


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
            name: str | None=None
        ):
        """Initialize a DatedMarker.

        Args    age - PDF defining the age of a dated marker
                displacement - PDF defining the displacement of a dated marker
        """
        # Record age and displacement data
        self.age = age
        self.displacement = displacement

        # Record metadata
        self.name = name

        # Check units
        self.__check_units__()


    def __check_units__(self):
        """Check that the age measurement is some multiple of years,
        and the displacement unit is some multiple of meters.
        """
        # Check age
        if self.age.unit is None:
            warnings.warn(
                "Age unit not specified. "
                "It is highly recommended to specify units.",
                stacklevel=2
            )
        else:
            if units.check_pdf_base_unit(self.age) != 'y':
                raise ValueError("Age base unit must be y")

        # Check displacement
        if self.displacement.unit is None:
            warnings.warn(
                "Displacement unit not specified. "
                "It is highly recommended to specify units.",
                stacklevel=2
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