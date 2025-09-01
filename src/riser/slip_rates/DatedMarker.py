# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

# Constants


# Import modules
from riser.probability_functions import PDF
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
    def __init__(self, age:PDF, displacement:PDF):
        """Initialize a DatedMarker.

        Args    age - PDF defining the age of a dated marker
                displacement - PDF defining the displacement of a dated marker
        """
        # Record age and displacement data
        self.age = age
        self.displacement = displacement

        # Check units
        self.__check_units__()


    def __check_units__(self):
        """Check that the age measurement is some multiple of years,
        and the displacement unit is some multiple of meters.
        """
        # Check age
        if units.check_pdf_base_unit(self.age) != 'y':
            raise ValueError("Age base unit must be y")

        # Check displacement
        if units.check_pdf_base_unit(self.displacement) != 'm':
            raise ValueError("Displacement base unit must be m")

        return True


    def __str__(self):
        print_str = "DatedMarker comprising:"

        # Report age

        # Report displacement

        return print_str


# end of file