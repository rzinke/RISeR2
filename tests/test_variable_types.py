# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

# Import modules
import pytest

from riser import variable_types


# Tests
class TestCheckVariableTypeSupported:
    def test_warn_if_none(self):
        with pytest.warns(UserWarning):
            variable_types.check_variable_type_supported(None)

    @pytest.mark.parametrize(
        "variable_type", ["age", "displacement", "slip rate"]
    )
    def test_silent_if_valid(self, variable_type, recwarn):
        variable_types.check_variable_type_supported(variable_type)
        assert len(recwarn) == 0

    @pytest.mark.parametrize("variable_type", ["foo"])
    def test_raise_if_not_supported(self, variable_type):
        with pytest.raises(ValueError):
            variable_types.check_variable_type_supported(variable_type)



# end of file
