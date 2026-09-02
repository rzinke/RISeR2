# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

# Import modules
import dataclasses

import pytest

from riser.probability_functions import metadata


# Tests
class TestPDFmetadata:
    """
    PDF metadata validation only tests for type, not value.

    A PDFmetadata object should construct with default values.

    It has items `name`, `variable_type`, and `unit`.

    None or str values must be accepted for each meta-datum.

    It is immutable.

    It should be transformable to a `dict` in a way that preserves the metadata.
    """

    name = "x"

    variable_type = "age"

    unit = "y"

    def test_construction_with_values(self):
        meta = metadata.PDFmetadata(
            name=self.name,
            variable_type=self.variable_type,
            unit=self.unit,
        )
        assert meta == metadata.PDFmetadata(
            name=self.name,
            variable_type=self.variable_type,
            unit=self.unit,
        )

    def test_construction_with_defaults(self):
        meta = metadata.PDFmetadata()
        assert meta == metadata.PDFmetadata(
            name=None,
            variable_type=None,
            unit=None,
        )

    def test_dataclass_fields_match_metadata_items(self):
        meta = metadata.PDFmetadata()
        assert metadata.METADATA_ITEMS == ["name", "variable_type", "unit"]

    def test_metadata_items_hold_assignment(self):
        meta = metadata.PDFmetadata(
            name=self.name,
            variable_type=self.variable_type,
            unit=self.unit,
        )
        assert meta.name == self.name
        assert meta.variable_type == self.variable_type
        assert meta.unit == self.unit

    def test_only_type_validated(self):
        meta = metadata.PDFmetadata(
            name="defacto name",
            variable_type="defacto variable type",
            unit="defacto unit",
        )
        assert meta.name == "defacto name"
        assert meta.variable_type == "defacto variable type"
        assert meta.unit == "defacto unit"

    @pytest.mark.parametrize(
        "field",
        [
            {"name": 123},
            {"variable_type": 123},
            {"unit": 123},
        ]
    )
    def test_invalid_type_raises(self, field):
        with pytest.raises(TypeError):
            metadata.PDFmetadata(**field)

    def test_immutable(self):
        meta = metadata.PDFmetadata(
            name=self.name,
            variable_type=self.variable_type,
            unit=self.unit,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            meta.name = "y"

    def test_as_dict_preserves_metadata(self):
        meta = metadata.PDFmetadata(
            name=self.name,
            variable_type=self.variable_type,
            unit=self.unit,
        )
        assert meta.as_dict() == {
            "name": self.name,
            "variable_type": self.variable_type,
            "unit": self.unit,
        }


class TestGetCommonMetadata:

    meta0 = metadata.PDFmetadata(
        name="0",
    )

    meta1 = metadata.PDFmetadata(
        name="1",
        variable_type="age",
        unit="y",
    )

    def test_None_metadata(self):
        meta2 = metadata.PDFmetadata()
        meta_cmmn = metadata.get_common_metadata(
            metadata_list=[self.meta0, meta2]
        )
        assert meta_cmmn.name is None
        assert meta_cmmn.variable_type is None
        assert meta_cmmn.unit is None

    def test_two_common_metadata_remain_common(self):
        meta2 = metadata.PDFmetadata(
            name="1",
            variable_type="age",
            unit="y",
        )
        meta_cmmn = metadata.get_common_metadata(
            metadata_list=[self.meta1, meta2]
        )
        assert meta_cmmn.name == "1"
        assert meta_cmmn.variable_type == self.meta1.variable_type
        assert meta_cmmn.unit == self.meta1.unit

    def test_three_common_metadata_remain_common(self):
        meta2 = metadata.PDFmetadata(
            name="1",
            variable_type="age",
            unit="y",
        )
        meta3 = metadata.PDFmetadata(
            name="1",
            variable_type="age",
            unit="y",
        )
        meta_cmmn = metadata.get_common_metadata(
            metadata_list=[self.meta1, meta2, meta3]
        )
        assert meta_cmmn.name == "1"
        assert meta_cmmn.variable_type == self.meta1.variable_type
        assert meta_cmmn.unit == self.meta1.unit

    def test_common_metadata_remain_common_with_rename(self):
        meta2 = metadata.PDFmetadata(
            name="1",
            variable_type="age",
            unit="y",
        )
        meta_cmmn = metadata.get_common_metadata(
            metadata_list=[self.meta1, meta2],
            name="common",
        )
        assert meta_cmmn.name == "common"
        assert meta_cmmn.variable_type == self.meta1.variable_type
        assert meta_cmmn.unit == self.meta1.unit

    @pytest.mark.parametrize(
        "differing_field", ["name", "variable_type", "unit"]
    )
    def test_differing_fields_warn_and_reset(self, differing_field):
        meta_dict = self.meta1.as_dict()
        meta_dict[differing_field] = "DIFFERENT"
        meta2 = metadata.PDFmetadata(**meta_dict)

        with pytest.warns(UserWarning):
            result = metadata.get_common_metadata(
                metadata_list=[self.meta1, meta2], warn=True
            )
        assert getattr(result, differing_field) is None

    def test_three_different_metadata_default_to_None(self):
        meta2 = metadata.PDFmetadata(
            name="2",
            variable_type="age",
            unit="y",
        )
        meta3 = metadata.PDFmetadata(
            name="3",
            variable_type="displacement",
            unit="m",
        )
        meta_cmmn = metadata.get_common_metadata(
            metadata_list=[self.meta1, meta2, meta3],
        )
        assert meta_cmmn.name is None
        assert meta_cmmn.variable_type is None
        assert meta_cmmn.unit is None

    def test_case_sensitivity(self):
        meta2 = metadata.PDFmetadata(
            name="2",
            variable_type="Age",
            unit="Y",
        )
        meta_cmmn = metadata.get_common_metadata(
            metadata_list=[self.meta1, meta2]
        )
        assert meta_cmmn.name is None
        assert meta_cmmn.variable_type is None
        assert meta_cmmn.unit is None

    def test_silent_by_default_if_different(self, recwarn):
        meta_cmmn = metadata.get_common_metadata(
            metadata_list=[self.meta0, self.meta1]
        )
        assert len(recwarn) == 0

    def test_warn_if_different(self):
        with pytest.warns(UserWarning):
            meta_cmmn = metadata.get_common_metadata(
                metadata_list=[self.meta0, self.meta1],
                warn=True,
            )


# end of file
