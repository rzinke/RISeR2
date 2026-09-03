# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

# Import modules
import numpy as np
import pytest

from riser import probability_functions as PDFs


# Tests
class TestCheckExtension:
    def test_correct_extension_silent(self, recwarn):
        fname = "xx.txt"
        ext = "txt"
        PDFs.readers.check_extension(fname, ext)
        assert len(recwarn) == 0

    def test_wrong_extension_raises(self):
        fname = "xx.png"
        ext = "txt"
        with pytest.raises(ValueError, match="Filename must have extension"):
            PDFs.readers.check_extension(fname, ext)

    def test_multiple_dots_uses_final_extension(self, recwarn):
        fname = "xx.v2.txt"
        ext = "txt"
        PDFs.readers.check_extension(fname, ext)
        assert len(recwarn) == 0


class TestParseMetadataFromHeader:
    def test_parses_all_metadata_items(self):
        header_lines = [
            "# name: X\n",
            "# variable_type: displacement\n",
            "# unit: m\n",
        ]
        metadata = PDFs.readers.parse_metadata_from_header(header_lines)
        assert metadata == {
            "name": "X",
            "variable_type": "displacement",
            "unit": "m",
        }

    @pytest.mark.parametrize(
        "header_line",
        ["# NAME: X\n", "# Name: X\n", "# name: X\n"],
    )
    def test_key_matching_is_case_insensitive(self, header_line):
        metadata = PDFs.readers.parse_metadata_from_header([header_line])
        assert metadata["name"] == "X"

    def test_value_case_is_preserved(self):
        header_lines = ["# NAME: x\n"]
        metadata = PDFs.readers.parse_metadata_from_header(header_lines)
        assert metadata["name"] == "x"

    def test_similar_prefix_not_falsely_matched_regardless_of_case(self):
        header_lines = ["# UNITS: kg\n"]
        metadata = PDFs.readers.parse_metadata_from_header(header_lines)
        assert "unit" not in metadata

    def test_write_then_read_round_trip(self):
        pdf = PDFs.PDF(
            x=np.linspace(0, 2, 3),
            px=np.array([0.0, 1.0, 0.0]),
            name="X",
            variable_type="displacement",
            unit="m",
        )
        header = PDFs.readers.create_header_from_pdf(pdf)
        metadata = PDFs.readers.parse_metadata_from_header(
            header.splitlines(keepends=True)
        )
        assert metadata == {
            "name": "X",
            "variable_type": "displacement",
            "unit": "m",
        }


class TestReadPDF:
    def test_read_pdf_basic(self, tmp_path):
        fname = tmp_path / "test.txt"
        fname.write_text(
            "# name: x\n"
            "# variable_type: displacement\n"
            "# unit: m\n"
            "0.0,0.0\n"
            "1.0,1.0\n"
            "2.0,0.0\n"
        )
        pdf = PDFs.readers.read_pdf(str(fname))
        np.testing.assert_allclose(pdf.x, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(pdf.px, [0.0, 1.0, 0.0])
        assert pdf.name == "x"

    def test_last_line_without_trailing_newline_not_dropped(self, tmp_path):
        fname = tmp_path / "test.txt"
        fname.write_text("0.0,0.0\n1.0,1.0\n2.0,0.0")
        pdf = PDFs.readers.read_pdf(str(fname))
        assert len(pdf.x) == 3

    def test_user_metadata_overrides_file_metadata(self, tmp_path):
        fname = tmp_path / "test.txt"
        fname.write_text(
            "# name: from_file\n"
            "0.0,0.0\n"
            "1.0,1.0\n"
            "2.0,0.0\n"
        )
        with pytest.warns(UserWarning, match="differs from metadata in file"):
            pdf = PDFs.readers.read_pdf(str(fname), name="from_user")
        assert pdf.name == "from_user"

    def test_write_then_read_round_trip(self, tmp_path):
        pdf = PDFs.PDF(
            x=np.linspace(0, 2, 3),
            px=np.array([0.0, 1.0, 0.0]),
            name="x",
            variable_type="displacement",
            unit="m",
        )
        fname = tmp_path / "roundtrip.txt"
        PDFs.readers.save_pdf(str(fname), pdf)
        pdf_read = PDFs.readers.read_pdf(str(fname))

        np.testing.assert_allclose(pdf_read.x, pdf.x)
        np.testing.assert_allclose(pdf_read.px, pdf.px)
        assert pdf_read.name == pdf.name
        assert pdf_read.variable_type == pdf.variable_type
        assert pdf_read.unit == pdf.unit


class TestReadCalendarFile:
    def test_read_calendar_file_basic(self, tmp_path):
        fname = tmp_path / "test.txt"
        fname.write_text(
            "# name: cal file\n"
            "# unit: y\n"
            "1950,0.1\n"
            "1960,5.0\n"
            "1970,0.1\n"
        )
        calyr, calpx, metadata = PDFs.readers.read_calendar_file(str(fname))
        np.testing.assert_allclose(calyr, [1950.0, 1960.0, 1970.0])
        np.testing.assert_allclose(calpx, [0.1, 5.0, 0.1])
        assert metadata == {"name": "cal file", "unit": "y"}

    def test_last_line_without_trailing_newline_not_dropped(self, tmp_path):
        fname = tmp_path / "test.txt"
        fname.write_text("0,0\n1,1\n2,0")
        calyr, calpx, metadata = PDFs.readers.read_calendar_file(str(fname))
        assert len(calyr) == 3

    def test_does_not_enforce_unit_area(self, tmp_path):
        fname = tmp_path / "test.txt"
        fname.write_text("1950,0.1\n1960,5.0\n1970,0.1\n")
        calyr, calpx, metadata = PDFs.readers.read_calendar_file(str(fname))
        np.testing.assert_allclose(calpx, [0.1, 5.0, 0.1])


class TestReadPdfs:
    def test_reads_multiple_files_in_order(self, tmp_path):
        fname1 = tmp_path / "first.txt"
        fname1.write_text("# name: first\n0.0,0.0\n1.0,1.0\n2.0,0.0\n")

        fname2 = tmp_path / "second.txt"
        fname2.write_text(
            "# name: second\n"
            "0.0,0.0\n1.0,0.5\n2.0,1.0\n3.0,0.5\n4.0,0.0\n"
        )

        pdfs = PDFs.readers.read_pdfs([str(fname1), str(fname2)])
        assert len(pdfs) == 2
        assert pdfs[0].name == "first"
        assert pdfs[1].name == "second"


class TestSavePdf:
    def test_writes_expected_raw_content(self, tmp_path):
        pdf = PDFs.PDF(
            x=np.linspace(0, 2, 3),
            px=np.array([0.0, 1.0, 0.0]),
            name="fault A",
            variable_type="displacement",
            unit="m",
        )
        fname = tmp_path / "test.txt"
        PDFs.readers.save_pdf(str(fname), pdf)

        raw = fname.read_text()
        assert raw == (
            "# name: fault A\n"
            "# variable_type: displacement\n"
            "# unit: m\n"
            "0.0,0.0\n"
            "1.0,1.0\n"
            "2.0,0.0\n"
        )

    def test_requires_txt_extension(self, tmp_path):
        pdf = PDFs.PDF(x=np.linspace(0, 2, 3), px=np.array([0.0, 1.0, 0.0]))
        fname = tmp_path / "test.csv"
        with pytest.raises(ValueError, match="Filename must have extension"):
            PDFs.readers.save_pdf(str(fname), pdf)


# end of file
