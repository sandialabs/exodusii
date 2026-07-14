# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import exodusii
from exodusii.api.copy import copy_file
from exodusii.api.file import ExodusFile
from exodusii.api.writer import ExodusWriter


def test_writer_reader_info_and_qa_records(tmp_path: Path) -> None:
    path = tmp_path / "records.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize("records", 2, 0, 0)
        writer.write_info_records(["line one", "line two"])
        writer.write_qa_records([["code", "1.0", "2026-07-14", "12:00:00"]])

    with ExodusFile.open(path) as exo:
        assert exo.info_records() == ("line one", "line two")
        assert exo.qa_records() == (("code", "1.0", "2026-07-14", "12:00:00"),)


def test_legacy_info_and_qa_records(tmp_path: Path) -> None:
    path = tmp_path / "legacy_records.exo"

    with exodusii.File(path, mode="w") as exo:
        exo.put_init("legacy records", 2, 0, 0, 0, 0, 0)
        exo.put_info(2, ["line one", "line two"])
        exo.put_qa(1, [["code", "1.0", "2026-07-14", "12:00:00"]])

    with exodusii.File(path) as exo:
        assert exo.get_info_records() == ["line one", "line two"]
        assert exo.get_qa_records() == [("code", "1.0", "2026-07-14", "12:00:00")]


def test_copy_preserves_info_and_qa_records(tmp_path: Path) -> None:
    source = tmp_path / "source.exo"
    target = tmp_path / "target.exo"

    with ExodusWriter.create(source) as writer:
        writer.initialize("records", 2, 0, 0)
        writer.write_info_records(["line one"])
        writer.write_qa_records([["code", "1.0", "2026-07-14", "12:00:00"]])

    copy_file(source, target)

    with ExodusFile.open(target) as exo:
        assert exo.info_records() == ("line one",)
        assert exo.qa_records() == (("code", "1.0", "2026-07-14", "12:00:00"),)
