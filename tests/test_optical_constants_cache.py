"""Pins OpticalConstants' sharing guarantee: N references to the same material
load and interpolate ONE table, not N. See tmp/USAGE.md "Verifying the sharing
guarantee" — this is a memory/performance contract a future edit could easily
break without looking wrong on a casual read, so it's pinned by a real test
rather than left to inspection.
"""

from __future__ import annotations

import polars as pl
import pytest

from refloxide.data import OpticalConstants


@pytest.fixture(autouse=True)
def _clear_cache():
    OpticalConstants._cache.clear()
    yield
    OpticalConstants._cache.clear()


def _write_ooc_csv(path):
    pl.DataFrame(
        {
            "energy": [500.0, 700.0, 900.0],
            "n_xx": [4.0e-6, 5.0e-6, 6.0e-6],
            "n_ixx": [1.0e-7, 1.2e-7, 1.4e-7],
            "n_zz": [8.0e-6, 9.0e-6, 1.0e-5],
            "n_izz": [2.0e-7, 2.2e-7, 2.4e-7],
        }
    ).write_csv(path)


def test_three_from_file_calls_on_same_path_share_one_instance(tmp_path):
    ooc_path = tmp_path / "znpc_ooc.csv"
    _write_ooc_csv(ooc_path)

    a = OpticalConstants.from_file(ooc_path)
    b = OpticalConstants.from_file(ooc_path)
    c = OpticalConstants.from_file(str(ooc_path))

    assert a is b is c
    assert a.table is b.table is c.table
    assert OpticalConstants.cache_size() == 1


def test_relative_and_absolute_path_spelling_still_share_one_instance(
    tmp_path, monkeypatch
):
    ooc_path = tmp_path / "znpc_ooc.csv"
    _write_ooc_csv(ooc_path)
    monkeypatch.chdir(tmp_path)

    absolute = OpticalConstants.from_file(str(ooc_path))
    relative = OpticalConstants.from_file("znpc_ooc.csv")

    assert absolute is relative
    assert OpticalConstants.cache_size() == 1


def test_different_materials_are_not_merged(tmp_path):
    znpc_path = tmp_path / "znpc_ooc.csv"
    c60_path = tmp_path / "c60_ooc.csv"
    _write_ooc_csv(znpc_path)
    _write_ooc_csv(c60_path)

    znpc = OpticalConstants.from_file(znpc_path)
    c60 = OpticalConstants.from_file(c60_path)

    assert znpc is not c60
    assert OpticalConstants.cache_size() == 2


def test_from_dataframe_shares_by_identity_not_by_equal_content():
    table_a = pl.DataFrame(
        {
            "energy": [500.0, 700.0],
            "n_xx": [4e-6, 5e-6],
            "n_ixx": [1e-7, 1.2e-7],
            "n_zz": [8e-6, 9e-6],
            "n_izz": [2e-7, 2.2e-7],
        }
    )
    table_b = table_a.clone()  # equal content, distinct object

    same_object_twice_a = OpticalConstants.from_dataframe(table_a)
    same_object_twice_b = OpticalConstants.from_dataframe(table_a)
    different_object = OpticalConstants.from_dataframe(table_b)

    assert same_object_twice_a is same_object_twice_b
    assert same_object_twice_a is not different_object
    assert OpticalConstants.cache_size() == 2


def test_from_source_accepts_existing_instance_dataframe_or_path(tmp_path):
    ooc_path = tmp_path / "znpc_ooc.csv"
    _write_ooc_csv(ooc_path)

    from_path = OpticalConstants.from_source(str(ooc_path))
    from_existing_instance = OpticalConstants.from_source(from_path)

    assert from_path is from_existing_instance
    assert OpticalConstants.cache_size() == 1


def test_missing_required_column_raises():
    incomplete = pl.DataFrame({"energy": [500.0], "n_xx": [4e-6]})
    with pytest.raises(ValueError, match="missing columns"):
        OpticalConstants(incomplete, source="<test>")
