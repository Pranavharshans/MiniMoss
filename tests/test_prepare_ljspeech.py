import pytest

from minimoss.prepare_ljspeech import select_split


def test_select_split_is_deterministic_and_disjoint():
    records = [{"id": f"LJ-{index:04d}"} for index in range(20)]
    train_a, validation_a = select_split(records, 10, 5, seed=42)
    train_b, validation_b = select_split(list(reversed(records)), 10, 5, seed=42)

    assert train_a == train_b
    assert validation_a == validation_b
    assert len(train_a) == 10
    assert len(validation_a) == 5
    assert {row["id"] for row in train_a}.isdisjoint(row["id"] for row in validation_a)


def test_select_split_rejects_insufficient_records():
    with pytest.raises(ValueError, match="Need 11 records"):
        select_split([{"id": "one"}], 10, 1, seed=42)
