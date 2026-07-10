import torch

from minimoss.dataset import collate_fn
from minimoss.prepare_hf_dataset import split_records


def test_collate_pads_audio_and_returns_frame_mask():
    batch = [
        (torch.tensor([1, 2]), torch.ones(3, 4, dtype=torch.long)),
        (torch.tensor([3]), torch.full((5, 4), 2, dtype=torch.long)),
    ]

    text, text_mask, rvq, audio_mask = collate_fn(batch)

    assert text.shape == (2, 2)
    assert text_mask.tolist() == [[1, 1], [1, 0]]
    assert rvq.shape == (2, 5, 4)
    assert audio_mask.tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, True],
    ]
    assert torch.equal(rvq[0, 3:], torch.zeros(2, 4, dtype=torch.long))


def test_split_records_is_deterministic_and_disjoint():
    records = [{"id": str(index)} for index in range(10)]

    train_a, validation_a = split_records(records, validation_count=2, seed=42)
    train_b, validation_b = split_records(list(reversed(records)), validation_count=2, seed=42)

    assert train_a == train_b
    assert validation_a == validation_b
    assert len(train_a) == 8
    assert len(validation_a) == 2
    assert {item["id"] for item in train_a}.isdisjoint(
        item["id"] for item in validation_a
    )
