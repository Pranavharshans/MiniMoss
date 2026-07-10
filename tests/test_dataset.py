import torch

from minimoss.dataset import collate_fn


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
