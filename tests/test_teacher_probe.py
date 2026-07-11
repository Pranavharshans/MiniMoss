import torch

from minimoss.validate_moss_teacher import (
    CoarseStateProbe,
    probe_loss,
    valid_audio_target_mask,
)


def test_valid_audio_target_mask_rejects_padding_and_special_tokens():
    targets = torch.zeros((1, 3, 33), dtype=torch.long)
    targets[0, 1, 5] = 1024
    targets[0, 2, 8] = -100

    mask = valid_audio_target_mask(targets, audio_pad_code=1024)

    assert mask.tolist() == [[True, False, False]]


def test_coarse_probe_predicts_four_codebooks_and_backpropagates():
    probe = CoarseStateProbe(input_size=12, hidden_size=16, codebook_size=8)
    states = torch.randn(5, 12)
    targets = torch.randint(0, 8, (5, 4))

    logits = probe(states)
    loss = probe_loss(logits, targets)
    loss.backward()

    assert len(logits) == 4
    assert all(channel.shape == (5, 8) for channel in logits)
    assert probe.trunk[1].weight.grad is not None
