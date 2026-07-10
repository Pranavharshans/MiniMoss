from types import SimpleNamespace

import torch
import torch.nn as nn

from minimoss.config import MiniMossConfig
from minimoss.model import LocalRMSNorm, MiniMossModel


class FakeBackbone(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, **kwargs):
        del attention_mask, kwargs
        return SimpleNamespace(last_hidden_state=self.layer(inputs_embeds))


def tiny_config():
    return MiniMossConfig(
        backbone_hidden_size=12,
        n_codebooks=8,
        n_groups=2,
        codebooks_per_group=4,
        codebook_size=16,
        local_hidden_size=16,
        local_num_layers=2,
        local_num_heads=4,
        local_num_kv_heads=2,
        local_ffn_hidden_size=32,
        max_frames=16,
        group_loss_weights=(1.0, 1.0),
    )


def test_forward_backward_keeps_backbone_frozen_and_masks_padding():
    model = MiniMossModel(tiny_config())
    model._backbone = FakeBackbone(vocab_size=32, hidden_size=12)
    model._backbone.requires_grad_(False)

    text = torch.tensor([[1, 2, 3], [4, 5, 0]])
    text_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    audio = torch.randint(0, 16, (2, 5, 8))
    audio_mask = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool
    )

    logits, losses = model(text, audio, text_mask, audio_mask)
    assert len(logits) == 8
    assert logits[0].shape == (2, 5, 16)
    assert len(losses) == 2
    assert all(torch.isfinite(loss) for loss in losses)

    sum(losses).backward()
    assert model.frame_to_backbone.weight.grad is not None
    assert all(parameter.grad is None for parameter in model._backbone.parameters())


def test_rmsnorm_scales_initialize_to_one():
    model = MiniMossModel(tiny_config())
    norms = [module for module in model.modules() if isinstance(module, LocalRMSNorm)]
    assert norms
    assert all(torch.equal(norm.weight, torch.ones_like(norm.weight)) for norm in norms)


def test_frame_conditioning_is_shifted_without_target_leakage():
    torch.manual_seed(3)
    model = MiniMossModel(tiny_config())
    model._backbone = FakeBackbone(vocab_size=32, hidden_size=12)
    model._backbone.requires_grad_(False)
    audio_a = torch.randint(0, 16, (1, 4, 8))
    audio_b = audio_a.clone()
    audio_b[:, 2] = (audio_b[:, 2] + 1) % 16

    hidden_a = model.encode_frames(torch.tensor([[1, 2]]), audio_a)
    hidden_b = model.encode_frames(torch.tensor([[1, 2]]), audio_b)

    assert torch.allclose(hidden_a[:, 2], hidden_b[:, 2])
    assert not torch.allclose(hidden_a[:, 3], hidden_b[:, 3])


def test_tiny_batch_loss_decreases():
    torch.manual_seed(7)
    model = MiniMossModel(tiny_config())
    model._backbone = FakeBackbone(vocab_size=32, hidden_size=12)
    model._backbone.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-2,
        weight_decay=0.0,
    )

    text = torch.tensor([[1, 2, 3]])
    audio = torch.randint(0, 16, (1, 4, 8))
    mask = torch.ones(1, 4, dtype=torch.bool)
    losses_seen = []
    for _ in range(120):
        _, group_losses = model(text, audio, audio_frame_mask=mask)
        loss = sum(group_losses) / len(group_losses)
        losses_seen.append(float(loss.detach()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert losses_seen[-1] < losses_seen[0] * 0.25
