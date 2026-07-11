from types import SimpleNamespace

import torch
import torch.nn as nn

from minimoss.extract_moss_teacher_targets import original_local_logits


class IdentityLocal(nn.Module):
    def forward(self, inputs_embeds, **kwargs):
        del kwargs
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class FakeTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            n_vq=4,
            audio_vocab_size=8,
            audio_assistant_gen_slot_token_id=3,
        )
        self.model = SimpleNamespace(
            embedding_list=nn.ModuleList([nn.Embedding(8, 6) for _ in range(4)])
        )
        self.speech_embedding_to_local_mlp = nn.Identity()
        self.local_transformer = IdentityLocal()
        self.local_to_speech_embedding_mlps = nn.ModuleList([nn.Identity() for _ in range(5)])
        self.layer_norm_before_lm_heads = nn.ModuleList([nn.Identity() for _ in range(5)])
        self.lm_heads = nn.ModuleList([nn.Linear(6, 8, bias=False) for _ in range(5)])


def test_original_local_logits_align_audio_heads_after_text_channel():
    teacher = FakeTeacher()
    states = torch.randn(3, 6)
    targets = torch.randint(0, 8, (3, 4))

    logits = original_local_logits(teacher, states, targets)

    assert len(logits) == 4
    assert all(channel.shape == (3, 8) for channel in logits)
