"""Grouped local student driven by cached official-MOSS global frame states."""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import LocalTransformer


COARSE_FIRST_GROUPS = (
    (0,),
    (1,),
    (2,),
    (3,),
    (4, 5, 6, 7),
    (8, 9, 10, 11),
    (12, 13, 14, 15),
    (16, 17, 18, 19),
    (20, 21, 22, 23),
    (24, 25, 26, 27),
    (28, 29, 30, 31),
)


@dataclass
class GroupedStudentConfig:
    global_hidden_size: int
    local_hidden_size: int = 512
    local_num_layers: int = 4
    local_num_heads: int = 8
    local_num_kv_heads: int = 2
    local_ffn_hidden_size: int = 1024
    local_dropout: float = 0.1
    codebook_size: int = 1024
    n_codebooks: int = 32
    groups: tuple[tuple[int, ...], ...] = COARSE_FIRST_GROUPS

    def __post_init__(self):
        flattened = [codebook for group in self.groups for codebook in group]
        if flattened != list(range(self.n_codebooks)):
            raise ValueError("groups must cover every codebook exactly once in order")
        if self.local_hidden_size % self.local_num_heads:
            raise ValueError("local_hidden_size must be divisible by local_num_heads")
        if self.local_num_heads % self.local_num_kv_heads:
            raise ValueError("local_num_heads must be divisible by local_num_kv_heads")

    @property
    def local_head_dim(self):
        return self.local_hidden_size // self.local_num_heads

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        values["groups"] = tuple(tuple(group) for group in values["groups"])
        return cls(**values)


class GroupedLocalStudent(nn.Module):
    def __init__(self, config: GroupedStudentConfig):
        super().__init__()
        self.config = config
        self.global_projection = nn.Sequential(
            nn.LayerNorm(config.global_hidden_size),
            nn.Linear(config.global_hidden_size, config.local_hidden_size, bias=False),
        )
        self.codebook_embeddings = nn.ModuleList([
            nn.Embedding(config.codebook_size, config.local_hidden_size)
            for _ in range(config.n_codebooks)
        ])
        self.local_decoder = LocalTransformer(config)
        self.output_heads = nn.ModuleList([
            nn.Linear(config.local_hidden_size, config.codebook_size, bias=False)
            for _ in range(config.n_codebooks)
        ])
        self._initialize()

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def embed_group(self, codes: torch.Tensor, group: tuple[int, ...]):
        embedding = 0
        for codebook in group:
            embedding = embedding + self.codebook_embeddings[codebook](
                codes[:, codebook]
            )
        return embedding

    def teacher_logits(self, global_states: torch.Tensor, targets: torch.Tensor):
        inputs = [self.global_projection(global_states)]
        for group in self.config.groups[:-1]:
            inputs.append(self.embed_group(targets, group))
        decoder_input = torch.stack(inputs, dim=1)
        positions = torch.arange(len(inputs), device=decoder_input.device).unsqueeze(0)
        hidden = self.local_decoder(decoder_input, position_ids=positions)
        logits = [None] * self.config.n_codebooks
        for group_index, group in enumerate(self.config.groups):
            for codebook in group:
                logits[codebook] = self.output_heads[codebook](hidden[:, group_index])
        return logits

    def loss(self, global_states, targets, label_smoothing: float = 0.0):
        logits = self.teacher_logits(global_states, targets)
        losses = [
            F.cross_entropy(
                logits[codebook],
                targets[:, codebook],
                label_smoothing=label_smoothing,
            )
            for codebook in range(self.config.n_codebooks)
        ]
        return torch.stack(losses).mean(), losses

    def distillation_loss(
        self,
        logits,
        teacher_indices: torch.Tensor,
        teacher_values: torch.Tensor,
        temperature: float = 1.0,
    ):
        student = torch.stack(logits, dim=1)
        teacher_probabilities = F.softmax(
            teacher_values.float() / temperature, dim=-1
        )
        student_log_probabilities = F.log_softmax(
            student.float() / temperature, dim=-1
        )
        selected = student_log_probabilities.gather(
            dim=-1, index=teacher_indices.long()
        )
        return -(teacher_probabilities * selected).sum(dim=-1).mean() * temperature ** 2

    def combined_loss(
        self,
        global_states,
        targets,
        teacher_indices,
        teacher_values,
        ground_truth_weight: float,
        distillation_weight: float,
        temperature: float,
        label_smoothing: float = 0.0,
    ):
        logits = self.teacher_logits(global_states, targets)
        channel_losses = [
            F.cross_entropy(
                logits[codebook],
                targets[:, codebook],
                label_smoothing=label_smoothing,
            )
            for codebook in range(self.config.n_codebooks)
        ]
        ground_truth_loss = torch.stack(channel_losses).mean()
        teacher_loss = self.distillation_loss(
            logits, teacher_indices, teacher_values, temperature
        )
        total = ground_truth_weight * ground_truth_loss + distillation_weight * teacher_loss
        return total, ground_truth_loss, teacher_loss, channel_losses

    @torch.inference_mode()
    def predict(self, global_states: torch.Tensor, teacher_targets=None):
        """Predict all codebooks, optionally conditioning on ground-truth prior groups."""
        inputs = [self.global_projection(global_states)]
        predictions = torch.zeros(
            (global_states.shape[0], self.config.n_codebooks),
            dtype=torch.long,
            device=global_states.device,
        )
        for group_index, group in enumerate(self.config.groups):
            decoder_input = torch.stack(inputs, dim=1)
            positions = torch.arange(len(inputs), device=decoder_input.device).unsqueeze(0)
            hidden = self.local_decoder(decoder_input, position_ids=positions)[:, -1]
            for codebook in group:
                predictions[:, codebook] = self.output_heads[codebook](hidden).argmax(-1)
            if group_index + 1 < len(self.config.groups):
                context = teacher_targets if teacher_targets is not None else predictions
                inputs.append(self.embed_group(context, group))
        return predictions
