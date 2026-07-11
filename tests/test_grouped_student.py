import pytest
import torch

from minimoss.grouped_student import (
    COARSE_FIRST_GROUPS,
    GroupedLocalStudent,
    GroupedStudentConfig,
)


def tiny_config():
    return GroupedStudentConfig(
        global_hidden_size=12,
        local_hidden_size=16,
        local_num_layers=2,
        local_num_heads=4,
        local_num_kv_heads=2,
        local_ffn_hidden_size=32,
        codebook_size=8,
        n_codebooks=4,
        groups=((0,), (1,), (2, 3)),
    )


def test_coarse_first_layout_has_eleven_steps_and_all_codebooks():
    assert len(COARSE_FIRST_GROUPS) == 11
    assert [codebook for group in COARSE_FIRST_GROUPS for codebook in group] == list(range(32))


def test_grouped_student_forward_backward_and_prediction_modes():
    model = GroupedLocalStudent(tiny_config())
    states = torch.randn(5, 12)
    targets = torch.randint(0, 8, (5, 4))

    loss, channel_losses = model.loss(states, targets)
    loss.backward()
    teacher = model.predict(states, teacher_targets=targets)
    free = model.predict(states)

    assert len(channel_losses) == 4
    assert teacher.shape == free.shape == targets.shape
    assert model.global_projection[1].weight.grad is not None


def test_topk_distillation_loss_backpropagates():
    model = GroupedLocalStudent(tiny_config())
    states = torch.randn(5, 12)
    targets = torch.randint(0, 8, (5, 4))
    teacher_indices = torch.randint(0, 8, (5, 4, 3))
    teacher_values = torch.randn(5, 4, 3)

    total, ground_truth, distillation, _ = model.combined_loss(
        states,
        targets,
        teacher_indices,
        teacher_values,
        ground_truth_weight=0.5,
        distillation_weight=0.5,
        temperature=2.0,
    )
    total.backward()

    assert torch.isfinite(ground_truth)
    assert torch.isfinite(distillation)
    assert model.output_heads[0].weight.grad is not None


def test_group_layout_rejects_missing_or_reordered_codebooks():
    with pytest.raises(ValueError, match="cover every codebook"):
        GroupedStudentConfig(global_hidden_size=12, n_codebooks=4, groups=((0,), (2, 3)))
