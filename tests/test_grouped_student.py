import pytest
import torch

from minimoss.grouped_student import (
    ADJACENT2_GROUPS,
    COARSE_FIRST_GROUPS,
    EXPERIMENT_SPECS,
    GroupedLocalStudent,
    GroupedStudentConfig,
    get_experiment_spec,
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


def test_experiment_matrix_has_five_valid_layouts():
    assert set(EXPERIMENT_SPECS) == {
        "baseline11",
        "gt_only11",
        "kd_only11",
        "adjacent16",
        "large11",
        "rollout11",
    }
    assert len(ADJACENT2_GROUPS) == 16
    for name in EXPERIMENT_SPECS:
        spec = get_experiment_spec(name)
        flattened = [codebook for group in spec["groups"] for codebook in group]
        assert flattened == list(range(32))
        assert spec["ground_truth_weight"] + spec["distillation_weight"] > 0


def test_unknown_experiment_has_actionable_error():
    with pytest.raises(ValueError, match="choose from"):
        get_experiment_spec("does_not_exist")


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


def test_rollout_loss_backpropagates_under_free_context():
    model = GroupedLocalStudent(tiny_config())
    states = torch.randn(5, 12)
    targets = torch.randint(0, 8, (5, 4))

    loss, channel_losses = model.rollout_loss(
        states,
        targets,
        teacher_forcing_probability=0.0,
    )
    loss.backward()

    assert len(channel_losses) == 4
    assert torch.isfinite(loss)
    assert model.output_heads[0].weight.grad is not None


def test_group_layout_rejects_missing_or_reordered_codebooks():
    with pytest.raises(ValueError, match="cover every codebook"):
        GroupedStudentConfig(global_hidden_size=12, n_codebooks=4, groups=((0,), (2, 3)))
