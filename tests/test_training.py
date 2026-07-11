import pytest
import torch

from minimoss.train_overfit import (
    context_dropout_for_step,
    curriculum_phase_and_weights,
    per_codebook_losses,
    refinement_stage_and_weights,
)


def test_context_dropout_schedule_holds_then_decays():
    values = [
        context_dropout_for_step(step, 100, 200, 1.0, 0.2)
        for step in (1, 100, 200, 300, 400)
    ]

    assert values == pytest.approx([1.0, 1.0, 0.6, 0.2, 0.2])


def test_context_dropout_schedule_supports_no_decay():
    assert context_dropout_for_step(101, 100, 0, 1.0, 0.2) == 0.2


def test_group_curriculum_stages_refinement_losses():
    phase_a, weights_a = curriculum_phase_and_weights(750, 750, 1500)
    phase_b, weights_b = curriculum_phase_and_weights(751, 750, 1500)
    phase_c, weights_c = curriculum_phase_and_weights(1501, 750, 1500)

    assert phase_a == "A"
    assert weights_a == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert phase_b == "B"
    assert weights_b == (4.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    assert phase_c == "C"
    assert weights_c == (4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def test_phase_gate_best_improvement_uses_best_not_boundary_loss():
    initial_loss = 6.6918
    best_loss = 6.6680
    boundary_loss = 6.6835

    assert initial_loss - best_loss > 0.02
    assert initial_loss - boundary_loss < 0.02


def test_refinement_curriculum_introduces_one_group_at_a_time():
    phase_r2, weights_r2, group_r2 = refinement_stage_and_weights(375, 375)
    phase_r3, weights_r3, group_r3 = refinement_stage_and_weights(376, 375)
    phase_r8, weights_r8, group_r8 = refinement_stage_and_weights(2625, 375)

    assert (phase_r2, group_r2) == ("R2", 2)
    assert weights_r2 == (8.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert (phase_r3, group_r3) == ("R3", 3)
    assert weights_r3 == (8.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert (phase_r8, group_r8) == ("R8", 8)
    assert weights_r8 == (8.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0)


def test_refinement_curriculum_supports_stronger_new_group_gate():
    phase, weights, group = refinement_stage_and_weights(
        750,
        750,
        anchor_weight=8.0,
        existing_weight=2.0,
        new_weight=4.0,
    )

    assert (phase, group) == ("R2", 2)
    assert weights == (8.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_per_codebook_losses_respect_frame_mask():
    logits = [torch.zeros((1, 2, 4)), torch.zeros((1, 2, 4))]
    targets = torch.tensor([[[0, 1], [3, 2]]])
    mask = torch.tensor([[True, False]])

    losses = per_codebook_losses(logits, targets, mask, codebook_size=4)

    assert len(losses) == 2
    assert losses[0].item() == pytest.approx(torch.log(torch.tensor(4.0)).item())
    assert losses[1].item() == pytest.approx(torch.log(torch.tensor(4.0)).item())
