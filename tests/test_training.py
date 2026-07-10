import pytest

from minimoss.train_overfit import context_dropout_for_step


def test_context_dropout_schedule_holds_then_decays():
    values = [
        context_dropout_for_step(step, 100, 200, 1.0, 0.2)
        for step in (1, 100, 200, 300, 400)
    ]

    assert values == pytest.approx([1.0, 1.0, 0.6, 0.2, 0.2])


def test_context_dropout_schedule_supports_no_decay():
    assert context_dropout_for_step(101, 100, 0, 1.0, 0.2) == 0.2
