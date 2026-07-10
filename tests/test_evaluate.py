import torch

from minimoss.evaluate import numbered_prefix, token_metrics


def test_numbered_prefix_uses_at_least_two_digits():
    assert numbered_prefix(1, 10) == "01"
    assert numbered_prefix(10, 10) == "10"
    assert numbered_prefix(1, 100) == "001"


def test_token_metrics_reports_token_group_and_frame_accuracy():
    target = torch.zeros((2, 8), dtype=torch.long)
    prediction = target.clone()
    prediction[1, 0] = 1

    metrics = token_metrics(prediction, target, group_size=4)

    assert metrics["token_accuracy"] == 15 / 16
    assert metrics["frame_accuracy"] == 0.5
    assert metrics["group_accuracy"] == [7 / 8, 1.0]
