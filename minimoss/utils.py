import random
import torch
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_metrics(step: int, total_loss: float, group_losses: list[float], lr: float, step_time: float) -> str:
    parts = [
        f"step={step}",
        f"loss={total_loss:.4f}",
    ]
    for i, gl in enumerate(group_losses):
        parts.append(f"g{i+1}={gl:.4f}")
    parts.append(f"lr={lr:.2e}")
    parts.append(f"t={step_time:.2f}s")
    return " | ".join(parts)
