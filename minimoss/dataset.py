import json
import torch
import torchaudio
from pathlib import Path
from typing import Optional

from .codec import AudioCodec


def load_manifest(path: str) -> list[dict]:
    """Load a JSONL manifest file."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


class MiniMossDataset(torch.utils.data.Dataset):
    """Loads pre-tokenized RVQ codes + text from disk."""

    def __init__(
        self,
        manifest_path: str,
        token_dir: str,
        max_frames: Optional[int] = None,
    ):
        self.manifest = load_manifest(manifest_path)
        self.token_dir = Path(token_dir)
        self.max_frames = max_frames

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        item = self.manifest[idx]
        tok_path = self.token_dir / f"{item['id']}.pt"
        data = torch.load(tok_path, map_location="cpu", weights_only=True)
        text_tokens = data["text_tokens"]
        rvq = data["rvq"]  # [n_frames, n_codebooks]

        if self.max_frames is not None and rvq.shape[0] > self.max_frames:
            rvq = rvq[:self.max_frames]

        return text_tokens, rvq


def collate_fn(batch):
    """Pad text_tokens and truncate/align to common audio length."""
    text_tokens_list, rvq_list = zip(*batch)

    # Pad text tokens and build attention mask
    max_text_len = max(t.shape[0] for t in text_tokens_list)
    pad_id = 151643
    text_padded = torch.full((len(batch), max_text_len), pad_id, dtype=torch.long)
    text_mask = torch.zeros((len(batch), max_text_len), dtype=torch.long)
    for i, t in enumerate(text_tokens_list):
        text_padded[i, :t.shape[0]] = t
        text_mask[i, :t.shape[0]] = 1

    # Take min audio length
    min_audio_len = min(r.shape[0] for r in rvq_list)
    rvq_batch = torch.stack([r[:min_audio_len] for r in rvq_list])

    return text_padded, text_mask, rvq_batch
