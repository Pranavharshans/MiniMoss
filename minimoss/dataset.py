import json
import torch
from pathlib import Path
from typing import Optional


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
        n_codebooks: Optional[int] = None,
        codebook_size: Optional[int] = None,
    ):
        self.manifest = load_manifest(manifest_path)
        self.token_dir = Path(token_dir)
        self.max_frames = max_frames
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        if not self.manifest:
            raise ValueError(f"Manifest is empty: {manifest_path}")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        item = self.manifest[idx]
        tok_path = self.token_dir / f"{item['id']}.pt"
        data = torch.load(tok_path, map_location="cpu", weights_only=True)
        text_tokens = data["text_tokens"]
        rvq = data["rvq"]  # [n_frames, n_codebooks]

        if text_tokens.ndim != 1 or text_tokens.numel() == 0:
            raise ValueError(f"{tok_path}: text_tokens must be a non-empty 1-D tensor")
        if rvq.ndim != 2 or rvq.shape[0] == 0:
            raise ValueError(f"{tok_path}: rvq must have shape [frames, codebooks] with frames > 0")
        if self.n_codebooks is not None and rvq.shape[1] != self.n_codebooks:
            raise ValueError(
                f"{tok_path}: found {rvq.shape[1]} codebooks, expected {self.n_codebooks}"
            )
        if self.codebook_size is not None:
            min_code, max_code = int(rvq.min()), int(rvq.max())
            if min_code < 0 or max_code >= self.codebook_size:
                raise ValueError(
                    f"{tok_path}: RVQ codes must be in [0, {self.codebook_size - 1}], "
                    f"found [{min_code}, {max_code}]"
                )

        if self.max_frames is not None and rvq.shape[0] > self.max_frames:
            rvq = rvq[:self.max_frames]

        return text_tokens, rvq


def collate_fn(batch):
    """Pad text and audio while returning masks for both modalities."""
    text_tokens_list, rvq_list = zip(*batch)

    # Pad text tokens and build attention mask
    max_text_len = max(t.shape[0] for t in text_tokens_list)
    pad_id = 151643
    text_padded = torch.full((len(batch), max_text_len), pad_id, dtype=torch.long)
    text_mask = torch.zeros((len(batch), max_text_len), dtype=torch.long)
    for i, t in enumerate(text_tokens_list):
        text_padded[i, :t.shape[0]] = t
        text_mask[i, :t.shape[0]] = 1

    max_audio_len = max(r.shape[0] for r in rvq_list)
    n_codebooks = rvq_list[0].shape[1]
    if any(r.shape[1] != n_codebooks for r in rvq_list):
        raise ValueError("All RVQ tensors in a batch must have the same codebook count")
    rvq_batch = torch.zeros(
        (len(batch), max_audio_len, n_codebooks), dtype=torch.long
    )
    audio_mask = torch.zeros((len(batch), max_audio_len), dtype=torch.bool)
    for i, rvq in enumerate(rvq_list):
        rvq_batch[i, :rvq.shape[0]] = rvq.long()
        audio_mask[i, :rvq.shape[0]] = True

    return text_padded, text_mask, rvq_batch, audio_mask
