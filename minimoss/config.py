from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MiniMossConfig:
    # Backbone
    backbone_name: str = "Qwen/Qwen2.5-0.5B"
    backbone_hidden_size: int = 896  # Qwen2.5-0.5B hidden dim

    # Codec
    codec_name: str = "descript/dac_24khz"
    codec_sample_rate: int = 24000
    n_codebooks: int = 16
    codebook_size: int = 1024  # DAC uses 1024

    # Grouped local decoder
    n_groups: int = 4
    codebooks_per_group: int = 4
    local_hidden_size: int = 512
    local_num_layers: int = 4
    local_num_heads: int = 8
    local_ffn_hidden_size: int = 1024
    local_dropout: float = 0.0

    # Frame position embeddings
    max_frames: int = 2048

    # Projection MLP
    projection_ffn_hidden_size: int = 1024

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 5000
    batch_size: int = 4
    grad_accum_steps: int = 1
    group_loss_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    use_amp: bool = True

    # Logging
    log_every: int = 10
    sample_every: int = 500
    checkpoint_every: int = 1000

    # Paths
    output_dir: str = "./checkpoints"
    manifest_path: str = "data/manifest.jsonl"
    token_dir: str = "data/tokens"

    # Text tokenizer (from backbone)
    pad_token_id: int = 151643  # Qwen2.5 pad token
    audio_pad_code: int = 1024

    @property
    def local_head_dim(self) -> int:
        return self.local_hidden_size // self.local_num_heads
