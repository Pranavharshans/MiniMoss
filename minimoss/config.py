from dataclasses import dataclass


@dataclass
class MiniMossConfig:
    # Backbone
    backbone_name: str = "Qwen/Qwen2.5-0.5B"
    backbone_hidden_size: int = 896  # Qwen2.5-0.5B hidden dim
    use_qwen_lora: bool = False
    qwen_lora_rank: int = 8
    qwen_lora_alpha: float = 16.0
    qwen_lora_dropout: float = 0.0
    qwen_lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    # Audio tokenizer / codec
    # MOSS-Audio-Tokenizer uses 32 RVQ layers, codebook size 1024, and 12.5 fps
    # according to the MOSS-TTS technical report.
    codec_name: str = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
    codec_sample_rate: int = 24000
    n_codebooks: int = 32
    codebook_size: int = 1024
    use_nonlinear_frame_conditioner: bool = False

    # Grouped local decoder
    n_groups: int = 8
    codebooks_per_group: int = 4
    local_hidden_size: int = 512
    local_num_layers: int = 4
    local_num_heads: int = 8
    local_num_kv_heads: int = 2
    local_ffn_hidden_size: int = 1024
    local_dropout: float = 0.0

    # Maximum temporal context used by the lean overfit experiment
    max_frames: int = 2048

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 5000
    batch_size: int = 4
    group_loss_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    use_amp: bool = True

    # Logging
    log_every: int = 10
    checkpoint_every: int = 1000

    # Paths
    output_dir: str = "./checkpoints"
    manifest_path: str = "data/manifest.jsonl"
    token_dir: str = "data/tokens"

    def __post_init__(self):
        if self.n_codebooks != self.n_groups * self.codebooks_per_group:
            raise ValueError(
                "n_codebooks must equal n_groups * codebooks_per_group "
                f"({self.n_codebooks} != {self.n_groups} * {self.codebooks_per_group})"
            )
        if self.local_hidden_size % self.local_num_heads != 0:
            raise ValueError("local_hidden_size must be divisible by local_num_heads")
        if self.local_num_heads % self.local_num_kv_heads != 0:
            raise ValueError("local_num_heads must be divisible by local_num_kv_heads")
        if len(self.group_loss_weights) != self.n_groups:
            raise ValueError("group_loss_weights length must equal n_groups")
        if self.codebook_size <= 0:
            raise ValueError("codebook_size must be positive")
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if self.qwen_lora_rank <= 0:
            raise ValueError("qwen_lora_rank must be positive")

    @property
    def local_head_dim(self) -> int:
        return self.local_hidden_size // self.local_num_heads
