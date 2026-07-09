import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import MiniMossConfig


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.register_buffer("inv_freq", self._compute_inv_freq(), persistent=False)

    def _compute_inv_freq(self, device=None):
        return 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )

    def forward(self, position_ids: torch.LongTensor, device: torch.device, dtype: torch.dtype):
        inv_freq = self._compute_inv_freq(device=device)
        freqs = torch.einsum("bs,d->bsd", position_ids.to(dtype=inv_freq.dtype), inv_freq)
        cos = freqs.cos().repeat_interleave(2, dim=-1).unsqueeze(1).to(dtype=dtype)
        sin = freqs.sin().repeat_interleave(2, dim=-1).unsqueeze(1).to(dtype=dtype)
        return cos, sin


def rotate_half(x):
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).reshape_as(x)


def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


# ---------------------------------------------------------------------------
# Local transformer blocks
# ---------------------------------------------------------------------------

class LocalRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight


class LocalAttention(nn.Module):
    def __init__(self, config: MiniMossConfig):
        super().__init__()
        self.n_heads = config.local_num_heads
        self.head_dim = config.local_head_dim
        self.hidden_size = config.local_hidden_size

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.local_dropout)
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x, position_ids=None):
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if position_ids is not None:
            cos, sin = self.rotary(position_ids, device=x.device, dtype=x.dtype)
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.dropout(self.o_proj(y))


class LocalMLP(nn.Module):
    def __init__(self, config: MiniMossConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.local_hidden_size, config.local_ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(config.local_hidden_size, config.local_ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(config.local_ffn_hidden_size, config.local_hidden_size, bias=False)
        self.dropout = nn.Dropout(config.local_dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class LocalTransformerBlock(nn.Module):
    def __init__(self, config: MiniMossConfig):
        super().__init__()
        self.ln_1 = LocalRMSNorm(config.local_hidden_size)
        self.attn = LocalAttention(config)
        self.ln_2 = LocalRMSNorm(config.local_hidden_size)
        self.mlp = LocalMLP(config)

    def forward(self, x, position_ids=None):
        x = x + self.attn(self.ln_1(x), position_ids=position_ids)
        x = x + self.mlp(self.ln_2(x))
        return x


class LocalTransformer(nn.Module):
    def __init__(self, config: MiniMossConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            LocalTransformerBlock(config) for _ in range(config.local_num_layers)
        ])
        self.ln_f = LocalRMSNorm(config.local_hidden_size)

    def forward(self, x, position_ids=None):
        for layer in self.layers:
            x = layer(x, position_ids=position_ids)
        return self.ln_f(x)


# ---------------------------------------------------------------------------
# SwiGLU projection MLP (global -> local)
# ---------------------------------------------------------------------------

class ProjectionMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MiniMossModel(nn.Module):
    def __init__(self, config: MiniMossConfig):
        super().__init__()
        self.config = config

        # Frozen backbone (lazy init)
        self._backbone = None

        # Projection: backbone hidden -> local hidden
        self.projection = ProjectionMLP(
            input_size=config.backbone_hidden_size,
            hidden_size=config.projection_ffn_hidden_size,
            output_size=config.local_hidden_size,
        )

        # Frame position embeddings
        self.frame_pos_embed = nn.Embedding(config.max_frames, config.local_hidden_size)

        # Codebook embeddings (one per codebook)
        self.codebook_embeddings = nn.ModuleList([
            nn.Embedding(config.codebook_size, config.local_hidden_size)
            for _ in range(config.n_codebooks)
        ])

        # Local decoder
        self.local_decoder = LocalTransformer(config)

        # Output heads (one per codebook)
        self.output_heads = nn.ModuleList([
            nn.Linear(config.local_hidden_size, config.codebook_size, bias=False)
            for _ in range(config.n_codebooks)
        ])

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        for module in [self.projection, self.local_decoder, *self.output_heads]:
            for p in module.parameters():
                if p.dim() >= 2:
                    nn.init.normal_(p, mean=0.0, std=std)
                elif p.dim() == 1:
                    nn.init.zeros_(p)
        nn.init.normal_(self.frame_pos_embed.weight, mean=0.0, std=std)
        for emb in self.codebook_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=std)

    @property
    def backbone(self):
        if self._backbone is None:
            from transformers import AutoModel
            self._backbone = AutoModel.from_pretrained(
                self.config.backbone_name,
                torch_dtype=torch.float32,
            )
            for p in self._backbone.parameters():
                p.requires_grad = False
            self._backbone.eval()
        return self._backbone

    @property
    def device(self):
        return next(self.parameters()).device

    def encode_text(self, text_tokens: torch.LongTensor, attention_mask: Optional[torch.LongTensor] = None) -> torch.Tensor:
        """Encode text tokens through frozen Qwen backbone, return pooled representation.

        NOTE: This mean-pools all text positions into a single vector, then repeats it
        across audio frames with frame position embeddings. This is a simplification
        vs. the PRD's "selected hidden states" per frame — it tests whether a
        text-conditioned decoder can overfit, NOT whether Qwen produces per-frame
        acoustic hidden states. A stronger version would interleave text and audio
        tokens in the backbone sequence (as the official MOSS-TTS does).
        """
        backbone = self.backbone.to(text_tokens.device)
        with torch.no_grad():
            out = backbone(text_tokens, attention_mask=attention_mask, output_hidden_states=False)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(dtype=out.last_hidden_state.dtype)
            pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = out.last_hidden_state.mean(dim=1)
        return pooled

    def embed_group(self, group_codes: torch.LongTensor) -> torch.Tensor:
        """Embed a group of codebook tokens.

        Args:
            group_codes: [..., codebooks_per_group] codebook indices

        Returns:
            [..., local_hidden_size] summed embedding
        """
        g = group_codes.shape[-1]  # codebooks_per_group
        emb = 0
        for i in range(g):
            emb = emb + self.codebook_embeddings[i](group_codes[..., i])
        return emb

    def _get_group_embeddings(self, audio_codes: torch.LongTensor) -> list[torch.Tensor]:
        """Compute ground-truth group embeddings for all groups.

        Args:
            audio_codes: [B, T_audio, n_codebooks]

        Returns:
            list of [B, T_audio, local_hidden_size], one per group
        """
        B, T_audio, n_cb = audio_codes.shape
        n_groups = self.config.n_groups
        cbg = self.config.codebooks_per_group
        group_embs = []
        for g in range(n_groups):
            start = g * cbg
            g_codes = audio_codes[:, :, start:start + cbg]
            g_emb = 0
            for i in range(cbg):
                g_emb = g_emb + self.codebook_embeddings[start + i](g_codes[:, :, i])
            group_embs.append(g_emb)
        return group_embs

    def forward(
        self,
        text_tokens: torch.LongTensor,
        audio_codes: torch.LongTensor,
        text_attention_mask: Optional[torch.LongTensor] = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass with teacher forcing.

        Args:
            text_tokens: [B, T_text] text token ids
            audio_codes: [B, T_audio, n_codebooks] ground-truth RVQ tokens
            text_attention_mask: [B, T_text] mask for valid (non-pad) text positions

        Returns:
            logits: list of n_codebooks tensors, each [B, T_audio, codebook_size]
            group_losses: list of 4 scalar tensors (one per group)
        """
        B, T_audio, n_cb = audio_codes.shape
        n_groups = self.config.n_groups
        cbg = self.config.codebooks_per_group

        # 1. Encode text through frozen backbone
        text_hidden = self.encode_text(text_tokens, attention_mask=text_attention_mask)  # [B, D_backbone]

        # 2. Project to local hidden size
        h = self.projection(text_hidden)  # [B, D_local]

        # 3. Add frame position embeddings
        positions = torch.arange(T_audio, device=h.device).unsqueeze(0)  # [1, T_audio]
        frame_h = h.unsqueeze(1) + self.frame_pos_embed(positions)  # [B, T_audio, D_local]

        # 4. Compute ground-truth group embeddings (for teacher forcing)
        group_embs = self._get_group_embeddings(audio_codes)  # list of [B, T_audio, D_local]

        # 5. Build local decoder input sequence per frame:
        #    [h_t] + group_embs[:-1] -> predict groups 0..n_groups-1
        #    Frame dimensions become batch: [B*T_audio, n_groups, D_local]
        input_parts = [frame_h.reshape(B * T_audio, 1, self.config.local_hidden_size)]
        for g_emb in group_embs[:-1]:
            input_parts.append(g_emb.reshape(B * T_audio, 1, self.config.local_hidden_size))
        decoder_input = torch.cat(input_parts, dim=1)
        # [B*T_audio, n_groups, D_local]

        # 6. Run local decoder (causal)
        group_position_ids = torch.arange(n_groups, device=decoder_input.device).unsqueeze(0)
        # [1, n_groups]
        decoder_output = self.local_decoder(decoder_input, position_ids=group_position_ids)
        # [B*T_audio, n_groups, D_local]

        # 7. Compute logits per codebook from corresponding group output
        logits = []
        for cb in range(n_cb):
            group_idx = cb // cbg
            cb_logits = self.output_heads[cb](decoder_output[:, group_idx, :])
            cb_logits = cb_logits.reshape(B, T_audio, self.config.codebook_size)
            logits.append(cb_logits)

        # 8. Compute per-group loss
        per_group_losses = []
        for g in range(n_groups):
            loss_g = 0.0
            for cb in range(g * cbg, (g + 1) * cbg):
                loss_g = loss_g + F.cross_entropy(
                    logits[cb].reshape(-1, self.config.codebook_size),
                    audio_codes[:, :, cb].reshape(-1),
                )
            per_group_losses.append(loss_g / cbg)

        return logits, per_group_losses
