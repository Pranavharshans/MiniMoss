import torch
import torch.nn as nn
from typing import Optional


class AudioCodec:
    """Wrapper for audio codec (encode + decode). Uses DAC by default."""

    def __init__(self, model_name: str = "descript/dac_24khz", n_quantizers: int = 16):
        self.model_name = model_name
        self.n_quantizers = n_quantizers
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import dac
            self._model = dac.DAC.from_pretrained(self.model_name)
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad = False
        return self._model

    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate

    @staticmethod
    def _unwrap(obj, key):
        """Handle both dict-style and attribute-style returns from DAC."""
        if isinstance(obj, dict):
            return obj[key]
        return getattr(obj, key)

    @torch.inference_mode()
    def encode(self, wav: torch.Tensor, n_quantizers: Optional[int] = None) -> torch.Tensor:
        """Encode waveform to RVQ codes.

        Args:
            wav: [B, 1, T_samples] or [B, T_samples] or [1, T_samples] waveform
            n_quantizers: number of quantizers to use (default: self.n_quantizers)

        Returns:
            codes: [B, n_quantizers, T_frames]
        """
        nq = n_quantizers if n_quantizers is not None else self.n_quantizers
        model = self.model.to(wav.device)

        if wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)
        elif wav.dim() == 2:
            wav = wav.unsqueeze(1)

        enc = model.encode(wav, n_quantizers=nq)
        return self._unwrap(enc, "codes")  # [B, nq, T_frames]

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode RVQ codes to waveform.

        Args:
            codes: [n_quantizers, T_frames] or [B, n_quantizers, T_frames]

        Returns:
            wav: [1, T_samples]
        """
        model = self.model.to(codes.device)
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        dec = model.decode(codes)
        return self._unwrap(dec, "audio")  # [B, 1, T_samples]
