from typing import Optional

import torch


class AudioCodec:
    """Official MOSS-Audio-Tokenizer wrapper.

    The public MOSS model uses [Q, B, T] audio codes. This wrapper exposes one
    internal contract to the rest of MiniMoss: [B, Q, T].
    """

    def __init__(
        self,
        model_name: str = "OpenMOSS-Team/MOSS-Audio-Tokenizer",
        n_quantizers: int = 32,
    ):
        self.model_name = model_name
        self.n_quantizers = n_quantizers
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_moss_tokenizer()
            if hasattr(self._model, "eval"):
                self._model.eval()
            if hasattr(self._model, "parameters"):
                for p in self._model.parameters():
                    p.requires_grad = False
        return self._model

    @property
    def sample_rate(self) -> int:
        return int(getattr(self.model, "sampling_rate", getattr(self.model, "sample_rate", 24000)))

    def _load_moss_tokenizer(self):
        from transformers import AutoModel

        return AutoModel.from_pretrained(self.model_name, trust_remote_code=True)

    @staticmethod
    def _unwrap(obj, *keys):
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, dict):
            for key in keys:
                if key in obj:
                    return obj[key]
        for key in keys:
            if hasattr(obj, key):
                return getattr(obj, key)
        raise TypeError(f"Tokenizer output does not contain any of {keys}")

    @torch.inference_mode()
    def encode(self, wav: torch.Tensor, n_quantizers: Optional[int] = None) -> torch.Tensor:
        """Encode waveform to RVQ codes [B, n_quantizers, T_frames]."""
        nq = n_quantizers if n_quantizers is not None else self.n_quantizers
        model = self.model
        if hasattr(model, "to"):
            model = model.to(wav.device)

        if wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)
        elif wav.dim() == 2:
            wav = wav.unsqueeze(1)

        encoded = model.encode(wav, num_quantizers=nq, return_dict=True)
        codes_qbt = self._unwrap(encoded, "audio_codes")
        if codes_qbt.ndim != 3 or codes_qbt.shape[0] < nq:
            raise RuntimeError(
                f"MOSS tokenizer returned shape {tuple(codes_qbt.shape)}; expected [Q, B, T] with Q >= {nq}"
            )
        return codes_qbt[:nq].permute(1, 0, 2).contiguous().long()

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode RVQ codes to waveform [B, 1, T_samples]."""
        model = self.model
        if hasattr(model, "to"):
            model = model.to(codes.device)
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        if codes.dim() != 3:
            raise ValueError(f"codes must have shape [B, Q, T] or [Q, T], got {tuple(codes.shape)}")

        codes_qbt = codes.permute(1, 0, 2).contiguous().long()
        decoded = model.decode(
            codes_qbt,
            num_quantizers=codes_qbt.shape[0],
            return_dict=True,
        )
        audio = self._unwrap(decoded, "audio")
        if audio.ndim != 3:
            raise RuntimeError(f"Tokenizer decoded shape {tuple(audio.shape)}; expected [B, C, samples]")
        return audio
