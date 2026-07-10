from types import SimpleNamespace

import torch

from minimoss.codec import AudioCodec


class FakeMossTokenizer(torch.nn.Module):
    sampling_rate = 24000

    def __init__(self):
        super().__init__()
        self.last_decode_codes = None

    def encode(self, wav, num_quantizers, return_dict):
        assert return_dict is True
        batch = wav.shape[0]
        codes = torch.arange(num_quantizers * batch * 3).reshape(num_quantizers, batch, 3)
        return SimpleNamespace(audio_codes=codes)

    def decode(self, audio_codes, num_quantizers, return_dict):
        assert return_dict is True
        assert num_quantizers == audio_codes.shape[0]
        self.last_decode_codes = audio_codes
        return SimpleNamespace(audio=torch.zeros(audio_codes.shape[1], 1, 24))


def test_moss_codec_normalizes_code_layout():
    codec = AudioCodec(n_quantizers=4)
    codec._model = FakeMossTokenizer()

    codes = codec.encode(torch.zeros(2, 1, 100))
    assert codes.shape == (2, 4, 3)
    assert codec.sample_rate == 24000

    audio = codec.decode(codes)
    assert audio.shape == (2, 1, 24)
    assert codec.model.last_decode_codes.shape == (4, 2, 3)
