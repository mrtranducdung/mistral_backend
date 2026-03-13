from kokoro_onnx import Kokoro
from misaki import ja
import soundfile as sf
import torch

kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
jf_alpha = torch.load("jf_alpha.pt", weights_only=True).numpy()

# Phonemize trước
g2p = ja.JAG2P()
phonemes, _ = g2p("こんにちは、私はゆき先生です。")

print("Phonemes:", phonemes)

samples, sample_rate = kokoro.create(
    phonemes,
    voice=jf_alpha,
    speed=1.0,
    lang="ja",
    is_phonemes=True  # báo đây là phonemes rồi
)

sf.write("test.wav", samples, sample_rate)
print("Done!")