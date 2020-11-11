import torch
import torchaudio
from denoiser.utils import deserialize_model
from denoiser.enhance import write

path = "./ckpt/dummy.th"
pkg = torch.load(path)
model = deserialize_model(pkg)
model.eval()

audio_path = "../examples/audio_samples/p232_052_noisy.wav"
audio, sr = torchaudio.load(audio_path)

with torch.no_grad():
    enhanced = model(audio)
write(enhanced, "./ckpt/test/enhanced.wav", sr=sr)
