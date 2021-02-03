import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from denoiser.enhance import write
from denoiser.utils import deserialize_model
from torch.nn import functional as F

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def eval_with_overlap(model, audio, length, overlap):
    model.eval()
    index = 0
    output = torch.zeros_like(audio)
    audio_length = audio.shape[-1]
    while index < audio.shape[-1]:
        chunk = audio[..., index:min(index + length, audio_length)]
        chunk_length = chunk.shape[-1]
        chunk = F.pad(chunk, (0, length - chunk_length))
        with torch.no_grad():
            chunk_output = model(chunk)
        output_beg = index + overlap // 2
        output_end = index + length - overlap // 2
        chunk_beg = overlap // 2
        chunk_end = length - overlap // 2
        if index == 0:
            output_beg = 0
            chunk_beg = 0
        if index + length > audio_length:
            output_end = audio_length
            chunk_end = audio_length - index
        output[..., output_beg:output_end] = chunk_output[..., chunk_beg:chunk_end]
        index = index + length - overlap
    return output


def compute_snr(x, y):
    # SNR between x (enhanced) and y (ground truth).
    ref = np.power(y, 2)
    if len(x) == len(y):
        diff = np.power(x - y, 2)
    else:
        stop = min(len(x), len(y))
        diff = np.power(x[:stop] - y[:stop], 2)
    ratio = np.sum(ref) / np.sum(diff)
    value = 10 * np.log10(ratio)
    return value


# Load the PyTorch model
path = "./ckpt/best_48_3_12.th"
pkg = torch.load(path)
model = deserialize_model(pkg)
model.eval()

# Audio for evaluation
audio_paths = ["../examples/audio_samples/p232_052_noisy.wav",
               "../examples/audio_samples/p257_054_noisy.wav",
               "../examples/audio_samples/p257_143_noisy.wav",
               "../examples/from_matt/Sequence_01.wav",
               "../examples/from_matt/vlog_noisy.wav",
               "../examples/from_videos/16000/ftv1_noisy_16000.wav",
               "../examples/from_videos/16000/ftv2_noisy_16000.wav",
               "../examples/from_videos/16000/ftv3_noisy_16000.wav"]

max_overlap = 8200
step_overlap = 200
all_overlaps = np.arange(0, max_overlap, step_overlap)
all_metrics = np.zeros((len(audio_paths), len(all_overlaps)))
for i, audio_path in enumerate(audio_paths):
    print(audio_path)
    audio, sr = torchaudio.load(audio_path)
    length = audio.shape[-1]
    audio = F.pad(audio, (0, model.valid_length(length) - length))
    length = audio.shape[-1]
    audio = audio.unsqueeze(1)

    # Test of the PyTorch model
    with torch.no_grad():
        enhanced = model(audio)[0]
    write(enhanced, "./tmp/" + os.path.splitext(os.path.basename(audio_path))[0] + "_enhanced.wav")

    for j, overlap in enumerate(all_overlaps):
        enhanced_with_overlap = eval_with_overlap(model, audio, 16040, overlap)[0]
        write(enhanced_with_overlap, "./tmp/" + os.path.splitext(os.path.basename(audio_path))[0] + "_enhanced_overlap_" + str(overlap) + ".wav")
        snr = compute_snr(enhanced_with_overlap.numpy(), enhanced.numpy())
        all_metrics[i, j] = snr

    plt.plot(all_overlaps, all_metrics[i, :])

plt.xlabel("Overlap")
plt.ylabel("SNR (dB)")
plt.savefig("./tmp/all_snr_48_3_12.png")
plt.clf()

plt.plot(all_overlaps, np.mean(all_metrics, axis=0))
plt.xlabel("Overlap")
plt.ylabel("Mean SNR (dB)")
plt.savefig("./tmp/mean_snr_48_3_12.png")
