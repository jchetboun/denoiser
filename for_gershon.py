import os
import torchaudio
import numpy as np
from coremltools.models import MLModel
from torch.nn import functional as F
from scipy.io import wavfile


def eval_with_overlap(ml_model, model_length, audio, length, overlap):
    index = 0
    audio = audio.mean(dim=0, keepdim=True)
    output = np.zeros_like(audio.detach().cpu().numpy())
    audio_length = audio.shape[-1]
    while index < audio.shape[-1]:
        chunk = audio[..., index:min(index + length, audio_length)]
        chunk_length = chunk.shape[-1]
        chunk = F.pad(chunk, (0, model_length - chunk_length))
        chunk = chunk.unsqueeze(1)
        std = chunk.std(dim=-1, keepdim=True)
        chunk = chunk / (1.e-3 + std)
        data = {"input": chunk.detach().cpu().numpy()}
        chunk_output = ml_model.predict(data)["output"][0] * std.detach().cpu().numpy()
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


video = "./denoisingInput.mov"
model = MLModel("./best_48_3_12_from_keras.mlmodel")
input_length = model.get_spec().description.input[0].type.multiArrayType.shape[-1]

# Audio extraction and conversion to 16 KHz with FFmpeg

audio_path = video[:-3] + "_extracted_audio.wav"
cmd = "ffmpeg -i " + video + " -ar 16000 " + audio_path
os.system(cmd)

# Reading the audio and denoising

all_audio, sr = torchaudio.load(audio_path)
all_output = eval_with_overlap(model, input_length, all_audio, sr, 0)
os.remove(audio_path)
wavfile.write("./output.wav", sr, all_output[0])




