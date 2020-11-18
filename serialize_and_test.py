import os
import onnx
import torch
import torchaudio
import onnxruntime
import onnx_coreml
from scipy.io import wavfile
from onnxsim import simplify
from denoiser.enhance import write
from denoiser.utils import deserialize_model
from denoiser.resample import upsample2, downsample2
from serialization.pytorch_converter import convert
from serialization.utils import create_preprocess_dict, compress_and_save

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

path = "./ckpt/dummy.th"
pkg = torch.load(path)
model = deserialize_model(pkg)
model.resample = 1
model.eval()

audio_path = "../examples/audio_samples/p232_052_noisy.wav"
audio, sr = torchaudio.load(audio_path)
length = audio.shape[-1]
from torch.nn import functional as F
audio = F.pad(audio, (0, model.valid_length(length) - length))
audio = audio.unsqueeze(1)
audio = upsample2(upsample2(audio))

with torch.no_grad():
    enhanced = model(audio)[0]
enhanced = downsample2(downsample2(enhanced))
write(enhanced, "./ckpt/test/enhanced.wav", sr=16000)

onnx_path = "./ckpt/dummy.onnx"
torch.onnx.export(model,
                  audio,
                  onnx_path,
                  input_names=["input"],
                  output_names=["output"],
                  opset_version=11)

onnx_model = onnx.load(onnx_path)
model_simple, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx_path = "./ckpt/dummy_simple.onnx"
onnx.save(model_simple, onnx_path)

ort_session = onnxruntime.InferenceSession(onnx_path)
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
ort_inputs = {ort_session.get_inputs()[0].name: audio.detach().cpu().numpy()}
ort_out = ort_session.run(None, ort_inputs)[0]
wavfile.write("./ckpt/test/enhanced_with_onnx.wav", 16000, ort_out)

# Convert from serialization is KO
# mlmodel = convert(onnx_path,
#                   image_input_names=("input"),
#                   minimum_ios_deployment_target='12')

# Convert from onnx_coreml is OK
mlmodel = onnx_coreml.convert(onnx.load_model(onnx_path), minimum_ios_deployment_target="13")
pd = create_preprocess_dict(model.valid_length(length),
                            "Fixed",
                            side_length=model.valid_length(length),
                            output_classes="irrelevant")
compress_and_save(mlmodel,
                  save_path="./ckpt",
                  model_name="dummy",
                  version=1.0,
                  ckpt_location=onnx_path,
                  preprocess_dict=pd,
                  model_description="Audio Denoising",
                  convert_to_float16=False)
