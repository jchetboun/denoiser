import onnx
import torch
import torchaudio
import onnxruntime
import onnx_coreml
import coremltools
from scipy.io import wavfile
from onnxsim import simplify
from denoiser.enhance import write
from denoiser.utils import deserialize_model
from denoiser.resample import upsample2, downsample2
from serialization.pytorch_converter import convert
from serialization.utils import create_preprocess_dict, compress_and_save
from coremltools.models import MLModel
from coremltools.models.neural_network import flexible_shape_utils


# Function to change the CoreML input to flexible size
def update_input_range(model):
    spec = model.get_spec()
    input_size_ranges = flexible_shape_utils.NeuralNetworkMultiArrayShapeRange()
    input_size_ranges.add_channel_range((1, 1))
    input_size_ranges.add_height_range((1, 1))
    # Maximum duration is 2 min
    input_size_ranges.add_width_range((256, 1920000))
    flexible_shape_utils.update_multiarray_shape_range(spec, feature_name="input", shape_range=input_size_ranges)
    flexible_shape_utils.update_multiarray_shape_range(spec, feature_name="output", shape_range=input_size_ranges)
    result = coremltools.models.MLModel(spec)
    return result


# Load the PyTorch model
path = "./ckpt/best.th"
pkg = torch.load(path)
model = deserialize_model(pkg)
# model.resample = 1
model.eval()

# Audio for evaluation
audio_path = "../examples/audio_samples/p232_052_noisy.wav"
audio, sr = torchaudio.load(audio_path)
length = audio.shape[-1]
from torch.nn import functional as F
audio = F.pad(audio, (0, model.valid_length(length) - length))
audio = audio.unsqueeze(1)
half_audio = audio[:, :, :int(audio.shape[-1] / 2)]
# audio = upsample2(upsample2(audio))
# audio = upsample2(audio)

# Test of the PyTorch model
with torch.no_grad():
    enhanced = model(audio)[0]
# enhanced = downsample2(downsample2(enhanced))
# enhanced = downsample2(enhanced)
write(enhanced, "./ckpt/test/enhanced.wav", sr=16000)

# Conversion to ONNX
onnx_path = "./ckpt/best.onnx"
torch.onnx.export(model,
                  audio,
                  onnx_path,
                  input_names=["input"],
                  output_names=["output"],
                  opset_version=11)

# Simplification of the ONNX model
onnx_model = onnx.load(onnx_path)
model_simple, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx_path = "./ckpt/best_simple.onnx"
onnx.save(model_simple, onnx_path)

# Test of the ONNX model
ort_session = onnxruntime.InferenceSession(onnx_path)
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
ort_inputs = {ort_session.get_inputs()[0].name: audio.detach().cpu().numpy()}
ort_out = ort_session.run(None, ort_inputs)[0]
# ort_out = downsample2(downsample2(torch.tensor(ort_out))).numpy()
# ort_out = downsample2(torch.tensor(ort_out)).numpy()
wavfile.write("./ckpt/test/enhanced_with_onnx.wav", 16000, ort_out)

# Conversion to CoreML
# Convert from serialization is KO
# mlmodel = convert(onnx_path,
#                   image_input_names=("input"),
#                   minimum_ios_deployment_target='13')
# Convert from onnx_coreml is OK
mlmodel = onnx_coreml.convert(onnx.load_model(onnx_path), minimum_ios_deployment_target="13")
pd = create_preprocess_dict(model.valid_length(length),
                            "Fixed",
                            side_length=model.valid_length(length),
                            output_classes="irrelevant")
compress_and_save(mlmodel,
                  save_path="./ckpt",
                  model_name="best",
                  version=1.0,
                  ckpt_location=onnx_path,
                  preprocess_dict=pd,
                  model_description="Audio Denoising",
                  convert_to_float16=True)

# Input size update
model = MLModel("./ckpt/best.mlmodel")
model = update_input_range(model)
model.save("./ckpt/best.mlmodel")

# Test of the CoreML model
data = {"input": audio.detach().cpu().numpy()}
ml_output = model.predict(data)["output"]
# ml_output = downsample2(downsample2(torch.tensor(ml_output))).numpy()
# ml_output = downsample2(torch.tensor(ml_output)).numpy()
wavfile.write("./ckpt/test/enhanced_with_mlmodel.wav", 16000, ml_output)
# Test on half audio file to check flexible input size
data = {"input": half_audio.detach().cpu().numpy()}
ml_output = model.predict(data, useCPUOnly=True)["output"]
# ml_output = downsample2(downsample2(torch.tensor(ml_output))).numpy()
# ml_output = downsample2(torch.tensor(ml_output)).numpy()
wavfile.write("./ckpt/test/enhanced_with_mlmodel_half.wav", 16000, ml_output)

