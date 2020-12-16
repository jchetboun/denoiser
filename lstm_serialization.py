import torch
import onnx
# import onnx_coreml
from onnxsim import simplify
import coremltools as ct

m = torch.nn.LSTM(bidirectional=True, num_layers=2, hidden_size=256, input_size=256, batch_first=False)
torch.onnx.export(m, torch.zeros((100, 1, 256)), "lstm.onnx", opset_version=11)
onnx_model = onnx.load("lstm.onnx")
model_simple, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simple, "lstm.onnx")
mlmodel = ct.converters.onnx.convert(model='lstm.onnx')
mlmodel.save("lstm.mlmodel")
# onnx_model = onnx.load("lstm.onnx")
# mlmodel = onnx_coreml.convert(onnx_model, minimum_ios_deployment_target="12")
