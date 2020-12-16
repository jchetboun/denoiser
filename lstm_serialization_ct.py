import torch
import coremltools

m = torch.nn.LSTM(bidirectional=True, num_layers=1, hidden_size=256, input_size=256, batch_first=False)
m.eval()
x = torch.rand(100, 1, 256)
tm = torch.jit.trace(m, x)
ml_model = coremltools.convert(tm, inputs=[coremltools.TensorType(name="input", shape=x.shape)], minimum_deployment_target=coremltools.target.iOS13)
