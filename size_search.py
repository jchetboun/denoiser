import os
import onnx
import torch
import itertools
# import onnx_coreml
from onnxsim import simplify
from denoiser.demucs import Demucs
from serialization.utils import create_preprocess_dict, compress_and_save


def my_func(**args):

    f = open("./tmp/Log.txt", "a")

    n_hidden = args['n_hidden']
    n_depth = args['n_depth']
    n_kernel = args['n_kernel']
    n_causal = args['n_causal']
    n_lstm = args['n_lstm']

    if n_lstm is False and n_causal is False:
        return

    name = "model_h=" + str(n_hidden) + "_l=" + str(n_depth) + "_k=" + str(n_kernel)
    if n_lstm:
        if n_causal:
            name = name + "_with_lstm"
        else:
            name = name + "_with_bidirectional_lstm"
    else:
        name = name + "_no_lstm"

    model = Demucs(hidden=n_hidden,
                   depth=n_depth,
                   kernel_size=n_kernel,
                   causal=n_causal,
                   resample=1,
                   use_lstm=n_lstm)

    s = 0
    for p in model.parameters():
        s += p.numel()
    size_param = s

    model.eval()
    length = 16128
    audio = torch.rand((1, 1, length))
    torch.onnx.export(model, audio, "./tmp/" + name + ".onnx", opset_version=11)

    onnx_model = onnx.load("./tmp/" + name + ".onnx")
    model_simple, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simple, "./tmp/" + name + ".onnx")

    # mlmodel = onnx_coreml.convert(onnx.load_model("./tmp/" + name + ".onnx"), minimum_ios_deployment_target="13")
    import coremltools as ct
    mlmodel = ct.converters.onnx.convert(model="./tmp/" + name + ".onnx")
    pd = create_preprocess_dict(length, "Fixed", side_length=length, output_classes="irrelevant")
    compress_and_save(mlmodel,
                      save_path="./tmp",
                      model_name=name,
                      version=1.0,
                      ckpt_location="./tmp/" + name + ".onnx",
                      preprocess_dict=pd,
                      model_description="Audio Denoising",
                      convert_to_float16=True)

    size_mb = os.path.getsize("./tmp/" + name + ".mlmodel") // 10000 / 100

    if n_lstm:
        f.write(
        str(n_hidden) + "\t\t" + str(n_depth) + "\t\t" + str(n_kernel) + "\t\t" + str(n_lstm) + "\t" + str(n_causal) + "\t" + str(size_param) + "\t\t" + str(size_mb) + "\n")
    else:
        f.write(
        str(n_hidden) + "\t\t" + str(n_depth) + "\t\t" + str(n_kernel) + "\t\t" + str(n_lstm) + "\t-\t\t" + str(size_param) + "\t\t" + str(size_mb) + "\n")
    f.close()


os.remove("./tmp/Log.txt")
f = open("./tmp/Log.txt", "x")
f.write("Hidden\tDepth\tKernel\tLSTM\tCausal\tParameters\tSize\n")
f.close()

params = {
    'n_hidden': [64],
    'n_depth': [5],
    'n_kernel': [8],
    'n_lstm': [True],
    'n_causal': [True]
}

keys = list(params)
for values in itertools.product(*map(params.get, keys)):
    my_func(**dict(zip(keys, values)))

