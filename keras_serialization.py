import torch
import torchaudio
import tensorflow as tf
import keras
from torch.nn import functional as F
from denoiser.utils import deserialize_model
import coremltools


@tf.function
def glu(x, dim=-1):
    a, b = tf.split(x, 2, axis=dim)
    return a * tf.math.sigmoid(b)


def encoder_block(x, hidden, kernel_size, stride, ch_scale, activation):
    x = keras.layers.Conv1D(hidden, kernel_size, strides=stride, activation=keras.activations.relu)(x)
    x = keras.layers.Conv1D(hidden * ch_scale, 1, activation=activation)(x)
    return x


def decoder_block(x, ch_scale, hidden, activation, chout, kernel_size, stride, index):
    x = keras.layers.Conv1D(ch_scale * hidden, 1, activation=activation)(x)
    x = keras.layers.Reshape((int(x.shape[1]), 1, int(x.shape[2])))(x)
    if index > 0:
        x = keras.layers.Conv2DTranspose(chout, (kernel_size, 1), strides=(stride, 1),
                                         activation=keras.activations.relu)(x)
    else:
        x = keras.layers.Conv2DTranspose(1, (kernel_size, 1), strides=(stride, 1))(x)
    x = keras.layers.Reshape((int(x.shape[1]), int(x.shape[3])))(x)
    return x


def demucs_keras(x,
                 hidden=32,
                 depth=4,
                 kernel_size=4,
                 stride=4,
                 causal=False,
                 growth=2,
                 use_glu=False):

    skips = []
    activation = glu if use_glu else keras.activations.relu
    ch_scale = 2 if use_glu else 1
    for index in range(depth):
        x = encoder_block(x, hidden, kernel_size, stride, ch_scale, activation)
        skips.append(x)
        hidden = growth * hidden
    hidden = int(hidden / growth)
    if causal:
        x = keras.layers.LSTM(hidden, return_sequences=True)(x)
    else:
        x = keras.layers.Bidirectional(keras.layers.LSTM(hidden, return_sequences=True))(x)
        x = keras.layers.Dense(hidden)(x)
    for index in reversed(range(depth)):
        hidden = int(hidden / growth)
        skip = skips.pop(-1)
        x = keras.layers.Add()([x, skip])
        x = decoder_block(x, ch_scale, hidden * growth, activation, hidden, kernel_size, stride, index)
    return x


# Load the PyTorch model
path = "./ckpt/best.th"
pkg = torch.load(path)
model = deserialize_model(pkg)
model.eval()

# Audio for evaluation
# audio_path = "../examples/audio_samples/p232_052_noisy.wav"
# audio, sr = torchaudio.load(audio_path)
# length = audio.shape[-1]
# audio = F.pad(audio, (0, model.valid_length(length) - length))
# audio = audio.unsqueeze(1)

# Run the PyTorch model
# x = torch.rand((1, 1, 25600))
# with torch.no_grad():
#     enhanced_pytorch = model(x)[0]

# Build the Keras model
input = keras.Input(batch_shape=(1, 2560, 1))
output = demucs_keras(input)
model = keras.Model(inputs=input, outputs=output)
model.build(input_shape=(1, 2560, 1))
model.summary()
coreml_model = coremltools.converters.keras.convert(
    model,
    input_names=['input'],
    output_names=["output"],
)
coreml_model.save("test.mlmodel")

