import keras
import torch
import torchaudio
import coremltools
import numpy as np
from scipy.io import wavfile
from torch.nn import functional as F
from denoiser.utils import deserialize_model


def encoder_block(x, hidden, kernel_size, stride, use_glu):
    x = keras.layers.Conv1D(hidden, kernel_size, strides=stride, activation=keras.activations.relu)(x)
    if use_glu:
        # Split
        a = keras.layers.Conv1D(hidden, 1)(x)
        b = keras.layers.Conv1D(hidden, 1, activation=keras.activations.sigmoid)(x)
        x = keras.layers.Multiply()([a, b])
    else:
        x = keras.layers.Conv1D(hidden, 1, activation=keras.activations.relu)(x)
    return x


def decoder_block(x, hidden, use_glu, chout, kernel_size, stride, index):
    if use_glu:
        # Split
        a = keras.layers.Conv1D(hidden, 1)(x)
        b = keras.layers.Conv1D(hidden, 1, activation=keras.activations.sigmoid)(x)
        x = keras.layers.Multiply()([a, b])
    else:
        x = keras.layers.Conv1D(hidden, 1, activation=keras.activations.relu)(x)
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
                 use_glu=True):

    skips = []
    for index in range(depth):
        x = encoder_block(x, hidden, kernel_size, stride, use_glu)
        skips.append(x)
        hidden = growth * hidden
    hidden = int(hidden / growth)
    if causal:
        x = keras.layers.LSTM(hidden, return_sequences=True)(x)
        x = keras.layers.LSTM(hidden, return_sequences=True)(x)
    else:
        x = keras.layers.Bidirectional(keras.layers.LSTM(hidden, return_sequences=True))(x)
        x = keras.layers.Bidirectional(keras.layers.LSTM(hidden, return_sequences=True))(x)
        x = keras.layers.Dense(hidden)(x)
    for index in reversed(range(depth)):
        hidden = int(hidden / growth)
        skip = skips.pop(-1)
        x = keras.layers.Add()([x, skip])
        x = decoder_block(x, hidden * growth, use_glu, hidden, kernel_size, stride, index)
    return x


def transfer_weights(pytorch_model, keras_model, depth=4, use_glu=True, causal=False):
    pytorch_dict = pytorch_model.state_dict()
    # Encoder
    for index in range(depth):
        if use_glu:
            key = 'encoder.' + str(index) + '.0.'
            keras_model.layers[4 * index + 1].set_weights([pytorch_dict[key + 'weight'].numpy().transpose(2, 1, 0),
                                                           pytorch_dict[key + 'bias'].numpy()])
            key = 'encoder.' + str(index) + '.2.'
            w1, w2 = np.split(pytorch_dict[key + 'weight'].numpy().transpose(2, 1, 0), 2, axis=-1)
            b1, b2 = np.split(pytorch_dict[key + 'bias'].numpy(), 2, axis=-1)
            keras_model.layers[4 * index + 2].set_weights([w1, b1])
            keras_model.layers[4 * index + 3].set_weights([w2, b2])
        else:
            key = 'encoder.' + str(index) + '.0.'
            keras_model.layers[2 * index + 1].set_weights([pytorch_dict[key + 'weight'].numpy().transpose(2, 1, 0),
                                                           pytorch_dict[key + 'bias'].numpy()])
            key = 'encoder.' + str(index) + '.2.'
            keras_model.layers[2 * index + 2].set_weights([pytorch_dict[key + 'weight'].numpy().transpose(2, 1, 0),
                                                           pytorch_dict[key + 'bias'].numpy()])
    # LSTM
    index = 4 * depth + 1 if use_glu else 2 * depth + 1
    if causal:
        keras_model.layers[index].set_weights([pytorch_dict['lstm.lstm.weight_ih_l0'].numpy().transpose(),
                                               pytorch_dict['lstm.lstm.weight_hh_l0'].numpy().transpose(),
                                               pytorch_dict['lstm.lstm.bias_ih_l0'].numpy() +
                                               pytorch_dict['lstm.lstm.bias_hh_l0'].numpy()])
        keras_model.layers[index + 1].set_weights([pytorch_dict['lstm.lstm.weight_ih_l1'].numpy().transpose(),
                                                   pytorch_dict['lstm.lstm.weight_hh_l1'].numpy().transpose(),
                                                   pytorch_dict['lstm.lstm.bias_ih_l1'].numpy() +
                                                   pytorch_dict['lstm.lstm.bias_hh_l1'].numpy()])
    else:
        keras_model.layers[index].set_weights([pytorch_dict['lstm.lstm.weight_ih_l0'].numpy().transpose(),
                                               pytorch_dict['lstm.lstm.weight_hh_l0'].numpy().transpose(),
                                               pytorch_dict['lstm.lstm.bias_ih_l0'].numpy() +
                                               pytorch_dict['lstm.lstm.bias_hh_l0'].numpy(),
                                               pytorch_dict['lstm.lstm.weight_ih_l0_reverse'].numpy().transpose(),
                                               pytorch_dict['lstm.lstm.weight_hh_l0_reverse'].numpy().transpose(),
                                               pytorch_dict['lstm.lstm.bias_ih_l0_reverse'].numpy() +
                                               pytorch_dict['lstm.lstm.bias_hh_l0_reverse'].numpy()])
        keras_model.layers[index + 1].set_weights([pytorch_dict['lstm.lstm.weight_ih_l1'].numpy().transpose(),
                                                   pytorch_dict['lstm.lstm.weight_hh_l1'].numpy().transpose(),
                                                   pytorch_dict['lstm.lstm.bias_ih_l1'].numpy() +
                                                   pytorch_dict['lstm.lstm.bias_hh_l1'].numpy(),
                                                   pytorch_dict['lstm.lstm.weight_ih_l1_reverse'].numpy().transpose(),
                                                   pytorch_dict['lstm.lstm.weight_hh_l1_reverse'].numpy().transpose(),
                                                   pytorch_dict['lstm.lstm.bias_ih_l1_reverse'].numpy() +
                                                   pytorch_dict['lstm.lstm.bias_hh_l1_reverse'].numpy()])
        keras_model.layers[index + 2].set_weights([pytorch_dict['lstm.linear.weight'].numpy().transpose(),
                                                   pytorch_dict['lstm.linear.bias'].numpy()])
    # Decoder
    last_index = index + 3 if causal else index + 4
    for index in range(depth):
        if use_glu:
            key = 'decoder.' + str(index) + '.0.'
            w1, w2 = np.split(pytorch_dict[key + 'weight'].numpy().transpose(2, 1, 0), 2, axis=-1)
            b1, b2 = np.split(pytorch_dict[key + 'bias'].numpy(), 2, axis=-1)
            keras_model.layers[7 * index + last_index].set_weights([w1, b1])
            keras_model.layers[7 * index + last_index + 1].set_weights([w2, b2])
            key = 'decoder.' + str(index) + '.2.'
            keras_model.layers[7 * index + last_index + 4].set_weights([
                np.expand_dims(pytorch_dict[key + 'weight'].numpy().transpose(2, 1, 0), axis=1),
                pytorch_dict[key + 'bias'].numpy()])
        else:
            key = 'decoder.' + str(index) + '.0.'
            keras_model.layers[5 * index + last_index].set_weights([pytorch_dict[key + 'weight'].numpy().transpose(2, 1, 0), pytorch_dict[key + 'bias'].numpy()])
            key = 'decoder.' + str(index) + '.2.'
            keras_model.layers[5 * index + last_index + 2].set_weights([
                np.expand_dims(pytorch_dict[key + 'weight'].numpy().transpose(2, 1, 0), axis=1),
                pytorch_dict[key + 'bias'].numpy()])

# Load the PyTorch model
path = "./ckpt/best.th"
pkg = torch.load(path)
model_pt = deserialize_model(pkg)
model_pt.eval()

# Audio for evaluation
audio_path = "../examples/audio_samples/p232_052_noisy.wav"
audio, sr = torchaudio.load(audio_path)
length = audio.shape[-1]
audio = F.pad(audio, (0, model_pt.valid_length(length) - length))
audio = audio[:, :2560]
length = audio.shape[-1]
audio = audio.unsqueeze(1)

# Build the Keras model
input = keras.Input(batch_shape=(1, length, 1))
output = demucs_keras(input)
model_ke = keras.Model(inputs=input, outputs=output)
model_ke.build(input_shape=(1, length, 1))
model_ke.summary()
transfer_weights(model_pt, model_ke)
enhanced = model_ke.predict(audio.numpy().transpose(0, 2, 1))
wavfile.write('test.wav', sr, enhanced[0, :, 0])
coreml_model = coremltools.converters.keras.convert(
    model_ke,
    input_names=['input'],
    output_names=["output"],
)
coreml_model.save("test.mlmodel")
# data = {"input": audio.detach().cpu().numpy()}
# ml_output = coreml_model.predict(data)["output"]
# wavfile.write('test_ml.wav', sr, enhanced[0, :, 0])
