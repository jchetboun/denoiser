import keras
import torch


input = keras.Input(batch_shape=(1, 100, 256))
output = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True))(input)
output = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True))(output)
keras_lstm = keras.Model(inputs=input, outputs=output)
keras_lstm.build(input_shape=(1, 100, 256))

torch_lstm = torch.nn.LSTM(bidirectional=True, num_layers=2, hidden_size=256, input_size=256)
x = torch.rand((100, 1, 256))
with torch.no_grad():
    y, _ = torch_lstm(x)

torch_dict = torch_lstm.state_dict()
keras_lstm.layers[1].set_weights([torch_dict['weight_ih_l0'].numpy().transpose(),
                                  torch_dict['weight_hh_l0'].numpy().transpose(),
                                  torch_dict['bias_ih_l0'].numpy() +
                                  torch_dict['bias_hh_l0'].numpy(),
                                  torch_dict['weight_ih_l0_reverse'].numpy().transpose(),
                                  torch_dict['weight_hh_l0_reverse'].numpy().transpose(),
                                  torch_dict['bias_ih_l0_reverse'].numpy() +
                                  torch_dict['bias_hh_l0_reverse'].numpy()])
keras_lstm.layers[2].set_weights([torch_dict['weight_ih_l1'].numpy().transpose(),
                                  torch_dict['weight_hh_l1'].numpy().transpose(),
                                  torch_dict['bias_ih_l1'].numpy() +
                                  torch_dict['bias_hh_l1'].numpy(),
                                  torch_dict['weight_ih_l1_reverse'].numpy().transpose(),
                                  torch_dict['weight_hh_l1_reverse'].numpy().transpose(),
                                  torch_dict['bias_ih_l1_reverse'].numpy() +
                                  torch_dict['bias_hh_l1_reverse'].numpy()])
keras_y = keras_lstm.predict(x.numpy().transpose(1, 0, 2))

print('ok')
