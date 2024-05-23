import numpy as np

from lab3.common import preprocess_data, train_lstm, save_weights, mse, count_metrics
from lab3.lstm import LSTMLayer

train_data, valid_data, test_data = preprocess_data("resources/Steel_industry_data.csv")

# Step 2: Initialize your GRU model
layer_conf = [
    {"type": "input", "units": 6},
    {"type": "rnn", "hidden": 6, "output": 3}
]

lr = 1e-5
num_epochs=10

lstm_model = LSTMLayer(layer_conf, "saved_weights/lstm_layers_weights.pkl")


#
# train_lstm = train_lstm(lstm_model, *train_data, *valid_data, num_epochs, lr)
# save_weights(train_lstm, "lstm_layers_weights.pkl")


def test_lstm(lstm_model, test_x, test_y):
    outputs=[]
    sequence_len = 7
    epoch_loss = 0
    test_y2 =np.eye(3)[test_y]
    c_prev = np.zeros((lstm_model.hidden_units,))

    for j in range(test_x.shape[0] - sequence_len):
        seq_x = test_x[j:(j + sequence_len), :]
        seq_y = test_y2[j:(j + sequence_len), :]
        h_prev = np.zeros((lstm_model.hidden_units,))
        hiddens, c_prev, output, fs, is_, cs_, os = lstm_model.forward(seq_x, h_prev, c_prev)

        outputs.append(output[-1][0][0])
        epoch_loss += mse(seq_y, output[-1])

    print("LSTM:")
    count_metrics(test_y[:-7], [np.argmax(i[-1]) for i in outputs])

test_lstm(lstm_model, test_data[0], test_data[1])
