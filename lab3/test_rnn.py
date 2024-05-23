import numpy as np

from lab3.common import preprocess_data, train_rnn, save_weights, count_metrics, mse_grad
from lab3.rnn import SimpleRNN, RNNLayer, mse

train_data, valid_data, test_data = preprocess_data("resources/Steel_industry_data.csv")

epochs = 1
lr = 1e-5

layer_conf = [
    {"type": "input", "units": 6},
    {"type": "rnn", "hidden": 6, "output": 1}
]

rnn = SimpleRNN(layer_conf, "saved_weights/layers_weights.pkl")
# trained_rnn = train_rnn(rnn, *train_data, *valid_data, epochs, lr)

# save_weights(trained_rnn.layers, "layers_weights.pkl")


# Тест для RNN
def test_rnn(rnn_model, test_x, test_y):
    outputs=[]
    sequence_len = 7
    epoch_loss = 0
    for j in range(test_x.shape[0] - sequence_len):
        seq_x = test_x[j:(j + sequence_len), :]
        seq_y = test_y[j:(j + sequence_len), :]
        hiddens, output = rnn_model.forward(seq_x)
        outputs.append(output[-1][0])
        epoch_loss += mse(seq_y, output[-1])
    print("RNN:")
    predicted_labels_rounded = np.round(outputs).astype(int)
    count_metrics(test_y[:-7], predicted_labels_rounded)

test_rnn(rnn, test_data[0],test_data[1])

