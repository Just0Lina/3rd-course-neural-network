import numpy as np

from lab3.common import preprocess_data, train_gru, save_weights, mse, count_metrics
from lab3.gru import GRULayer

train_data, valid_data, test_data  = preprocess_data("resources/Steel_industry_data.csv")

# Step 2: Initialize your GRU model
layer_conf = [
    {"type": "input", "units": 6},
    {"type": "rnn", "hidden": 6, "output": 3}
]

lr = 1e-5
num_epochs=10

gru_model = GRULayer(layer_conf, "saved_weights/gru_layers_weights.pkl")
# trained_gru = train_gru(gru_model, *train_data, *valid_data, num_epochs, lr)
# save_weights(trained_gru, "gru_layers_weights.pkl")

def test_gru(gru_model, test_x, test_y):
    outputs=[]
    sequence_len = 7
    epoch_loss = 0
    test_y2 =np.eye(3)[test_y]

    for j in range(test_x.shape[0] - sequence_len):
        seq_x = test_x[j:(j + sequence_len), :]
        seq_y = test_y2[j:(j + sequence_len), :]
        h_prev = np.zeros((gru_model.hidden_units,))
        output = gru_model.forward(seq_x, h_prev)
        outputs.append(output[-1][0][0][0])
        epoch_loss += mse(seq_y, output[-1])

    print("GRU:")
    count_metrics(test_y[:-7], [np.argmax(i[-1]) for i in outputs])

test_gru(gru_model, test_data[0],test_data[1])