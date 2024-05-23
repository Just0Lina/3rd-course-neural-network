import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


def mse(actual, predicted):
    # print(actual, predicted, np.mean((actual - predicted) ** 2))
    return np.mean((actual - predicted) ** 2)


def mse_grad(actual, predicted):
    return predicted - actual

def load_weights(filename):
    with open(filename, 'rb') as f:
        weights = pickle.load(f)
    return weights

def save_weights(weights, filename):
    with open(filename, 'wb') as f:
        pickle.dump(weights, f)

def preprocess_data(filename):
    data = pd.read_csv(filename)
    data = data.ffill()
    label_encoder = LabelEncoder()

    for label in data.columns:
        data[label + '_code'] = label_encoder.fit_transform(data[label])
        data = data.drop(label, axis=1)

    PREDICTORS = ["NSM_code", "Usage_kWh_code", "date_code", "Leading_Current_Power_Factor_code", "Day_of_week_code",
                  "Leading_Current_Reactive_Power_kVarh_code"]
    TARGET = "Load_Type_code"

    scaler = StandardScaler()
    data[PREDICTORS] = scaler.fit_transform(data[PREDICTORS])

    np.random.seed(0)
    split_data = np.split(data, [int(.7 * len(data)), int(.85 * len(data))])
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = [[d[PREDICTORS].to_numpy(), d[[TARGET]].to_numpy()] for d in split_data]

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def train_rnn(rnn, train_x, train_y, valid_x, valid_y, epochs, lr):
    for epoch in range(epochs):
        sequence_len = 7
        epoch_loss = 0
        for j in range(train_x.shape[0] - sequence_len):
            seq_x = train_x[j:(j + sequence_len), :]
            seq_y = train_y[j:(j + sequence_len), :]
            hiddens, outputs = rnn.forward(seq_x)
            grad = mse_grad(seq_y, outputs[-1])
            rnn.backward(seq_x, grad, hiddens, lr)
            epoch_loss += mse(seq_y, outputs[-1])

        if epoch % 2 == 0:
            sequence_len = 7
            valid_loss = 0
            for j in range(valid_x.shape[0] - sequence_len):
                seq_x = valid_x[j:(j + sequence_len), :]
                seq_y = valid_y[j:(j + sequence_len), :]
                _, outputs = rnn.forward(seq_x)
                valid_loss += mse(seq_y, outputs[-1])

            print(f"Epoch: {epoch} train loss {epoch_loss / len(train_x)} valid loss {valid_loss / len(valid_x)}")

    return rnn



def train_gru(gru, train_x, train_y, valid_x, valid_y, epochs, lr):
    train_y = np.eye(3)[train_y]
    valid_y =np.eye(3)[valid_y]
    for epoch in range(epochs):
        sequence_len = 7
        epoch_loss = 0
        # for j in range(train_x.shape[0] - sequence_len):
        #     seq_x = train_x[j:(j + sequence_len), :]
        #     seq_y = train_y[j:(j + sequence_len), :]
        #     # Прямой проход через GRU слой
        #     h_prev = np.zeros((gru.hidden_units,))
        #     hiddens, h_tildes, zs, rs, outputs = gru.forward(seq_x, h_prev)
        #     # Вычисление градиента ошибки
        #     grad = (mse_grad(seq_y, outputs[-1])).reshape(7, 3)
        #     # Обратный проход и обновление параметров
        #     gru.backward(seq_x, grad, hiddens, h_tildes, zs, rs, lr)
        #     # Вычисление функции потерь
        #     epoch_loss += mse(seq_y, outputs[-1])

        if epoch % 2 == 0:
            sequence_len = 7
            valid_loss = 0
            for j in range(valid_x.shape[0] - sequence_len):
                seq_x = valid_x[j:(j + sequence_len), :]
                seq_y = valid_y[j:(j + sequence_len), :]
                h_prev = np.zeros((gru.hidden_units,))
                outputs = gru.forward(seq_x, h_prev)
                valid_loss += mse(seq_y, outputs[-1])

            print(f"Epoch: {epoch} train loss {epoch_loss / len(train_x)} valid loss {valid_loss / len(valid_x)}")

    return gru


def train_lstm(lstm, train_x, train_y, valid_x, valid_y, epochs, lr):
    train_y = np.eye(3)[train_y]
    valid_y = np.eye(3)[valid_y]
    for epoch in range(epochs):
        sequence_len = 7
        epoch_loss = 0
        c_prev = np.zeros((lstm.hidden_units,))
        for j in range(train_x.shape[0] - sequence_len):
            seq_x = train_x[j:(j + sequence_len), :]
            seq_y = train_y[j:(j + sequence_len), :]
            # Forward pass through LSTM layer
            h_prev = np.zeros((lstm.hidden_units,))
            hiddens, c_prev, outputs, fs, is_, cs_, os = lstm.forward(seq_x, h_prev, c_prev)
            # Compute error gradient
            grad = (mse_grad(train_x[j+1:(j + sequence_len+1), :], hiddens))
            # Backward pass and parameter update
            lstm.backward(seq_x, grad, hiddens, c_prev, fs, is_, cs_, os, lr)
            # Compute loss
            epoch_loss += mse(seq_y, outputs[-1])

        if epoch % 2 == 0:
            sequence_len = 7
            valid_loss = 0
            c_prev = np.zeros((lstm.hidden_units,))
            for j in range(valid_x.shape[0] - sequence_len):
                seq_x = valid_x[j:(j + sequence_len), :]
                seq_y = valid_y[j:(j + sequence_len), :]
                h_prev = np.zeros((lstm.hidden_units,))
                hiddens, c_prev, outputs, fs, is_, cs_, os  = lstm.forward(seq_x, h_prev, c_prev)
                valid_loss += mse(seq_y, outputs[-1])

            print(f"Epoch: {epoch} train loss {epoch_loss / len(train_x)} valid loss {valid_loss / len(valid_x)}")

    return lstm


# Функция подсчета метрик
def count_metrics(true, predicted):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(true, predicted)
    r2 = cust_r2_score(true, predicted)
    rmse = RMSE(true,predicted)

    print("Accuracy:", accuracy)
    print("R2:", r2)
    print("Rmse:", rmse)



def cust_r2_score(true, pred):
    num = np.nansum(((true-pred)**2), axis=0, dtype=np.float64)
    denum = np.nansum(((true-np.nanmean(true,axis=0))**2), axis=0, dtype=np.float64)
    return np.mean(1-num/denum)


def RMSE(true, pred):
    mse = np.mean((true-pred)**2)
    rmse_val = np.sqrt(mse)
    return rmse_val
