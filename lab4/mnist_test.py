import numpy as np
import pandas as pd

from lab4.Activations import Sigmoid, LinearLayer, binary_cross_entropy, binary_cross_entropy_prime
from lab4.cnn import ConvolutionalLayer, Reshape
from lab4.predictions import train, predict_testint


def preprocess_data(x, y):
    # Reshape input data x into 4D array with shape (number_of_samples, 1, 28, 28)
    x = x.values.reshape(-1, 1, 28, 28).astype("float32") / 255
    # y = np.eye(10)[y].astype('float32')  # Assuming y is one-hot encoded
    y = np.eye(10)[y].reshape(len(y), 10, 1).astype('float32')

    return x[:1000], y[:1000]


# Load data
train_data = pd.read_csv('resources/mnist_train.csv')
test_data = pd.read_csv('resources/mnist_test.csv')

# Preprocess data
x_train, y_train = preprocess_data(train_data.drop('label', axis=1), train_data['label'])
x_test, y_test = preprocess_data(test_data.drop('label', axis=1), test_data['label'])


# neural network
network = [
    ConvolutionalLayer((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    LinearLayer(5 * 26 * 26, 100),
    Sigmoid(),
    LinearLayer(100, 10),
    Sigmoid()
]

train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=10,
    learning_rate=0.1
)
predict_testint(x_test, y_test, network)

# # test
# for x, y in zip(x_test, y_test):
#     output = predict(network, x)
#     print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
