import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


class Activation:
    def __init__(self, activation_function, activation_function_derivative):
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

    def forward(self, input_data):
        self.input_data = input_data
        return self.activation_function(self.input_data)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.activation_function_derivative(self.input_data)

class Tanh(Activation):
    def __init__(self):
        super().__init__(np.tanh, lambda x: 1 - np.tanh(x) ** 2)

class LinearLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input_data):
        self.input_data = input_data
        return np.dot(self.weights, self.input_data) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input_data.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient

def mean_squared_error(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mean_squared_error_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

