import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import math

from lab3.common import load_weights


class RNNLayer:
    def __init__(self, input_units, hidden_units, output_units, is_not_inited):
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units
        if is_not_inited:
            self.initialize_weights()


    def initialize_weights(self):
        k = 1 / math.sqrt(self.hidden_units)
        self.i_weight = np.random.rand(self.input_units, self.hidden_units) * 2 * k - k
        self.h_weight = np.random.rand(self.hidden_units, self.hidden_units) * 2 * k - k
        self.h_bias = np.random.rand(1, self.hidden_units) * 2 * k - k
        self.o_weight = np.random.rand(self.hidden_units, self.output_units) * 2 * k - k
        self.o_bias = np.random.rand(1, self.output_units) * 2 * k - k

    def forward(self, x):
        hidden = np.zeros((x.shape[0], self.hidden_units))
        output = np.zeros((x.shape[0], self.output_units))
        for j in range(x.shape[0]):
            input_x = x[j, :][np.newaxis, :] @ self.i_weight
            hidden_x = input_x + hidden[max(j - 1, 0), :][np.newaxis, :] @ self.h_weight + self.h_bias
            hidden_x = np.tanh(hidden_x)
            hidden[j, :] = hidden_x
            output_x = hidden_x @ self.o_weight + self.o_bias
            output[j, :] = output_x
        return hidden, output

    def backward(self, x, grad, hiddens, lr):
        next_h_grad = None
        i_weight_grad, h_weight_grad, h_bias_grad, o_weight_grad, o_bias_grad = [0] * 5
        for j in range(x.shape[0] - 1, -1, -1):
            out_grad = grad[j, :][np.newaxis, :]
            o_weight_grad += hiddens[j][:, np.newaxis] @ out_grad
            o_bias_grad += out_grad
            h_grad = out_grad @ self.o_weight.T
            if j < x.shape[0] - 1:
                hh_grad = next_h_grad @ self.h_weight.T
                h_grad += hh_grad
            tanh_deriv = 1 - hiddens[j][np.newaxis, :] ** 2
            h_grad = np.multiply(h_grad, tanh_deriv)
            next_h_grad = h_grad.copy()
            if j > 0:
                h_weight_grad += hiddens[j - 1][:, np.newaxis] @ h_grad
                h_bias_grad += h_grad
            i_weight_grad += x[j, :][:, np.newaxis] @ h_grad

        lr = lr / x.shape[0]
        self.i_weight -= i_weight_grad * lr
        self.h_weight -= h_weight_grad * lr
        self.h_bias -= h_bias_grad * lr
        self.o_weight -= o_weight_grad * lr
        self.o_bias -= o_bias_grad * lr


class SimpleRNN:
    def __init__(self, layer_conf, weights_filename=None):
        self.layers = []
        if weights_filename:
            weights = load_weights(weights_filename)
        for i in range(1, len(layer_conf)):
            if weights_filename:
                layer_weights = weights[i-1]  # Assuming weights are stored in order for each layer
                layer = RNNLayer(layer_conf[i-1]["units"], layer_conf[i]["hidden"], layer_conf[i]["output"], 0)
                layer.i_weight = layer_weights.i_weight
                layer.h_weight = layer_weights.h_weight
                layer.h_bias = layer_weights.h_bias
                layer.o_weight = layer_weights.o_weight
                layer.o_bias = layer_weights.o_bias
            else:
                layer = RNNLayer(layer_conf[i-1]["units"], layer_conf[i]["hidden"], layer_conf[i]["output"],1)
            self.layers.append(layer)

    def forward(self, x):
        hiddens = []
        outputs = []
        for layer in self.layers:
            hidden, output = layer.forward(x)
            hiddens.append(hidden)
            outputs.append(output)
            x = hidden
        return hiddens, outputs

    def backward(self, x, grad, hiddens, lr):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hiddens_layer = hiddens[i]
            grad = layer.backward(x, grad, hiddens_layer, lr)
            x = hiddens_layer

    def predict_rnn(model, x):
        _, outputs = model.forward(x)
        return outputs[-1]

def mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)