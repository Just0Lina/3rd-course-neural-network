import numpy as np


class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_hidden_layers = len(hidden_sizes)
        self.weights = []
        # Инициализация весов
        layers = [input_size] + hidden_sizes + [output_size]
        for i in np.arange(0, len(layers) - 1):
            w = np.random.randn(layers[i], layers[i + 1])
            self.weights.append(w / np.sqrt(layers[i]))

    def save_weights(self, filename):
        np.savez(filename, *self.weights)

    def load_weights(self, filename):
        loaded_data = np.load(filename)
        self.weights = [loaded_data[key] for key in loaded_data.keys()]

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, x):
        self.activations = [np.atleast_2d(x)]
        for layer in np.arange(0, len(self.weights) - 1):
            net = self.activations[layer].dot(self.weights[layer])
            out = self.sigmoid(net)
            self.activations.append(out)
        net = self.activations[-1].dot(self.weights[-1])
        self.activations.append(net)
        return self.activations[-1]

    def backward_propagation(self, x, y, learning_rate):
        deltas = [None] * (self.num_hidden_layers + 1)
        deltas[-1] = (y - self.activations[-1])
        for i in range(self.num_hidden_layers, 0, -1):
            deltas[i - 1] = np.dot(deltas[i], self.weights[i].T) * self.sigmoid_derivative(self.activations[i])

        # Обновление весов и смещений
        for i in range(self.num_hidden_layers + 1):
            self.weights[i] += learning_rate * np.dot(self.activations[i].T, deltas[i])

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            total_loss = 0
            for (x, target) in zip(X, y):
                output = self.forward_propagation(x)
                self.backward_propagation(x, target, learning_rate)
                total_loss += np.mean(np.abs(target - output))
            if epoch % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(X)}')

    def predict(self, X):
        return self.forward_propagation(X)

    def calculate_loss(self, x, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(x, add_bias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)
        return loss
