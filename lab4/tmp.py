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

def preprocess_data(x, y, limit):
    x = x.values.reshape(x.shape[0], 28 * 28, 1).astype("float32") / 255
    y = np.eye(10)[y].reshape(len(y), 10, 1).astype('float32')
    return x[:limit], y[:limit]

def predict(network, input_data):
    output = input_data
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss_function, loss_function_derivative, x_train, y_train, epochs=1000, learning_rate=0.01, verbose=True, filename='tmp'):
    for epoch in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # Forward pass
            output = predict(network, x)

            # Compute loss
            error += loss_function(y, output)

            # Backward pass
            gradient = loss_function_derivative(y, output)
            for layer in reversed(network):
                gradient = layer.backward(gradient, learning_rate)

        error /= len(x_train)
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Error: {error}")

# Load data
train_data = pd.read_csv('resources/mnist_train.csv')
test_data = pd.read_csv('resources/mnist_test.csv')

# Preprocess data
x_train, y_train = preprocess_data(train_data.drop('label', axis=1), train_data['label'], 1000)
x_test, y_test = preprocess_data(test_data.drop('label', axis=1), test_data['label'], 20)

# Neural network architecture
network = [
    LinearLayer(28 * 28, 40),
    Tanh(),
    LinearLayer(40, 10),
    Tanh()
]


def predict_testint(x_test, y_test, network):
    # Тестирование сети
    total_error = 0
    correct_predictions = 0
    true_labels = []
    predicted_labels = []
    for x, y in zip(x_test, y_test):
        output = predict(network, x)
        predicted_label = np.argmax(output)
        true_label = np.argmax(y)
        total_error += mean_squared_error(y, output)
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)
        if predicted_label == true_label:
            correct_predictions += 1
        print(f'Prediction: {predicted_label}\tTrue Label: {true_label}')

    accuracy = correct_predictions / len(x_test)
    average_error = total_error / len(x_test)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    print(f'Accuracy: {accuracy:.2%}')
    print(f'Average Error: {average_error:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)


# Train the network
train(network, mean_squared_error, mean_squared_error_derivative, x_train, y_train, epochs=1000, learning_rate=0.1)
predict_testint(x_test, y_test, network)
