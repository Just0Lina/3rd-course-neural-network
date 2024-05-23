import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error


def preprocess_data(x, y):
    x = x.values.reshape(x.shape[0], 28 * 28, 1).astype("float32") / 255
    y = np.eye(10)[y].reshape(len(y), 10, 1).astype('float32')
    return x, y

def predict(network, input_data):
    output = input_data
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss_function, loss_function_derivative, x_train, y_train, epochs=1000, learning_rate=0.01, verbose=True):
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


