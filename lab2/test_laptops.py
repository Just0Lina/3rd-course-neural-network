import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from metrics import count_metrics
from perceptron import MLP

def preprocess_data():
    label_encoder = LabelEncoder()
    df = pd.read_csv('resources/data.csv')

    df['Storage_Capacity' + '_code'] = label_encoder.fit_transform(df['Storage_Capacity'])
    df = df.drop('Storage_Capacity',axis=1)

    column_names = ['Storage_Capacity_code', 'Processor_Speed', 'Weight', 'Screen_Size', 'RAM_Size','Price']
    selected_columns = df[column_names]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(selected_columns)
    normalized_data = pd.DataFrame(scaled_data, columns=selected_columns.columns)

    target = normalized_data['Price']
    target = pd.qcut(target, q=5, labels=['Very low', 'Low', 'Medium', 'High', 'Very high'])
    target = label_encoder.fit_transform(target)
    target_array = np.array([[x] for x in target])

    normalized_data = normalized_data.drop('Price',axis=1)
    incoming_params = normalized_data.to_numpy()

    return incoming_params, target_array

def test_data():
    mlp = MLP(input_size=5, hidden_sizes=[20,20,20], output_size=5)
    mlp.load_weights("saved_weights/new_new_weight_data_laptop.npz")
    total_error = 0

    predicted = []  # Список для предсказанных значений
    true = []  # Список для истинных значений

    for (x, target) in zip(X_test, y_test):
        prediction = mlp.predict(x)
        true.append(target)
        ans = np.argmax(prediction)
        predicted.append(ans)
        total_error += 0 if ans == target[0] else 1
        # print("[INFO] data={}, ground-truth={}, pred={:.4f}".format(x, target[0], ans))
    print("Total error:", total_error / X_test.size)
    count_metrics(true, predicted)



def train_data():
    mlp = MLP(input_size=5, hidden_sizes=[20,20,20], output_size=5)
    # mlp.load_weights("new_new_weight_data_laptop.npz")
    for j in range(20):
        target = []
        print("Текущая: ", j)
        for i in target_array:
            array = [0] * 5
            array[int(i)] = 1
            target.append(array)
        X_train, X_test, y_train, y_test = train_test_split(incoming_params, target, test_size=0.2, random_state=42)
        mlp.train(X_train, y_train, epochs=200, learning_rate=0.1)
    mlp.save_weights("new_new_weight_data_laptop")


incoming_params, target_array = preprocess_data()
X_train, X_test, y_train, y_test = train_test_split(incoming_params, target_array, test_size=0.2, random_state=42)

# train_data()
test_data()
