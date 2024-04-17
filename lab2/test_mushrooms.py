import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from metrics import count_metrics, custom_roc_curve, sklearn_roc_curve
from perceptron import MLP



def preprocess_agaricus():
    label_encoder = LabelEncoder()
    df = pd.read_csv('resources/agaricus-lepiota.csv')
    for label in df.columns:
        most_common_value = df[label].mode()[0]
        df[label] = df[label].replace('?', most_common_value)


    for label in df.columns:
        df[label + '_code'] = label_encoder.fit_transform(df[label])
        df = df.drop(label, axis=1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    normalized_data = pd.DataFrame(scaled_data, columns=df.columns)

    column_names = ['gill-color_code', 'spore-print-color_code', 'population_code', 'gill-size_code']
    selected_columns = normalized_data[column_names]
    incoming_params = selected_columns.to_numpy()


    target = normalized_data['poisonous_code'].to_numpy()
    target_array = target.reshape(-1, 1)
    return incoming_params, target_array

def test_mush():
    true_positive = 0
    true_negative = 0
    false_negative = 0
    false_positive = 0
    mlp = MLP(input_size=3, hidden_sizes=[3], output_size=1)
    mlp.load_weights("saved_weights/weights_mushrooms.npz")
    total_error = 0
    preds = []
    for (x, target) in zip(X_test, y_test):
        pred = mlp.predict(x)[0][0]
        preds.append(pred)
        step = 1 if pred > 0.7 else 0
        # print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(x, target[0], pred, step))
        total_error += abs(step - target[0])
        if step == target == 1:
            true_positive += 1
        if step == target == 0:
            true_negative += 1
        if target == 0 and step != target:
            false_positive += 1
        if target == 1 and step != target:
            false_negative += 1
    print("Total error:", total_error / X_test.size)
    count_metrics(true_positive, true_negative, false_positive, false_negative)
    custom_roc_curve(preds, y_test)
    sklearn_roc_curve(preds, y_test)



def train_mush():
    mlp = MLP(input_size=4, hidden_sizes=[3,4], output_size=1)
    X_train, X_test, y_train, y_test = train_test_split(incoming_params, target_array, test_size=0.2, random_state=42)
    mlp.load_weights("weights_mushrooms.npz")
    print("Тренировка началась!")
    mlp.train(X_train, y_train, epochs=1000, learning_rate=0.1)
    print("Weights were saved to weights_mushrooms")
    mlp.save_weights("weights_mushrooms")



incoming_params, target_array = preprocess_agaricus()
X_train, X_test, y_train, y_test = train_test_split(incoming_params, target_array, test_size=0.2, random_state=42)
# train_mush()
test_mush()


