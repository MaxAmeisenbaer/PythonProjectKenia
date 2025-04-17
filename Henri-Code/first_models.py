import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from common_functions import create_model
from data_preprocessing import create_final_ds, create_tf_dataset

"""
This was used to generate some first models and get a better feel for the data
"""

# uses Early Stopping
EPOCHS = 70
# MODEL CONFIG
nodes_lstm = 20
dropout = 0.1
learning_rate = 0.001
metric = "r_square"
batch_size = 32
seq_length = 1

model_filename = f"rmse_{nodes_lstm}_001_{seq_length}_{batch_size}_01.h5"


train_ds, val_ds, test_ds, train_df, test_df, val_df = create_final_ds(
    "SHA", ["SHA", "WSH"], "SHA_nit", batch_size=batch_size, seq_length=seq_length, interval="24h")

model, early_stopping = create_model(nodes_lstm, None, dropout, metric, learning_rate)


def train_model():
    history = model.fit(train_ds, epochs=EPOCHS,
                        validation_data=val_ds)

    model.save(model_filename)

    # list all data in history
    print(history.history.keys())
    # visualize history for accuracy
    plt.plot(history.history['r_square'])
    plt.plot(history.history['val_r_square'])
    plt.title('model MSE')
    plt.ylabel('MSE')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # visualize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def visualize_model():
    model.evaluate(train_ds.take(1))
    model.load_weights(model_filename)
    print(model.evaluate(test_ds))
    print(model.evaluate(train_ds))
    print(model.evaluate(val_ds))
    # visualize specific length of data
    """predictions = []
    labels = []
    for batch in test_ds.take(2000):
        prediction = model.predict(batch)

        prediction = list(prediction)
        for ind, p in enumerate(prediction):
            predictions.append(np.max(p))

        y = [arr.numpy() for arr in batch][1]
        labels.append(y[0])

    x = range(0, len(predictions))
    print(predictions)
    print(labels)"""

    for ds, df in ((train_ds, train_df), (val_ds, val_df), (test_ds, test_df)):
        # visualize
        predictions = model.predict(ds).flatten()
        labels = np.array(df["SHA_nit"])

        x = range(0, len(predictions))
        x1 = range(0, len(labels))

        print(predictions)
        print(labels)

        sns.set()
        fig, ax1 = plt.subplots()

        ax1.plot(x1, labels, color="blue", label="actual", linewidth=1)
        ax1.plot(x, predictions, color="orange", label="Predictions", linewidth=1.5)
        plt.title("Train Data")
        plt.legend()

        plt.show()


def visualize_whole_dataset_model():
    model.evaluate(train_ds.take(1))
    model.load_weights(model_filename)

    full_ds = train_ds.concatenate(val_ds)
    full_ds = full_ds.concatenate(test_ds)
    full_df = train_df.append(val_df)
    full_df = full_df.append(test_df)
    # visualize
    predictions = model.predict(full_ds).flatten()
    labels = np.array(full_df["SHA_nit"])

    x = range(0, len(predictions))
    x1 = range(0, len(labels))

    print(predictions)
    print(labels)

    sns.set()
    fig, ax1 = plt.subplots()

    ax1.plot(x1, labels, color="blue", label="actual", linewidth=1)
    ax1.plot(x, predictions, color="orange", label="Predictions", linewidth=1.5)
    plt.title("Train data | val data | test data")
    plt.legend()

    for i in range(3):
        plt.axvline(len(train_df) + i, color="r")
    for i in range(3):
        plt.axvline(len(train_df) + len(val_df) + i, color="r")

    plt.show()


def extract_feature_importance():
    model.evaluate(test_ds.take(1))
    model.load_weights("RMSE_interval=24h.h5")

    feature_df = pd.DataFrame(columns=["Feature removed", "loss", "RMSE"])

    loss, metric = model.evaluate(train_ds)
    feature_df.loc[0] = ["Normal"] + [loss] + [metric]

    for ind, feature in enumerate(train_df.columns[1:]):
        # shuffle the feature
        n_df = train_df.copy()
        np.random.shuffle(n_df[feature])

        feature_train = n_df.drop(["nit"], axis=1)

        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        feature_scaler.fit(feature_train.to_numpy())
        feature_train_scaled = feature_scaler.transform(feature_train)
        target_train = np.array(n_df["nit"], ndmin=2).T

        feature_dataset = create_tf_dataset(feature_train_scaled, target_train, batch_size=32, seq_length=8)

        loss, metric = model.evaluate(feature_dataset)
        feature_df.loc[ind + 1] = [feature] + [loss] + [metric]

    feature_df.to_pickle("features.pkl")
    print(feature_df)


def calculate_important_features():
    feature_df = pd.read_pickle("features.pkl")

    norm = feature_df["RMSE"][0]
    differences = [norm]
    divided = [norm]

    for rmse in feature_df["RMSE"][1:]:
        differences.append(rmse - norm)
        divided.append(rmse / norm)

    feature_df["Difference"] = differences
    feature_df["Divided"] = divided
    feature_df.sort_values("Difference", inplace=True, ignore_index=True, ascending=False)
    feature_df.to_pickle("feature_importance.pkl")


def show_important_features():
    feature_df = pd.read_pickle("feature_importance.pkl")
    print(feature_df)


train_model()
extract_feature_importance()
calculate_important_features()
show_important_features()