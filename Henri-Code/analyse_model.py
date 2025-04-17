import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from common_functions import create_model
from data_preprocessing import create_final_ds, create_tf_dataset

"""
Check which features play the most important role in the model
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
    model.fit(train_ds, epochs=EPOCHS,
              validation_data=val_ds)

    model.save(model_filename)


def extract_feature_importance():
    model.evaluate(test_ds.take(1))
    model.load_weights(model_filename)

    feature_df = pd.DataFrame(columns=["Feature removed", "loss", "RMSE"])

    loss, metric = model.evaluate(train_ds)
    feature_df.loc[0] = ["Normal"] + [loss] + [metric]

    for ind, feature in enumerate(train_df.columns[1:]):
        # shuffle the feature
        n_df = train_df.copy()
        np.random.shuffle(n_df[feature])
        print(n_df.head())
        feature_train = n_df.drop(["SHA_nit"], axis=1)

        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        feature_scaler.fit(feature_train.to_numpy())
        feature_train_scaled = feature_scaler.transform(feature_train)
        target_train = np.array(n_df["SHA_nit"], ndmin=2).T

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


if __name__ == "__main__":
    # training is only needed once
    train_model()
    extract_feature_importance()
    calculate_important_features()
    show_important_features()


