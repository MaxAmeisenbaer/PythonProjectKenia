import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common_functions import create_model
from data_preprocessing import create_final_ds, create_tf_dataset

"""
Choose a model config to visualize its predictions.
You can find good configurations in "final_best_models_only_SHA_data.xlsx".
"""

# MODEL CONFIG
EPOCHS = 70

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
    plt.plot(history.history[f'{metric}'])
    plt.plot(history.history[f'val_{metric}'])
    plt.title(f'model {metric}')
    plt.ylabel(f'{metric}')
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

        ax1.plot(x1, labels, color="blue", label="Tatsächliche Werte", linewidth=1)
        ax1.plot(x, predictions, color="orange", label="Vorhergesagte Werte", linewidth=1.5)
        plt.title("Nitratkonzentration")
        plt.legend()

        plt.show()


def visualize_whole_dataset_model():
    model.evaluate(train_ds.take(1))
    model.load_weights(model_filename)

    full_ds = train_ds.concatenate(val_ds)
    full_ds = full_ds.concatenate(test_ds)
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # visualize
    predictions = model.predict(full_ds).flatten()
    labels = np.array(full_df["SHA_nit"])

    x = range(0, len(predictions))
    x1 = range(0, len(labels))

    sns.set()
    fig, ax1 = plt.subplots()

    ax1.plot(x1, labels, color="blue", label="Tatsächliche Werte", linewidth=1)
    ax1.plot(x, predictions, color="orange", label="Vorhergesagte Werte", linewidth=1.5)
    plt.title("Nitratkonzentration", fontsize=20)  # Schriftgröße des Titels ändern
    plt.xlabel("Zeit", fontsize=18)  # X-Achsenbeschriftung hinzufügen und Schriftgröße ändern
    plt.ylabel("Nitratkonzentration", fontsize=18)  # Y-Achsenbeschriftung hinzufügen und Schriftgröße ändern
    plt.legend(fontsize=16)

    ax1.tick_params(axis='x', labelsize=14)  # Schriftgröße der Daten auf der X-Achse ändern
    ax1.tick_params(axis='y', labelsize=14)

    for i in range(3):
        plt.axvline(len(train_df) + i, color="r")
    for i in range(3):
        plt.axvline(len(train_df) + len(val_df) + i, color="r")

    plt.show()


if __name__ == '__main__':
    visualize_whole_dataset_model()