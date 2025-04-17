import pandas as pd

from keras.callbacks import Callback

from common_functions import create_model
from data_preprocessing import create_final_ds

"""
Try out different model configurations by looping over different predefined parameters to find the best one.
"""

# uses Early Stopping
EPOCHS = 70


class PrintTrainingOnTextEvery10EpochsCallback(Callback):
    def __init__(self, metric):
        super().__init__()
        self.metric = metric

    def on_epoch_end(self, epoch, logs=None):
        if (int(epoch) % 20) == 0:
            print(f"Epoch: {epoch} | Loss: {logs['loss']} | {self.metric}: {logs[self.metric]}")


def optimize_model():
    # model_df = pd.read_excel("model_comparison.xlsx", engine='openpyxl')
    data = {"root_mean_squared_error": [], "mean_absolute_error": [], "r_square": [], "Nodes_lstm": [],
            "Nodes_dense": [], "learning_rate": [],
            "sequence_length": [], "batch_size": [], "dropout": [], "interval": []
            }

    nodes_lstm = [80]
    nodes_dense = [80]
    learning_rate = [0.01, 0.001, 0.0001]
    sequence_length = [1, 2, 3, 4, 5, 10, 20]
    batch_size = [32, 64, 128, 256]
    dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
    interval = ["24h"]
    metrics = ["r_square", "root_mean_squared_error", "mean_absolute_error", ]

    parameters = list(data.keys())
    count = 8

    for n_lstm, n_dense in zip(nodes_lstm, nodes_dense):

        for l_r in learning_rate:
            for s_l in sequence_length:
                for b_s in batch_size:
                    for dp in dropout:
                        for iv in interval:
                            data["Nodes_lstm"].append(n_lstm)
                            data["Nodes_dense"].append(n_dense)
                            data["learning_rate"].append(l_r)
                            data["sequence_length"].append(s_l)
                            data["batch_size"].append(b_s)
                            data["dropout"].append(dp)
                            data["interval"].append(iv)

                            for metric in metrics:
                                train_ds, val_ds, test_ds, train_df, test_df, val_df = create_final_ds(
                                    "SHA", ["SHA", "WSH"], "SHA_nit", batch_size=b_s, seq_length=s_l, interval=iv)

                                model, early_stopping = create_model(n_lstm, n_dense, dp, metric, l_r)
                                print(f"Training model: {n_lstm, n_dense, l_r, s_l, b_s, dp, iv, metric}")
                                model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, verbose=0,
                                          callbacks=[PrintTrainingOnTextEvery10EpochsCallback(metric)],
                                          )
                                loss, metric_val = model.evaluate(test_ds)

                                data[metric].append(metric_val)

        model_df = pd.DataFrame.from_dict(data)
        model_df.to_pickle(f"SHa-nit-prediction-model_{count}.pkl")
        count += 1

    model_df = pd.DataFrame.from_dict(data)
    model_df.to_pickle("SHa-nit-prediction_full.pkl")
    print(model_df.head())
    model_df.to_excel("SHa-nit-prediction_full.xlsx")


if __name__ == '__main__':
    optimize_model()
