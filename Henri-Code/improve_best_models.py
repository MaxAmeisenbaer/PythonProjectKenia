import pandas as pd

from common_functions import create_model
from data_preprocessing import create_final_ds

# uses Early Stopping
EPOCHS = 50


def optimize_model():
    # model_df = pd.read_excel("model_comparison.xlsx", engine='openpyxl')
    data = {"root_mean_squared_error": [], "mean_absolute_error": [], "r_square": [], "Nodes_lstm": [],
            "Nodes_dense": [], "learning_rate": [],
            "sequence_length": [], "batch_size": [], "dropout": [], "interval": []
            }

    metrics = ["root_mean_squared_error", "r_square", "mean_absolute_error"]

    df = pd.read_excel("best_model_configurations.xlsx")
    for ind, setting in df.iterrows():
        n_lstm = setting["Nodes_lstm"]
        n_dense = setting["Nodes_dense"]
        l_r = setting["learning_rate"]
        s_l = setting["sequence_length"]
        b_s = setting["batch_size"]
        dp = setting["dropout"]
        iv = setting["interval"]

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
            model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
            loss, metric_val = model.evaluate(test_ds)

            data[metric].append(metric_val)

    model_df = pd.DataFrame.from_dict(data)
    model_df.to_pickle("model_comparison_1.pkl")
    print(model_df.head())
    model_df.to_excel("model_comparison_1.xlsx")


if __name__ == "__main__":
    optimize_model()
