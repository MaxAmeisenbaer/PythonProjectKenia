import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import numpy as np
import torch
import pickle
import pandas as pd

from log_transformation import inverse_log_transform


def calculate_all_metrics(model, test_loader, log_target="Kein log"):
    """
    Bewertet ein trainiertes Modell auf einem Test-Datensatz und berechnet verschiedene Regressionsmetriken.

    :param model:       Das trainierte PyTorch-Modell
    :param test_loader: DataLoader für den Testdatensatz
    :param log_target: Infos über potentielle logarithmisierung der Zielvariable
    :return: Dictionary mit MSE, RMSE, MAE, R2, NSE, MBE, KGE
    """
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = model(x_batch)
            y_true.extend(y_batch.numpy().flatten())
            y_pred.extend(preds.numpy().flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ── Rücktransformation aus Log-Raum ──
    if log_target=="log_eps":
        y_true, y_pred = inverse_log_transform(y_true, y_pred, method= "log_eps")
    elif log_target=="log1p":
        y_true, y_pred = inverse_log_transform(y_true, y_pred, method= "log1p")

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    lin_model = LinearRegression().fit(y_pred.reshape(-1, 1), y_true)
    y_reg = lin_model.predict(y_pred.reshape(-1, 1))
    ss_res = np.sum((y_true - y_reg) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Nash-Sutcliffe Efficiency
    sse = np.sum((y_true - y_pred) ** 2)
    var = np.sum((y_true - np.mean(y_true)) ** 2)
    nse = 1 - sse / (var + 1e-8)

    # Mean Bias Error
    mbe = np.mean(y_pred - y_true)

    # Kling-Gupta Efficiency
    r_corr = np.corrcoef(y_pred, y_true)[0, 1]
    alpha = np.std(y_pred) / (np.std(y_true) + 1e-8)
    beta = np.mean(y_pred) / (np.mean(y_true) + 1e-8)
    kge = 1 - np.sqrt((r_corr - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "NSE": nse,
        "MBE": mbe,
        "KGE": kge
    }

def save_split_boundaries(train_df, val_df, test_df, save_path):
    """
    Speichert Start- und Endzeitpunkte von Trainings-, Validierungs- und Testset.

    :param train_df: DataFrame mit Trainingsdaten
    :param val_df: Validierungsdaten
    :param test_df: Testdaten
    :param save_path: Pfad zur Zieldatei (CSV)
    """
    split_info = {
        "set": ["train", "val", "test"],
        "start": [train_df.index[0], val_df.index[0], test_df.index[0]],
        "end": [train_df.index[-1], val_df.index[-1], test_df.index[-1]]
    }
    pd.DataFrame(split_info).to_csv(save_path, index=False)
    print(f"Zeitbereiche der Splits gespeichert unter: {save_path}")


def evaluate_and_store_full_predictions(model, full_ds, output_dir,
                                        x_full, scaler_y, log_target= "Kein log", batch_size: int = 256):
    """
    Führt Vorhersage auf dem gesamten Datensatz durch und speichert:
    - predictions_full.npy
    - y_true_full.npy
    - dates_full.npy
    - X_full.npy
    - scaler_y.pkl

    :param model:      Das trainierte PyTorch-Modell
    :param full_ds:    TimeSeriesDatasetWithTimestamps (iterierbares Dataset)
    :param output_dir: Zielverzeichnis für die gespeicherten Dateien
    :param x_full:     Vollständige skalierte Eingabematrix
    :param scaler_y:   Scaler für die Zielvariable
    :param log_target: Infos über potentielle logarithmisierung der Zielvariable
    :param batch_size: Batch-Größe für den DataLoader (beeinflusst nur Speicher nicht Ergebnis)
    """
    model.eval()
    y_true = []
    y_pred = []
    timestamps_collected = []

    full_loader = torch.utils.data.DataLoader(full_ds, batch_size, shuffle=False)

    with torch.no_grad():
        for x_batch, y_batch, t_batch in full_loader:
            preds = model(x_batch)
            y_true.extend(y_batch.numpy().flatten())
            y_pred.extend(preds.numpy().flatten())
            timestamps_collected.extend(t_batch)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    timestamps_collected = np.array(timestamps_collected).reshape(-1)

    assert len(y_true) == len(timestamps_collected), "Länge von y_true und Zeitachse passt nicht!"

    os.makedirs(output_dir, exist_ok=True)


    # ── Log-Raum-Werte speichern (für Debugging) ──
    np.save(os.path.join(output_dir, "predictions_log.npy"), y_pred)
    np.save(os.path.join(output_dir, "y_true_log.npy"), y_true)

    # ── Rücktransformation ──
    if log_target == "log_eps":
        y_true_orig, y_pred_orig = inverse_log_transform(y_true,y_pred, method= "log_eps")
    elif log_target == "log1p":
        y_true_orig, y_pred_orig = inverse_log_transform(y_true,y_pred, method= "log1p")
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred

    # ── Originalskala-Werte speichern (für Plots und Metriken) ──
    np.save(os.path.join(output_dir, "predictions_full.npy"), y_pred_orig)
    np.save(os.path.join(output_dir, "y_true_full.npy"), y_true_orig)
    np.save(os.path.join(output_dir, "dates_full.npy"), timestamps_collected)
    np.save(os.path.join(output_dir, "X_full.npy"), x_full)

    with open(os.path.join(output_dir, "scaler_y.pkl"), "wb") as f:
        pickle.dump(scaler_y, f)