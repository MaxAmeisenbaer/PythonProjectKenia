import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import tensorflow as tf
import pickle


def calculate_all_metrics(model, test_ds):
    """
    Bewertet ein trainiertes Modell auf einem Test-Datensatz und berechnet verschiedene Regressionsmetriken.

    Args:
        model (tf.keras.Model): Das trainierte Modell.
        test_ds (tf.data.Dataset): Der Test-Datensatz als tf.data.Dataset.

    Returns:
        dict: Ein Dictionary mit folgenden Metriken:
            - "MSE": Mean Squared Error
            - "RMSE": Root Mean Squared Error
            - "MAE": Mean Absolute Error
            - "R2": R²-Koeffizient
            - "NSE": Nash-Sutcliffe Efficiency
            - "MBE": Mean Bias Error
            - "KGE": Kling-Gupta Efficiency
    """
    y_true = []
    y_pred = []

    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0)
        y_true.extend(y_batch.numpy().flatten())
        y_pred.extend(preds.flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

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


def evaluate_and_store_full_predictions(model, full_ds, timestamps, output_dir,
                                        x_full, scaler_y):
    """
    Führt Vorhersage auf dem gesamten Datensatz durch und speichert:
    - predictions_full.npy
    - y_true_full.npy
    - dates_full.npy
    - X_full.npy
    - scaler_y.pkl
    """

    y_true = []
    y_pred = []
    timestamps_collected = []

    for x_batch, y_batch, t_batch in full_ds:
        preds = model.predict(x_batch, verbose=0)
        y_true.extend(y_batch.numpy().flatten())
        y_pred.extend(preds.flatten())
        timestamps_collected.extend(t_batch.numpy().flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    timestamps_collected = np.array(timestamps_collected).reshape(-1)

    assert len(y_true) == len(timestamps), "Länge von y_true und Zeitachse passt nicht!"

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "predictions_full.npy"), y_pred)
    np.save(os.path.join(output_dir, "y_true_full.npy"), y_true)
    np.save(os.path.join(output_dir, "dates_full.npy"), timestamps_collected)
    np.save(os.path.join(output_dir, "X_full.npy"), x_full)

    with open(os.path.join(output_dir, "scaler_y.pkl"), "wb") as f:
        pickle.dump(scaler_y, f)

