from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import tensorflow as tf


def evaluate_model(model, test_ds):
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
