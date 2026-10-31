import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import numpy as np
import torch
import pickle
import pandas as pd

from log_transformation import inverse_log_transform


def _nanmean(values: list[float]) -> float:
    """Mittelwert über alle endlichen Werte; NaN, wenn keiner vorhanden ist."""
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def calculate_all_metrics(model, test_loader, target_features, target_transformations):
    """
    Bewertet ein trainiertes Modell auf einem Test-Datensatz und berechnet verschiedene Regressionsmetriken.

    Nicht beobachtete Zielwerte (Target-Maske == 0 bzw. fehlend/imputiert) werden
    bei der Metrikberechnung ausgeschlossen.

    :param model:       Das trainierte PyTorch-Modell
    :param test_loader: DataLoader für den Testdatensatz
    :param target_features: Liste der Spaltennamen der Target-Features (z.B. ["SHA_nit", "SHA_temp"])
    :param target_transformations: Dict mit den Methoden pro Target-Features
    :return: Dictionary mit gemittelten MSE, RMSE, MAE, R2, NSE, MBE, KGE über alle Target-Features
    """
    model.eval()
    y_true_list = []
    y_pred_list = []
    y_mask_list = []

    with torch.no_grad():
        for x_batch, y_batch, y_mask_batch in test_loader:
            preds = model(x_batch)
            y_true_list.append(y_batch.cpu().numpy())
            y_pred_list.append(preds.cpu().numpy())
            y_mask_list.append(y_mask_batch.cpu().numpy())

    y_true_all = np.concatenate(y_true_list, axis=0)
    y_pred_all = np.concatenate(y_pred_list, axis=0)
    y_mask_all = np.concatenate(y_mask_list, axis=0)

    # Leere Listen in die die Werte über alle Target-Features reingeschrieben werden
    mse_vals, rmse_vals, mae_vals, r2_vals, nse_vals, mbe_vals, kge_vals = [], [], [], [], [], [], []

    for i, col_name in enumerate(target_features):
        # Nur tatsächlich beobachtete Werte bewerten (0 = fehlend/imputiert)
        valid = y_mask_all[:, i] > 0
        yt_col = y_true_all[valid, i]
        yp_col = y_pred_all[valid, i]

        if yt_col.size == 0:
            print(f"  Warnung: {col_name} hat keine beobachteten Testwerte - Metriken = NaN")
            for values in (mse_vals, rmse_vals, mae_vals, r2_vals, nse_vals, mbe_vals, kge_vals):
                values.append(np.nan)
            continue

        # ── Rücktransformation aus Log-Raum (nur beobachtete Werte) ──
        trans_info = target_transformations.get(col_name, {"type": "none", "params": None})
        epsilon = trans_info["params"].get("epsilon", 1e-6) if trans_info["params"] else 1e-6
        yt, yp = inverse_log_transform(
            yt_col,
            yp_col,
            method=trans_info,
            epsilon=epsilon
        )

        # Durch die Rücktransformation können ungültige Werte entstehen (z.B. Box-Cox)
        finite = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[finite]
        yp = yp[finite]

        if yt.size == 0:
            print(f"  Warnung: {col_name} hat nach Rücktransformation keine gültigen Testwerte - Metriken = NaN")
            for values in (mse_vals, rmse_vals, mae_vals, r2_vals, nse_vals, mbe_vals, kge_vals):
                values.append(np.nan)
            continue

        # Mean Error Maße
        mse = mean_squared_error(yt, yp)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(yt, yp)

        # R-Square
        ss_tot = np.sum((yt - np.mean(yt)) ** 2) if yt.size >= 2 else 0.0
        if yt.size >= 2 and ss_tot > 1e-12:
            lin_model = LinearRegression().fit(yp.reshape(-1, 1), yt)
            y_reg = lin_model.predict(yp.reshape(-1, 1))
            ss_res = np.sum((yt - y_reg) ** 2)
            r2 = 1 - (ss_res / ss_tot)
        else:
            r2 = np.nan

        # Nash-Sutcliffe Efficiency
        if yt.size >= 2 and ss_tot > 1e-12:
            sse = np.sum((yt - yp) ** 2)
            nse = 1 - (sse / ss_tot)
        else:
            nse = np.nan

        # Mean Bias Error
        mbe = np.mean(yp - yt)

        # Kling-Gupta Efficiency
        if yt.size >= 2:
            std_yt = np.std(yt)
            std_yp = np.std(yp)
            mean_yt = np.mean(yt)
            if std_yt > 1e-12 and std_yp > 1e-12 and abs(mean_yt) > 1e-12:
                r_corr = np.corrcoef(yp, yt)[0, 1]
                if np.isfinite(r_corr):
                    alpha = std_yp / std_yt
                    beta = np.mean(yp) / mean_yt
                    kge = 1 - np.sqrt((r_corr - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
                else:
                    kge = np.nan
            else:
                kge = np.nan
        else:
            kge = np.nan

        # Anhängen und zurück in die Schleife
        mse_vals.append(mse)
        rmse_vals.append(rmse)
        mae_vals.append(mae)
        r2_vals.append(r2)
        nse_vals.append(nse)
        mbe_vals.append(mbe)
        kge_vals.append(kge)

    return {
        "MSE": _nanmean(mse_vals),
        "RMSE": _nanmean(rmse_vals),
        "MAE": _nanmean(mae_vals),
        "R2": _nanmean(r2_vals),
        "NSE": _nanmean(nse_vals),
        "MBE": _nanmean(mbe_vals),
        "KGE": _nanmean(kge_vals)
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
                                        x_full, scaler_y, target_features,
                                        target_transformations, batch_size: int = 256):
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
    :param target_features: Liste der Zielvariablen
    :param target_transformations: Infors über Normalisierung der Target-Features
    :param batch_size: Batch-Größe für den DataLoader (beeinflusst nur Speicher nicht Ergebnis)
    """
    model.eval()
    y_true_list = []
    y_pred_list = []
    timestamps_collected = []

    full_loader = torch.utils.data.DataLoader(full_ds, batch_size, shuffle=False)

    with torch.no_grad():
        for x_batch, y_batch, t_batch in full_loader:
            preds = model(x_batch)
            y_true_list.append(y_batch.cpu().numpy())
            y_pred_list.append(preds.cpu().numpy())
            timestamps_collected.extend(t_batch)

    y_true_all = np.concatenate(y_true_list, axis=0)
    y_pred_all = np.concatenate(y_pred_list, axis=0)
    timestamps_collected = np.array(timestamps_collected).reshape(-1)

    assert len(y_true_all) == len(timestamps_collected), "Länge von y_true und Zeitachse passt nicht!"

    os.makedirs(output_dir, exist_ok=True)

    # ── Log-Raum-Werte speichern (für Debugging) ──
    np.save(os.path.join(output_dir, "predictions_log.npy"), y_pred_all)
    np.save(os.path.join(output_dir, "y_true_log.npy"), y_true_all)

    # ── Multi-Target Rücktransformation ──
    y_true_orig = np.zeros_like(y_true_all)
    y_pred_orig = np.zeros_like(y_pred_all)

    for i, col_name in enumerate(target_features):
        trans_info = target_transformations.get(col_name, {"type": "none", "params": None})

        yt_col = y_true_all[:, i]
        yp_col = y_pred_all[:, i]

        epsilon = trans_info["params"].get("epsilon", 1e-6) if trans_info["params"] else 1e-6
        yt_orig, yp_orig = inverse_log_transform(
            yt_col,
            yp_col,
            method=trans_info,
            epsilon=epsilon
        )
        y_true_orig[:, i] = yt_orig
        y_pred_orig[:, i] = yp_orig

    # ── Originalskala-Werte speichern (für Plots und Metriken) ──
    np.save(os.path.join(output_dir, "predictions_full.npy"), y_pred_orig)
    np.save(os.path.join(output_dir, "y_true_full.npy"), y_true_orig)
    np.save(os.path.join(output_dir, "dates_full.npy"), timestamps_collected)
    np.save(os.path.join(output_dir, "X_full.npy"), x_full)

    with open(os.path.join(output_dir, "scaler_y.pkl"), "wb") as f:
        pickle.dump(scaler_y, f)