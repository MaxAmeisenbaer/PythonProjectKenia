import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import numpy as np
import torch
import pickle
import pandas as pd

from log_transformation import inverse_log_transform


def _nanmean(values: list[float]) -> float:
    """Mean over all finite values; NaN if none are present."""
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


METRIC_NAMES = ("MSE", "RMSE", "MAE", "R2", "NSE", "MBE", "KGE")
NITRATE_TARGET_SUFFIX = "_nit"


def _empty_metrics() -> dict[str, float]:
    """Returns a metric dictionary with NaN values."""
    return {name: float("nan") for name in METRIC_NAMES}


def _compute_metrics_per_target(y_true_all, y_pred_all, y_mask_all,
                                target_features, target_transformations):
    """
    Computes all regression metrics separately for each target feature.

    :param y_true_all: True values [n_samples, n_targets]
    :param y_pred_all: Predictions [n_samples, n_targets]
    :param y_mask_all: Observation mask [n_samples, n_targets] (1 = observed)
    :param target_features: List of column names of the target features
    :param target_transformations: Dict with the methods per target feature
    :return: Dictionary {target_name: {metric_name: value}}
    """
    metrics_per_target = {}

    for i, col_name in enumerate(target_features):
        # Only evaluate actually observed values (0 = missing/imputed)
        valid = y_mask_all[:, i] > 0
        yt_col = y_true_all[valid, i]
        yp_col = y_pred_all[valid, i]

        if yt_col.size == 0:
            print(f"  Warnung: {col_name} hat keine beobachteten Testwerte - Metriken = NaN")
            metrics_per_target[col_name] = _empty_metrics()
            continue

        # ── Inverse transformation from log space (only observed values) ──
        trans_info = target_transformations.get(col_name, {"type": "none", "params": None})
        epsilon = trans_info["params"].get("epsilon", 1e-6) if trans_info["params"] else 1e-6
        yt, yp = inverse_log_transform(
            yt_col,
            yp_col,
            method=trans_info,
            epsilon=epsilon
        )

        # The inverse transformation can produce invalid values (e.g., Box-Cox)
        finite = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[finite]
        yp = yp[finite]

        if yt.size == 0:
            print(f"  Warnung: {col_name} hat nach Rücktransformation keine gültigen Testwerte - Metriken = NaN")
            metrics_per_target[col_name] = _empty_metrics()
            continue

        # Mean error measurements
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

        metrics_per_target[col_name] = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "NSE": nse,
            "MBE": mbe,
            "KGE": kge,
        }

    return metrics_per_target


def _aggregate_metrics(metrics_per_target, target_features, suffix=""):
    """
    Averages the metrics over the specified target features.

    :param metrics_per_target: Dictionary from _compute_metrics_per_target
    :param target_features: Targets over which to average
    :param suffix: Optional suffix for the metric names (e.g., "_nit")
    :return: Dictionary with averaged metrics
    """
    aggregated = {}
    for metric_name in METRIC_NAMES:
        values = [metrics_per_target[feature][metric_name] for feature in target_features]
        aggregated[f"{metric_name}{suffix}"] = _nanmean(values)
    return aggregated


def calculate_all_metrics(model, test_loader, target_features, target_transformations,
                          include_nitrate_only: bool = False,
                          include_per_target: bool = False):
    """
    Evaluates a trained model on a test dataset and computes various regression metrics.

    Unobserved target values (target mask == 0 or missing/imputed) are
    excluded from metric computation.

    :param model:       The trained PyTorch model
    :param test_loader: DataLoader for the test dataset
    :param target_features: List of column names of the target features (e.g., ["SHA_nit", "SHA_temp"])
    :param target_transformations: Dict with the methods per target feature
    :param include_nitrate_only: If True, in addition to the metrics, the metrics
        exclusively over the nitrate targets (suffix "_nit") are computed and
        returned with suffix "_nit" (e.g., "MSE_nit"). This keeps runs with
        additional output variables (e.g., discharge) comparable to
        nitrate-only runs.
    :param include_per_target: If True, in addition, the metrics for each
        individual target feature are returned, named "<metric>_<target>"
        (e.g., "MSE_NF_nit" or "KGE_SHA_disch").
    :return: Dictionary with averaged MSE, RMSE, MAE, R2, NSE, MBE, KGE over all
        target features; optionally also the same metrics with suffix "_nit"
        over the nitrate targets and/or per individual target feature.
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

    metrics_per_target = _compute_metrics_per_target(
        y_true_all, y_pred_all, y_mask_all, target_features, target_transformations
    )

    # Metrics over all target features
    result = _aggregate_metrics(metrics_per_target, target_features)

    # Optional: metrics only over the nitrate targets
    if include_nitrate_only:
        nitrate_features = [
            feature for feature in target_features
            if feature.endswith(NITRATE_TARGET_SUFFIX)
        ]
        if nitrate_features:
            result.update(_aggregate_metrics(
                metrics_per_target,
                nitrate_features,
                suffix=NITRATE_TARGET_SUFFIX
            ))
        else:
            print(f"  Warnung: Keine Targets mit Endung '{NITRATE_TARGET_SUFFIX}' gefunden - "
                  f"Nitrat-Metriken = NaN")
            result.update({
                f"{metric_name}{NITRATE_TARGET_SUFFIX}": float("nan")
                for metric_name in METRIC_NAMES
            })

    # Optional: metrics for each individual target feature
    if include_per_target:
        for feature in target_features:
            for metric_name in METRIC_NAMES:
                result[f"{metric_name}_{feature}"] = metrics_per_target[feature][metric_name]

    return result

def save_split_boundaries(train_df, val_df, test_df, save_path):
    """
    Saves start and end timestamps of the training, validation, and test sets.

    :param train_df: DataFrame with training data
    :param val_df: Validation data
    :param test_df: Test data
    :param save_path: Path to the target file (CSV)
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
    Performs prediction on the full dataset and saves:
    - predictions_full.npy
    - y_true_full.npy
    - dates_full.npy
    - X_full.npy
    - scaler_y.pkl

    :param model:      The trained PyTorch model
    :param full_ds:    TimeSeriesDatasetWithTimestamps (iterable dataset)
    :param output_dir: Target directory for the saved files
    :param x_full:     Full scaled input matrix
    :param scaler_y:   Scaler for the target variable
    :param target_features: List of target variables
    :param target_transformations: Information about normalization of the target features
    :param batch_size: Batch size for the DataLoader (affects only memory, not the result)
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

    # ── Save log-space values (for debugging) ──
    np.save(os.path.join(output_dir, "predictions_log.npy"), y_pred_all)
    np.save(os.path.join(output_dir, "y_true_log.npy"), y_true_all)

    # ── Multi-target inverse transformation ──
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

    # ── Save original-scale values (for plots and metrics) ──
    np.save(os.path.join(output_dir, "predictions_full.npy"), y_pred_orig)
    np.save(os.path.join(output_dir, "y_true_full.npy"), y_true_orig)
    np.save(os.path.join(output_dir, "dates_full.npy"), timestamps_collected)
    np.save(os.path.join(output_dir, "X_full.npy"), x_full)

    with open(os.path.join(output_dir, "scaler_y.pkl"), "wb") as f:
        pickle.dump(scaler_y, f)