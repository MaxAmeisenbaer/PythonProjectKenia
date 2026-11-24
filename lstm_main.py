import os
import re
import torch
import pandas as pd
from datetime import datetime

from lstm_model import create_model, train_model, set_seed
from data_prepro import create_final_ds
from scenario_config import *
from evaluate_model import (
    calculate_all_metrics,
    save_split_boundaries,
    evaluate_and_store_full_predictions,
    NITRATE_TARGET_SUFFIX,
)


def _save_model_metadata(model_name, params, output_path="model_log.xlsx"):
    """
    Saves metadata of a trained model to an Excel file.
    If the file exists, a new entry is appended.

    :param model_name: Name of the saved model
    :param params: Dictionary with model parameters and metrics
    :param output_path: Path to the Excel log file (default="model_log.xlsx")
    """
    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        **params
    }

    if os.path.exists(output_path):
        df = pd.read_excel(output_path)
        df = pd.concat([df, pd.DataFrame([metadata])], ignore_index=True)
    else:
        df = pd.DataFrame([metadata])

    df.to_excel(output_path, index=False)


def _generate_model_name(config_name, target_features, output_path="model_log.xlsx"):
    """
    Generates a unique model name based on the configuration and target variable.
    Numbering continues automatically if models already exist.

    :param config_name: Name of the configuration (e.g., "benchmark")
    :param target_features: List of target variables
    :param output_path: Path to the Excel log file for existing models

    :return: String with the new model name
    """
    stations = []
    features = []
    for variable in target_features:
        station, feature = variable.split("_")
        stations.append(station)
        features.append(feature)
    full_stations = ["NF", "SHA", "TTP"]
    unique_stations = sorted(list(set(stations)))
    unique_stations = [x for x in unique_stations if x in full_stations]
    if unique_stations == full_stations:
        station = "ALL"
    else:
        station = "_".join(unique_stations)
    unique_features = sorted(list(set(features)))
    target = "_".join(unique_features)


    prefix = f"LSTM_{station}_{config_name}_{target}"

    existing_numbers = []
    if os.path.exists(output_path):
        try:
            df = pd.read_excel(output_path)
            pattern = rf"^{re.escape(prefix)}_(\d+)"
            for name in df["model_name"]:
                match = re.match(pattern, str(name))
                if match:
                    existing_numbers.append(int(match.group(1)))
        except Exception as e:
            print(f"Warnung beim Lesen der Logdatei für Nummerierung: {e}")


    next_number = max(existing_numbers, default=0) + 1
    return f"{prefix}_{next_number:03d}"


def _prepare_data(config, target_features, stations, measurements):
    """
    Prepares the datasets for training, validation, and test.

    :param config: Dictionary with model configuration (batch size, sequence length, etc.)
    :param target_features: List of target variables (e.g., "SHA_Nit")
    :param stations: List of stations used
    :param measurements: List of measurements

    :return: Prepared data coming from create_final_ds (datasets and -frames for train,val,test + timestamps, transformation-info and scaler-info)
    """
    train_ds, val_ds, test_ds, train_df, test_df, val_df, x_full, full_ds, timestamps_full, target_transformations, scaler_y = create_final_ds(
        stations=stations,
        target_features=target_features,
        batch_size=config["batch_size"],
        seq_length=config["seq_length"],
        measurements=measurements
    )
    return train_ds, val_ds, test_ds, train_df, test_df, val_df, x_full, full_ds, timestamps_full, target_transformations, scaler_y


def _build_and_train_model(train_loader, val_loader, config, n_features: int, n_targets: int):
    """
    Creates an LSTM model and trains it.

    :param train_loader: Training DataLoader
    :param val_loader:   Validation DataLoader
    :param config:       Dictionary with model parameters
    :param n_features:   Number of input features
    :param n_targets:    Target variables of the model

    :return:             Trained model, history dictionary
    """
    model, optimizer, loss_fn = create_model(
        n_features=n_features,
        n_targets=n_targets,
        nodes_lstm=config["nodes_lstm"],
        nodes_dense=config["nodes_dense"],
        dropout=config["dropout"],
        learning_rate=config["learning_rate"],
        euclidean=config.get("euclidean", False)
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=config["epochs"],
        patience=5
    )
    return model, history


def run(scenario, nitrate_only_metrics: bool = False, per_target_metrics: bool = False,
        euclidean: bool = False):
    """
    Runs the full model-training workflow for a specific scenario:
    - Load the scenario configuration
    - Prepare the data
    - Train the model
    - Compute the metrics
    - Save the model, log entry, and split boundaries
    - Evaluate on the full dataset

    :param scenario: Name of the scenario ("benchmark", "low_input", "test_code")
    :param nitrate_only_metrics: Opt-in. If True, in addition to the metrics
        over all targets, the metrics are computed exclusively over the nitrate
        targets (suffix "_nit") and logged with suffix "_nit" (e.g., "MSE_nit").
        This keeps runs with additional output variables (e.g., discharge)
        comparable to nitrate-only runs.
    :param per_target_metrics: Opt-in. If True, the metrics for each individual
        target feature are also logged, named "<metric>_<target>"
        (e.g., "MSE_NF_nit", "KGE_SHA_disch").
    :param euclidean: Opt-in. If True, the MSE is computed per target, then its
        square root is taken and the mean over the targets is squared:
        (mean_t sqrt(MSE_t))^2. As a result, all targets contribute equally to
        the loss regardless of their number of observations.

    :return: Dictionary with model name and computed metrics
    """
    # Reproducible initialization and training order
    set_seed(42)

    if scenario == "benchmark":
        stations, measurements, target_features, config_name = get_benchmark_config()
    elif scenario == "disch":
        stations, measurements, target_features, config_name = get_disch_config()
    elif scenario == "not_prec":
        stations, measurements, target_features, config_name = get_not_prec_config()
    elif scenario == "not_soil":
        stations, measurements, target_features, config_name = get_not_soil_config()
    elif scenario == "not_water_qual":
        stations, measurements, target_features, config_name = get_not_water_qual_config()
    elif scenario == "not_water_lvl":
        stations, measurements, target_features, config_name = get_not_water_lvl_config()
    elif scenario == "not_temp":
        stations, measurements, target_features, config_name = get_not_temp_config()
    elif scenario == "not_wind":
        stations, measurements, target_features, config_name = get_not_wind_config()
    elif scenario == "not_ec":
        stations, measurements, target_features, config_name = get_not_ec_config()
    elif scenario == "not_atmos":
        stations, measurements, target_features, config_name = get_not_atmos_config()
    elif scenario == "test_code":
        stations, measurements, target_features, config_name = get_test_code_config()
    else:
        raise ValueError(f"Unbekanntes Szenario: {scenario}")

#Legacy code: not_nit_config, not_lyser_config, low_input_config

    # Request nitrate-only metrics only if nitrate targets are present
    nitrate_targets = [
        target for target in target_features
        if target.endswith(NITRATE_TARGET_SUFFIX)
    ]
    if nitrate_only_metrics and not nitrate_targets:
        raise ValueError(
            f"nitrate_only_metrics=True, aber keine Targets mit Endung "
            f"'{NITRATE_TARGET_SUFFIX}' in {target_features} gefunden."
        )

    model_config = {
        "nodes_lstm": 20,
        "nodes_dense": 0,
        "dropout": 0.2,
        "learning_rate": 0.01,
        "num_layers": 3,
        "batch_size": 32,
        "seq_length": 6,
        "epochs": 70,
        "euclidean": euclidean
    }

    # Prepare data
    (train_ds, val_ds, test_ds, train_df, test_df, val_df, x_full, full_ds, timestamps_full, target_transformations, scaler_y) = _prepare_data(
        model_config, target_features, stations, measurements)

    # Derive n_features from x_full
    n_features = x_full.shape[1]
    n_targets = len(target_features)

    # Build and train model
    model, history = _build_and_train_model(
        train_ds, val_ds, model_config, n_features, n_targets)

    # Compute metrics (optionally nitrate-only and/or per individual target)
    metrics_result = calculate_all_metrics(
        model,
        test_ds,
        target_features,
        target_transformations=target_transformations,
        include_nitrate_only=nitrate_only_metrics,
        include_per_target=per_target_metrics
    )
    model_name = _generate_model_name(config_name, target_features)

    # Save model
    output_dir = os.path.join("models", config_name)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, f"{model_name}.pt"))

    # Read early-stopping epochs from history
    early_stopped = len(history["train_loss"])

    # Save model metadata
    _save_model_metadata(
        model_name=model_name,
        params={
            **model_config,
            "early_stopped": early_stopped,
            "metrics_scope": "all+nitrate" if nitrate_only_metrics else "all",
            "nitrate_targets": ", ".join(nitrate_targets) if nitrate_only_metrics else None,
            "per_target_metrics": per_target_metrics,
            "per_target_metrics_targets": ", ".join(target_features) if per_target_metrics else None,
            **metrics_result
        }
    )

    # Save the split time windows
    split_save = os.path.join(output_dir, f"{model_name}_split_boundaries.csv")
    save_split_boundaries(train_df, val_df, test_df, save_path=split_save)

    # Evaluation and prediction on the full dataset
    if full_ds is not None and timestamps_full is not None and x_full is not None and scaler_y is not None:
        evaluate_and_store_full_predictions(
            model=model,
            full_ds=full_ds,
            output_dir=output_dir,
            x_full=x_full,
            scaler_y=scaler_y,
            target_features=target_features,
            target_transformations=target_transformations
        )

    return {
        "model_name": model_name,
        "metrics": metrics_result
    }


if __name__ == "__main__":
    run(scenario="benchmark")