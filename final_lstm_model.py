import os
import re
import pandas as pd
from datetime import datetime

from lstm_model import create_model, train_model
from data_prepro import create_final_ds
from benchmark_szenario_sha import get_benchmark_config
from low_input_szenario_sha import get_low_input_config
from not_nit_szenario_sha import get_not_nit_config
from evaluate_model import calculate_all_metrics
from not_lyser import get_not_spectro_config


def save_model_metadata(model_name, params, output_path="model_log.xlsx"):
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

def generate_model_name(config_name, target_feature, output_path="model_log.xlsx"):
    station, target = target_feature.split("_")
    prefix = f"LSTM_{station}_{config_name}_{target}"

    existing_numbers = []
    if os.path.exists(output_path):
        df = pd.read_excel(output_path)
        pattern = rf"{re.escape(prefix)}_(\d+)"
        for name in df["model_name"]:
            match = re.match(pattern, name)
            if match:
                existing_numbers.append(int(match.group(1)))

    next_number = max(existing_numbers, default=0) + 1
    return f"{prefix}_{next_number:03d}"

def prepare_data(config, target_feature, stations, measurements):
    train_ds, val_ds, test_ds, *_ = create_final_ds(
        station="SHA",
        stations=stations,
        target_feature=target_feature,
        batch_size=config["batch_size"],
        seq_length=config["seq_length"],
        measurements=measurements
    )
    return train_ds, val_ds, test_ds

def build_and_train_model(train_ds, val_ds, config):
    model, early_stopping = create_model(
        nodes_lstm=config["nodes_lstm"],
        nodes_dense=config["nodes_dense"],
        dropout=config["dropout"],
        metric=config["metric"],
        learning_rate=config["learning_rate"]
    )

    history, train_loss, val_loss = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        early_stopping=early_stopping,
        metric=config["metric"],
        epochs=config["epochs"]
    )
    return model, train_loss, val_loss

def run(scenario):
    if scenario == "benchmark":
        stations, measurements, target_feature, config_name = get_benchmark_config()
    elif scenario == "low_input":
        stations, measurements, target_feature, config_name = get_low_input_config()
    elif scenario == "not_nit":
        stations, measurements, target_feature, config_name = get_not_nit_config()
    elif scenario == "not_lyser":
        stations, measurements, target_feature, config_name = get_not_lyser_config()
    else:
        raise ValueError(f"Unbekanntes Szenario: {scenario}")

    model_config = {
        "nodes_lstm": 100,
        "nodes_dense": 64,
        "dropout": 0.1,
        "metric": "mean_squared_error",
        "learning_rate": 0.0001,
        "batch_size": 16,
        "seq_length": 18,
        "epochs": 70
    }

    train_ds, val_ds, test_ds = prepare_data(model_config, target_feature, stations, measurements)
    model, train_loss, val_loss = build_and_train_model(train_ds, val_ds, model_config)

    metrics_result = calculate_all_metrics(model, test_ds)
    model_name = generate_model_name(config_name, target_feature)

    os.makedirs("models", exist_ok=True)
    model.save(f"models/{model_name}.keras")

    save_model_metadata(
        model_name=model_name,
        params={
            "train_loss": train_loss,
            "val_loss": val_loss,
            **model_config,
            **metrics_result
        }
    )

    return {
        "model_name": model_name,
        "metrics": metrics_result,
        "train_loss": train_loss,
        "val_loss": val_loss
    }

if __name__ == "__main__":
    run(scenario= "not_lyser")
