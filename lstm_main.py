import os
import re
import torch
import pandas as pd
from datetime import datetime

from lstm_model import create_model, train_model
from data_prepro import create_final_ds
from benchmark_szenario_sha import get_benchmark_config
from low_input_szenario_sha import get_low_input_config
from not_nit_szenario_sha import get_not_nit_config
from test_code_szenario import get_test_code_config
from evaluate_model import calculate_all_metrics, save_split_boundaries
from not_lyser_szenario_sha import get_not_lyser_config


def save_model_metadata(model_name, params, output_path="model_log.xlsx"):
    """
    Speichert Metadaten eines trainierten Modells in einer Excel-Datei.
    Falls die Datei existiert, wird ein neuer Eintrag hinzugefügt.

    :param model_name: Name des gespeicherten Modells
    :param params: Dictionary mit Modellparametern und Metriken
    :param output_path: Pfad zur Excel-Logdatei (default="model_log.xlsx")
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


def generate_model_name(config_name, target_feature, output_path="model_log.xlsx"):
    """
    Generiert einen eindeutigen Modellnamen basierend auf Konfiguration und Zielvariable.
    Die Nummerierung wird automatisch fortgeführt, falls bereits Modelle existieren.

    :param config_name: Name der Konfiguration (z. B. "benchmark")
    :param target_feature: Zielgröße im Format "STATION_FEATURE" (z. B. "SHA_NO3")
    :param output_path: Pfad zur Excel-Logdatei für bestehende Modelle
    :return: String mit neuem Modellnamen
    """
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
    """
    Bereitet die Datensätze für Training, Validierung und Test vor.

    :param config: Dictionary mit Modellkonfiguration (Batchgröße, Sequenzlänge etc.)
    :param target_feature: Zielvariable (z. B. "SHA_Nit")
    :param stations: Liste der verwendeten Stationen
    :param measurements: Liste der Messgrößen
    :return: train_ds, val_ds, test_ds, train_df, test_df, val_df, x_full, full_ds, timestamps_full, scaler_y
    """
    train_ds, val_ds, test_ds, train_df, test_df, val_df, x_full, full_ds, timestamps_full, scaler_y = create_final_ds(
        station="SHA",
        stations=stations,
        target_feature=target_feature,
        batch_size=config["batch_size"],
        seq_length=config["seq_length"],
        measurements=measurements
    )
    return train_ds, val_ds, test_ds, train_df, test_df, val_df, x_full, full_ds, timestamps_full, scaler_y


def build_and_train_model(train_loader, val_loader, config, n_features: int):
    """
    Erstellt ein LSTM-Modell und trainiert es.

    :param train_loader: Trainings-DataLoader
    :param val_loader:   Validierungs-DataLoader
    :param config:       Dictionary mit Modellparametern
    :param n_features:   Anzahl der Eingabe-Features
    :return:             Trainiertes Modell, History-Dictionary
    """
    model, optimizer, loss_fn = create_model(
        n_features=n_features,
        nodes_lstm=config["nodes_lstm"],
        nodes_dense=config["nodes_dense"],
        dropout=config["dropout"],
        learning_rate=config["learning_rate"]
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


def run(scenario):
    """
    Führt den gesamten Modelltrainings-Workflow für ein bestimmtes Szenario aus:
    - Laden der Szenario-Konfiguration
    - Datenvorbereitung
    - Modelltraining
    - Berechnung der Metriken
    - Speichern von Modell, Log-Eintrag und Split-Grenzen
    - Evaluation auf vollem Datensatz

    :param scenario: Name des Szenarios ("benchmark", "low_input", "not_nit", "not_lyser", "test_code")
    :return: Dictionary mit Modellname und berechneten Metriken
    """
    if scenario == "benchmark":
        stations, measurements, target_feature, config_name = get_benchmark_config()
    elif scenario == "low_input":
        stations, measurements, target_feature, config_name = get_low_input_config()
    elif scenario == "not_nit":
        stations, measurements, target_feature, config_name = get_not_nit_config()
    elif scenario == "not_lyser":
        stations, measurements, target_feature, config_name = get_not_lyser_config()
    elif scenario == "test_code":
        stations, measurements, target_feature, config_name = get_test_code_config()
    else:
        raise ValueError(f"Unbekanntes Szenario: {scenario}")

    model_config = {
        "nodes_lstm": 100,
        "nodes_dense": 64,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "num_layers": 2,
        "batch_size": 64,
        "seq_length": 18,
        "epochs": 70
    }

    # Daten vorbereiten
    (train_ds, val_ds, test_ds, train_df, test_df, val_df, x_full, full_ds, timestamps_full, scaler_y) = prepare_data(
        model_config, target_feature, stations, measurements)

    # n_features aus x_full ableiten
    n_features = x_full.shape[1]

    # Modell bauen und trainieren
    model, history = build_and_train_model(
        train_ds, val_ds, model_config, n_features)

    # Metriken berechnen
    metrics_result = calculate_all_metrics(model, test_ds)
    model_name = generate_model_name(config_name, target_feature)

    # Modell speichern
    output_dir = os.path.join("models", config_name)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, f"{model_name}.pt"))

    # Early-Stopping-Epochen aus History auslesen
    early_stopped = len(history["train_loss"])

    # Modell-Metadaten speichern
    save_model_metadata(
        model_name=model_name,
        params={
            **model_config,
            "early_stopped": early_stopped,
            **metrics_result
        }
    )

    # Speichern der Split-Zeiträume
    split_save = os.path.join(output_dir, f"{model_name}_split_boundaries.csv")
    save_split_boundaries(train_df, val_df, test_df, save_path=split_save)

    # Evaluation und Vorhersage auf vollem Datensatz
    if full_ds is not None and timestamps_full is not None and x_full is not None and scaler_y is not None:
        from evaluate_model import evaluate_and_store_full_predictions
        evaluate_and_store_full_predictions(
            model=model,
            full_ds=full_ds,
            output_dir=output_dir,
            x_full=x_full,
            scaler_y=scaler_y
        )

    return {
        "model_name": model_name,
        "metrics": metrics_result
    }


if __name__ == "__main__":
    run(scenario="benchmark")