import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from benchmark_szenario_sha import get_benchmark_config
from low_input_szenario_sha import get_low_input_config
from not_lyser import get_not_lyser_config
from not_nit_szenario_sha import get_not_nit_config

from data_prepro import (
    load_data,
    scale_features,
    split_dataset,
    prepare_targets,
)

def get_config_for_model(model_folder):
    """
    Gibt die passende Konfigurationsfunktion für das Modell zurück.
    """
    if model_folder == "benchmark":
        return get_benchmark_config
    elif model_folder == "low_input":
        return get_low_input_config
    elif model_folder == "not_lyser":
        return get_not_lyser_config
    elif model_folder == "not_nit":
        return get_not_nit_config
    else:
        raise ValueError(f"Unbekannter Modellordner: {model_folder}")

def load_val_data_from_config(model_folder, seq_length=18):
    """
    Lädt val_df, y_val, x_val aus der Konfiguration, wie im Training.
    """
    config_func = get_config_for_model(model_folder)
    stations, measurements, target_feature, config_name = config_func()

    # Rohdaten laden
    df = load_data(stations, measurements)
    train_df, val_df, test_df = split_dataset(df, target_feature)
    y_train, y_val, y_test = prepare_targets(train_df, val_df, test_df, target_feature)
    x_train, x_val, x_test, _ = scale_features(train_df, val_df, test_df, target_feature)

    return x_val, y_val, val_df


def plot_val_period(x_val, y_val, val_df, start="2018-09-01", end="2018-12-31", seq_length=18):
    """
    Visualisiert Zielwerte, Feature-Mittelwerte und Heatmap für den gegebenen Zeitraum.
    """
    val_df = val_df.copy()
    val_df["time"] = pd.to_datetime(val_df.index)

    # Rekonstruktion der Zeitachse passend zu x_val/y_val
    time_val = val_df["time"].iloc[seq_length - 1: seq_length - 1 + len(y_val)].reset_index(drop=True)
    y_val = y_val.squeeze()[:len(time_val)]

    assert len(time_val) == len(y_val) == x_val.shape[0], "Datenlängen stimmen nicht überein."

    # Maske basierend auf time_val erzeugen
    mask = (time_val >= pd.to_datetime(start)) & (time_val <= pd.to_datetime(end))

    x_seq = x_val[mask]
    y_seq = y_val[mask]
    time_seq = time_val[mask]

    # Plot 1: y_val über Zeit
    plt.figure(figsize=(12, 4))
    plt.plot(time_seq, y_seq, label="y_val", color="tab:red")
    plt.title("Zielwert (y_val) im Validierungszeitraum")
    plt.xlabel("Datum")
    plt.ylabel("Skalierter Zielwert")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()

    # Plot 2: Feature-Mittelwert pro Sequenz
    feature_means = x_seq.mean(axis=2).mean(axis=1)
    plt.figure(figsize=(12, 4))
    plt.plot(time_seq, feature_means, label="Feature-Mittelwert", color="tab:blue")
    plt.title("Durchschnittlicher Feature-Wert pro Sequenz")
    plt.xlabel("Datum")
    plt.ylabel("Skalierter Mittelwert")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()

    # Plot 3: Heatmap der Features (letzter Zeitschritt pro Sequenz)
    feature_names = val_df.columns.tolist()
    if len(feature_names) <= 60:
        df_heat = pd.DataFrame(x_seq[:, -1, :], columns=feature_names)
        df_heat.index = time_seq
        plt.figure(figsize=(12, 6))
        sns.heatmap(df_heat.T, cmap="viridis", cbar=True)
        plt.title("Heatmap der Features (letzter Zeitschritt jeder Sequenz)")
        plt.xlabel("Datum")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ Zu viele Features für Heatmap (>60) – übersprungen.")


if __name__ == "__main__":
    model_folder = "benchmark"  # z.B. "benchmark", "low_input", "not_nit", "not_lyser"
    seq_length = 18

    x_val, y_val, val_df = load_val_data_from_config(model_folder, seq_length=seq_length)
    plot_val_period(x_val, y_val, val_df, start="2018-09-01", end="2018-12-31", seq_length=seq_length)
