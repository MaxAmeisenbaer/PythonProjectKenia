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

def load_val_data_from_config(model_folder):
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

    print(val_df.describe())
    return x_val, y_val, val_df


def plot_val_period_simplified(df, start="2018-09-01", end="2018-12-31"):
    """
    Plot für alle skalierten Nicht-Target-Features im gewünschten Zeitraum.
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df.index)

    # Zeitraum einschränken
    df = df[(df["time"] >= start) & (df["time"] <= end)]

    # Alle Spalten außer "time" plotten
    feature_cols = df.columns.difference(["time"])
    plt.figure(figsize=(14, 6))
    for col in feature_cols:
        plt.plot(df["time"], df[col], label=col, alpha=0.7)

    plt.title(f"Skalierte Features ({start} bis {end})")
    plt.xlabel("Zeit")
    plt.ylabel("Skalierter Wert")
    plt.legend(loc="upper right", ncol=3, fontsize="small")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model_folder = ("low_input")

    _, _, val_df = load_val_data_from_config(model_folder)
    plot_val_period_simplified(val_df, start="2018-09-01", end="2018-12-31")
