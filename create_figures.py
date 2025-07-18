import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras


def get_all_timestamps(filepath="Data/SHA-nit.csv", interval="10min", valid_time_suffix="0:00"):
    """
    Gibt alle Zeitstempel des Zielzeitraums zurück, z. B. für Vorhersageplots über den gesamten Zeitraum.

    :param filepath: Pfad zur CSV-Datei mit der Zielgröße.
    :param interval: Zeitintervall für Resampling (Standard: "10min").
    :param valid_time_suffix: Optionaler Zeitfilter, um z. B. nur Messwerte mit Uhrzeit „0:00“ zu behalten.
    :return: pd.DatetimeIndex mit vollständigen Zeitstempeln.
    """
    start_date = "2015-04-28 11:00:00"
    end_date = "2019-11-21 12:00:00"

    df = pd.read_csv(filepath)
    if valid_time_suffix:
        df = df[df["date"].str.endswith(valid_time_suffix)]

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")
    df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]
    df.set_index("date", inplace=True)
    df = df.resample(interval).mean()
    df = df.interpolate()

    return df.index


def load_model_and_predictions(model_folder, keras_file):
    model_name = os.path.basename(model_folder)
    model = keras.models.load_model(os.path.join(model_folder, keras_file), compile=False)

    predictions_train = np.load(os.path.join(model_folder, "predictions_train.npy"))
    predictions_val = np.load(os.path.join(model_folder, "predictions_val.npy"))
    predictions_test = np.load(os.path.join(model_folder, "predictions_test.npy"))

    y_train = np.load(os.path.join(model_folder, "y_train.npy"))
    y_val = np.load(os.path.join(model_folder, "y_val.npy"))
    y_test = np.load(os.path.join(model_folder, "y_test.npy"))

    return model_name, predictions_train, predictions_val, predictions_test, y_train, y_val, y_test


def plot_predictions_full_timeline(model_folder, keras_file, output_folder, full_timestamps):
    model_name, pred_train, pred_val, pred_test, y_train, y_val, y_test = load_model_and_predictions(model_folder,
                                                                                                     keras_file)

    predictions = np.concatenate([pred_train, pred_val, pred_test])
    targets = np.concatenate([y_train, y_val, y_test])

    offset = len(full_timestamps) - len(predictions)
    timestamps = full_timestamps[offset:]

    if len(timestamps) != len(predictions):
        raise ValueError("Länge der Zeitstempel stimmt nicht mit Vorhersage-/Zielwerten überein.")

    plt.figure(figsize=(16, 6))
    plt.plot(timestamps, targets, label='Messwerte', color='black', linewidth=2)
    plt.plot(timestamps, predictions, label='Vorhersage', color='red', linestyle='--')

    n = len(targets)
    train_idx = len(pred_train)
    val_idx = train_idx + len(pred_val)

    plt.axvline(timestamps[train_idx], color='blue', linestyle=':', label='Train/Val-Grenze')
    plt.axvline(timestamps[val_idx], color='green', linestyle=':', label='Val/Test-Grenze')

    plt.title(f"Vorhersage vs Messung – Modell: {model_name}")
    plt.xlabel("Zeit")
    plt.ylabel("Nitratkonzentration [mg/L]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{model_name}_zeitreihe_full.png")
    plt.savefig(output_path)
    plt.close()


def plot_scatter(model_folder, output_folder, keras_file):
    model_name, pred_train, pred_val, pred_test, y_train, y_val, y_test = load_model_and_predictions(model_folder,
                                                                                                     keras_file)

    y_all = np.concatenate([y_train, y_val, y_test])
    pred_all = np.concatenate([pred_train, pred_val, pred_test])

    plt.figure(figsize=(6, 6))
    plt.scatter(y_all, pred_all, color="dodgerblue", edgecolor='k', alpha=0.75)
    plt.plot([y_all.min(), y_all.max()], [y_all.min(), y_all.max()], 'k--', lw=2)

    plt.xlabel("Observed (mg/l)", fontsize=12)
    plt.ylabel("Predicted (mg/l)", fontsize=12)
    plt.title(f"Nitrate for station {model_name}", fontsize=14)
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{model_name}_scatter.png")
    plt.savefig(output_path)
    plt.close()


def plot_all_models(szenarien, base_model_dir, output_zeitreihe_dir, output_scatter_dir, full_timestamps):
    for szenario in szenarien:
        model_path = os.path.join(base_model_dir, szenario)
        if szenario == "benchmark":
            keras_file = "LSTM_SHA_benchmark_nit_001.keras"
        elif szenario == "low_input":
            keras_file = "LSTM_SHA_low_input_nit_001.keras"
        elif szenario == "not_lyser":
            keras_file = "LSTM_SHA_not_lyser_nit_001.keras"
        elif szenario == "not_nit":
            keras_file = "LSTM_SHA_not_nit_nit_001.keras"
        else:
            raise ValueError(f"Unbekannter Modellordner: {szenario}")

        plot_predictions_full_timeline(model_path, keras_file, output_zeitreihe_dir, full_timestamps)
        plot_scatter(model_path, output_scatter_dir, keras_file)


def main():
    szenarien = ["benchmark", "low_input", "not_lyser", "not_nit"]
    base_model_dir = "models"
    output_zeitreihe_dir = "figures/zeitreihe"
    output_scatter_dir = "figures/scatter"

    full_timestamps = get_all_timestamps(filepath="Data/SHA-nit.csv", interval="10min")

    plot_all_models(
        szenarien,
        base_model_dir,
        output_zeitreihe_dir,
        output_scatter_dir,
        full_timestamps
    )


if __name__ == "__main__":
    main()
