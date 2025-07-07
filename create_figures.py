import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras


def get_timestamps_from_target_csv(y_train, y_val, y_test, filepath="Data/SHA-nit.csv", interval="10min", valid_time_suffix="0:00", seq_length=18):
    """
    Erzeugt Zeitstempel, die exakt zur Länge von y_train, y_val, y_test passen.

    :param y_train: Array der Trainings-Zielwerte (z. B. aus np.load).
    :param y_val: Array der Validierungs-Zielwerte.
    :param y_test: Array der Test-Zielwerte.
    :param filepath: Pfad zur CSV-Datei mit der Zielgröße.
    :param interval: Zeitintervall für Resampling (Standard: "10min").
    :param valid_time_suffix: Optionaler Zeitfilter, um z. B. nur Messwerte mit Uhrzeit „0:00“ zu behalten.
    :param seq_length: Sequenzlänge, wird standardmäßig abgezogen.
    :return: timestamps_train, timestamps_val, timestamps_test (pandas Index-Objekte)
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

    timestamps = df.index

    n = len(timestamps)
    train_end = int(n * 0.6)
    val_end = train_end + int(n * 0.2)

    timestamps_train = timestamps[:train_end][seq_length:seq_length + len(y_train)]
    timestamps_val = timestamps[train_end:val_end][seq_length:seq_length + len(y_val)]
    timestamps_test = timestamps[val_end:][seq_length:seq_length + len(y_test)]

    return timestamps_train, timestamps_val, timestamps_test



def load_model_and_predictions(model_folder, keras_file):
    """
    Lädt ein Keras-Modell und zugehörige Vorhersagen und Zielwerte aus einem Modellordner.

    :param model_folder: Pfad zum Modellunterordner.
    :return: Modellname, Vorhersagen (train/val/test) und Zielwerte (train/val/test).
    """
    model_name = os.path.basename(model_folder)
    model = keras.models.load_model(os.path.join(model_folder, keras_file), compile=False)

    predictions_train = np.load(os.path.join(model_folder, "predictions_train.npy"))
    predictions_val = np.load(os.path.join(model_folder, "predictions_val.npy"))
    predictions_test = np.load(os.path.join(model_folder, "predictions_test.npy"))

    y_train = np.load(os.path.join(model_folder, "y_train.npy"))
    y_val = np.load(os.path.join(model_folder, "y_val.npy"))
    y_test = np.load(os.path.join(model_folder, "y_test.npy"))

    return model_name, predictions_train, predictions_val, predictions_test, y_train, y_val, y_test


def plot_predictions_with_timestamps(model_folder, keras_file, output_folder, timestamps_train, timestamps_val, timestamps_test):
    """
    Erstellt einen Zeitreihenplot mit Vorhersagen und Messwerten eines Modells.

    :param model_folder: Pfad zum Modellordner mit Vorhersagen.
    :param output_folder: Zielordner für das Plotbild.
    :param timestamps_train: Zeitstempel der Trainingsdaten.
    :param timestamps_val: Zeitstempel der Validierungsdaten.
    :param timestamps_test: Zeitstempel der Testdaten.
    """
    model_name, pred_train, pred_val, pred_test, y_train, y_val, y_test = load_model_and_predictions(model_folder, keras_file)

    predictions = np.concatenate([pred_train, pred_val, pred_test])
    targets = np.concatenate([y_train, y_val, y_test])
    timestamps = np.concatenate([timestamps_train, timestamps_val, timestamps_test])

    plt.figure(figsize=(16, 6))
    plt.plot(timestamps, targets, label='Messwerte', color='black', linewidth=2)
    plt.plot(timestamps, predictions, label='Vorhersage', color='red', linestyle='--')

    plt.title(f"Vorhersage vs Messung – Modell: {model_name}")
    plt.xlabel("Zeit")
    plt.ylabel("Nitratkonzentration [mg/L]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{model_name}_zeitreihe.png")
    plt.savefig(output_path)
    plt.close()


def plot_scatter(model_folder, output_folder, keras_file):
    """
    Erstellt einen Scatter-Plot: beobachtet vs. vorhergesagt.

    :param model_folder: Pfad zum Modellordner.
    :param output_folder: Zielordner für die Scatter-Grafik.
    """
    model_name, pred_train, pred_val, pred_test, y_train, y_val, y_test = load_model_and_predictions(model_folder, keras_file)

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


def plot_all_models(szenarien, base_model_dir, output_zeitreihe_dir, output_scatter_dir, timestamps_train,
                    timestamps_val, timestamps_test):
    """
    Erstellt für alle Szenarien alle Zeitreihen- und Scatterplots.

    :param szenarien: Liste der Modellunterordner (z.B. ['benchmark', 'low_input']).
    :param base_model_dir: Basisverzeichnis, das alle Szenarienordner enthält.
    :param output_zeitreihe_dir: Zielordner für Zeitreihenplots.
    :param output_scatter_dir: Zielordner für Scatterplots.
    :param timestamps_train: Zeitstempel Training.
    :param timestamps_val: Zeitstempel Validierung.
    :param timestamps_test: Zeitstempel Test.
    """
    for szenario in szenarien:
        model_path = os.path.join(base_model_dir, szenario)
        if szenario == "benchmark":
            keras_file = "LSTM_SHA_benchmark_nit_001.keras"
            plot_predictions_with_timestamps(model_path, keras_file, output_zeitreihe_dir, timestamps_train,
                                             timestamps_val,
                                             timestamps_test)
            plot_scatter(model_path, output_scatter_dir, keras_file)
        elif szenario == "low_input":
            keras_file = "LSTM_SHA_low_input_nit_001.keras"
            plot_predictions_with_timestamps(model_path, keras_file, output_zeitreihe_dir, timestamps_train,
                                             timestamps_val,
                                             timestamps_test)
            plot_scatter(model_path, output_scatter_dir, keras_file)
        elif szenario == "not_lyser":
            keras_file = "LSTM_SHA_not_lyser_nit_001.keras"
            plot_predictions_with_timestamps(model_path, keras_file, output_zeitreihe_dir, timestamps_train,
                                             timestamps_val,
                                             timestamps_test)
            plot_scatter(model_path, output_scatter_dir, keras_file)
        elif szenario == "not_nit":
            keras_file = "LSTM_SHA_not_nit_nit_001.keras"
            plot_predictions_with_timestamps(model_path, keras_file, output_zeitreihe_dir, timestamps_train,
                                             timestamps_val,
                                             timestamps_test)
            plot_scatter(model_path, output_scatter_dir, keras_file)
        else:
            raise ValueError(f"Unbekannter Modellordner: {szenario}")


def main():
    """
    Hauptfunktion: Lädt Zeitstempel und erstellt alle Plots für definierte Szenarien.
    """
    szenarien = ["benchmark", "low_input", "not_lyser", "not_nit"]
    base_model_dir = "models"
    output_zeitreihe_dir = "figures/zeitreihe"
    output_scatter_dir = "figures/scatter"

    benchmark_folder = os.path.join(base_model_dir, "benchmark")
    y_train = np.load(os.path.join(benchmark_folder, "y_train.npy"))
    y_val = np.load(os.path.join(benchmark_folder, "y_val.npy"))
    y_test = np.load(os.path.join(benchmark_folder, "y_test.npy"))

    timestamps_train, timestamps_val, timestamps_test = get_timestamps_from_target_csv(
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        filepath="Data/SHA-nit.csv",
        interval="10min",
    )


    plot_all_models(
        szenarien,
        base_model_dir,
        output_zeitreihe_dir,
        output_scatter_dir,
        timestamps_train,
        timestamps_val,
        timestamps_test
    )


if __name__ == "__main__":
    main()
