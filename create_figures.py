import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras


def get_timestamps_from_target_csv(filepath="Data/SHA-nit.csv", interval="10min", seq_length=18):
    """
    Extrahiert Zeitstempel aus der Originaldatei mit der Zielvariable (z.B. Nitrat).
    Die Daten werden resampled, interpoliert und dann um die Sequenzlänge reduziert,
    um zu den Modellvorhersagen zu passen.

    :param filepath: Pfad zur CSV-Datei mit der Zielgröße
    :param interval: Resamplingintervall, z.B. '10min'
    :param seq_length: Anzahl an Zeitschritten pro Sequenz im LSTM-Modell
    :return: Drei Pandas Index-Objekte: timestamps_train, timestamps_val, timestamps_test
    """
    start_date = "2015-04-28 11:00:00"
    end_date = "2019-11-21 12:00:00"

    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.strftime("%H:%M:%S") == "00:00:00"]
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    df.set_index("date", inplace=True)

    df = df.resample(interval).mean()
    df = df.interpolate()

    timestamps = df.index[seq_length:]

    n = len(timestamps)
    train_end = int(n * 0.6)
    val_end = train_end + int(n * 0.2)

    timestamps_train = timestamps[:train_end]
    timestamps_val = timestamps[train_end:val_end]
    timestamps_test = timestamps[val_end:]

    return timestamps_train, timestamps_val, timestamps_test


def load_model_and_predictions(model_folder):
    """
    Lädt das Modell und die gespeicherten Vorhersagen/Zielwerte aus einem gegebenen Modellordner.

    :param model_folder: Pfad zum Ordner, der das Modell und die .npy-Dateien enthält
    :return: Modellname, predictions_train/val/test, y_train/val/test
    """
    model_name = os.path.basename(model_folder)
    model = keras.models.load_model(os.path.join(model_folder, "model.keras"), compile=False)

    predictions_train = np.load(os.path.join(model_folder, "predictions_train.npy"))
    predictions_val = np.load(os.path.join(model_folder, "predictions_val.npy"))
    predictions_test = np.load(os.path.join(model_folder, "predictions_test.npy"))

    y_train = np.load(os.path.join(model_folder, "y_train.npy"))
    y_val = np.load(os.path.join(model_folder, "y_val.npy"))
    y_test = np.load(os.path.join(model_folder, "y_test.npy"))

    return model_name, predictions_train, predictions_val, predictions_test, y_train, y_val, y_test


def plot_predictions_with_timestamps(model_folder, output_folder, timestamps_train, timestamps_val, timestamps_test):
    """
    Erstellt einen Zeitreihenplot für ein gegebenes Modell mit Vorhersage- und Messwerten
    in den drei Bereichen: Training, Validierung und Test.

    :param model_folder: Pfad zum Ordner mit dem Modell und den gespeicherten Vorhersagen/Zielen
    :param output_folder: Zielordner zum Speichern des Plots
    :param timestamps_train: Zeitstempel der Trainingsdaten (Index oder Liste)
    :param timestamps_val: Zeitstempel der Validierungsdaten (Index oder Liste)
    :param timestamps_test: Zeitstempel der Testdaten (Index oder Liste)
    """
    model_name, predictions_train, predictions_val, predictions_test, y_train, y_val, y_test = load_model_and_predictions(model_folder)

    predictions = np.concatenate([predictions_train, predictions_val, predictions_test])
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


def plot_all_timeline(models_directory, output_directory, timestamps_train, timestamps_val, timestamps_test):
    """
    Erstellt Zeitreihenplots für alle Modelle im gegebenen Verzeichnis.

    :param models_directory: Ordner mit den Modellunterordnern
    :param output_directory: Ordner für die Ausgabeplots
    :param timestamps_train: Zeitstempel für Trainingsdaten
    :param timestamps_val: Zeitstempel für Validierungsdaten
    :param timestamps_test: Zeitstempel für Testdaten
    """
    for model_folder in os.listdir(models_directory):
        model_path = os.path.join(models_directory, model_folder)
        if os.path.isdir(model_path):
            plot_predictions_with_timestamps(model_path, output_directory, timestamps_train, timestamps_val, timestamps_test)


def plot_scatter(model_folder, output_folder):
    """
    Erstellt einen Scatter-Plot der beobachteten vs. vorhergesagten Nitratwerte
    für ein gegebenes Modell.

    :param model_folder: Pfad zum Ordner mit dem Modell und den gespeicherten Vorhersagen/Zielwerten
    :param output_folder: Zielordner zum Speichern des Scatter-Plots
    """
    model_name, pred_train, pred_val, pred_test, y_train, y_val, y_test = load_model_and_predictions(model_folder)

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


def plot_all_scatter(models_directory, output_directory):
    """
    Erstellt Scatterplots für alle Modelle im gegebenen Verzeichnis.

    :param models_directory: Ordner mit den Modellunterordnern
    :param output_directory: Zielordner für die Scatterplots
    """
    for model_folder in os.listdir(models_directory):
        model_path = os.path.join(models_directory, model_folder)
        if os.path.isdir(model_path):
            plot_scatter(model_path, output_directory)


def main():
    models_directory = "models"
    output_zeitreihe = "results/zeitreihe"
    output_scatter = "results/scatter"

    timestamps_train, timestamps_val, timestamps_test = get_timestamps_from_target_csv(
        filepath="Data/SHA-nit.csv",
        interval="10min",
        seq_length=18
    )

    # Zeitreihenplots
    plot_all_timeline(models_directory, output_zeitreihe, timestamps_train, timestamps_val, timestamps_test)

    # Scatterplots
    plot_all_scatter(models_directory, output_scatter)


if __name__ == "__main__":
    main()
