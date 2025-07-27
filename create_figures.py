import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras


def load_split_boundaries(model_path, split_info):
    """
    Lädt die Start- und Endzeitpunkte der Splits aus CSV-Datei.

    :param split_info: Dateiname der CSV-Datei mit Spalten "set", "start", "end"
    :return: dict mit 'train', 'val', 'test' jeweils als (start, end) Timestamp-Tuple
    """
    df = pd.read_csv(os.path.join(model_path, split_info))
    boundaries = {}
    for split in ["train", "val", "test"]:
        row = df[df["set"] == split].iloc[0]
        start = pd.to_datetime(row["start"])
        end = pd.to_datetime(row["end"])
        boundaries[split] = (start, end)
    return boundaries



def load_model_and_predictions(model_folder, keras_file):
    """
    Lädt das Modell und die konsolidierten Vorhersage- und Zeitreihendateien aus dem Modellordner.

    :param model_folder: Pfad zum Modellordner (z.B. 'models/benchmark')
    :param keras_file: Name der .keras-Datei im Modellordner
    :return: model_name, predictions_full, y_true_full, timestamps_full
    """
    model_path = os.path.join(model_folder, keras_file)
    model_name = os.path.splitext(keras_file)[0]

    predictions = np.load(os.path.join(model_folder, "predictions_full.npy"))
    y_true = np.load(os.path.join(model_folder, "y_true_full.npy"))
    timestamps = np.load(os.path.join(model_folder, "dates_full.npy"), allow_pickle=True)

    # Bytes zu Strings umwandeln, falls nötig
    timestamps_str = [ts.decode('utf-8') if isinstance(ts, bytes) else ts for ts in timestamps]
    timestamps = pd.to_datetime(timestamps_str)

    return model_name, predictions, y_true, timestamps



def plot_predictions_full_timeline(model_folder, keras_file, output_path, szenario, boundaries):
    model_name, y_pred, y_true, full_timestamps = load_model_and_predictions(model_folder, keras_file)

    if len(full_timestamps) != len(y_pred):
        raise ValueError(f"Längen passen nicht: {len(full_timestamps)} vs. {len(y_pred)}")

    # Split-Grenzen aus CSV laden
    val_start, _ = boundaries["val"]
    test_start, _ = boundaries["test"]

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(full_timestamps, y_true, label='Messwert', color="black", linewidth=1.2)
    ax.plot(full_timestamps, y_pred, label='Vorhersage', color="red", linewidth=1.2, linestyle='--')

    # Vertikale Linien mit exakten Zeitpunkten
    if full_timestamps.min() < val_start < full_timestamps.max():
        ax.axvline(val_start, color='green', linestyle='--', linewidth=1.5)
    if full_timestamps.min() < test_start < full_timestamps.max():
        ax.axvline(test_start, color='blue', linestyle='--', linewidth=1.5)

    ax.set_title(szenario)
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Nitrat [mg/L]")
    ax.legend()
    plt.tight_layout()

    os.makedirs(output_path, exist_ok=True)
    fig.savefig(os.path.join(output_path, f"{szenario}_zeitreihe.png"))
    plt.close(fig)





def plot_scatter(model_folder, output_folder, keras_file, szenario):
    model_name, predictions, y_true, _ = load_model_and_predictions(model_folder, keras_file)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, predictions, color="dodgerblue", edgecolor='k', alpha=0.75)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)

    plt.xlabel("Gemessen (mg/l)", fontsize=12)
    plt.ylabel("Vorhergesagt (mg/l)", fontsize=12)
    plt.title(szenario, fontsize=14)
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()

    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, f"{szenario}_scatter.png")
    plt.savefig(output_path)
    plt.close()



def plot_all_models(szenarien, base_model_dir, output_zeitreihe_dir, output_scatter_dir):
    for szenario in szenarien:
        model_path = os.path.join(base_model_dir, szenario)
        if szenario == "benchmark":
            keras_file = "LSTM_SHA_benchmark_nit_001.keras"
            split_info = "LSTM_SHA_benchmark_nit_001_split_boundaries.csv"
        elif szenario == "low_input":
            keras_file = "LSTM_SHA_low_input_nit_001.keras"
            split_info = "LSTM_SHA_low_input_nit_001_split_boundaries.csv"
        elif szenario == "not_lyser":
            keras_file = "LSTM_SHA_not_lyser_nit_001.keras"
            split_info = "LSTM_SHA_not_lyser_nit_001_split_boundaries.csv"
        elif szenario == "not_nit":
            keras_file = "LSTM_SHA_not_nit_nit_001.keras"
            split_info = "LSTM_SHA_not_nit_nit_001_split_boundaries.csv"
        else:
            raise ValueError(f"Unbekannter Modellordner: {szenario}")

        boundaries = load_split_boundaries(model_path, split_info)
        plot_predictions_full_timeline(model_path, keras_file, output_zeitreihe_dir, szenario, boundaries)
        plot_scatter(model_path, output_scatter_dir, keras_file, szenario)


def main():
    szenarien = ["benchmark", "low_input", "not_lyser", "not_nit"]
    base_model_dir = "models"
    output_zeitreihe_dir = "figures/zeitreihe"
    output_scatter_dir = "figures/scatter"


    plot_all_models(
        szenarien,
        base_model_dir,
        output_zeitreihe_dir,
        output_scatter_dir
    )


if __name__ == "__main__":
    main()

def test_single_model():
    szenarien = ["benchmark"]
    base_model_dir = "models"
    output_zeitreihe_dir = "figures/zeitreihe"
    output_scatter_dir = "figures/scatter"


    plot_all_models(
        szenarien,
        base_model_dir,
        output_zeitreihe_dir,
        output_scatter_dir
    )


#if __name__ == "__main__":
#    test_single_model()
