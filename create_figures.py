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
    """
    Lädt das Modell und die konsolidierten Vorhersage- und Zeitreihendateien aus dem Modellordner.

    :param model_folder: Pfad zum Modellordner (z. B. 'models/benchmark')
    :param keras_file: Name der .keras-Datei im Modellordner
    :return: model_name, predictions_full, y_true_full, timestamps_full
    """
    model_path = os.path.join(model_folder, keras_file)
    model_name = os.path.splitext(keras_file)[0]

    model = keras.models.load_model(model_path)

    predictions = np.load(os.path.join(model_folder, "predictions_full.npy"))
    y_true = np.load(os.path.join(model_folder, "y_true_full.npy"))
    timestamps = np.load(os.path.join(model_folder, "dates_full.npy"), allow_pickle=True)

    return model_name, predictions, y_true, timestamps


def plot_predictions_full_timeline(model_folder, keras_file, output_path, full_timestamps, seq_length=18):
    model_name, y_pred, y_true, _ = load_model_and_predictions(model_folder, keras_file)

    # Zeitachse anpassen
    diff = len(full_timestamps) - len(y_pred)
    if diff == seq_length - 1:
        full_timestamps = full_timestamps[(seq_length - 1):]
        print(f"ℹ️ Zeitachse automatisch um seq_length-1={seq_length - 1} gekürzt.")
    elif diff == seq_length:
        full_timestamps = full_timestamps[seq_length:]
        print(f"ℹ️ Zeitachse automatisch um seq_length={seq_length} gekürzt.")
    else:
        print(f"❌ Länge der Vorhersage ({len(y_pred)}) passt nicht zur Zeitachse ({len(full_timestamps)})!")
        return

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(full_timestamps, y_true, label='Wahr', linewidth=1.2)
    ax.plot(full_timestamps, y_pred, label='Vorhersage', linewidth=1.2, linestyle='--')
    ax.set_title(f"Modellvorhersage: {model_name}")
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Konzentration")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_path, f"{model_name}_full_plot.png"))
    plt.close(fig)




def plot_scatter(model_folder, output_folder, keras_file):
    """
    Plottet einen Scatterplot zwischen gemessenen und vorhergesagten Werten.

    :param model_folder: Pfad zum Modellordner
    :param output_folder: Zielordner für die Grafik
    :param keras_file: Modell-Dateiname (.keras)
    """
    model_name, predictions, y_true, _ = load_model_and_predictions(model_folder, keras_file)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, predictions, color="dodgerblue", edgecolor='k', alpha=0.75)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)

    plt.xlabel("Gemessen (mg/l)", fontsize=12)
    plt.ylabel("Vorhergesagt (mg/l)", fontsize=12)
    plt.title(f"Nitrat: {model_name}", fontsize=14)
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
        elif szenario == "test_code":
            keras_file = "LSTM_SHA_test_code_nit_001.keras"
        else:
            raise ValueError(f"Unbekannter Modellordner: {szenario}")

        plot_predictions_full_timeline(model_path, keras_file, output_zeitreihe_dir, full_timestamps)
        plot_scatter(model_path, output_scatter_dir, keras_file)


def main():
    szenarien = ["benchmark", "low_input", "not_lyser", "not_nit", "test_code"]
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


#if __name__ == "__main__":
#    main()
def test_single_model():
    szenarien = ["test_code"]
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
    test_single_model()
