import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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



def load_predictions(model_folder):
    """
    Lädt das Modell und die konsolidierten Vorhersage- und Zeitreihendateien aus dem Modellordner.

    :param model_folder: Pfad zum Modellordner (z.B. 'models/benchmark')
    :return: predictions_full, y_true_full, timestamps_full
    """
    predictions = np.load(os.path.join(model_folder, "predictions_full.npy"))
    y_true = np.load(os.path.join(model_folder, "y_true_full.npy"))
    timestamps = np.load(os.path.join(model_folder, "dates_full.npy"), allow_pickle=True)

    # Bytes zu Strings umwandeln, falls nötig
    timestamps_str = [ts.decode('utf-8') if isinstance(ts, bytes) else ts for ts in timestamps]
    timestamps = pd.to_datetime(timestamps_str)

    return predictions, y_true, timestamps



def plot_predictions_full_timeline(model_folder, output_path, szenario, boundaries):
    """
    Erstellt eine Zeitreihengrafik für Messwerte und Vorhersagen über den gesamten Zeitraum.
    Die Split-Grenzen (Validierung/Test) werden als vertikale Linien markiert.

    :param model_folder: Pfad zum Modellordner
    :param output_path: Pfad zum Ausgabeverzeichnis
    :param szenario: Name des Szenarios (z.B. "benchmark")
    :param boundaries: Dictionary mit Split-Grenzen (train/val/test)
    """
    y_pred, y_true, full_timestamps = load_predictions(model_folder)

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

    ax.set_title(szenario, fontsize = 18)
    ax.set_xlabel("Zeit", fontsize = 14)
    ax.set_ylabel("Nitrat [mg/L]", fontsize = 14)
    ax.legend(fontsize = 10)
    plt.tight_layout()

    os.makedirs(output_path, exist_ok=True)
    fig.savefig(os.path.join(output_path, f"{szenario}_zeitreihe.png"))
    plt.close(fig)

def plot_predictions_test_timeline(model_folder, output_path, szenario, boundaries):
    """
    Erstellt eine Zeitreihengrafik für Messwerte und Vorhersagen über den Testdatenzeitraum

    :param model_folder: Pfad zum Modellordner
    :param output_path: Pfad zum Ausgabeverzeichnis
    :param szenario: Name des Szenarios (z.B. "benchmark")
    """
    y_pred, y_true, full_timestamps = load_predictions(model_folder)

    if len(full_timestamps) != len(y_pred):
        raise ValueError(f"Längen passen nicht: {len(full_timestamps)} vs. {len(y_pred)}")

    # Start-Zeitpunkt aus Boundaries holen und Start-Index daraus ermitteln
    test_start_time = pd.to_datetime(boundaries["test"][0])
    start_index = np.searchsorted(full_timestamps, test_start_time)

    # Synchrones Schneiden aller Arrays
    y_pred = y_pred[start_index:]
    y_true = y_true[start_index:]
    full_timestamps = full_timestamps[start_index:]

    fig, ax = plt.subplots(figsize=(16, 6.5))
    ax.plot(full_timestamps, y_true, label='Measurement', color="black", linewidth=1.2)
    ax.plot(full_timestamps, y_pred, label='Prediction', color="red", linewidth=1.2, linestyle='--')

    ax.set_title(szenario, fontsize = 28)
    ax.set_xlabel("Time", fontsize = 24)
    ax.set_ylabel("nitrate [mg L⁻¹]", fontsize = 24)
    ax.legend(fontsize = 20)
    plt.tight_layout()

    os.makedirs(output_path, exist_ok=True)
    fig.savefig(os.path.join(output_path, f"{szenario}_zeitreihe.png"))
    plt.close(fig)



def plot_scatter(model_folder, output_folder, szenario):
    """
    Erstellt ein Scatterplot-Diagramm (gemessen vs. vorhergesagt) für ein Modell.

    :param model_folder: Pfad zum Modellordner
    :param output_folder: Pfad zum Ausgabeverzeichnis
    :param szenario: Name des Szenarios (z.B. "benchmark")
    """
    predictions, y_true, _ = load_predictions(model_folder)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, predictions, color="dodgerblue", edgecolor='k', alpha=0.75)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)

    plt.xlabel("Gemessen (mg/l)", fontsize=14)
    plt.ylabel("Vorhergesagt (mg/l)", fontsize=14)
    plt.title(szenario, fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()

    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, f"{szenario}_scatter.png")
    plt.savefig(output_path, dpi=300)
    plt.close()


def combine_scatter_plots(image_names, input_folder, output_path, dpi=300):
    """
    Kombiniert mehrere Scatterplots zu einem 2x2-Rasterbild,
    ohne zusätzliche Beschriftungen (für kompakte Darstellung).

    :param image_names: Liste der Bilddateinamen der Scatterplots
    :param input_folder: Pfad zum Ordner mit den Scatterplot-Bildern
    :param output_path: Pfad für das kombinierte Ausgabebild
    :param dpi: Auflösung des Ausgabebildes (default=300)
    """
    # Zielgröße in cm → Umrechnung in Zoll
    target_width_cm = 16
    target_height_cm = 10
    cm_to_inch = 1 / 2.54
    fig_width_in = target_width_cm * cm_to_inch
    fig_height_in = target_height_cm * cm_to_inch

    fig, axes = plt.subplots(2, 2, figsize=(fig_width_in, fig_height_in), dpi=dpi)

    for ax, img_name in zip(axes.flat, image_names):
        img_path = os.path.join(input_folder, img_name)
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis('off')

    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Kombinierte Scattergrafik gespeichert unter: {output_path}")



def plot_models_bachelor(szenarien, base_model_dir, output_zeitreihe_dir, output_scatter_dir):
    """
    Führt die Erstellung von Zeitreihen- und Scatterplots für alle angegebenen Szenarien durch.

    :param szenarien: Liste der Szenarien (z.B. ["benchmark", "low_input", ...])
    :param base_model_dir: Basisordner, in dem die Modelle gespeichert sind
    :param output_zeitreihe_dir: Zielordner für Zeitreihengrafiken
    :param output_scatter_dir: Zielordner für Scatterplots
    """
    for szenario in szenarien:
        model_path = os.path.join(base_model_dir, szenario)
        if szenario == "benchmark":
            split_info = "LSTM_SHA_benchmark_nit_001_split_boundaries.csv"
        elif szenario == "low_input":
            split_info = "LSTM_SHA_low_input_nit_001_split_boundaries.csv"
        elif szenario == "not_lyser":
            split_info = "LSTM_SHA_not_lyser_nit_001_split_boundaries.csv"
        elif szenario == "not_nit":
            split_info = "LSTM_SHA_not_nit_nit_001_split_boundaries.csv"
        else:
            raise ValueError(f"Unbekannter Modellordner: {szenario}")

        boundaries = load_split_boundaries(model_path, split_info)
        plot_predictions_full_timeline(model_path,output_zeitreihe_dir, szenario, boundaries)
        plot_scatter(model_path, output_scatter_dir, szenario)


def plot_models_data_type_exclusion(szenarien, base_model_dir, output_zeitreihe_dir):
    """
    Führt die Erstellung von Zeitreihenplots für alle angegebenen Szenarien durch.

    :param szenarien: Liste der Szenarien (z.B. ["benchmark", "low_input", ...])
    :param base_model_dir: Basisordner, in dem die Modelle gespeichert sind
    :param output_zeitreihe_dir: Zielordner für Zeitreihengrafiken
    """
    for szenario in szenarien:
        model_path = os.path.join(base_model_dir, szenario)
        if szenario == "benchmark":
            split_info = "LSTM_SHA_benchmark_nit_001_split_boundaries.csv"
        elif szenario == "not_prec":
            split_info = "LSTM_SHA_not_prec_nit_001_split_boundaries.csv"
        elif szenario == "not_soil":
            split_info = "LSTM_SHA_not_soil_nit_001_split_boundaries.csv"
        elif szenario == "not_water_qual":
            split_info = "LSTM_SHA_not_water_qual_nit_001_split_boundaries.csv"
        elif szenario == "not_water_lvl":
            split_info = "LSTM_SHA_not_water_lvl_nit_001_split_boundaries.csv"
        elif szenario == "not_temp":
            split_info = "LSTM_SHA_not_temp_nit_001_split_boundaries.csv"
        elif szenario == "not_wind":
            split_info = "LSTM_SHA_not_wind_nit_001_split_boundaries.csv"
        elif szenario == "not_ec":
            split_info = "LSTM_SHA_not_ec_nit_001_split_boundaries.csv"
        elif szenario == "not_atmos":
            split_info = "LSTM_SHA_not_atmos_nit_001_split_boundaries.csv"
        else:
            raise ValueError(f"Unbekannter Modellordner: {szenario}")

        boundaries = load_split_boundaries(model_path, split_info)
        plot_predictions_test_timeline(model_path, output_zeitreihe_dir, szenario, boundaries)


def main():
    """
    Hauptfunktion: erstellt Zeitreihen- und Scatterplots für alle Szenarien
    und kombiniert die Scatterplots in einem 2x2-Gesamtbild.
    """
    szenarien = ["benchmark", "low_input", "not_lyser", "not_nit","not_prec","not_soil", "not_water_qual",
                 "not_water_lvl","not_temp", "not_wind", "not_ec", "not_atmos"]
    base_model_dir = "models"
    output_zeitreihe_dir = "figures/zeitreihe"
    output_scatter_dir = "figures/scatter"


    plot_models_bachelor(szenarien, base_model_dir, output_zeitreihe_dir, output_scatter_dir)

    image_list = [
        "benchmark_scatter.png",
        "not_nit_scatter.png",
        "not_lyser_scatter.png",
        "low_input_scatter.png"
    ]

    combine_scatter_plots(
        image_names=image_list,
        input_folder="figures/scatter",
        output_path="figures/combined_scatter_2x2.png"
    )


#if __name__ == "__main__":
#    main()


def test_single_model():
    """
    Testfunktion: führt die Plot-Erstellung nur für das Benchmark-Szenario aus.
    """
    szenarien = ["benchmark"]
    base_model_dir = "models"
    output_zeitreihe_dir = "figures/zeitreihe"
    output_scatter_dir = "figures/scatter"


    plot_models_bachelor(szenarien, base_model_dir, output_zeitreihe_dir, output_scatter_dir)


#if __name__ == "__main__":
#    test_single_model()

def data_type_exclusion_graphs():
    """
    Plot-Erstellung für alle Szenarien der data_typ exklusion (nur der test-Zeitabschnitt) (nur zeitreihe)
    """
    szenarien = ["benchmark", "not_prec","not_soil", "not_water_qual",
                 "not_water_lvl","not_temp", "not_wind", "not_ec", "not_atmos"]
    base_model_dir = "models"
    output_zeitreihe_dir = "figures/zeitreihe"

    plot_models_data_type_exclusion(szenarien, base_model_dir, output_zeitreihe_dir)

if __name__ == "__main__":
    data_type_exclusion_graphs()