import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#TODO: This script needs a full overhaul, as it is not up to date!
def _load_split_boundaries(model_path, split_info):
    """
    Loads the start and end timestamps of the splits from a CSV file.

    :param model_path: [MISPARAM]
    :param split_info: Filename of the CSV file with columns "set", "start", "end"
    :return: dict with 'train', 'val', 'test' each as (start, end) timestamp tuple
    """
    df = pd.read_csv(os.path.join(model_path, split_info))
    boundaries = {}
    for split in ["train", "val", "test"]:
        row = df[df["set"] == split].iloc[0]
        start = pd.to_datetime(row["start"])
        end = pd.to_datetime(row["end"])
        boundaries[split] = (start, end)
    return boundaries



def _load_predictions(model_folder):
    """
    Loads the model and the consolidated prediction and time series files from the model folder. [MISDESC]

    :param model_folder: Path to the model folder (e.g., 'models/benchmark')
    :return: predictions_full, y_true_full, timestamps_full
    """
    predictions = np.load(os.path.join(model_folder, "predictions_full.npy"))
    y_true = np.load(os.path.join(model_folder, "y_true_full.npy"))
    timestamps = np.load(os.path.join(model_folder, "dates_full.npy"), allow_pickle=True)

    # Convert bytes to strings if necessary
    timestamps_str = [ts.decode('utf-8') if isinstance(ts, bytes) else ts for ts in timestamps]
    timestamps = pd.to_datetime(timestamps_str)

    return predictions, y_true, timestamps



def _plot_predictions_full_timeline(model_folder, output_path, szenario, boundaries):
    """
    Creates a time series plot for measurements and predictions over the entire period.
    The split boundaries (validation/test) are marked as vertical lines.

    :param model_folder: Path to the model folder
    :param output_path: Path to the output directory
    :param szenario: Name of the scenario (e.g., "benchmark")
    :param boundaries: Dictionary with split boundaries (train/val/test)
    """
    y_pred, y_true, full_timestamps = _load_predictions(model_folder)

    if len(full_timestamps) != len(y_pred):
        raise ValueError(f"Längen passen nicht: {len(full_timestamps)} vs. {len(y_pred)}")

    # Load split boundaries from CSV
    val_start, _ = boundaries["val"]
    test_start, _ = boundaries["test"]

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(full_timestamps, y_true, label='Messwert', color="black", linewidth=1.2)
    ax.plot(full_timestamps, y_pred, label='Vorhersage', color="red", linewidth=1.2, linestyle='--')

    # Vertical lines at exact timestamps
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

def _plot_predictions_test_timeline(model_folder, output_path, szenario, boundaries):
    """
    Creates a time series plot for measurements and predictions over the test data period

    :param model_folder: Path to the model folder
    :param output_path: Path to the output directory
    :param szenario: Name of the scenario (e.g., "benchmark")
    :param boundaries: [MISPARAM]
    """
    y_pred, y_true, full_timestamps = _load_predictions(model_folder)

    if len(full_timestamps) != len(y_pred):
        raise ValueError(f"Längen passen nicht: {len(full_timestamps)} vs. {len(y_pred)}")

    # Get start timestamp from boundaries and determine start index from it
    test_start_time = pd.to_datetime(boundaries["test"][0])
    start_index = np.searchsorted(full_timestamps, test_start_time)

    # Synchronously slice all arrays
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
    fig.savefig(os.path.join(output_path, f"{szenario}_zeitreihe.svg"), format = "svg", bbox_inches='tight')
    plt.close(fig)



def _plot_scatter(model_folder, output_folder, szenario):
    """
    Creates a scatter plot (measured vs. predicted) for a model.

    :param model_folder: Path to the model folder
    :param output_folder: Path to the output directory
    :param szenario: Name of the scenario (e.g., "benchmark")
    """
    predictions, y_true, _ = _load_predictions(model_folder)

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


def _combine_scatter_plots(image_names, input_folder, output_path, dpi=300):
    """
    Combines multiple scatter plots into a 2x2 grid image,
    without additional labels (for a compact presentation).

    :param image_names: List of image filenames of the scatter plots
    :param input_folder: Path to the folder with the scatter plot images
    :param output_path: Path for the combined output image
    :param dpi: Resolution of the output image (default=300)
    """
    # Target size in cm → conversion to inches
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



def _plot_models_bachelor(szenarien, base_model_dir, output_zeitreihe_dir, output_scatter_dir):
    """
    Performs the creation of time series and scatter plots for all specified scenarios.

    :param szenarien: List of scenarios (e.g., ["benchmark", "low_input", ...])
    :param base_model_dir: Base folder where the models are stored
    :param output_zeitreihe_dir: Target folder for time series plots
    :param output_scatter_dir: Target folder for scatter plots
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

        boundaries = _load_split_boundaries(model_path, split_info)
        _plot_predictions_full_timeline(model_path,output_zeitreihe_dir, szenario, boundaries)
        _plot_scatter(model_path, output_scatter_dir, szenario)


def _plot_models_data_type_exclusion(szenarien, base_model_dir, output_zeitreihe_dir):
    """
    Performs the creation of time series plots for all specified scenarios.

    :param szenarien: List of scenarios (e.g., ["benchmark", "low_input", ...])
    :param base_model_dir: Base folder where the models are stored
    :param output_zeitreihe_dir: Target folder for time series plots
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

        boundaries = _load_split_boundaries(model_path, split_info)
        _plot_predictions_test_timeline(model_path, output_zeitreihe_dir, szenario, boundaries)


def main():
    """
    Main function: creates time series and scatter plots for all scenarios [MISDESC]
    and combines the scatter plots into a 2x2 overall image.
    """
    szenarien = ["benchmark", "low_input", "not_lyser", "not_nit","not_prec","not_soil", "not_water_qual",
                 "not_water_lvl","not_temp", "not_wind", "not_ec", "not_atmos"]
    base_model_dir = "models"
    output_zeitreihe_dir = "figures/zeitreihe"
    output_scatter_dir = "figures/scatter"


    _plot_models_bachelor(szenarien, base_model_dir, output_zeitreihe_dir, output_scatter_dir)

    image_list = [
        "benchmark_scatter.png",
        "not_nit_scatter.png",
        "not_lyser_scatter.png",
        "low_input_scatter.png"
    ]

    _combine_scatter_plots(
        image_names=image_list,
        input_folder="figures/scatter",
        output_path="figures/combined_scatter_2x2.png"
    )


#if __name__ == "__main__":
#    main()


def test_single_model():
    """
    Test function: runs plot creation only for the benchmark scenario.
    """
    szenarien = ["benchmark"]
    base_model_dir = "models"
    output_zeitreihe_dir = "figures/zeitreihe"
    output_scatter_dir = "figures/scatter"


    _plot_models_bachelor(szenarien, base_model_dir, output_zeitreihe_dir, output_scatter_dir)


#if __name__ == "__main__":
#    test_single_model()

def data_type_exclusion_graphs():
    """
    Plot creation for all scenarios of the not_x scenarios (only the test time period) (only time series)
    """
    szenarien = ["benchmark", "not_prec","not_soil", "not_water_qual",
                 "not_water_lvl","not_temp", "not_wind", "not_ec", "not_atmos"]
    base_model_dir = "models"
    output_zeitreihe_dir = "figures/zeitreihe"

    _plot_models_data_type_exclusion(szenarien, base_model_dir, output_zeitreihe_dir)

if __name__ == "__main__":
    data_type_exclusion_graphs()