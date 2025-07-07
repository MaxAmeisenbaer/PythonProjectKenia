import os
import numpy as np
import tensorflow as tf

from benchmark_szenario_sha import get_benchmark_config
from low_input_szenario_sha import get_low_input_config
from not_lyser import get_not_lyser_config
from not_nit_szenario_sha import get_not_nit_config

from data_prepro import (
    load_data,
    scale_features,
    split_dataset,
    prepare_targets,
    make_tf_datasets,
)

def get_model_info(folder_name):
    """
    Gibt den Modellnamen und die passende Konfigurationsfunktion basierend auf dem Ordnernamen zurück.
    """
    if folder_name == "benchmark":
        return "LSTM_SHA_benchmark_nit_001.keras", get_benchmark_config
    elif folder_name == "low_input":
        return "LSTM_SHA_low_input_nit_001.keras", get_low_input_config
    elif folder_name == "not_lyser":
        return "LSTM_SHA_not_lyser_nit_001.keras", get_not_lyser_config
    elif folder_name == "not_nit":
        return "LSTM_SHA_not_nit_nit_001.keras", get_not_nit_config
    else:
        raise ValueError(f"Unbekannter Ordnername: {folder_name}")

def recreate_predictions_for_folder(model_root, model_folder):
    """
    Führt die Vorhersage und das Abspeichern der .npy-Dateien für ein einzelnes Modell durch –
    mit exakt derselben Logik wie beim ursprünglichen Training.
    """
    model_filename, config_func = get_model_info(model_folder)
    model_path = os.path.join(model_root, model_folder)
    model_file_path = os.path.join(model_path, model_filename)

    # Konfigurationsdaten laden
    stations, measurements, target_feature, config_name = config_func()

    # Standardwerte: station, batch size, sequence length
    station = "SHA"
    batch_size = 16
    seq_length = 18

    # Rohdaten laden und vorbereiten
    df = load_data(stations, measurements)
    train_df, val_df, test_df = split_dataset(df, target_feature)

    # Zielgrößen vorbereiten
    y_train, y_val, y_test = prepare_targets(train_df, val_df, test_df, target_feature)

    # Merkmale skalieren
    x_train, x_val, x_test, _ = scale_features(train_df, val_df, test_df, target_feature)

    # TensorFlow-Datasets generieren
    ds_train, ds_val, ds_test = make_tf_datasets(
        x_train, x_val, x_test,
        y_train, y_val, y_test,
        seq_length, batch_size
    )

    # Modell laden
    model = tf.keras.models.load_model(model_file_path, compile=False)

    # Vorhersagen
    predictions_train = model.predict(ds_train, verbose=0)
    predictions_val = model.predict(ds_val, verbose=0)
    predictions_test = model.predict(ds_test, verbose=0)

    # Zielwerte aus tf.data-Datasets extrahieren
    y_train_seq = np.concatenate([y for _, y in ds_train], axis=0)
    y_val_seq = np.concatenate([y for _, y in ds_val], axis=0)
    y_test_seq = np.concatenate([y for _, y in ds_test], axis=0)

    # Abspeichern der Vorhersagen und Zielwerte
    np.save(os.path.join(model_path, "predictions_train.npy"), predictions_train)
    np.save(os.path.join(model_path, "predictions_val.npy"), predictions_val)
    np.save(os.path.join(model_path, "predictions_test.npy"), predictions_test)

    np.save(os.path.join(model_path, "y_train.npy"), y_train_seq)
    np.save(os.path.join(model_path, "y_val.npy"), y_val_seq)
    np.save(os.path.join(model_path, "y_test.npy"), y_test_seq)

    print(f"✅ Vorhersagen für '{model_folder}' gespeichert unter: {model_path}")

def recreate_all_predictions(model_root="models"):
    """
    Führt die Vorhersageerzeugung für alle bekannten Modellordner aus.
    """
    folders = ["benchmark", "low_input", "not_lyser", "not_nit"]
    for folder in folders:
        recreate_predictions_for_folder(model_root, folder)

if __name__ == "__main__":
    recreate_all_predictions()
