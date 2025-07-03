import os
import numpy as np
import tensorflow as tf
from data_prepro import create_final_ds, scale_features
from benchmark_szenario_sha import get_benchmark_config
from low_input_szenario_sha import get_low_input_config
from not_nit_szenario_sha import get_not_nit_config
from not_lyser import get_not_lyser_config

def get_model_info(model_folder):
    """
    Gibt den Modellnamen und die zugehörige Konfigurationsfunktion für einen gegebenen Ordner zurück.
    """
    if model_folder == "benchmark":
        return "LSTM_SHA_benchmark_nit_001.keras", get_benchmark_config
    elif model_folder == "low_input":
        return "LSTM_SHA_low_input_nit_001.keras", get_low_input_config
    elif model_folder == "not_lyser":
        return "LSTM_SHA_not_lyser_nit_001.keras", get_not_lyser_config
    elif model_folder == "not_nit":
        return "LSTM_SHA_not_nit_nit_001.keras", get_not_nit_config
    else:
        raise ValueError(f"Unbekannter Modellordner: {model_folder}")


def recreate_predictions_for_folder(model_root, model_folder):
    """
    Führt die Vorhersage und das Abspeichern der .npy-Dateien für ein einzelnes Modell durch.
    """
    model_filename, config_func = get_model_info(model_folder)
    model_path = os.path.join(model_root, model_folder)
    model_file_path = os.path.join(model_path, model_filename)

    stations, measurements, target_feature, config_name = config_func()

    station = "SHA"
    batch_size = 16
    seq_length = 18

    # Daten vorbereiten
    _, _, _, train_df, val_df, test_df = create_final_ds(
        station, stations, measurements, target_feature,
        batch_size, seq_length
    )

    y_train = train_df[target_feature].values
    y_val = val_df[target_feature].values
    y_test = test_df[target_feature].values

    x_train, x_val, x_test, _ = scale_features(train_df, val_df, test_df, target_feature)

    def to_sequences(x, y, seq_length):
        x_seq = np.array([x[i:i + seq_length] for i in range(len(x) - seq_length)])
        y_seq = y[seq_length:len(x)]
        return x_seq, y_seq

    x_train_seq, y_train_seq = to_sequences(x_train, y_train, seq_length)
    x_val_seq, y_val_seq = to_sequences(x_val, y_val, seq_length)
    x_test_seq, y_test_seq = to_sequences(x_test, y_test, seq_length)

    # Modell laden
    model = tf.keras.models.load_model(model_file_path, compile=False)

    # Vorhersagen
    predictions_train = model.predict(x_train_seq, verbose=0)
    predictions_val = model.predict(x_val_seq, verbose=0)
    predictions_test = model.predict(x_test_seq, verbose=0)

    # Speichern
    np.save(os.path.join(model_path, "predictions_train.npy"), predictions_train)
    np.save(os.path.join(model_path, "predictions_val.npy"), predictions_val)
    np.save(os.path.join(model_path, "predictions_test.npy"), predictions_test)

    np.save(os.path.join(model_path, "y_train.npy"), y_train_seq)
    np.save(os.path.join(model_path, "y_val.npy"), y_val_seq)
    np.save(os.path.join(model_path, "y_test.npy"), y_test_seq)

    print(f"✅ Fertig: {model_folder} → Dateien gespeichert in {model_path}")


def recreate_all_predictions(model_root="models"):
    """
    Durchläuft alle bekannten Modellordner und erstellt dort Vorhersagedateien.
    """
    for model_folder in ["benchmark", "low_input", "not_nit", "not_lyser"]:
        recreate_predictions_for_folder(model_root, model_folder)


if __name__ == "__main__":
    recreate_all_predictions()
