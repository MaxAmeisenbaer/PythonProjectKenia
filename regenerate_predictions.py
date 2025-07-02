import os
import numpy as np
import tensorflow as tf

from benchmark_szenario_sha import get_benchmark_config
from low_input_szenario_sha import get_low_input_config
from not_nit_szenario_sha import get_not_nit_config
from not_lyser import get_not_lyser_config
from data_prepro import create_final_ds


SCENARIOS = [
    {
        "name": "benchmark",
        "model_file": "LSTM_SHA_benchmark_nit_001.keras",
        "get_config": get_benchmark_config
    },
    {
        "name": "low_input",
        "model_file": "LSTM_SHA_low_input_nit_001.keras",
        "get_config": get_low_input_config
    },
    {
        "name": "not_lyser",
        "model_file": "LSTM_SHA_not_lyser_nit_001.keras",
        "get_config": get_not_lyser_config
    },
    {
        "name": "not_nit",
        "model_file": "LSTM_SHA_not_nit_nit_001.keras",
        "get_config": get_not_nit_config
    },
]


def generate_and_save_predictions():
    for scenario in SCENARIOS:
        name = scenario["name"]
        model_path = os.path.join("models", name, scenario["model_file"])
        print(f"\nVerarbeite Szenario: {name} -> {model_path}")

        # Konfiguration abrufen
        config = scenario["get_config"]()

        # Daten erzeugen
        _, _, test_ds, _, test_df, _ = create_final_ds(
            station=config["station"],
            stations=config["stations"],
            measurements=config["measurements"],
            target_feature=config["target_feature"],
            batch_size=config["batch_size"],
            seq_length=config["seq_length"],
            interval=config["interval"]
        )

        # Modell laden
        model = tf.keras.models.load_model(model_path, compile=False)

        # Vorhersagen erzeugen
        predictions = model.predict(test_ds)

        # Wahre Werte extrahieren
        true_values = np.concatenate([y for x, y in test_ds], axis=0)

        # Ergebnisse speichern
        output_dir = os.path.join("models", name)
        np.save(os.path.join(output_dir, "predictions_test.npy"), predictions)
        np.save(os.path.join(output_dir, "true_values_test.npy"), true_values)
        print(f"Gespeichert in: {output_dir}")


if __name__ == "__main__":
    generate_and_save_predictions()

#muss noch überarbeitet werden