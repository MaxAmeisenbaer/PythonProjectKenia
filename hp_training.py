import keras_tuner as kt
import numpy as np
from data_prepro import create_final_ds
from lstm_model import create_model, train_model
from benchmark_szenario_sha import get_benchmark_config

stations, measurements, target_feature = get_benchmark_config()

class LSTMHyperModel(kt.HyperModel):
    def build(self, hp):
        nodes_lstm = hp.Choice("lstm_nodes", values=[10, 20, 50, 100])
        dropout = hp.Choice("dropout", values=[0.1, 0.2, 0.3, 0.5])
        learning_rate = hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4])

        model, early_stopping = create_model(
            nodes_lstm=nodes_lstm,
            nodes_dense=None,
            dropout=dropout,
            metric="r_square",
            learning_rate=learning_rate,
        )

        # Wir geben early_stopping mit zurück, für später
        self.early_stopping = early_stopping
        return model

    def fit(self, hp, model, *args, **kwargs):
        batch_size = hp.Choice("batch_size", values=[16, 32, 128])
        seq_length = hp.Choice("seq_length", values=[2, 6, 18, 432])
        epochs = hp.Choice("epochs", values=[20, 30, 50, 70])

        train_ds, val_ds, test_ds, _, _, _ = create_final_ds(
            station="SHA",
            stations=stations,
            measurements=measurements,
            target_feature=target_feature,
            batch_size=batch_size,
            seq_length=seq_length,
            interval="10min"
        )

        return model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[self.early_stopping],
            verbose=0
        )

# Tuner initialisieren
tuner = kt.GridSearch(
    hypermodel=LSTMHyperModel(),
    objective=kt.Objective("val_r_square", direction="max"),
    max_trials=20,
    executions_per_trial=1,
    directory="kt_tuner",
    project_name="lstm_tuning"
)

# Suche starten (Dummy-Daten, weil fit() intern create_final_ds nutzt)
dummy_x = np.zeros((10, 2, 3))
dummy_y = np.zeros((10, 1))

tuner.search(dummy_x, dummy_y)

# Bestes Modell extrahieren
best_hp = tuner.get_best_hyperparameters(1)[0]
best_model = tuner.hypermodel.build(best_hp)

# Optional: Modell mit optimalen Parametern neu trainieren und speichern
train_ds, val_ds, test_ds, _, _, _ = create_final_ds(
    station="SHA",
    stations=stations,
    measurements=measurements,
    target_feature=target_feature,
    batch_size=best_hp["batch_size"],
    seq_length=best_hp["seq_length"]
)

_, early_stopping = create_model(
    nodes_lstm=best_hp["lstm_nodes"],
    nodes_dense=None,
    dropout=best_hp["dropout"],
    metric="r_square",
    learning_rate=best_hp["learning_rate"]
)

train_model(best_model, train_ds, val_ds, early_stopping,
            metric="r_square", epochs=best_hp["epochs"])

print("Beste Hyperparameter:")
for hp_name in best_hp.values.keys():
    print(f"{hp_name}: {best_hp.get(hp_name)}")
