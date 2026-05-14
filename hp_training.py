import optuna
import torch

from data_prepro import create_final_ds
from lstm_model import create_model, train_model
from benchmark_szenario_sha import get_benchmark_config


# Szenario laden
stations, measurements, target_feature, config_name = get_benchmark_config()


def objective(trial):
    """
    Optuna-Zielfunktion: Trainiert ein LSTM mit den vorgeschlagenen
    Hyperparametern und gibt den besten Validierungsverlust zurück.

    :param trial: Optuna Trial-Objekt
    :return:      Bester Validierungsverlust (val_loss) als float
    """

    # --- Hyperparameter definieren ---
    nodes_lstm    = trial.suggest_categorical("nodes_lstm",    [10, 20, 50, 100])
    dropout       = trial.suggest_categorical("dropout",       [0.1, 0.2, 0.3, 0.5])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-2, 1e-3, 1e-4])
    num_layers    = trial.suggest_categorical("num_layers",    [1,2,3])
    batch_size    = trial.suggest_categorical("batch_size",    [32, 64, 128])
    seq_length    = trial.suggest_categorical("seq_length",    [6, 18, 72, 432])
    epochs        = trial.suggest_categorical("epochs",        [20, 30, 50, 70])
    nodes_dense   = trial.suggest_categorical("nodes_dense",   [0, 32, 64])

    # --- Daten vorbereiten ---
    (train_ds, val_ds, test_ds,
     train_df, test_df, val_df,
     x_full, full_ds,
     timestamps_full, _ , _ ) = create_final_ds(
        station="SHA",
        stations=stations,
        measurements=measurements,
        target_feature=target_feature,
        batch_size=batch_size,
        seq_length=seq_length
    )

    n_features = x_full.shape[1]

    # --- Modell erstellen ---
    model, optimizer, loss_fn = create_model(
        n_features=n_features,
        nodes_lstm=nodes_lstm,
        nodes_dense=nodes_dense,
        dropout=dropout,
        learning_rate=learning_rate,
        num_layers=num_layers
    )

    # --- Training ---
    history = train_model(
        model=model,
        train_loader=train_ds,
        val_loader=val_ds,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        patience=5
    )

    # Besten Validierungsverlust zurückgeben
    return min(history["val_loss"])


# --- Studie erstellen und Suche starten ---
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42), #Bayessche Optimierung
    study_name="lstm_tuning"
)

study.optimize(objective, n_trials=20)

# --- Beste Parameter ausgeben ---
print(f"Bester Validierungsverlust: {study.best_value:.4f}")
print("Beste Hyperparameter:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# --- Bestes Modell neu trainieren ---
best = study.best_params

(train_ds, val_ds, test_ds,
 train_df, test_df, val_df,
 x_full, full_ds,
 timestamps_full, _ , _ ) = create_final_ds(
    station="SHA",
    stations=stations,
    measurements=measurements,
    target_feature=target_feature,
    batch_size=best["batch_size"],
    seq_length=best["seq_length"]
)

n_features = x_full.shape[1]

best_model, optimizer, loss_fn = create_model(
    n_features=n_features,
    nodes_lstm=best["nodes_lstm"],
    nodes_dense=best["nodes_dense"],
    dropout=best["dropout"],
    learning_rate=best["learning_rate"],
    num_layers=best["num_layers"]
)

history = train_model(
    model=best_model,
    train_loader=train_ds,
    val_loader=val_ds,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=best["epochs"],
    patience=5
)

# --- Bestes Modell speichern ---
torch.save(best_model.state_dict(), "best_model.pt")
print("\nBestes Modell gespeichert unter: best_model.pt")