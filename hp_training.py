import optuna
import torch

from data_prepro import prepare_data_arrays, make_dataloaders
from lstm_model import create_model, train_model
from scenario_config import get_benchmark_config


# Szenario laden
stations, measurements, target_features, config_name = get_benchmark_config()
n_targets = len(target_features)

# Teure Vorverarbeitung einmalig ausführen; die Trials bauen nur noch die DataLoader.
DATA = prepare_data_arrays(
    stations=stations,
    measurements=measurements,
    target_features=target_features
)
n_features = DATA["n_features"]


def objective(trial):
    """
    Optuna-Zielfunktion: Trainiert ein LSTM mit den vorgeschlagenen
    Hyperparametern und gibt den besten Validierungsverlust zurück.

    :param trial: Optuna Trial-Objekt
    :return:      Bester Validierungsverlust (val_loss) als float
    """

    # --- Hyperparameter definieren ---
    nodes_lstm    = trial.suggest_categorical("nodes_lstm",    [10, 20, 50])
    dropout       = trial.suggest_categorical("dropout",       [0.1, 0.2, 0.3, 0.5])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-2, 1e-3, 1e-4])
    num_layers    = trial.suggest_categorical("num_layers",    [1,2,3])
    batch_size    = trial.suggest_categorical("batch_size",    [32, 64])
    seq_length    = trial.suggest_categorical("seq_length",    [6, 18, 72])
    epochs        = trial.suggest_categorical("epochs",        [20, 30, 50, 70])
    nodes_dense   = trial.suggest_categorical("nodes_dense",   [0, 32, 64])

    # --- Daten vorbereiten ---
    # Nur die von seq_length/batch_size abhängigen DataLoader neu bauen.
    train_ds, val_ds, test_ds = make_dataloaders(
        DATA["x_train"], DATA["x_val"], DATA["x_test"],
        DATA["y_train"], DATA["y_val"], DATA["y_test"],
        DATA["y_train_mask"], DATA["y_val_mask"], DATA["y_test_mask"],
        seq_length=seq_length,
        batch_size=batch_size
    )

    # --- Modell erstellen ---
    model, optimizer, loss_fn = create_model(
        n_features=n_features,
        n_targets=n_targets,
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
    study_name="lstm_tuning",
    storage="sqlite:///optuna_study.db",
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42), #Bayessche Optimierung
    load_if_exists=True
)

study.optimize(objective, n_trials=20)

# --- Beste Parameter ausgeben ---
print(f"Bester Validierungsverlust: {study.best_value:.4f}")
print("Beste Hyperparameter:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# --- Bestes Modell neu trainieren ---
best = study.best_params

train_ds, val_ds, test_ds = make_dataloaders(
    DATA["x_train"], DATA["x_val"], DATA["x_test"],
    DATA["y_train"], DATA["y_val"], DATA["y_test"],
    DATA["y_train_mask"], DATA["y_val_mask"], DATA["y_test_mask"],
    seq_length=best["seq_length"],
    batch_size=best["batch_size"]
)

best_model, optimizer, loss_fn = create_model(
    n_features=n_features,
    n_targets=n_targets,
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