import json
import argparse
import os
import pickle
from datetime import datetime

import optuna
import torch

from data_prepro import prepare_data_arrays, make_dataloaders
from lstm_model import create_model, train_model, set_seed
from scenario_config import get_benchmark_config, get_disch_config


SEED = 42
STORAGE = "sqlite:///optuna_study.db"
OUTPUT_ROOT = os.path.join("models", "hp_tuning")

SCENARIO_GETTERS = {
    "benchmark": get_benchmark_config,
    "disch": get_disch_config,
}


def parse_args():
    """
    Funktion, die Argumente aus Aufrufen in der Konsole einliest
    Arugumente:
    --scenario      Szenario, welches für Benchmark genutzt werden soll
    n_trials        Menge an Bayes-Kombinationen, die getestet werden
    """
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for LSTM scenarios"
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIO_GETTERS,
        default="benchmark",
        help="Scenario to tune (default: benchmark)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of Optuna trials (default: 20)"
    )
    return parser.parse_args()


def objective(trial, data, n_features, n_targets):
    """
    Optuna-Zielfunktion: Trainiert ein LSTM mit den vorgeschlagenen
    Hyperparametern und gibt den besten Validierungsverlust zurück.

    :param trial:      Optuna Trial-Objekt
    :param data:       Vorbereitete Arrays aus prepare_data_arrays()
    :param n_features: Anzahl der Eingabe-Features
    :param n_targets:  Anzahl der Zielvariablen
    :return:           Bester Validierungsverlust (val_loss) als float
    """
    # Jeder Trial bekommt einen reproduzierbaren, aber eigenen Seed.
    set_seed(SEED + trial.number)

    # --- Hyperparameter definieren ---
    nodes_lstm    = trial.suggest_categorical("nodes_lstm",    [10, 20, 50])
    dropout       = trial.suggest_categorical("dropout",       [0.1, 0.2, 0.3, 0.5])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-2, 1e-3, 1e-4])
    num_layers    = trial.suggest_categorical("num_layers",    [1, 2, 3])
    batch_size    = trial.suggest_categorical("batch_size",    [32, 64])
    seq_length    = trial.suggest_categorical("seq_length",    [6, 18, 72])
    epochs        = trial.suggest_categorical("epochs",        [20, 30, 50, 70])
    nodes_dense   = trial.suggest_categorical("nodes_dense",   [0, 32, 64])

    # --- Daten vorbereiten ---
    # Nur die von seq_length/batch_size abhängigen DataLoader neu bauen.
    train_ds, val_ds, test_ds = make_dataloaders(
        data["x_train"], data["x_val"], data["x_test"],
        data["y_train"], data["y_val"], data["y_test"],
        data["y_train_mask"], data["y_val_mask"], data["y_test_mask"],
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


def save_best_artifacts(study, best_model, history, data, config_name,
                        target_features, stations, measurements,
                        output_root=None):
    """
    Speichert das beste Modell sowie alle für eine spätere Wiederverwendung
    benötigten Preprocessing- und Konfigurationsartefakte.

    :param study:            Abgeschlossene Optuna-Studie
    :param best_model:       Neu trainiertes Modell mit den besten Parametern
    :param history:          Trainings-History des besten Modells
    :param data:             Dictionary aus prepare_data_arrays()
    :param config_name:      Szenario-/Konfigurationsname
    :param target_features:  Liste der Zielvariablen
    :param stations:         Verwendete Stationen
    :param measurements:     Verwendete Messgrößen pro Station
    :param output_root:      Wurzelverzeichnis für HP-Tuning-Artefakte (default: OUTPUT_ROOT)
    :return:                 (best_config, output_dir)
    """
    if output_root is None:
        output_root = OUTPUT_ROOT

    output_dir = os.path.join(output_root, config_name)
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "best_model.pt")
    scaler_path = os.path.join(output_dir, "feature_scaler.pkl")
    transformations_path = os.path.join(output_dir, "target_transformations.pkl")
    config_path = os.path.join(output_dir, "best_config.json")

    torch.save(best_model.state_dict(), model_path)

    # Feature-Scaler und Target-Transformationen für Inferenz/Retraining
    with open(scaler_path, "wb") as f:
        pickle.dump(data["scaler"], f)

    with open(transformations_path, "wb") as f:
        pickle.dump(data["target_transformations"], f)

    best_config = {
        "config_name": config_name,
        "study_name": study.study_name,
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "model_config": {
            **study.best_params,
            "early_stopped": len(history["train_loss"]),
        },
        "n_features": data["n_features"],
        "n_targets": len(target_features),
        "target_features": target_features,
        "stations": stations,
        "measurements": measurements,
        "feature_cols": data["feature_cols"],
        "split_sizes": {
            "total": len(data["df"]),
            "train": len(data["train_df"]),
            "val": len(data["val_df"]),
            "test": len(data["test_df"]),
        },
        "seed": SEED,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "artifacts": {
            "model": os.path.basename(model_path),
            "feature_scaler": os.path.basename(scaler_path),
            "target_transformations": os.path.basename(transformations_path),
        },
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)

    return best_config, output_dir


def run_hp_training(scenario: str = "benchmark", n_trials: int = 20):
    """
    Führt die Optuna-Studie aus und speichert das beste Modell samt
    vollständiger Konfiguration und Preprocessing-Objekte.

    :param scenario:        Szenario, welches für Benchmark genutzt werden soll
    :param n_trials:        Menge an Bayes-Kombinationen, die getestet werden
    """
    # Szenario laden
    stations, measurements, target_features, config_name = SCENARIO_GETTERS[scenario]()

    n_targets = len(target_features)

    set_seed(SEED)

    # Teure Vorverarbeitung einmalig ausführen; die Trials bauen nur die DataLoader.
    data = prepare_data_arrays(
        stations=stations,
        measurements=measurements,
        target_features=target_features
    )
    n_features = data["n_features"]

    # Konfigurationsspezifischer Studienname, damit Szenarien nicht kollidieren.
    study_name = f"lstm_tuning_{config_name}"
    study = optuna.create_study(
        study_name=study_name,
        storage=STORAGE,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),  # Bayessche Optimierung
        load_if_exists=True
    )

    study.set_user_attr("config_name", config_name)
    study.set_user_attr("target_features", target_features)
    study.set_user_attr("seed", SEED)

    study.optimize(
        lambda trial: objective(trial, data, n_features, n_targets),
        n_trials=n_trials
    )

    # --- Beste Parameter ausgeben ---
    print(f"Bester Validierungsverlust: {study.best_value:.4f}")
    print("Beste Hyperparameter:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # --- Bestes Modell neu trainieren ---
    best = study.best_params
    set_seed(SEED)

    train_ds, val_ds, test_ds = make_dataloaders(
        data["x_train"], data["x_val"], data["x_test"],
        data["y_train"], data["y_val"], data["y_test"],
        data["y_train_mask"], data["y_val_mask"], data["y_test_mask"],
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

    # --- Bestes Modell + vollständige Artefakte speichern ---
    best_config, output_dir = save_best_artifacts(
        study=study,
        best_model=best_model,
        history=history,
        data=data,
        config_name=config_name,
        target_features=target_features,
        stations=stations,
        measurements=measurements,
    )

    print(f"\nBestes Modell gespeichert unter: {os.path.join(output_dir, 'best_model.pt')}")
    print(f"Konfiguration gespeichert unter: {os.path.join(output_dir, 'best_config.json')}")
    print(f"Feature-Scaler gespeichert unter: {os.path.join(output_dir, 'feature_scaler.pkl')}")
    print(f"Target-Transformationen gespeichert unter: {os.path.join(output_dir, 'target_transformations.pkl')}")

    return best_config


if __name__ == "__main__":
    args = parse_args()
    run_hp_training(scenario=args.scenario, n_trials=args.trials)
