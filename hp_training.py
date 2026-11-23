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


def _parse_args():
    """
    Function that reads arguments from console calls.
    Arguments:
    --scenario      Scenario to be used for the benchmark
    --trials        Number of Bayesian combinations to be tested
    --euclidean     Loss-Function with Euclidean Distance
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
    parser.add_argument(
        "--euclidean",
        action="store_true",
        help=(
            "Use the per-target Euclidean MSE aggregation "
            "(mean_t sqrt(MSE_t))^2 instead of the global MSE."
        )
    )
    return parser.parse_args()


def _objective(trial, data, n_features, n_targets, euclidean: bool = False):
    """
    Optuna objective function: Trains an LSTM with the proposed
    hyperparameters and returns the best validation loss.

    :param trial:      Optuna trial object
    :param data:       Prepared arrays from prepare_data_arrays()
    :param n_features: Number of input features
    :param n_targets:  Number of target variables
    :param euclidean:  If True, the per-target Euclidean MSE variant is used
                       as the loss.
    :return:           Best validation loss (val_loss) as float
    """
    # Each trial gets a reproducible but separate seed.
    set_seed(SEED + trial.number)

    # --- Define hyperparameters ---
    nodes_lstm    = trial.suggest_categorical("nodes_lstm",    [10, 20, 50])
    dropout       = trial.suggest_categorical("dropout",       [0.1, 0.2, 0.3, 0.5])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-2, 1e-3, 1e-4])
    num_layers    = trial.suggest_categorical("num_layers",    [1, 2, 3])
    batch_size    = trial.suggest_categorical("batch_size",    [32, 64])
    seq_length    = trial.suggest_categorical("seq_length",    [6, 18, 72])
    epochs        = trial.suggest_categorical("epochs",        [20, 30, 50, 70])
    nodes_dense   = trial.suggest_categorical("nodes_dense",   [0, 32, 64])

    # --- Prepare data ---
    # Only rebuild the DataLoaders that depend on seq_length/batch_size.
    train_ds, val_ds, test_ds = make_dataloaders(
        data["x_train"], data["x_val"], data["x_test"],
        data["y_train"], data["y_val"], data["y_test"],
        data["y_train_mask"], data["y_val_mask"], data["y_test_mask"],
        seq_length=seq_length,
        batch_size=batch_size
    )

    # --- Create model ---
    model, optimizer, loss_fn = create_model(
        n_features=n_features,
        n_targets=n_targets,
        nodes_lstm=nodes_lstm,
        nodes_dense=nodes_dense,
        dropout=dropout,
        learning_rate=learning_rate,
        num_layers=num_layers,
        euclidean=euclidean
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

    # Return the best validation loss
    return min(history["val_loss"])


def _save_best_artifacts(study, best_model, history, data, config_name,
                        target_features, stations, measurements,
                        output_root=None, euclidean: bool = False):
    """
    Saves the best model as well as all preprocessing and configuration
    artifacts needed for later reuse.

    :param study:            Completed Optuna study
    :param best_model:       Newly trained model with the best parameters
    :param history:          Training history of the best model
    :param data:             Dictionary from prepare_data_arrays()
    :param config_name:      Scenario/configuration name
    :param target_features:  List of target variables
    :param stations:         Stations used
    :param measurements:     Measurements used per station
    :param output_root:      Root directory for HP-tuning artifacts (default: OUTPUT_ROOT)
    :param euclidean:        If True, the artifacts are stored in a separate
                             subdirectory so that MSE and Euclidean runs do not
                             overwrite each other.
    :return:                 (best_config, output_dir)
    """
    if output_root is None:
        output_root = OUTPUT_ROOT

    run_name = f"{config_name}_euclidean" if euclidean else config_name
    output_dir = os.path.join(output_root, run_name)
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "best_model.pt")
    scaler_path = os.path.join(output_dir, "feature_scaler.pkl")
    transformations_path = os.path.join(output_dir, "target_transformations.pkl")
    config_path = os.path.join(output_dir, "best_config.json")

    torch.save(best_model.state_dict(), model_path)

    # Feature scaler and target transformations for inference/retraining
    with open(scaler_path, "wb") as f:
        pickle.dump(data["scaler"], f)

    with open(transformations_path, "wb") as f:
        pickle.dump(data["target_transformations"], f)

    best_config = {
        "config_name": config_name,
        "run_name": run_name,
        "euclidean": euclidean,
        "study_name": study.study_name,
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "model_config": {
            **study.best_params,
            "early_stopped": len(history["train_loss"]),
            "euclidean": euclidean,
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


def run_hp_training(scenario: str = "benchmark", n_trials: int = 20,
                    euclidean: bool = False):
    """
    Runs the Optuna study and saves the best model together with the full
    configuration and preprocessing objects.

    :param scenario:        Scenario to be used for the benchmark
    :param n_trials:        Number of Bayesian combinations to be tested
    :param euclidean:       If True, the per-target Euclidean MSE variant is used
                            as the loss. MSE and Euclidean runs are stored in
                            separate Optuna studies and output directories.
    """
    # Load scenario
    stations, measurements, target_features, config_name = SCENARIO_GETTERS[scenario]()

    n_targets = len(target_features)

    set_seed(SEED)

    # Run expensive preprocessing once; the trials only build the DataLoaders.
    data = prepare_data_arrays(
        stations=stations,
        measurements=measurements,
        target_features=target_features
    )
    n_features = data["n_features"]

    # Configuration-specific study name so that scenarios and loss variants
    # do not collide. MSE and Euclidean have different loss scales
    # and therefore must not be mixed in the same study.
    study_name = f"lstm_tuning_{config_name}" + ("_euclidean" if euclidean else "")
    study = optuna.create_study(
        study_name=study_name,
        storage=STORAGE,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),  # Bayesian optimization
        load_if_exists=True
    )

    study.set_user_attr("config_name", config_name)
    study.set_user_attr("target_features", target_features)
    study.set_user_attr("seed", SEED)
    study.set_user_attr("euclidean", euclidean)

    study.optimize(
        lambda trial: _objective(trial, data, n_features, n_targets,
                                 euclidean=euclidean),
        n_trials=n_trials
    )

    # --- Print best parameters ---
    print(f"Bester Validierungsverlust: {study.best_value:.4f}")
    print("Beste Hyperparameter:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # --- Retrain the best model ---
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
        num_layers=best["num_layers"],
        euclidean=euclidean
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

    # --- Save best model + full artifacts ---
    best_config, output_dir = _save_best_artifacts(
        study=study,
        best_model=best_model,
        history=history,
        data=data,
        config_name=config_name,
        target_features=target_features,
        stations=stations,
        measurements=measurements,
        euclidean=euclidean,
    )

    print(f"\nBestes Modell gespeichert unter: {os.path.join(output_dir, 'best_model.pt')}")
    print(f"Konfiguration gespeichert unter: {os.path.join(output_dir, 'best_config.json')}")
    print(f"Feature-Scaler gespeichert unter: {os.path.join(output_dir, 'feature_scaler.pkl')}")
    print(f"Target-Transformationen gespeichert unter: {os.path.join(output_dir, 'target_transformations.pkl')}")

    return best_config


if __name__ == "__main__":
    args = _parse_args()
    run_hp_training(
        scenario=args.scenario,
        n_trials=args.trials,
        euclidean=args.euclidean
    )
