# Forecasting Nitrate Concentrations in Kenyan Highland Water Bodies Using Neural Networks

## Project Description

This project investigates **nitrate concentration forecasting in Kenyan highland water bodies**. Several **LSTM models (Long Short-Term Memory neural networks)** are developed and evaluated to assess their suitability for time-series forecasting of water-quality data.

The current pipeline supports **multi-target forecasting**: the default targets are nitrate at the NF, SHA, and TTP stations, and additional targets such as discharge can be enabled through the scenario configuration.

Project goals:
- Develop and compare different LSTM models
- Forecast nitrate from environmental factors (water, weather, soil, and wind data)
- Investigate the impact of including discharge in the output parameters
- Evaluate model performance using multiple performance metrics
- Provide a reproducible workflow for preprocessing, hyperparameter tuning, training, and evaluation
- data-reductionist approach to the input parameters to guarantee good models on low data input

---

## Technologies

- **Python 3.10** or **Python 3.12**
- **PyTorch** (the `requirements.txt` pins the CUDA 13.0 build `torch==2.11.0+cu130`)
- pandas, NumPy, SciPy, scikit-learn, Optuna, matplotlib, seaborn, and openpyxl
- All required packages are listed in `requirements.txt`

Install the dependencies with:

```bash
pip install -r requirements.txt
```

> If you use CPU-only PyTorch or a different CUDA version, adjust the PyTorch entry in `requirements.txt`.

---

## Requirements

- A `Data/` directory containing:
  - raw measurement time series named `<station>-<measure>.csv`
  - precipitation time series named `<station>-prec.csv`
  - `Mau_instrumentation.csv` with station coordinates and metadata
- Sufficient compute power; a GPU is recommended for training
- Optionally, access to cloud compute infrastructure

The main pipeline uses relative paths (`Data/`, `models/`, `figures/`, etc.), so run the commands from the repository root.

---

## Current Pipeline

`data_prepro.py` performs the hyperparameter-independent preprocessing:

1. Load and resample the measurement and precipitation time series (default interval: `10min`).
2. Enrich Air data with Mar data where available.
3. Impute missing precipitation values using inverse distance weighting (IDW).
4. Split the data chronologically into train, validation, and test sets (default: 60/20/20).
5. MinMax-scale the input features and transform the target variables with log1p, log(x + epsilon), Box-Cox with shift, or no transformation.
6. Build sliding-window sequences with interleaved value/mask channels. The target is the last time step of each window.

`lstm_main.py` then trains the LSTM model and evaluates it on observed target values only.

Target masks use `1 = observed` and `0 = missing/imputed`. Target transformations are inverted before metrics are computed and before full-dataset predictions are exported.

---

## Scenarios

`scenario_config.py` defines the available model scenarios:

- `benchmark`: all stations and measurements, default nitrate targets
- `disch`: default targets plus all available discharge targets
- `not_atmos`, `not_ec`, `not_prec`, `not_soil`, `not_temp`, `not_water_lvl`, `not_water_qual`, `not_wind`: exclusion scenarios for specific measurement groups
- `test_code`: minimal scenario for testing

The legacy scenarios `not_nit`, `not_lyser`, and `low_input` are not wired into the current `lstm_main.run` dispatch.

`hp_training.py` currently supports the `benchmark` and `disch` scenarios.

---

## Usage

### 1. Optional: Hyperparameter Tuning

```bash
python hp_training.py --scenario benchmark --trials 20
python hp_training.py --scenario disch --trials 20 --euclidean
```

The tuning workflow uses Optuna with the TPE sampler and stores studies in `optuna_study.db`. The best model and preprocessing artifacts are saved under:

- `models/hp_tuning/<config_name>/`
- `models/hp_tuning/<config_name>_euclidean/` for Euclidean loss runs

Saved artifacts:
- `best_model.pt`
- `best_config.json`
- `feature_scaler.pkl`
- `target_transformations.pkl`

Hyperparameter tuning only determines and saves the best configuration and artifacts. Final test evaluation and full-dataset prediction export are performed by `lstm_main.py`.

### 2. Train and Evaluate a Model

The `__main__` block at the bottom of `lstm_main.py` currently runs:

```python
run(scenario="benchmark")
```

To use another scenario, edit that call or import `run` directly:

```python
from lstm_main import run

run(scenario="benchmark")
run(scenario="disch", nitrate_only_metrics=True, per_target_metrics=True)
run(scenario="disch", euclidean=True)
```

Training and evaluation write:
- model checkpoints and split boundaries to `models/<config_name>/`
- full-dataset predictions and metadata to the same folder
- one row per trained model to `model_log.xlsx`

### 3. Visualization

`create_figures.py` is currently a legacy/outdated script. Its `__main__` block is commented out, so running `python create_figures.py` does not start the plotting workflow. Update the script or call its functions manually before relying on the generated figures.

---

## Metrics and Options

The standard metrics are MSE, RMSE, MAE, R², NSE, MBE, and KGE. Only observed target values (mask `== 1`) are used.

- `nitrate_only_metrics=True` adds nitrate-only metrics with the `_nit` suffix (e.g., `MSE_nit`, `RMSE_nit`).
- `per_target_metrics=True` adds metrics for each individual target, named `<metric>_<target>` (e.g., `MSE_NF_nit`, `KGE_SHA_disch`).
- `euclidean=True` uses the per-target Euclidean loss `(mean_t sqrt(MSE_t))^2`; the default is normal MSE.
- For tuning, select the same loss variant with `python hp_training.py --euclidean`.
- Nitrate-only and per-target metrics can be combined.

---

## Project Structure

```text
Data/                         # raw data and station metadata
models/                       # trained models and tuning artifacts
figures/                      # plots from create_figures.py (legacy)
Additional Code/              # additional scripts, not part of the main pipeline
requirements.txt              # Python dependencies
scenario_config.py            # scenario definitions and input/target selection
hp_training.py                # Optuna hyperparameter tuning
log_transformation.py         # target transformations and inverse transformations
lstm_main.py                  # main training and evaluation workflow
lstm_model.py                 # LSTM model, losses, and training loop
data_prepro.py                # data loading, preprocessing, and datasets
evaluate_model.py             # metrics and prediction export
create_figures.py             # visualization script (legacy/outdated)
README.md                     # project documentation
```

## Generated Artifacts

```text
model_log.xlsx                # training log written by lstm_main.py
optuna_study.db               # Optuna study storage

models/<config_name>/
  <model_name>.pt
  <model_name>_split_boundaries.csv
  predictions_full.npy
  y_true_full.npy
  dates_full.npy
  X_full.npy
  scaler_y.pkl
  predictions_log.npy
  y_true_log.npy

models/hp_tuning/<config_name>/
  best_model.pt
  best_config.json
  feature_scaler.pkl
  target_transformations.pkl

models/hp_tuning/<config_name>_euclidean/
  ... same artifact layout for Euclidean loss runs ...
```

These files are created at runtime and may not exist until the corresponding workflow has been run.

---

## Author

- Project by Maximilian Mäder
- Based on work by Henri Breuer
- Under the supervision of Aaron Neill
- Data collected by Dr. Suzanne Jacobs

---