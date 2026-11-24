import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from log_transformation import log_transform_log1p, log_transform_eps, boxcox_transform, analyze_skewness

# ── Threshold: precipitation stations with a higher NaN ratio are excluded ──
MAX_PREC_NAN_RATIO = 0.20


def _load_and_trim_dataframe(filepath, valid_time_suffix="0:00"):
    """
    Loads a CSV file, filters by a specific time range and optionally by a time-of-day suffix,
    converts the dates to datetime objects, and sets the date as index.
    For Air data, the data is enriched with Mar data.

    :param filepath: Path to the CSV file
    :param valid_time_suffix: Time-of-day suffix for filtering (e.g., "0:00" for full 10-min), None disables filtering

    :return: Trimmed DataFrame with datetime index
    """
    start_date = "2015-04-28 11:00:00"
    end_date = "2019-11-21 12:00:00"

    chunk_iter = pd.read_csv(filepath,
                     dtype={"value": np.float32},
                     parse_dates=["date"],
                     chunksize=200_000
                     )

    filtered_chunks = []
    for chunk in chunk_iter:
        if valid_time_suffix:
            minutes = int(valid_time_suffix.split(":")[1])
            mask_min = chunk["date"].dt.minute % 10 == minutes
            chunk = chunk[mask_min]
        mask_date = (chunk["date"] >= pd.to_datetime(start_date)) & \
               (chunk["date"] <= pd.to_datetime(end_date))
        chunk = chunk.loc[mask_date]
        filtered_chunks.append(chunk.set_index("date", drop=True))

    df = pd.concat(filtered_chunks, axis=0, copy=False)

    # Merge Air and Mar
    filename = os.path.basename(filepath)
    if filename.startswith("Air-"):
        mar_filename = filename.replace("Air-", "Mar-")
        mar_filepath = os.path.join(os.path.dirname(filepath), mar_filename)
        if os.path.exists(mar_filepath):
            df_mar = _load_and_trim_dataframe(mar_filepath, valid_time_suffix=valid_time_suffix)
            df_mar.rename(columns={"value": "value_mar"}, inplace=True)
            df = df.join(df_mar, how='outer')
            df["value"] = df["value"].fillna(df["value_mar"])
            df.drop(columns=["value_mar"], inplace=True)

    return df


def _create_standard_measurement_df(filename, measure, interval, directory="Data"):
    """
    Creates a time-normalized DataFrame with a standardized column name from a measurement CSV.

    :param filename: Filename of the measurement series
    :param measure: Short code for the measurement (e.g., 'temp', 'ec')
    :param interval: Time interval for resampling (e.g., '10min')
    :param directory: Data directory

    :return: Prepared DataFrame with a standardized column name
    """
    filepath = os.path.join(directory, filename)
    df = _load_and_trim_dataframe(filepath)
    df = df.resample(interval).mean()

    station = filename.split("-")[0]
    colname = f"{station}_{measure}"
    df.rename(columns={"value": colname}, inplace=True)
    df[colname] = df[colname].astype('float32')

    return df


def _create_precipitation_df(station, interval, directory="Data"):
    """
    Creates a summed DataFrame from a precipitation file based on a predefined time index.

    :param station: Station short code (Identifiyer)
    :param interval: Time interval for resampling
    :param directory: Data directory

    :return: Prepared DataFrame with summed precipitation values
    """
    filepath = os.path.join(directory, f"{station}-prec.csv")
    df = _load_and_trim_dataframe(filepath, valid_time_suffix=None) #None because prec data must be summed
    df.rename(columns={"value": f"{station}_prec"}, inplace=True)

    df = df.resample(interval).sum(min_count=1)

    # Check NaN ratio
    nan_ratio = df.isna().sum().sum() / len(df)
    if nan_ratio > MAX_PREC_NAN_RATIO:
        print(f"  ⚠️  {station}_prec: {nan_ratio:.1%} NaN → Station wird ausgeschlossen")
        return None

    print(f"  ✓  {station}_prec: {nan_ratio:.1%} NaN")

    return df

def _get_station_coordinates(directory="Data"):
    """
    Reads coordinates from Mau_instrumentation.csv and creates a mapping over stations

    :param directory: Path to the folder for the file

    :return: Dictionary with station short code and lon/lat
    """
    filepath = os.path.join(directory, "Mau_instrumentation.csv")
    df = pd.read_csv(filepath)
    coords_dict = {
        row['ABB']: (float(row['lat']), float(row['lon']))
        for _, row in df.iterrows()
    }

    return coords_dict

def _impute_precipitation(df, station_coords, p = 2):
    """
    Weighted spatial interpolation (IDW) for precipitation.

    :param df: The combined DataFrame with all prec columns
    :param station_coords: Dict {station_name: (lat, lon)}
    :param p: Power parameter for weighting (higher = nearby stations dominate more)

    :return: Dataframe with all precipitation data
    """
    prec_cols = [c for c in df.columns if c.endswith("_prec")]

    for col in prec_cols:
        target_station = col.split("_")[0]
        if target_station not in station_coords:
            continue

        target_coord = np.array(station_coords[target_station])
        col_dtype = df[col].dtype.type
        mask = df[col].isna()

        if mask.any():
            for idx in df.index[mask]:
                weights = []
                values = []
                for other_col in prec_cols:
                    if other_col == col:
                        continue
                    other_station = other_col.split("_")[0]
                    if other_station not in station_coords:
                        continue
                    # Uses only stations that have a value for X
                    if not np.isnan(df.at[idx, other_col]):
                        # Computes distance (line-only implemented, Haversine would be more correct)
                        other_coord = np.array(station_coords[other_station])
                        dist = np.linalg.norm(target_coord - other_coord)
                        # Avoid division by 0
                        dist = max(dist, 0.000001)
                        weight = 1.0 / (dist ** p)
                        weights.append(weight)
                        values.append(df.at[idx, other_col])

                if weights:
                    # Weighted mean (explicitly cast to column dtype)
                    df.at[idx, col] = col_dtype(np.average(values, weights=weights))
                else:
                    df.at[idx, col] = col_dtype(0.0)

    return df

def _create_filenames(stations, measurements):
    """
    Creates a list of filenames for all station-measurement combinations.

    :param stations: List of station names
    :param measurements: Dictionary with measurement lists per station

    :return: List of filenames (e.g., 'SHA-temp.csv')
    """
    filenames = [
        f"{station}-{measure}.csv"
        for station in stations
        for measure in measurements.get(station, [])
        if measure != "prec"
    ]
    return filenames


def load_data(stations, measurements, target_features, interval="10min"):
    """
    Loads, combines, and processes all relevant time series (measurements and precipitation).

    :param stations: List of station names
    :param measurements: Dictionary with measurement lists per station
    :param target_features: List of target features
    :param interval: Time interval for resampling

    :return: Combined DataFrame of all time series (merge all other with prec) & dictionary with transformation of the target features
    """
    filenames = _create_filenames(stations, measurements)
    frames = []

    for filename in filenames:
        measure = filename.split("-")[1].split(".")[0]
        if measure == "prec":
            continue
        frames.append(_create_standard_measurement_df(filename, measure, interval=interval))

    # ── Load precipitation ──
    prec_frames = []
    for station in stations:
        if "prec" in measurements.get(station, []):
            prec_df = _create_precipitation_df(station, interval)
            if prec_df is not None:  # ← None check for excluded stations
                prec_frames.append(prec_df)

    # ── Merge and impute precipitation ──
    if prec_frames:
        prec_combined = pd.concat(prec_frames, axis=1)
        station_coords = _get_station_coordinates()
        prec_combined = _impute_precipitation(prec_combined,station_coords)
        frames.append(prec_combined)

    df = pd.concat(frames, axis=1)

    #Where values are not NA (must happen BEFORE any imputation)
    truth_mask = df.notna().astype(np.float32)

    # ── NaN handling BEFORE the transformation ──
    # Transformations (especially Box-Cox) require finite values, so fill first.
    n_before = len(df)
    nan_count_before = df.isna().sum().sum()

    # 1. Interpolation for small inner gaps (limit caps the maximum gap length)
    df = df.interpolate(method="time", limit=6)


    # 2. Fill remaining NaN rows
    df = df.ffill().bfill()
    df = df.fillna(df.mean())

    n_after = len(df)
    nan_count_after = df.isna().sum().sum()

    print(f"NaN-Bereinigung: {nan_count_before} NaN in {n_before} Zeilen"
          f" → {nan_count_after} NaN in {n_after} Zeilen"
          f" ({n_before - n_after} Zeilen entfernt)")
    # ── End of NaN handling ──

    # Skewness analysis (on the NaN-free data state)
    skew_dict = analyze_skewness(df)


    #log transformation depending on skew and share of zero values
    target_transformations = {}
    for col, stats in skew_dict.items():
        s = stats["skew"]
        pz = stats["pct_zero"]

        # Log transformations are only allowed for valid domains;
        # otherwise NaNs occur (e.g., log(negative)). In that case,
        # fall back to Box-Cox, which can handle negative values via a shift.
        min_val = df[col].min()

        if s > 1 and pz > 10 and min_val > -1:
            #log1p
            df = log_transform_log1p(df, col)
            trans_info = {"type": "log1p", "params": None}
        elif s > 1 and min_val > -1e-6:
            #log(x + 1e-6)
            df = log_transform_eps(df, col, epsilon=1e-6)
            trans_info = {"type": "log_eps", "params": {"epsilon": 1e-6}}
        elif s > 1 or 0.5 < s <= 1:
            #boxcox transformation (also a fallback for negative values)
            df, params = boxcox_transform(df, col)
            trans_info = {"type": "boxcox", "params": params}
        else:
            trans_info = {"type": "none", "params": None}

        if col in target_features:
            target_transformations[col] = trans_info

    truth_mask = truth_mask.reindex(index=df.index, columns=df.columns, fill_value=0).astype(np.float32)

    return df,truth_mask, target_transformations


def _split_dataset(df, split_ratios=(0.6, 0.2, 0.2)):
    """
    Splits the DataFrame into training, validation, and test sets according to the given ratio.

    :param df: Full DataFrame
    :param split_ratios: Tuple with (train, val, test) proportions

    :return: train_df, val_df, test_df
    """
    n = len(df)
    train_end = int(n * split_ratios[0])
    val_end = train_end + int(n * split_ratios[1])

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    return train_df, val_df, test_df


def _scale_features(train_df, val_df, test_df, target_features):
    """
    Scales all features except the target variable using a MinMax scaler.

    :param train_df: Training data
    :param val_df: Validation data
    :param test_df: Test data
    :param target_features: Name of the target variable (not to be scaled)

    :return: Scaled arrays (train, val, test) and the scaler used
    """
    scaler = MinMaxScaler()

    train_features = train_df.drop(columns=target_features)

    # ── Safety check ──
    if train_features.isna().any().any():
        nan_cols = train_features.columns[train_features.isna().any()].tolist()
        raise ValueError(
            f"NaN in Trainingsdaten vor Skalierung! Betroffene Spalten: {nan_cols}"
        )

    scaler.fit(train_features)

    train_scaled = scaler.transform(train_df.drop(columns=target_features))
    val_scaled = scaler.transform(val_df.drop(columns=target_features))
    test_scaled = scaler.transform(test_df.drop(columns=target_features))

    return train_scaled, val_scaled, test_scaled, scaler


def _interleave_inputs(x_scaled, truth_mask_values):
    """
    Creates the [value, mask, value, mask] format.

    :param x_scaled: Scaled x values
    :param truth_mask_values: Values that indicate interpolation of the variable

    :return: Masked input data (info on interpolation)
    """
    n_samples, n_features = x_scaled.shape
    x_interleaved = np.zeros((n_samples, n_features * 2), dtype=np.float32)

    for i in range(n_features):
        x_interleaved[:, 2 * i] = x_scaled[:, i]  # The scaled value
        x_interleaved[:, 2 * i + 1] = truth_mask_values[:, i]  # The 0/1 mask

    return x_interleaved



def _prepare_targets(train_df, val_df, test_df, target_features, truth_mask_df):
    """
    Extracts the target variables from the DataFrames and converts them into suitable NumPy arrays.

    :param train_df: Training data
    :param val_df: Validation data
    :param test_df: Test data
    :param target_features: Name of the target variable
    :param truth_mask_df: NA masking data

    :return: y_train, y_val, y_test as 2D arrays and their mask variants containing the NA information
    """
    y_train = train_df[target_features].values.astype(np.float32)
    y_val   = val_df[target_features].values.astype(np.float32)
    y_test  = test_df[target_features].values.astype(np.float32)
    #NA masking for loss function
    y_train_mask = truth_mask_df[target_features].loc[train_df.index].values.astype(np.float32)
    y_val_mask   = truth_mask_df[target_features].loc[val_df.index].values.astype(np.float32)
    y_test_mask  = truth_mask_df[target_features].loc[test_df.index].values.astype(np.float32)

    #Legacy code in case of problems
    #y_train = np.array(train_df[target_features], ndmin=2).T
    #y_val = np.array(val_df[target_features], ndmin=2).T
    #y_test = np.array(test_df[target_features], ndmin=2).T

    return y_train, y_val, y_test, y_train_mask, y_val_mask, y_test_mask


def make_dataloaders(
    x_train, x_val, x_test,
    y_train, y_val, y_test,
    y_train_mask, y_val_mask, y_test_mask,
    seq_length: int,
    batch_size: int,
    num_workers: int = 4
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for training, validation, and test.

    :param x_train:     Scaled input data training
    :param x_val:       Scaled input data validation
    :param x_test:      Scaled input data test
    :param y_train:     Target variable training
    :param y_val:       Target variable validation
    :param y_test:      Target variable test
    :param y_train_mask: Target variable training NA masking
    :param y_val_mask:   Target variable validation NA masking
    :param y_test_mask:  Target variable testing NA masking
    :param seq_length:  Sequence length for the LSTM
    :param batch_size:  Batch size
    :param num_workers: Number of parallel loading processes (0 = main process)

    :return: train_loader, val_loader, test_loader
    """
    train_ds = TimeSeriesDataset(x_train, y_train, y_train_mask, seq_length)
    val_ds   = TimeSeriesDataset(x_val,   y_val, y_val_mask, seq_length)
    test_ds  = TimeSeriesDataset(x_test,  y_test, y_test_mask, seq_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,          # ← True/False - can be changed - True is often common for time series in training
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,         # ← Never shuffle validation/test
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader, test_loader


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data in sliding-window format.

    Same-step reconstruction: the target value is the last time step of the window.

    :param data:       Input matrix (numpy array) [n_samples, n_features]
    :param target:     Target matrix (numpy array)    [n_samples, 1]
    :param target_mask: Masked Matrix for the Target variable (numpy array), 0=Sampled Data, 1=Interpolated
    :param seq_length: Count
    """
    def __init__(self, data: np.ndarray, target: np.ndarray,target_mask: np.ndarray, seq_length: int):
        # Sliding window: window i = data[i:i+seq_length].
        # The target is the last time step of the window (i+seq_length-1), i.e.
        # Same-step reconstruction - consistent with TimeSeriesDatasetWithTimestamps.
        if data.dtype != np.float32:
            data = data.astype(np.float32, copy=False)
        if target.dtype != np.float32:
            target = target.astype(np.float32, copy=False)
        self.data   = torch.from_numpy(
            np.lib.stride_tricks.sliding_window_view(data, seq_length, axis=0)).float()
        # Transpose feature set and seq-length into dimensions (standardization)
        if self.data.shape[2] == seq_length:
            self.data = self.data.transpose(1,2)

        # Target value is the last time step of the window
        self.target      = torch.from_numpy(target[seq_length-1:]).float()
        self.target_mask = torch.from_numpy(target_mask[seq_length-1:]).float()

        self.seq_length = seq_length

    def __len__(self) -> int:
        # Number of sliding-window lengths
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        # Return the tensor window and the target tensor
        x = self.data[idx]
        y = self.target[idx]
        mask = self.target_mask[idx]
        return x, y, mask

class TimeSeriesDatasetWithTimestamps(Dataset):
    """
    Input matrix [n_samples, n_features]. The target value is the last time step
    of the window (same-step reconstruction) - consistent with TimeSeriesDataset.

    :param data:        Input matrix (numpy array) [n_samples, n_features]
    :param target:      Target matrix   (numpy array) [n_samples, 1]
    :param timestamps:  Timestamp array (numpy, dtype datetime64 or str)
    :param seq_length:  Count
    """
    def __init__(self, data: np.ndarray, target: np.ndarray, timestamps: np.ndarray, seq_length: int):
        self.sequences  = []
        self.targets    = []
        self.timestamps = []

        for i in range(len(data) - seq_length + 1):
            self.sequences.append(data[i : i + seq_length])
            self.targets.append(target[i + seq_length - 1])
            self.timestamps.append(timestamps[i + seq_length - 1])

        self.sequences = torch.tensor(
            np.array(self.sequences), dtype=torch.float32
        )
        self.targets = torch.tensor(
            np.array(self.targets), dtype=torch.float32
        )
        # Timestamps remain as numpy/string – PyTorch does not know datetime
        self.timestamps = np.array(self.timestamps, dtype=str)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        # Timestamp is returned separately, not as a tensor
        return self.sequences[idx], self.targets[idx], self.timestamps[idx]

def prepare_data_arrays(stations: list, measurements: dict, target_features: str,
                        interval: str = "10min"):
    """
    Runs all hyperparameter-independent preprocessing steps once:
    load, transform, fill, split, scale, extract targets, and
    interleave input masks.

    The result can be used multiple times for different seq_length/batch_size
    (e.g., in the Optuna objective) without reloading the CSV files.

    :param stations:        List of Stations of the dataframe
    :param measurements:    Dictionary of Measurement of the stations
    :param target_features: String of Output-/Target-Features
    :param interval:        String indicating the time interval used in the model

    :return:            Dictionary with all prepared arrays and metadata
    """
    # Load data
    df, truth_mask, target_transformations = load_data(
        stations, measurements, target_features, interval=interval
    )

    df = df.loc[:, df.apply(pd.Series.nunique) > 0]
    df = df.apply(pd.to_numeric, errors="coerce")

    # Split
    train_df, val_df, test_df = _split_dataset(df)

    # Scaling
    x_train_scaled, x_val_scaled, x_test_scaled, scaler = _scale_features(
        train_df, val_df, test_df, target_features
    )
    y_train, y_val, y_test, y_train_mask, y_val_mask, y_test_mask = _prepare_targets(
        train_df, val_df, test_df, target_features, truth_mask
    )

    #Interleaving: combine input features with masking:
    feature_cols = [c for c in df.columns if c not in target_features]
    x_train_interleaved = _interleave_inputs(
        x_train_scaled, truth_mask.loc[train_df.index, feature_cols].values
    )
    x_val_interleaved = _interleave_inputs(
        x_val_scaled, truth_mask.loc[val_df.index, feature_cols].values
    )
    x_test_interleaved = _interleave_inputs(
        x_test_scaled, truth_mask.loc[test_df.index, feature_cols].values
    )

    return {
        "df": df,
        "truth_mask": truth_mask,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "x_train": x_train_interleaved,
        "x_val": x_val_interleaved,
        "x_test": x_test_interleaved,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_train_mask": y_train_mask,
        "y_val_mask": y_val_mask,
        "y_test_mask": y_test_mask,
        "target_transformations": target_transformations,
        "n_features": x_train_interleaved.shape[1],
    }


def create_final_ds(
    stations: list,
    measurements: dict,
    target_features: str,
    batch_size: int,
    seq_length: int,
    interval: str = "10min",
    include_full: bool = True
):
    """
    Complete pipeline: load (+ normalize) → split → scale → create DataLoader.

    The expensive, hyperparameter-independent preprocessing is moved to prepare_data_arrays()
    and can be reused for multiple calls.

    :param stations:        List of Stations of the dataframe
    :param measurements:    Dictionary of Measurement of the stations
    :param target_features: String of Output-/Target-Features
    :param seq_length:  Sequence length for the LSTM [REVIEW]
    :param batch_size:  Batch size [REVIEW]
    :param interval:        String indicating the time interval used in the model
    :param include_full: If False, the full dataset (x_full, full_dataset,
                         timestamps_full, scaler_y) is not created. It is not needed for the
                         hyperparameter search and saves a lot of memory depending on seq_length.

    :return: train_loader, val_loader, test_loader,
             train_df, test_df, val_df,
             x_full, full_dataset, timestamps_full, target_transformations, scaler_y
             (with include_full=False, x_full, full_dataset, timestamps_full, and scaler_y are None)
    """
    data = prepare_data_arrays(
        stations, measurements, target_features, interval=interval
    )

    # Create DataLoader (depends on seq_length and batch_size)
    train_loader, val_loader, test_loader = make_dataloaders(
        data["x_train"], data["x_val"], data["x_test"],
        data["y_train"], data["y_val"], data["y_test"],
        data["y_train_mask"], data["y_val_mask"], data["y_test_mask"],
        seq_length, batch_size
    )

    # Create the full dataset only when needed (e.g., not for the hyperparameter search)
    if not include_full:
        return (train_loader, val_loader, test_loader,
                data["train_df"], data["test_df"], data["val_df"],
                None, None, None, data["target_transformations"], None)

    df = data["df"]
    truth_mask = data["truth_mask"]
    feature_cols = data["feature_cols"]
    scaler = data["scaler"]

    # Scale the target variable separately
    timestamps_full = df.index.to_numpy()
    y_full          = df[target_features].values.astype(np.float32)
    scaler_y        = MinMaxScaler()
    scaler_y.fit(y_full)

    # Full input for the full dataset
    x_full_scaled = scaler.transform(df.drop(columns=target_features))
    x_full = _interleave_inputs(x_full_scaled, truth_mask[feature_cols].values)

    full_dataset = TimeSeriesDatasetWithTimestamps(
        x_full, y_full, timestamps_full, seq_length
    )

    return (train_loader, val_loader, test_loader,
            data["train_df"], data["test_df"], data["val_df"],
            x_full, full_dataset, timestamps_full, data["target_transformations"], scaler_y)