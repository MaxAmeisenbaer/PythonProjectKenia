import os
import numpy as np
import pandas as pd
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from log_transformation import log_transform_log1p, log_transform_eps, boxcox_transform, analyze_skewness

# ── Schwellenwert: Precipitation-Stationen mit mehr NaN-Anteil werden ausgeschlossen ──
MAX_PREC_NAN_RATIO = 0.20


def load_and_trim_dataframe(filepath, valid_time_suffix="0:00"):
    """
    Lädt eine CSV-Datei, filtert nach einem bestimmten Zeitbereich und optional nach einem Uhrzeit-Suffix,
    wandelt die Datumsangaben in datetime-Objekte um und setzt das Datum als Index.
    Bei Air-Daten werden die Daten mit Mar-Daten angereichert.

    :param filepath: Pfad zur CSV-Datei
    :param valid_time_suffix: Uhrzeit-Suffix für die Filterung (z.B. "0:00" für volle 10-min), None deaktiviert Filter
    :return: Getrimmter DataFrame mit datetime-Index
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

    # Zusammenführen Air und Mar
    filename = os.path.basename(filepath)
    if filename.startswith("Air-"):
        mar_filename = filename.replace("Air-", "Mar-")
        mar_filepath = os.path.join(os.path.dirname(filepath), mar_filename)
        if os.path.exists(mar_filepath):
            df_mar = load_and_trim_dataframe(mar_filepath, valid_time_suffix=valid_time_suffix)
            df_mar.rename(columns={"value": "value_mar"}, inplace=True)
            df = df.join(df_mar, how='outer')
            df["value"] = df["value"].fillna(df["value_mar"])
            df.drop(columns=["value_mar"], inplace=True)

    return df


def create_standard_measurement_df(filename, measure, interval, directory="Data"):
    """
    Erstellt aus einer Messwert-CSV einen zeitnormalisierten DataFrame mit standardisiertem Spaltennamen.

    :param filename: Dateiname der Messreihe
    :param measure: Kürzel des Messwertes (z.B. 'temp', 'ec')
    :param interval: Zeitintervall für das Resampling (z.B. '10min')
    :param directory: Datenverzeichnis
    :return: Aufbereiteter DataFrame mit standardisiertem Spaltennamen
    """
    filepath = os.path.join(directory, filename)
    df = load_and_trim_dataframe(filepath)
    df = df.resample(interval).mean()

    station = filename.split("-")[0]
    colname = f"{station}_{measure}"
    df.rename(columns={"value": colname}, inplace=True)
    df[colname] = df[colname].astype('float32')

    return df


def create_precipitation_df(station, interval, directory="Data"):
    """
    Erstellt aus einer Niederschlagsdatei einen summierten DataFrame, basierend auf einem vorgegebenen Zeitindex.

    :param station: Stationskürzel
    :param interval: Zeitintervall für das Resampling
    :param directory: Datenverzeichnis
    :return: Aufbereiteter DataFrame mit summierten Niederschlagswerten
    """
    filepath = os.path.join(directory, f"{station}-prec.csv")
    df = load_and_trim_dataframe(filepath, valid_time_suffix=None) #None, weil Prec-Daten summiert werden müssen
    df.rename(columns={"value": f"{station}_prec"}, inplace=True)

    df = df.resample(interval).sum(min_count=1)

    # Prüfe NaN-Anteil
    nan_ratio = df.isna().sum().sum() / len(df)
    if nan_ratio > MAX_PREC_NAN_RATIO:
        print(f"  ⚠️  {station}_prec: {nan_ratio:.1%} NaN → Station wird ausgeschlossen")
        return None

    print(f"  ✓  {station}_prec: {nan_ratio:.1%} NaN")

    return df

def impute_precipitation(df):
    """
    Füllt NaN in Niederschlagsspalten mit dem Mittelwert der anderen Stationen
    zum gleichen Zeitpunkt (räumliche Interpolation).
    Ausnahmefall: Falls alle Stationen gleichzeitig NaN haben → 0.
    """
    prec_cols = [c for c in df.columns if c.endswith("_prec")]

    if len(prec_cols) <= 1:
        df[prec_cols] = df[prec_cols].fillna(0)
        return df

    n_imputed_total = 0
    for col in prec_cols:
        other_cols = [c for c in prec_cols if c != col]
        mask = df[col].isna()
        n_imputed = mask.sum()

        if n_imputed > 0:
            df.loc[mask, col] = df.loc[mask, other_cols].mean(axis=1)
            n_imputed_total += n_imputed

    # Falls ALLE Stationen gleichzeitig NaN hatten → 0
    remaining_nan = df[prec_cols].isna().sum().sum()
    if remaining_nan > 0:
        df[prec_cols] = df[prec_cols].fillna(0)

    print(f"\n  Niederschlags-Imputation:")
    print(f"    {n_imputed_total} Werte durch Stationsmittel ersetzt")
    print(f"    {remaining_nan} Restwerte mit 0 gefüllt (alle Stationen gleichzeitig NaN)")

    return df

def create_filenames(stations, measurements):
    """
    Erzeugt eine Liste an Dateinamen für alle Stations-Messwert-Kombinationen.

    :param stations: Liste der Stationsnamen
    :param measurements: Dictionary mit Messwertlisten je Station
    :return: Liste der Dateinamen (z.B. 'SHA-temp.csv')
    """
    filenames = [
        f"{station}-{measure}.csv"
        for station in stations
        for measure in measurements.get(station, [])
    ]
    return filenames


def load_data(stations, measurements, target_features, interval="10min"):
    """
    Lädt, kombiniert und verarbeitet alle relevanten Zeitreihen (Messwerte und Niederschlag).

    :param stations: Liste der Stationsnamen
    :param measurements: Dictionary mit Messwertlisten je Station
    :param target_features: Liste von Target-Features
    :param interval: Zeitintervall für das Resampling
    :return: Kombinierter DataFrame aller Zeitreihen & Dictionary über Transformation der target-Features
    """
    filenames = create_filenames(stations, measurements)
    frames = []

    for filename in filenames:
        measure = filename.split("-")[1].split(".")[0]
        frames.append(create_standard_measurement_df(filename, measure, interval=interval))

    # ── Niederschlag laden ──
    prec_frames = []
    for station in stations:
        if "prec" in measurements.get(station, []):
            prec_df = create_precipitation_df(station, interval)
            if prec_df is not None:  # ← None-Check für ausgeschlossene Stationen
                prec_frames.append(prec_df)

    # ── Niederschlag zusammenführen und imputieren ──
    if prec_frames:
        prec_combined = pd.concat(prec_frames, axis=1)
        prec_combined = impute_precipitation(prec_combined)
        frames.append(prec_combined)

    df = pd.concat(frames, axis=1)

    #An welchen Stellen ist nicht NA
    truth_mask = df.notna().astype(np.float32)

    # Skewness-Analyse
    skew_dict = analyze_skewness(df)


    #log-Transformation je nach skew und Anteil von Nullwerten
    target_transformations = {}
    for col, stats in skew_dict.items():
        s = stats["skew"]
        pz = stats["pct_zero"]

        if s > 1 and pz > 10:
            #log1p
            df = log_transform_log1p(df, col)
            if col in target_features:
                target_transformations[col] = {"type": "log1p", "params": None}
        elif s > 1:
            #log(x + 1e-6)
            df = log_transform_eps(df, col, epsilon=1e-6)
            if col in target_features:
                target_transformations[col] = {"type": "log_eps", "params": {"epsilon": 1e-6}}
        elif 0.5 < s <= 1:
            #boxcox-transformation
            df, params = boxcox_transform(df, col)
            if col in target_features:
                target_transformations[col] = {"type": "boxcox", "params": params}
        else:
            if col in target_features:
                target_transformations[col] = {"type": "none", "params": None}

    # ── NaN-Behandlung nach Zusammenführung ──
    n_before = len(df)
    nan_count_before = df.isna().sum().sum()

    # 1. Interpolation für kleine innere Lücken (limit begrenzt max. Lückenlänge)
    df = df.interpolate(method="time", limit=6)


    # 2. Restliche NaN-Zeilen füllen
    df = df.ffill().bfill()
    df = df.fillna(df.mean())

    n_after = len(df)
    nan_count_after = df.isna().sum().sum()

    print(f"NaN-Bereinigung: {nan_count_before} NaN in {n_before} Zeilen"
          f" → {nan_count_after} NaN in {n_after} Zeilen"
          f" ({n_before - n_after} Zeilen entfernt)")
    # ── Ende NaN-Behandlung ──

    truth_mask = truth_mask.reindex(index=df.index, columns=df.columns, fill_value=0).astype(np.float32)

    return df,truth_mask, target_transformations


def split_dataset(df, split_ratios=(0.6, 0.2, 0.2)):
    """
    Teilt den DataFrame in Trainings-, Validierungs- und Testset gemäß gegebenem Verhältnis.

    :param df: Vollständiger DataFrame
    :param split_ratios: Tuple mit (Train, Val, Test)-Anteilen
    :return: train_df, val_df, test_df
    """
    n = len(df)
    train_end = int(n * split_ratios[0])
    val_end = train_end + int(n * split_ratios[1])

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    return train_df, val_df, test_df


def scale_features(train_df, val_df, test_df, target_features):
    """
    Skaliert alle Merkmale außer der Zielvariablen mithilfe eines MinMax-Scalers.

    :param train_df: Trainingsdaten
    :param val_df: Validierungsdaten
    :param test_df: Testdaten
    :param target_features: Name der Zielvariable (nicht zu skalieren)
    :return: Skalierte Arrays (train, val, test) und der verwendete Scaler
    """
    scaler = MinMaxScaler()

    train_features = train_df.drop(columns=target_features)

    # ── Sicherheitscheck ──
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


def interleave_inputs(x_scaled, truth_mask_values):
    """
    Erstellt das [Wert, Maske, Wert, Maske] Format.
    :param x_scaled: Skalierte x-Werte
    :param truth_mask_values: Werte, die Interpolierung der Variable angeben
    :return: Maskierte Input-Daten (Info zu Interpolierung
    """
    n_samples, n_features = x_scaled.shape
    x_interleaved = np.zeros((n_samples, n_features * 2), dtype=np.float32)

    for i in range(n_features):
        x_interleaved[:, 2 * i] = x_scaled[:, i]  # Der skalierte Wert
        x_interleaved[:, 2 * i + 1] = truth_mask_values[:, i]  # Die 0/1 Maske

    return x_interleaved



def prepare_targets(train_df, val_df, test_df, target_features, truth_mask_df):
    """
    Extrahiert die Zielvariablen aus den DataFrames und wandelt sie in passende Numpy-Arrays um.

    :param train_df: Trainingsdaten
    :param val_df: Validierungsdaten
    :param test_df: Testdaten
    :param target_features: Name der Zielvariable
    :param truth_mask_df: NA-Masking-Daten
    :return: y_train, y_val, y_test als 2D-Arrays und deren mask-varianten, die die NA-Infos haben
    """
    y_train = train_df[target_features].values.astype(np.float32)
    y_val   = val_df[target_features].values.astype(np.float32)
    y_test  = test_df[target_features].values.astype(np.float32)
    #NA-Masking für Loss-Funktion
    y_train_mask = truth_mask_df[target_features].loc[train_df.index].values.astype(np.float32)
    y_val_mask   = truth_mask_df[target_features].loc[val_df.index].values.astype(np.float32)
    y_test_mask  = truth_mask_df[target_features].loc[test_df.index].values.astype(np.float32)

    #Legacy-Code, falls Probleme
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
    Erzeugt PyTorch DataLoader für Training, Validierung und Test.

    :param x_train:     Skalierte Eingabedaten Training
    :param x_val:       Skalierte Eingabedaten Validierung
    :param x_test:      Skalierte Eingabedaten Test
    :param y_train:     Zielvariable Training
    :param y_val:       Zielvariable Validierung
    :param y_test:      Zielvariable Test
    :param y_train_mask: Zielvariable Training NA-Masking
    :param y_val_mask:   Zielvariable Validation NA-Masking
    :param y_test_mask:  Zielvariable Testing NA-Masking
    :param seq_length:  Sequenzlänge für das LSTM
    :param batch_size:  Batchgröße
    :param num_workers: Anzahl paralleler Ladeprozesse (0 = Hauptprozess)
    :return: train_loader, val_loader, test_loader
    """
    train_ds = TimeSeriesDataset(x_train, y_train, y_train_mask, seq_length)
    val_ds   = TimeSeriesDataset(x_val,   y_val, y_val_mask, seq_length)
    test_ds  = TimeSeriesDataset(x_test,  y_test, y_test_mask, seq_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,          # ← True/False - kann geändert werden - True bei Zeitreihen in Training oft üblich
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,         # ← Validierung/Test nie shufflen
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
    PyTorch Dataset für Zeitreihendaten im Sliding-Window-Format.

    :param data:       Eingabematrix (numpy array) [n_samples, n_features]
    :param target:     Zielmatrix (numpy array)    [n_samples, 1]
    :param seq_length: Anz np.ndarray, seq_length: int):
    """
    def __init__(self, data: np.ndarray, target: np.ndarray,target_mask: np.ndarray, seq_length: int):
        # Analog zu: data[:-seq_length], target[seq_length:]
        if data.dtype != np.float32:
            data = data.astype(np.float32, copy=False)
        if target.dtype != np.float32:
            target = target.astype(np.float32, copy=False)
        self.data   = torch.from_numpy(
            np.lib.stride_tricks.sliding_window_view(data, seq_length, axis=0)).float()
        self.target = torch.from_numpy(
            target[seq_length-1:]).float() #vllt zusätzlich .reshape(-1, 1)
        # Transposieren von Feature-Menge und Seq-Length in Dimensionen (Standartisierung)
        if self.data.shape[2] == seq_length:
            self.data = self.data.transpose(1,2)

        #Masking
        self.target = torch.from_numpy(target[seq_length-1:]).float()
        self.target_mask = torch.from_numpy(target_mask[seq_length-1:]).float()

        self.seq_length = seq_length

    def __len__(self) -> int:
        # Anzahl Sliding-Window-Länge
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        # Rückgabe des Tensor-Windows und das Ziel-Tensor
        x = self.data[idx]
        y = self.target[idx]
        mask = self.target_mask[idx]
        return x, y, mask

class TimeSeriesDatasetWithTimestamps(Dataset):
    """
    Eingabematrix [n_samples, n_features]
    :param target:      Zielmatrix   (numpy array) [n_samples, 1]
    :param timestamps:  Zeitstempel-Array (numpy, dtype datetime64 oder str)
    :param seq_length:  Anz int):
    """
    def __init__(self, data: np.ndarray, target: np.ndarray, timestamps: np.ndarray, seq_length: int):
        self.sequences  = []
        self.targets    = []
        self.timestamps = []

        for i in range(len(data) - seq_length):
            self.sequences.append(data[i : i + seq_length])
            self.targets.append(target[i + seq_length])
            self.timestamps.append(timestamps[i + seq_length])

        self.sequences = torch.tensor(
            np.array(self.sequences), dtype=torch.float32
        )
        self.targets = torch.tensor(
            np.array(self.targets), dtype=torch.float32
        )
        # Zeitstempel bleiben als numpy/string – PyTorch kennt kein datetime
        self.timestamps = np.array(self.timestamps, dtype=str)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        # Zeitstempel wird separat zurückgegeben, nicht als Tensor
        return self.sequences[idx], self.targets[idx], self.timestamps[idx]

def create_final_ds(
    station: str,
    stations: list,
    measurements: dict,
    target_features: str,
    batch_size: int,
    seq_length: int,
    interval: str = "10min"
):
    """
    Komplett-Pipeline: Laden (+ normalisieren) → Splitten → Skalieren → DataLoader erstellen.

    :return: train_loader, val_loader, test_loader,
             train_df, test_df, val_df,
             x_full, full_dataset, timestamps_full, log_target, scaler_y
    """
    # Daten laden (parquet aus Performance-Gründen)
    df, truth_mask, target_transformations = load_data(stations, measurements, target_features, interval=interval)
    df.reset_index().to_parquet(f"{station}.parquet",engine="pyarrow", compression="snappy",index=False)
    df.drop(columns=df.columns[df.columns.duplicated()], inplace=True)
    df = pd.read_parquet(f"{station}.parquet")
    df.set_index("date", inplace=True)

    truth_mask = truth_mask.reindex(index=df.index, columns=df.columns, fill_value=0).astype(np.float32)

    df = df.loc[:, df.apply(pd.Series.nunique) > 0]
    df = df.apply(pd.to_numeric,errors="coerce")

    # Split
    train_df, val_df, test_df = split_dataset(df)

    # Skalierung
    x_train_scaled, x_val_scaled, x_test_scaled, scaler = scale_features(
        train_df, val_df, test_df, target_features
    )
    y_train, y_val, y_test, y_train_mask, y_val_mask, y_test_mask = prepare_targets(
        train_df, val_df, test_df, target_features, truth_mask
    )

    #Interleaving: Kombinieren Input-Feature mit Masking:
    feature_cols = [c for c in df.columns if c not in target_features]
    x_train_interleaved = interleave_inputs(x_train_scaled, truth_mask.loc[train_df.index, feature_cols].values)
    x_val_interleaved = interleave_inputs(x_val_scaled, truth_mask.loc[val_df.index, feature_cols].values)
    x_test_interleaved = interleave_inputs(x_test_scaled, truth_mask.loc[test_df.index, feature_cols].values)

    # DataLoader erstellen
    train_loader, val_loader, test_loader = make_dataloaders(
        x_train_interleaved, x_val_interleaved, x_test_interleaved,
        y_train, y_val, y_test,
        y_train_mask, y_val_mask, y_test_mask,
        seq_length, batch_size
    )

    # Zielvariable separat skalieren
    timestamps_full = df.index.to_numpy()
    y_full          = df[target_features].values.astype(np.float32)
    scaler_y        = MinMaxScaler()
    scaler_y.fit(y_full)

    # Vollständige Eingabe für full-Dataset
    x_full_scaled = scaler.transform(df.drop(columns=target_features))
    x_full = interleave_inputs(x_full_scaled, truth_mask[feature_cols].values)

    full_dataset = TimeSeriesDatasetWithTimestamps(
        x_full, y_full, timestamps_full, seq_length
    )

    return (train_loader, val_loader, test_loader,
            train_df, test_df, val_df,
            x_full, full_dataset, timestamps_full, target_transformations, scaler_y)