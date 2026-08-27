import os
import numpy as np
import pandas as pd
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from log_transformation import log_transform_log1p, log_transform_eps, boxcox_transform, analyze_skewness

# ── Schwellenwert: Stationen mit mehr NaN-Anteil werden ausgeschlossen ──
MAX_PREC_NAN_RATIO = 0.20


def load_and_trim_dataframe(filepath, valid_time_suffix="0:00"):
    """
    Lädt eine CSV-Datei, filtert nach einem bestimmten Zeitbereich und optional nach einem Uhrzeit-Suffix,
    wandelt die Datumsangaben in datetime-Objekte um und setzt das Datum als Index.

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

    return pd.concat(filtered_chunks, axis=0, copy=False)


def create_standard_measurement_df(filename, measure, interval, directory="Data"):
    """
    Erstellt aus einer Messwert-CSV einen resampleten und interpolierten DataFrame mit standardisiertem Spaltennamen.

    :param filename: Dateiname der Messreihe
    :param measure: Kürzel des Messwertes (z.B. 'temp', 'ec')
    :param interval: Zeitintervall für das Resampling (z.B. '10min')
    :param directory: Datenverzeichnis
    :return: Aufbereiteter DataFrame mit standardisiertem Spaltennamen
    """
    filepath = os.path.join(directory, filename)
    df = load_and_trim_dataframe(filepath)
    df = df.resample(interval).mean()
    df = df.interpolate()

    station = filename.split("-")[0]
    colname = f"{station}_{measure}"
    df.rename(columns={"value": colname}, inplace=True)
    df[colname] = df[colname].astype('float32')

    return df


def create_precipitation_df(station, interval, directory="Data"):
    """
    Erstellt aus einer Niederschlagsdatei einen summierten, interpolierten DataFrame,
    basierend auf einem vorgegebenen Zeitindex.

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
    Erzeugt eine Liste an Dateinamen für alle Stations-Messwert-Kombinationen außer Niederschlag.

    :param stations: Liste der Stationsnamen
    :param measurements: Dictionary mit Messwertlisten je Station
    :return: Liste der Dateinamen (z.B. 'SHA-temp.csv')
    """
    filenames = [
        f"{station}-{measure}.csv"
        for station in stations
        for measure in measurements.get(station, [])
        if measure != "prec"
    ]
    return filenames


def load_data(stations, measurements, target_feature, interval="10min"):
    """
    Lädt, kombiniert und verarbeitet alle relevanten Zeitreihen (Messwerte und Niederschlag).

    :param stations: Liste der Stationsnamen
    :param measurements: Dictionary mit Messwertlisten je Station
    :param target_feature: Boolscher Wert, über Angabe der Logarithmisierung des target-Features
    :param interval: Zeitintervall für das Resampling
    :return: Kombinierter DataFrame aller Zeitreihen & Infos über target_feature logarithmisierung
    """
    filenames = create_filenames(stations, measurements)
    frames = []

    for filename in filenames:
        measure = filename.split("-")[1].split(".")[0]
        frames.append(create_standard_measurement_df(filename, measure, interval=interval))

    # ── Niederschlag laden ──
    print("\n=== Niederschlagsstationen ===")
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

    # Skewness-Analyse
    skew_dict = analyze_skewness(df)


    #log-Transformation je nach skew und Anteil von Nullwerten
    log_target = {"type": "none", "params": None}
    for col, stats in skew_dict.items():
        s = stats["skew"]
        pz = stats["pct_zero"]

        if s > 1 and pz > 10:
            #log1p
            df = log_transform_log1p(df, col)
            if col == target_feature:
                log_target = {"type": "log1p", "params": None}
        elif s > 1:
            #log(x + 1e-6)
            df = log_transform_eps(df, col, epsilon=1e-6)
            if col == target_feature:
                log_target = {"type": "log_eps", "params": {"epsilon": 1e-6}}
        elif 0.5 < s <= 1:
            #boxcox-transformation
            df, params = boxcox_transform(df, col)
            if col == target_feature:
                log_target = {"type": "boxcox", "params": params}
        else:
            if col == target_feature:
                log_target = {"type": "none", "params": None}

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

    return df, log_target


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


def scale_features(train_df, val_df, test_df, target_feature):
    """
    Skaliert alle Merkmale außer der Zielvariablen mithilfe eines MinMax-Scalers.

    :param train_df: Trainingsdaten
    :param val_df: Validierungsdaten
    :param test_df: Testdaten
    :param target_feature: Name der Zielvariable (nicht zu skalieren)
    :return: Skalierte Arrays (train, val, test) und der verwendete Scaler
    """
    scaler = MinMaxScaler()

    train_features = train_df.drop(columns=[target_feature])

    # ── Sicherheitscheck ──
    if train_features.isna().any().any():
        nan_cols = train_features.columns[train_features.isna().any()].tolist()
        raise ValueError(
            f"NaN in Trainingsdaten vor Skalierung! Betroffene Spalten: {nan_cols}"
        )

    scaler.fit(train_df.drop(columns=[target_feature]))

    train_scaled = scaler.transform(train_df.drop(columns=[target_feature]))
    val_scaled = scaler.transform(val_df.drop(columns=[target_feature]))
    test_scaled = scaler.transform(test_df.drop(columns=[target_feature]))

    return train_scaled, val_scaled, test_scaled, scaler


def prepare_targets(train_df, val_df, test_df, target_feature):
    """
    Extrahiert die Zielvariablen aus den DataFrames und wandelt sie in passende Numpy-Arrays um.

    :param train_df: Trainingsdaten
    :param val_df: Validierungsdaten
    :param test_df: Testdaten
    :param target_feature: Name der Zielvariable
    :return: y_train, y_val, y_test als 2D-Arrays
    """
    y_train = np.array(train_df[target_feature], ndmin=2).T
    y_val = np.array(val_df[target_feature], ndmin=2).T
    y_test = np.array(test_df[target_feature], ndmin=2).T
    return y_train, y_val, y_test


def make_dataloaders(
    x_train, x_val, x_test,
    y_train, y_val, y_test,
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
    :param seq_length:  Sequenzlänge für das LSTM
    :param batch_size:  Batchgröße
    :param num_workers: Anzahl paralleler Ladeprozesse (0 = Hauptprozess)
    :return: train_loader, val_loader, test_loader
    """
    train_ds = TimeSeriesDataset(x_train, y_train, seq_length)
    val_ds   = TimeSeriesDataset(x_val,   y_val,   seq_length)
    test_ds  = TimeSeriesDataset(x_test,  y_test,  seq_length)

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
    def __init__(self, data: np.ndarray, target: np.ndarray, seq_length: int):
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

        self.seq_length = seq_length

    def __len__(self) -> int:
        # Anzahl Sliding-Window-Länge
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        # Rückgabe des Tensor-Windows und das Ziel-Tensor
        x = self.data[idx]
        y = self.target[idx]
        return x, y

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
    target_feature: str,
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
    df, log_target = load_data(stations, measurements, target_feature, interval=interval)
    df.reset_index().to_parquet(f"{station}.parquet",engine="pyarrow", compression="snappy",index=False)
    df.drop(columns=df.columns[df.columns.duplicated()], inplace=True)
    df = pd.read_parquet(f"{station}.parquet")
    df.set_index("date", inplace=True)

    df = df.loc[:, df.apply(pd.Series.nunique) > 0]
    df = df.apply(pd.to_numeric,errors="coerce")

    # Split
    train_df, val_df, test_df = split_dataset(df)

    # Skalierung
    x_train, x_val, x_test, scaler = scale_features(
        train_df, val_df, test_df, target_feature
    )
    y_train, y_val, y_test = prepare_targets(
        train_df, val_df, test_df, target_feature
    )

    # DataLoader erstellen
    train_loader, val_loader, test_loader = make_dataloaders(
        x_train, x_val, x_test,
        y_train, y_val, y_test,
        seq_length, batch_size
    )

    # Zielvariable separat skalieren
    timestamps_full = df.index.to_numpy()
    y_full          = np.array(df[target_feature], ndmin=2).T
    scaler_y        = MinMaxScaler()
    scaler_y.fit(y_full)

    # Vollständige Eingabematrix skalieren
    x_full = scaler.transform(df.drop(columns=[target_feature]))
    x_full = np.clip(x_full, 0, 1)

    # Full-Dataset mit Zeitstempeln
    full_dataset = TimeSeriesDatasetWithTimestamps(
        x_full, y_full, timestamps_full, seq_length
    )
    print('x_train.shape :', x_train.shape)
    print('y_train.shape :', y_train.shape)

    return (train_loader, val_loader, test_loader,
            train_df, test_df, val_df,
            x_full, full_dataset, timestamps_full, log_target, scaler_y)