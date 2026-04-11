import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


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

    df = pd.read_csv(filepath)
    df = df[df["date"].str.endswith(valid_time_suffix)] if valid_time_suffix else df

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")
    df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]
    df.set_index("date", inplace=True)

    return df


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

    return df


def create_precipitation_df(reference_times, station, interval, directory="Data"):
    """
    Erstellt aus einer Niederschlagsdatei einen summierten, interpolierten DataFrame,
    basierend auf einem vorgegebenen Zeitindex.

    :param reference_times: Zeitindex zur Synchronisierung (z.B. von einer Referenzstation)
    :param station: Stationskürzel
    :param interval: Zeitintervall für das Resampling
    :param directory: Datenverzeichnis
    :return: Aufbereiteter DataFrame mit summierten Niederschlagswerten
    """
    filepath = os.path.join(directory, f"{station}-prec.csv")
    df = load_and_trim_dataframe(filepath, valid_time_suffix=None)
    df = df.interpolate()
    df.rename(columns={"value": f"{station}_prec"}, inplace=True)

    df = df[df.index.isin(pd.to_datetime(reference_times))]
    df = df.resample(interval).sum()

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


def load_data(stations, measurements, interval="10min"):
    """
    Lädt, kombiniert und verarbeitet alle relevanten Zeitreihen (Messwerte und Niederschlag).

    :param stations: Liste der Stationsnamen
    :param measurements: Dictionary mit Messwertlisten je Station
    :param interval: Zeitintervall für das Resampling
    :return: Kombinierter DataFrame aller Zeitreihen
    """
    filenames = create_filenames(stations, measurements)
    frames = []

    for filename in filenames:
        measure = filename.split("-")[1].split(".")[0]
        frames.append(create_standard_measurement_df(filename, measure, interval=interval))

    df_time = create_standard_measurement_df("SHA-nit.csv", "nit", interval=interval)

    for station in stations:
        if "prec" in measurements.get(station, []):
            prec_df = create_precipitation_df(df_time.index, station, interval)
            frames.append(prec_df)

    df = pd.concat(frames, axis=1)
    return df


def split_dataset(df, target_feature, split_ratios=(0.6, 0.2, 0.2)):
    """
    Teilt den DataFrame in Trainings-, Validierungs- und Testset gemäß gegebenem Verhältnis.

    :param df: Vollständiger DataFrame
    :param target_feature: Name der Zielvariable
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
    num_workers: int = 0
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
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,         # ← Validierung/Test nie shufflen
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
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
        # Analog zur alten Logik: data[:-seq_length], target[seq_length:]
        self.data   = torch.tensor(data[:-seq_length],   dtype=torch.float32)
        self.target = torch.tensor(target[seq_length:],  dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self) -> int:
        # Anzahl möglicher Sequenzen
        return len(self.data) - self.seq_length + 1

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_length]   # Shape: [seq_length, n_features]
        y = self.target[idx]                          # Shape: [1]
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
    Komplett-Pipeline: Laden → Splitten → Skalieren → DataLoader erstellen.

    :return: train_loader, val_loader, test_loader,
             train_df, test_df, val_df,
             x_full, full_dataset, timestamps_full, scaler_y
    """
    # Daten laden
    df = load_data(stations, measurements, interval=interval)
    df.to_pickle(f"{station}-dataframe.pkl")
    df.drop(columns=df.columns[df.columns.duplicated()], inplace=True)

    # Split
    train_df, val_df, test_df = split_dataset(df, target_feature)

    # Skalierung
    x_train, x_val, x_test, scaler = scale_features(
        train_df, val_df, test_df, target_feature
    )
    y_train, y_val, y_test = prepare_targets(
        train_df, val_df, test_df, target_feature
    )

    # DataLoader erstellen  ← einzige tf-Abhängigkeit ersetzt
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

    return (train_loader, val_loader, test_loader,
            train_df, test_df, val_df,
            x_full, full_dataset, timestamps_full, scaler_y)