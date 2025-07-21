import os
import numpy as np
import pandas as pd
import tensorflow as tf
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


def make_tf_datasets(x_train, x_val, x_test, y_train, y_val, y_test, seq_length, batch_size):
    """
    Erzeugt TensorFlow-Zeitreihen-Datensätze für Training, Validierung und Test.

    :param x_train: Skalierte Eingabedaten Training
    :param x_val: Validierung
    :param x_test: Test
    :param y_train: Zielvariable Training
    :param y_val: Zielvariable Val
    :param y_test: Zielvariable Test
    :param seq_length: Sequenzlänge für das LSTM
    :param batch_size: Batchgröße für das Training
    :return: train_ds, val_ds, test_ds als TensorFlow-Datensätze
    """
    ds_train = create_tf_dataset(x_train, y_train, seq_length, batch_size)
    ds_val = create_tf_dataset(x_val, y_val, seq_length, batch_size)
    ds_test = create_tf_dataset(x_test, y_test, seq_length, batch_size)
    return ds_train, ds_val, ds_test


def create_tf_dataset(data, target, seq_length, batch_size):
    """
    Erstellt einen TensorFlow-Datensatz im Zeitreihenformat aus Eingabe- und Zielwerten.

    :param data: Eingabematrix
    :param target: Zielmatrix
    :param seq_length: Sequenzlänge (Zeitschritte pro Sample)
    :param batch_size: Größe der Batches
    :return: tf.data.Dataset im Timeseries-Format
    """
    data = data[:-seq_length]
    target = target[seq_length:]

    ds = tf.keras.utils.timeseries_dataset_from_array(
        data, target,
        sequence_length=seq_length,
        sequence_stride=1,
        batch_size=batch_size
    )
    return ds

def create_full_dataset_with_timestamps(x_full, y_full, timestamps, seq_length, batch_size):
    """
    Erstellt ein tf.data.Dataset aus Sequenzen, Targets und zugehörigen Zeitstempeln.
    """
    x_seq = []
    y_seq = []
    t_seq = []

    for i in range(len(x_full) - seq_length):
        x_seq.append(x_full[i:i + seq_length])
        y_seq.append(y_full[i + seq_length])
        t_seq.append(timestamps[i + seq_length])  # Zeitpunkt des Zielwerts

    x_seq = np.array(x_seq)
    y_seq = np.array(y_seq)
    t_seq = np.array(t_seq, dtype=str)

    # Dataset mit (x, y, timestamp)
    ds = tf.data.Dataset.from_tensor_slices((x_seq, y_seq, t_seq))
    ds = ds.batch(batch_size)

    return ds

def create_final_ds(station, stations, measurements, target_feature,
                    batch_size, seq_length, interval="10min"):
    """
    Komplett-Pipeline zur Datenerstellung: Lädt und verarbeitet alle Messwerte, skaliert sie,
    erzeugt Zielvariablen und TensorFlow-Datensätze. Speichert das vollständige DataFrame als Pickle.

    :param station: Zielstation (für Namen des Pickle-Files)
    :param stations: Liste aller beteiligten Stationen
    :param measurements: Dictionary der Messwerte pro Station
    :param target_feature: Zielvariable
    :param batch_size: Batchgröße für das LSTM
    :param seq_length: Sequenzlänge
    :param interval: Resampling-Intervall
    :return: train_ds, val_ds, test_ds, train_df, test_df, val_df,
             X_full, y_full, timestamps_full, scaler_y
    """
    df = load_data(stations, measurements, interval=interval)
    df.to_pickle(f"{station}-dataframe.pkl")
    df.drop(columns=df.columns[df.columns.duplicated()], inplace=True)

    # Neue Outputs vorbereiten
    timestamps_full = df.index.to_numpy()
    x_full = df.drop(columns=[target_feature]).to_numpy()
    y_full = np.array(df[target_feature], ndmin=2).T
    full_ds = create_full_dataset_with_timestamps(x_full, y_full, timestamps_full, seq_length, batch_size)

    # Ziel-Scaler speichern
    scaler_y = MinMaxScaler()
    scaler_y.fit(y_full)


    # Bestehende Verarbeitung
    train_df, val_df, test_df = split_dataset(df, target_feature)
    x_train, x_val, x_test, _ = scale_features(train_df, val_df, test_df, target_feature)
    y_train, y_val, y_test = prepare_targets(train_df, val_df, test_df, target_feature)
    train_ds, val_ds, test_ds = make_tf_datasets(x_train, x_val, x_test, y_train, y_val, y_test, seq_length, batch_size)

    return train_ds, val_ds, test_ds, train_df, test_df, val_df, x_full, full_ds, timestamps_full, scaler_y

