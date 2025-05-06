import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def load_and_trim_dataframe(filepath, valid_time_suffix="0:00"):
    start_date = "2015-04-28 11:00:00"

    df = pd.read_csv(filepath)

    df = df[df["date"].str.endswith(valid_time_suffix)] if valid_time_suffix else df

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")
    df = df[df["date"] >= pd.to_datetime(start_date)]

    df.set_index("date", inplace=True)
    return df

def create_standard_measurement_df(filename, measure, interval, directory="Data"):
    filepath = os.path.join(directory, filename)
    df = load_and_trim_dataframe(filepath)

    df = df.resample(interval).mean()
    df = df.interpolate()

    station = filename.split("-")[0]
    colname = f"{station}_{measure}"
    df.rename(columns={"value": colname}, inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df

def create_precipitation_df(reference_times, station, interval, directory="Data"):
    filepath = os.path.join(directory, f"{station}-prec.csv")
    df = load_and_trim_dataframe(filepath, valid_time_suffix=None)

    df = df.interpolate()
    df.rename(columns={"value": f"{station}_prec"}, inplace=True)

    df = df[df.index.isin(pd.to_datetime(reference_times))]

    df = df.resample(interval).sum()
    df.reset_index(inplace=True)

    return df


def create_filenames(stations, measurements):
    filenames = [
        f"{station}-{measure}.csv"
        for station in stations
        for measure in measurements.get(station, [])
        if measure != "prec"  # "prec" wird separat behandelt
    ]
    return filenames

def load_data(stations, measurements, interval="10min"):
    filenames = create_filenames(stations, measurements)
    frames = []

    for filename in filenames:
        measure = filename.split("-")[1].split(".")[0]
        frames.append(create_standard_measurement_df(filename, measure, interval=interval)[0])

    #Referenz-Zeitreihe für Zeitindex aller Niederschlagsdaten
    _, df_time = create_standard_measurement_df("SHA-nit.csv", "nit", interval=interval)

    for station in stations:
        if "prec" in measurements.get(station, []):
            prec_df = create_precipitation_df(df_time, station, interval)
            frames.append(prec_df)

    df = pd.concat(frames, axis=1)
    return df


def split_dataset(df, target_feature, split_ratios=(0.6, 0.2, 0.2)):
    n = len(df)
    train_end = int(n * split_ratios[0])
    val_end = train_end + int(n * split_ratios[1])

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    return train_df, val_df, test_df

def scale_features(train_df, val_df, test_df, target_feature):
    scaler = MinMaxScaler()
    scaler.fit(train_df.drop(columns=[target_feature]))

    train_scaled = scaler.transform(train_df.drop(columns=[target_feature]))
    val_scaled = scaler.transform(val_df.drop(columns=[target_feature]))
    test_scaled = scaler.transform(test_df.drop(columns=[target_feature]))

    return train_scaled, val_scaled, test_scaled, scaler

def prepare_targets(train_df, val_df, test_df, target_feature):
    y_train = np.array(train_df[target_feature], ndmin=2).T
    y_val = np.array(val_df[target_feature], ndmin=2).T
    y_test = np.array(test_df[target_feature], ndmin=2).T
    return y_train, y_val, y_test

def make_tf_datasets(x_train, x_val, x_test, y_train, y_val, y_test, seq_length, batch_size):
    ds_train = create_tf_dataset(x_train, y_train, seq_length, batch_size)
    ds_val = create_tf_dataset(x_val, y_val, seq_length, batch_size)
    ds_test = create_tf_dataset(x_test, y_test, seq_length, batch_size)
    return ds_train, ds_val, ds_test

def create_tf_dataset(data, target, seq_length=3, batch_size=32):
    data = data[:-seq_length]
    target = target[seq_length:]

    ds = tf.keras.utils.timeseries_dataset_from_array(data, target, sequence_length=seq_length,
                                                      sequence_stride=1,
                                                      batch_size=batch_size, )

    return ds

def create_final_ds(station, stations, measurements, target_feature,
                    batch_size, seq_length, interval="10min"):
    df = load_data(stations, measurements, interval=interval)
    df.to_pickle(f"{station}-dataframe.pkl")
    df.drop(columns=df.columns[df.columns.duplicated()], inplace=True)

    train_df, val_df, test_df = split_dataset(df, target_feature)
    x_train, x_val, x_test, _ = scale_features(train_df, val_df, test_df, target_feature)
    y_train, y_val, y_test = prepare_targets(train_df, val_df, test_df, target_feature)
    train_ds, val_ds, test_ds = make_tf_datasets(x_train, x_val, x_test, y_train, y_val, y_test, seq_length, batch_size)

    return train_ds, val_ds, test_ds, train_df, test_df, val_df

from benchmark_szenario import get_benchmark_config

stations, measurements, target_feature = get_benchmark_config()

train_ds, val_ds, test_ds, train_df, test_df, val_df = create_final_ds(
    station="SHA",
    stations=stations,
    target_feature=target_feature,
    batch_size=32,
    seq_length=2,
    measurements=measurements
)

from lstm_model import create_model, train_model

model, early_stopping = create_model(
    nodes_lstm= 20,
    nodes_dense= None,
    dropout= 0.1,
    metric= "r_square",
    learning_rate= 0.001,
)

