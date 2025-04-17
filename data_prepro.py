import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def create_df_of_file(filename, measure, interval, directory="Data", ):
    """

    :param filename:
    :param measure:
    :param interval:
    :param directory:
    :return:
    """
    dataframe = pd.read_csv(os.path.join(directory, filename))

    # set start of dataframe to time when it was in Kenya
    time = dataframe["date"].tolist()
    start = time.index("2015-04-28 11:00:00")
    dataframe = dataframe[start:]

    # remove all non valid measure times (only measurements in 10minute steps are valid)
    dataframe = dataframe[dataframe["date"].str[-4:] == "0:00"]
    time = dataframe["date"].tolist()

    dataframe['date'] = pd.to_datetime(dataframe.pop('date'), format='%Y-%m-%d %H:%M:%S')
    dataframe.index = dataframe["date"]

    dataframe = dataframe.resample(interval).mean()

    # linear interpolation to fill NA values
    dataframe = dataframe.interpolate()

    # rename value column to actual measure category
    dataframe.rename(columns={"value": f"{filename[:3]}_{measure}"}, inplace=True) #Problem mit Datensatz NF -> if-Formulierung nötig

    dataframe.reset_index(inplace=True, drop=True)

    print(dataframe)

    return dataframe, time


def create_prec_df(time_df, station, interval):
    """
    This function is needed because the prec data had way more measurements than all the other data.
    """

    dataframe = pd.read_csv(os.path.join("Data", f"{station}-prec.csv"))
    time = dataframe["date"].tolist()
    start = time.index("2015-04-28 11:00:00")
    time = time[start:]
    dataframe = dataframe[start:]

    # linear interpolation to fill NA values
    dataframe = dataframe.interpolate("linear")

    dataframe.rename(columns={"value": f"{station}_prec"}, inplace=True)

    dataframe.set_index('date', inplace=True)

    # remove all measurement times that aren't in the data of the time_df parameter
    to_remove = list(set(time).difference(set(time_df)))
    dataframe.drop(labels=to_remove, inplace=True, errors="ignore")

    dataframe.reset_index(inplace=True, drop=False)

    dataframe['date'] = pd.to_datetime(dataframe.pop('date'), format='%Y-%m-%d %H:%M:%S')
    dataframe.index = dataframe["date"]
    dataframe.drop(labels="date", axis=1, inplace=True)

    dataframe = dataframe.resample(interval).sum()

    dataframe.reset_index(inplace=True)

    return dataframe


#"prec" ist absichtlich nicht in create_filenames vorhanden

def create_filenames(stations):
    categories_kfs = ["dir", "gust", "par", "rh", "stemp15", "temp", "vwc15", "wind"]
    categories_nf = ["disch", "doc", "elc", "nit", "tcd", "toc", "tsp", "tur", "wl"]
    categories_sha = ["disch", "doc", "elc", "nit", "tcd", "toc", "tsp", "tur", "wl"]
    categories_tf2 = ["temp"]
    categories_tf1 = ["temp"]
    categories_ttp = ["disch", "doc", "nit"]
    categories_wsh = ["dir", "ec15", "gust", "par", "rh", "stemp15", "temp", "vwc15", "wind"]
    filenames = []
    for station in stations:

        if station == "KFS":
            for category in categories_kfs:
                filenames.append(f"{station}-{category}.csv")

        elif station == "NF":
            for category in categories_nf:
                filenames.append(f"{station}-{category}.csv")

        elif station == "SHA":
            for category in categories_sha:
                filenames.append(f"{station}-{category}.csv")

        elif station == "TF2":
            for category in categories_tf2:
                filenames.append(f"{station}-{category}.csv")

        elif station == "TF1":
            for category in categories_tf1:
                filenames.append(f"{station}-{category}.csv")

        elif station == "TTP":
            for category in categories_ttp:
                filenames.append(f"{station}-{category}.csv")

        elif station == "WSH":
            for category in categories_wsh:
                filenames.append(f"{station}-{category}.csv")

    return filenames


def load_data(stations, interval= "10min"):
    filenames = create_filenames(stations)
    frames = []

    for filename in filenames:
        measure = filename.split("-")[1].split(".")[0]
        frames.append(create_df_of_file(filename, measure, interval=interval)[0])

    _, df_time = create_df_of_file("SHA-nit.csv", "nit", interval=interval)
    sha_prec_df = create_prec_df(df_time, "SHA", interval=interval)
    fun_prec_df = create_prec_df(df_time, "Fun", interval=interval)
    kur_prec_df = create_prec_df(df_time, "Kur", interval=interval)
    wsh_prec_df = create_prec_df(df_time, "WSH", interval=interval)
    cha_prec_df = create_prec_df(df_time, "Cha", interval=interval)
    chi_prec_df = create_prec_df(df_time, "Chi", interval=interval)
    fin_prec_df = create_prec_df(df_time, "Fin", interval=interval)
    kfs_prec_df = create_prec_df(df_time, "KFS", interval=interval)
    nf_prec_df = create_prec_df(df_time, "NF", interval=interval)
    tf2_prec_df = create_prec_df(df_time, "TF2", interval=interval)
    ttp_prec_df = create_prec_df(df_time, "TTP", interval=interval)
    frames.append(sha_prec_df)
    frames.append(fun_prec_df)
    frames.append(kur_prec_df)
    frames.append(wsh_prec_df)
    frames.append(cha_prec_df)
    frames.append(chi_prec_df)
    frames.append(fin_prec_df)
    frames.append(kfs_prec_df)
    frames.append(nf_prec_df)
    frames.append(tf2_prec_df)
    frames.append(ttp_prec_df)

    df = pd.concat(frames, axis=1)

    return df


def create_tf_dataset(data, target, seq_length=3, batch_size=32):
    data = data[:-seq_length]
    target = target[seq_length:]

    ds = tf.keras.utils.timeseries_dataset_from_array(data, target, sequence_length=seq_length,
                                                      sequence_stride=1,
                                                      batch_size=batch_size, )

    return ds


def create_final_ds(station, stations, target_feature, interval=None, batch_size=32, seq_length=3):

    df = load_data(stations)
    df.to_pickle(f"{station}-dataframe.pkl")

    df = df.drop([], axis=1)

    df.drop(columns=df.columns[df.columns.duplicated()], inplace=True)

    # split data into train (60%), val (20%) and test (20%) data
    n = len(df)
    train_df = df[0:int(n * 0.6)]
    val_df = df[int(n * 0.6):int(n * 0.8)]
    test_df = df[int(n * 0.8):]

    feature_train = train_df.drop([target_feature], axis=1)
    feature_val = val_df.drop([target_feature], axis=1)
    feature_test = test_df.drop([target_feature], axis=1)

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(feature_train.to_numpy())
    feature_train_scaled = feature_scaler.transform(feature_train)
    feature_val_scaled = feature_scaler.transform(feature_val)
    feature_test_scaled = feature_scaler.transform(feature_test)

    target_train = np.array(train_df[target_feature], ndmin=2).T
    target_val = np.array(val_df[target_feature], ndmin=2).T
    target_test = np.array(test_df[target_feature], ndmin=2).T

    # creating tensorflow time series datasets
    train_ds_norm = create_tf_dataset(feature_train_scaled, target_train, batch_size=batch_size, seq_length=seq_length)
    val_ds_norm = create_tf_dataset(feature_val_scaled, target_val, batch_size=batch_size, seq_length=seq_length)
    test_ds_norm = create_tf_dataset(feature_test_scaled, target_test, batch_size=batch_size, seq_length=seq_length)

    # only the first three return values are needed for training
    return train_ds_norm, val_ds_norm, test_ds_norm, train_df, test_df, val_df


train_ds, val_ds, test_ds, train_df, test_df, val_df = create_final_ds(
    "SHA", ["SHA", "WSH","TTP", "TF2", "NF", "Kur", "KFS", "Fun", "Fin", "Cha", "Che", "Chi"
            ], "SHA_nit", batch_size=32, seq_length=2, interval="24h")

print(train_df)