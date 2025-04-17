import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


#"prec" ist absichtlich nicht in create_filenames vorhanden

def create_filenames(stations):
    categories_air = ["dir", "ec15", "ec30", "ec45", "ec60", "ec90", "gust", "par", "rh", "stemp15", "stemp30",
                      "stemp45", "stemp60", "stemp90", "temp", "vwc15", "vwc30", "vwc45", "vwc60", "vwc90", "wind"]
    categories_kfs = ["dir", "ec15", "gust", "par", "rh", "stemp15", "temp", "vwc15", "wind"]
    categories_kur = ["prec"]
    categories_mar = ["dir", "ec15", "ec30", "ec45", "ec60", "ec90", "gust", "par", "rh", "stemp15", "stemp30",
                      "stemp45", "stemp60", "stemp90", "temp", "vwc15", "vwc30", "vwc45", "vwc60", "vwc90", "wind"]
    categories_nf = ["disch", "doc", "elc", "nit", "tcd", "toc", "tsp", "tur", "uv2", "uv4", "wl"]
    categories_out = ["disch", "doc", "elc", "nit", "tcd", "toc", "tsp", "tur", "wl"]
    categories_sha = ["disch", "doc", "elc", "nit", "tcd", "toc", "tsp", "tur", "wl"]
    categories_smf = ["dir", "ec15", "ec30", "ec45", "ec60", "ec90", "stemp15", "stemp30","stemp45", "stemp60",
                      "stemp90", "vwc15", "vwc30", "vwc45", "vwc60", "vwc90"]
    categories_tf1 = ["temp"]
    categories_tf2 = ["temp"]
    categories_ttp = ["disch", "dot", "nit"]
    categories_wsh = ["dir", "ec15", "gust", "par", "rh", "stemp15", "temp", "vwc15", "wind"]
    filenames = []
    for station in stations:
        if station == "Air":
            for category in categories_air:
                filenames.append(f"{station}-{category}.csv")

        elif station == "KFS":
            for category in categories_kfs:
                filenames.append(f"{station}-{category}.csv")

        elif station == "Kur":
            for category in categories_kur:
                filenames.append(f"{station}-{category}.csv")

        elif station == "Mar":
            for category in categories_mar:
                filenames.append(f"{station}-{category}.csv")

        elif station == "NF":
            for category in categories_nf:
                filenames.append(f"{station}-{category}.csv")

        elif station == "OUT":
            for category in categories_out:
                filenames.append(f"{station}-{category}.csv")

        elif station == "SHA":
            for category in categories_sha:
                filenames.append(f"{station}-{category}.csv")

        elif station == "SMF":
            for category in categories_smf:
                filenames.append(f"{station}-{category}.csv")

        elif station == "TF1":
            for category in categories_tf1:
                filenames.append(f"{station}-{category}.csv")

        elif station == "TF2":
            for category in categories_tf2:
                filenames.append(f"{station}-{category}.csv")

        elif station == "TTP":
            for category in categories_ttp:
                filenames.append(f"{station}-{category}.csv")

        elif station == "WSH":
            for category in categories_wsh:
                filenames.append(f"{station}-{category}.csv")

    return filenames


def load_data(stations, interval=None):
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
    air_prec_df = create_prec_df(df_time, "Air", interval=interval)
    cha_prec_df = create_prec_df(df_time, "Cha", interval=interval)
    che_prec_df = create_prec_df(df_time, "Che", interval=interval)
    chi_prec_df = create_prec_df(df_time, "Chi", interval=interval)
    fin_prec_df = create_prec_df(df_time, "Fin", interval=interval)
    kfs_prec_df = create_prec_df(df_time, "KFS", interval=interval)
    mar_prec_df = create_prec_df(df_time, "Mar", interval=interval)
    nf_prec_df = create_prec_df(df_time, "NF", interval=interval)
    out_prec_df = create_prec_df(df_time, "OUT", interval=interval)
    tf1_prec_df = create_prec_df(df_time, "TF1", interval=interval)
    tf2_prec_df = create_prec_df(df_time, "TF2", interval=interval)
    ttp_prec_df = create_prec_df(df_time, "TTP", interval=interval)
    frames.append(sha_prec_df)
    frames.append(fun_prec_df)
    frames.append(kur_prec_df)
    frames.append(wsh_prec_df)
    frames.append(air_prec_df)
    frames.append(cha_prec_df)
    frames.append(che_prec_df)
    frames.append(chi_prec_df)
    frames.append(fin_prec_df)
    frames.append(kfs_prec_df)
    frames.append(mar_prec_df)
    frames.append(nf_prec_df)
    frames.append(out_prec_df)
    frames.append(tf1_prec_df)
    frames.append(tf2_prec_df)
    frames.append(ttp_prec_df)

    df = pd.concat(frames, axis=1)

    return df
