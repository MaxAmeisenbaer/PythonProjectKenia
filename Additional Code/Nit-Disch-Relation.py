import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

def read_csv(region, directory = "Data"):
    nit_file = f"{region}-nit.csv"
    disch_file = f"{region}-disch.csv"

    df_nit = pd.read_csv(os.path.join(directory, nit_file))
    df_disch = pd.read_csv(os.path.join(directory, disch_file))

    df_nit["date"] = pd.to_datetime(df_nit["date"], format='%Y-%m-%d %H:%M:%S')
    df_disch["date"] = pd.to_datetime(df_disch["date"], format='%Y-%m-%d %H:%M:%S')

    return df_nit, df_disch

def sync_data(df_nit, df_disch):
    df_nit = df_nit[df_nit["date"] >= "2015-04-21 11:00:00"]
    df_disch = df_disch[df_disch["date"] >= "2015-04-01 11:00:00"]

    return df_nit, df_disch

def scaling(df_nit, df_disch):
    scaler = MinMaxScaler()
    df_nit["values"] = scaler.fit_transform(df_nit[["value"]])
    df_disch["values"] = scaler.fit_transform(df_disch[["value"]])

    return df_nit, df_disch

def plot_all_regions(regions, directory = "Data"):
    fig, axs = plt.subplots(len(regions), 1, figsize = (14, 4 * len(regions)), sharex = True)

    for i, region in enumerate(regions):
        df_nit, df_disch = read_csv(region, directory)
        df_nit, df_disch = sync_data(df_nit, df_disch)
        df_nit, df_disch = scaling(df_nit, df_disch)

        axs[i].plot(df_nit["date"], df_nit["values"], label="Nitrat", color="blue")
        axs[i].plot(df_disch["date"], df_disch["values"], label="Discharge", color="green")
        axs[i].set_title(f"Region: {region}")
        axs[i].set_ylabel("Normalisierte Werte")
        axs[i].grid(True)
        axs[i].legend()

    axs[-1].set_xlabel("Datum")
    plt.tight_layout()
    plt.show()

plot_all_regions(regions=["SHA", "NF", "TTP", "OUT"])

