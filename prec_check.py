import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from benchmark_szenario_sha import get_benchmark_config
from low_input_szenario_sha import get_low_input_config
from not_lyser import get_not_lyser_config
from not_nit_szenario_sha import get_not_nit_config
from data_prepro import load_data

def get_config_for_model(model_folder):
    if model_folder == "benchmark":
        return get_benchmark_config
    elif model_folder == "low_input":
        return get_low_input_config
    elif model_folder == "not_lyser":
        return get_not_lyser_config
    elif model_folder == "not_nit":
        return get_not_nit_config
    else:
        raise ValueError(f"Unbekannter Modellordner: {model_folder}")

def get_precipitation_series(df, start, end, scale=False):
    """
    Gibt den aufsummierten Niederschlag über *_prec-Spalten für den Zeitraum zurück.
    Optional skaliert (Min-Max über diesen Zeitraum).
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df[(df.index >= start) & (df.index <= end)]

    prec_cols = [col for col in df.columns if col.endswith("_prec")]
    if not prec_cols:
        raise ValueError("Keine *_prec Spalten gefunden.")

    series = df[prec_cols].sum(axis=1)

    if scale:
        scaler = MinMaxScaler()
        series = pd.Series(
            scaler.fit_transform(series.values.reshape(-1, 1)).flatten(),
            index=series.index
        )

    # Setze Index auf Tage seit Start
    days_since_start = (series.index - series.index[0]).days
    series.index = days_since_start

    return series


def plot_overlaid_precipitation_lines(df, scale=False):
    """
    Vergleicht drei Zeiträume durch übereinandergelegte Niederschlags-Zeitreihen (als Linienplots).
    """
    periods = {
        "Aug 2018 – Jan 2019": ("2018-08-01", "2019-01-31"),
        "Aug 2019 – Jan 2020": ("2019-08-01", "2020-01-31"),
        "März – Okt 2019":     ("2019-03-01", "2019-10-31")
    }

    plt.figure(figsize=(14, 6))

    for label, (start, end) in periods.items():
        series = get_precipitation_series(df, start, end, scale=scale)
        series = series.reset_index(drop=True)  # Setze Index auf 0, 1, 2, ... damit Zeitachsen vergleichbar sind
        plt.plot(series, label=label, linewidth=2)

    plt.title("Vergleich der Niederschlagsverläufe (übereinandergelegt, {}skalierte Werte)".format("" if scale else "un"))
    plt.xlabel("Tage seit Zeitraumstart")
    plt.ylabel("Gesamtniederschlag{}".format(" (skaliert 0–1)" if scale else ""))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    model_folder = "benchmark"
    df = load_data(*get_config_for_model(model_folder)()[:2])

    # Skaliert vergleichen
    plot_overlaid_precipitation_lines(df, scale=True)

    # Unskaliert vergleichen
    plot_overlaid_precipitation_lines(df, scale=False)

#Summenwerte der prec-Daten pro Tage? Also pro Zeiträume - weil übersicht schwierig!