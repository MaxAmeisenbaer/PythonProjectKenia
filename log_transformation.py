import numpy as np
import pandas as pd

def analyze_skewness(df: pd.DataFrame) -> dict:
    """
    Lädt die Schiefe (skewness) und die Prozent‑Null‑Rate (pct_zero) für
    jede Spalte im DataFrame. Gibt ein verschachteltes Python‑Dict zurück,
    das leicht als JSON serialisierbar ist.

    :param df  – DataFrame mit allen Zeitreihen
    :return   – { 'col_name': { 'skew': <float>, 'pct_zero': <float> } }
    """
    skew  = df.skew()                    # float Series
    pct   = (df == 0).mean() * 100       # Fractions → Prozent

    skew_dict = {
        col: {"skew": float(skew[col]), "pct_zero": float(pct[col])}
        for col in df.columns
    }
    return skew_dict


def log_transform_log1p(df, col):
    """
    Logarithmiert Niederschlagsspalten mit log1p (= ln(x + 1)).
    Null-Werte (kein Regen) bleiben exakt 0.
    Rücktransformation: np.expm1(x)

    :param df: DataFrame
    :param col: Name der zu transformierenden Spalte
    :return: DataFrame mit transformierten Spalten
    """
    df[col] = np.log1p(df[col])

    return df

def log_transform_rightskew(df, col, epsilon=1e-6):
    """
    Logarithmiert angegebene Spalten mit ln(x + epsilon).
    Geeignet für rechtsschief verteilte Werte wie Abfluss oder Nitrat.
    Rücktransformation: np.exp(x) - epsilon

    :param df: DataFrame
    :param col: Name der zu transformierenden Spalte
    :param epsilon: Kleiner Offset um log(0) zu vermeiden
    :return: DataFrame mit transformierten Spalten
    """
    df[col] = np.log(df[col] + epsilon)

    return df

def inverse_log_transform(values, method="log1p", epsilon=1e-6):
    """
    Rücktransformation der Log-Werte.

    :param values: Transformierte Werte (numpy array)
    :param method: "log1p" für Niederschlag, "log_eps" für Abfluss/Nitrat
    :param epsilon: Kleiner Offset um log(0) zu vermeiden
    :return: Rücktransformierte Werte
    """
    if method == "log1p":
        return np.expm1(values)     # Inverse von log1p
    elif method == "log_eps":
        return np.exp(values) - epsilon  # Inverse von log(x + epsilon)
    else:
        raise ValueError(f"Unbekannte Methode: {method}")