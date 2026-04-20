import numpy as np

def log_transform_precipitation(df):
    """
    Logarithmiert Niederschlagsspalten mit log1p (= ln(x + 1)).
    Null-Werte (kein Regen) bleiben exakt 0.
    Rücktransformation: np.expm1(x)

    :param df: DataFrame mit Niederschlagsspalten (*_prec)
    :return: DataFrame mit transformierten Niederschlagsspalten
    """
    prec_cols = [c for c in df.columns if c.endswith("_prec")]

    if prec_cols:
        for col in prec_cols:
            df[col] = np.log1p(df[col])

    return df


def log_transform_rightskew_columns(df, columns, epsilon=1e-6):
    """
    Logarithmiert angegebene Spalten mit ln(x + epsilon).
    Geeignet für rechtsschief verteilte Werte wie Abfluss oder Nitrat.
    Rücktransformation: np.exp(x) - epsilon

    :param df: DataFrame
    :param columns: Liste der zu transformierenden Spaltennamen
    :param epsilon: Kleiner Offset um log(0) zu vermeiden
    :return: DataFrame mit transformierten Spalten
    """
    for col in columns:

        df[col] = np.log(df[col] + epsilon)

    return df

def inverse_log_transform(values, method="log1p"):
    """
    Rücktransformation der Log-Werte.

    :param values: Transformierte Werte (numpy array)
    :param method: "log1p" für Niederschlag, "log_eps" für Abfluss/Nitrat
    :return: Rücktransformierte Werte
    """
    if method == "log1p":
        return np.expm1(values)     # Inverse von log1p
    elif method == "log_eps":
        return np.exp(values) - 1e-6  # Inverse von log(x + epsilon)
    else:
        raise ValueError(f"Unbekannte Methode: {method}")