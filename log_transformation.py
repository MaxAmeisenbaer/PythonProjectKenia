import numpy as np
import pandas as pd
from scipy.stats import boxcox

def analyze_skewness(df: pd.DataFrame) -> dict:
    """
    Lädt die Schiefe (skewness) und die Prozent‑Null‑Rate (pct_zero) für
    jede Spalte im DataFrame. Gibt ein verschachteltes Python‑Dict zurück.

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

def _ensure_positive(x: np.ndarray):
    """Hängt einen Shift an, damit alle endlichen Werte > 0 werden.

    NaN-Werte werden ignoriert (np.nanmin) und bleiben im Ergebnis NaN.
    """
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        raise ValueError("_ensure_positive: Array enthält keine endlichen Werte.")
    min_val = np.min(x[finite])
    shift = 0.0
    if min_val <= 0:
        shift = 1 - min_val
        x = x + shift
    return x, shift


def log_transform_log1p(df, col):
    """
    Logarithmiert Niederschlagsspalten mit log1p (= ln(x + 1)).
    Null-Werte (kein Regen) bleiben exakt 0.
    Rücktransformation: np.expm1(x)

    :param df: DataFrame
    :param col: Name der zu transformierenden Spalte
    :return: DataFrame mit transformierten Spalten
    """
    x = df[col]
    if (x.dropna() <= -1).any():
        raise ValueError(
            f"log1p ist für Spalte '{col}' nicht definiert: Werte <= -1 vorhanden."
        )
    df[col] = np.log1p(x)

    return df

def log_transform_eps(df, col, epsilon=1e-6):
    """
    Logarithmiert angegebene Spalten mit ln(x + epsilon).
    Geeignet für rechtsschief verteilte Werte wie Abfluss oder Nitrat.
    Rücktransformation: np.exp(x) - epsilon

    :param df: DataFrame
    :param col: Name der zu transformierenden Spalte
    :param epsilon: Kleiner Offset um log(0) zu vermeiden
    :return: DataFrame mit transformierten Spalten
    """
    x = df[col]
    if (x.dropna() + epsilon <= 0).any():
        raise ValueError(
            f"log(x + {epsilon}) ist für Spalte '{col}' nicht definiert: "
            f"Werte <= -{epsilon} vorhanden."
        )
    df[col] = np.log(x + epsilon)

    return df

def boxcox_transform(df, col):
    """
    Führt die Box‑Cox‑Transformation für eine Spalte durch, berechnet

    *   `lambda` – der *MLE*‑Parameter
    *   `shift`   – ggf. notwendiger Positives‑Shift

    Returns:
        tuple(DataFrame, {'lambda': float, 'shift': float})
    """
    x_raw = df[col].values.astype(float)

    # Nur endliche Werte transformieren (NaN bleibt NaN)
    finite = np.isfinite(x_raw)
    if not finite.any():
        raise ValueError(f"boxcox: Spalte '{col}' enthält keine endlichen Werte.")

    # Shift für negative/0‑Werte
    x, shift = _ensure_positive(x_raw[finite])

    # Transformation
    trans, lam = boxcox(x)

    # Resultat
    df = df.copy()
    values = np.full(len(x_raw), np.nan, dtype=float)
    values[finite] = trans
    df[col] = values

    # param-dict festlegen
    params = {"lambda": lam, "shift": shift}

    return df, params

def inverse_log_transform(y_true, y_pred, method, epsilon=1e-6):
    """
    Rücktransformation der Log-Werte.

    :param y_true: Transformierte wahre Werte (numpy array)
    :param y_pred: Transformierte vorhergesagte Werte (numpy array)
    :param method: dict mit Infos zu log-Typ und Parametern
    :param epsilon: Kleiner Offset um log(0) zu vermeiden
    :return: Rücktransformierte Werte
    """
    if method["type"] == "log1p":
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)
        y_pred = np.maximum(y_pred, 0)
        return y_true, y_pred    # Inverse von log1p
    elif method["type"] == "log_eps":
        eps = method.get("params", {}).get("epsilon", epsilon)
        y_true = np.exp(y_true) - eps
        y_pred = np.exp(y_pred) - eps
        y_pred = np.maximum(y_pred, 0)
        return y_true, y_pred  # Inverse von log(x + epsilon)
    elif method["type"] == "boxcox":
        lam = method["params"]["lambda"]
        shift = method["params"].get("shift", 0.0)

        def inv_boxcox(x):
            if np.isclose(lam, 0):
                return np.exp(x)
            return np.power(x * lam + 1, 1 / lam)

        y_true = inv_boxcox(y_true) - shift
        y_pred = inv_boxcox(y_pred) - shift
        return y_true, y_pred
    elif method["type"] == "none":
        return y_true, y_pred
    else:
        raise ValueError(f"Unbekannte Methode: {method}")
