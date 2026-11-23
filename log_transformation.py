import numpy as np
import pandas as pd
from scipy.stats import boxcox

def analyze_skewness(df: pd.DataFrame) -> dict:
    """
    Reads the skewness and the percentage-zero rate (pct_zero) for
    each column in the DataFrame. Returns a nested Python dict.

    :param df  – DataFrame with all time series
    :return   – { 'col_name': { 'skew': <float>, 'pct_zero': <float> } }
    """
    skew  = df.skew()                    # float Series
    pct   = (df == 0).mean() * 100       # Fractions → percent

    skew_dict = {
        col: {"skew": float(skew[col]), "pct_zero": float(pct[col])}
        for col in df.columns
    }
    return skew_dict

def _ensure_positive(x: np.ndarray):
    """Adds a shift so that all finite values become > 0.

    NaN values are ignored (np.nanmin) and remain NaN in the result.
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
    Applies the logarithm to precipitation columns using log1p (= ln(x + 1)).
    Zero values (no rain) remain exactly 0.
    Inverse transformation: np.expm1(x)

    :param df: DataFrame
    :param col: Name of the column to transform
    :return: DataFrame with transformed columns
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
    Applies the logarithm to the specified columns using ln(x + epsilon).
    Suitable for right-skewed values such as discharge or nitrate.
    Inverse transformation: np.exp(x) - epsilon

    :param df: DataFrame
    :param col: Name of the column to transform
    :param epsilon: Small offset to avoid log(0)
    :return: DataFrame with transformed columns
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
    Performs the Box-Cox transformation for a column, computes

    `lambda`  – the *MLE* parameter
    `shift`   – a positive shift if necessary

    Returns:
        tuple(DataFrame, {'lambda': float, 'shift': float})
    """
    x_raw = df[col].values.astype(float)

    # Transform only finite values (NaN remains NaN)
    finite = np.isfinite(x_raw)
    if not finite.any():
        raise ValueError(f"boxcox: Spalte '{col}' enthält keine endlichen Werte.")

    # Shift for negative/0 values
    x, shift = _ensure_positive(x_raw[finite])

    # Transformation
    trans, lam = boxcox(x)

    # Result
    df = df.copy()
    values = np.full(len(x_raw), np.nan, dtype=float)
    values[finite] = trans
    df[col] = values

    # Set param dict
    params = {"lambda": lam, "shift": shift}

    return df, params

def inverse_log_transform(y_true, y_pred, method, epsilon=1e-6):
    """
    Inverse transformation of the log values.

    :param y_true: Transformed true values (numpy array)
    :param y_pred: Transformed predicted values (numpy array)
    :param method: dict with information on log type and parameters
    :param epsilon: Small offset to avoid log(0)
    :return: Inverse-transformed values
    """
    if method["type"] == "log1p":
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)
        y_pred = np.maximum(y_pred, 0)
        return y_true, y_pred    # Inverse of log1p
    elif method["type"] == "log_eps":
        eps = method.get("params", {}).get("epsilon", epsilon)
        y_true = np.exp(y_true) - eps
        y_pred = np.exp(y_pred) - eps
        y_pred = np.maximum(y_pred, 0)
        return y_true, y_pred  # Inverse of log(x + epsilon)
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
