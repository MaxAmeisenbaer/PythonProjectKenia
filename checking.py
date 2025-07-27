import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def load_data(x_path, y_pred_path, y_true_path, dates_path):
    X = np.load(x_path)
    y_pred = np.load(y_pred_path)
    y_true = np.load(y_true_path)
    dates = np.load(dates_path, allow_pickle=True)

    # Bytes in Strings umwandeln (wenn nötig)
    if isinstance(dates[0], bytes):
        dates = [d.decode("utf-8") for d in dates]

    dates = pd.to_datetime(dates)
    return X, y_pred, y_true, dates



def find_dip_segment(y_pred, threshold=0.2, min_length=20):
    below_threshold = y_pred < threshold
    dip_start = None
    dip_end = None

    count = 0
    for i in range(len(below_threshold) - 1, -1, -1):
        if below_threshold[i]:
            count += 1
            dip_end = i
        elif count >= min_length:
            dip_start = dip_end - count + 1
            break
        else:
            count = 0
            dip_end = None

    if dip_start is None or dip_end is None:
        raise ValueError("Kein geeigneter Dip gefunden.")

    return dip_start, dip_end

def rebuild_sequences(X, seq_length):
    # X: (T, n_features)  ->  X_seq: (T-seq_length, seq_length, n_features)
    return np.array([X[i:i+seq_length] for i in range(len(X) - seq_length)])


def extract_feature_stats(X, indices, seq_length, feature_names=None):
    # X kann (T, n_features) oder (T, seq_len, n_features) sein
    if X.ndim == 2:
        X = rebuild_sequences(X, seq_length)  # -> (T', seq_len, n_features)

    X_sel = X[indices]  # (T_seg, seq_len, n_features)

    # Mittelwerte/Std pro Feature über beide Zeitachsen
    mean_feat = X_sel.mean(axis=(0, 1))  # (n_features,)
    std_feat  = X_sel.std(axis=(0, 1))   # (n_features,)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[2])]

    df = pd.DataFrame(
        {"mean": mean_feat, "std": std_feat},
        index=feature_names
    )
    return df



def compare_segments(X, y_pred, y_true, dates, seq_length, feature_names=None, threshold=0.2):
    dip_start, dip_end = find_dip_segment(y_pred, threshold=threshold)
    dip_indices = np.arange(dip_start, dip_end + 1)

    before_start = max(0, dip_start - (dip_end - dip_start + 1))
    before_indices = np.arange(before_start, dip_start)

    print(f"Dip:     {dates[dip_start]} bis {dates[dip_end]}  (len={len(dip_indices)})")
    print(f"Davor:   {dates[before_start]} bis {dates[dip_start - 1]}  (len={len(before_indices)})")

    df_before = extract_feature_stats(X, before_indices, seq_length, feature_names)
    df_dip    = extract_feature_stats(X, dip_indices,    seq_length, feature_names)

    df = df_before.join(df_dip, lsuffix="_before", rsuffix="_dip")
    df["Δmean"]    = df["mean_dip"] - df["mean_before"]
    df["Δmean[%]"] = 100 * df["Δmean"] / (df["mean_before"] + 1e-9)
    df["Δstd"]     = df["std_dip"] - df["std_before"]
    df["Δstd[%]"]  = 100 * df["Δstd"] / (df["std_before"] + 1e-9)

    return df.sort_values("Δmean[%]", key=lambda s: s.abs(), ascending=False)



# Beispielaufruf
if __name__ == "__main__":
    seq_length = 18  # an dein Setup anpassen

    X, y_pred, y_true, dates = load_data(
        "models/benchmark/X_full.npy",
        "models/benchmark/predictions_full.npy",
        "models/benchmark/y_true_full.npy",
        "models/benchmark/dates_full.npy"
    )

    feature_names = [f"f{i+1}" for i in range(X.shape[1])]  # falls vorhanden, echte Namen nehmen

    df_comp = compare_segments(X, y_pred, y_true, dates, seq_length, feature_names, threshold=0.2)
    df_comp.to_csv("dip_analysis_stats.csv")
    print(df_comp.round(3))




