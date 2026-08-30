import pandas as pd
import numpy as np
import os


def analyze_gaps(files, directory="Data", interval="10min"):
    """
    Analysiert Lücken in CSV-Dateien auf ein festes Zeitraster hin.
    """
    results = []

    for filename in files:
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            print(f"❌ Datei nicht gefunden: {filename}")
            continue

        print(f"🔍 Analysiere {filename}...")

        # 1. Datei laden (mit deinem Logik-Ansatz)
        # Wir nutzen hier eine vereinfachte Version deines Loaders für den Check
        df = pd.read_csv(path, parse_dates=["date"])
        df.set_index("date", inplace=True)

        # 2. WICHTIG: Auf das 10-Minuten-Raster bringen!
        # Das füllt die Zeitachse auf, sodass Lücken auch wirklich als NaNs erscheinen
        df = df.resample(interval).mean()

        # Spaltenname normalisieren (falls 'value' vorhanden)
        if 'value' in df.columns:
            df.rename(columns={'value': 'val'}, inplace=True)

        series = df['val']

        # 3. Statistiken berechnen
        total_nans = series.isna().sum()
        total_rows = len(series)
        nan_pct = (total_nans / total_rows) * 100 if total_rows > 0 else 0

        # 4. Lücken-Analyse (Gruppierung von aufeinanderfolgenden NaNs)
        # Wir markieren Blöcke von NaNs
        is_nan = series.isna()
        # Erzeugt Gruppen von aufeinanderfolgenden identischen Werten
        nan_groups = (is_nan != is_nan.shift()).cumsum()

        # Wir filtern nur die Gruppen, die tatsächlich NaNs sind
        nan_group_lengths = is_nan.groupby(nan_groups).sum()
        gap_lengths = nan_group_lengths[nan_group_lengths > 0]

        if not gap_lengths.empty:
            max_gap_count = gap_lengths.max()
            # Umrechnung in Stunden (basierend auf Intervall)
            # Ein 10min Intervall = 6 Werte pro Stunde
            freq_in_minutes = 10
            max_gap_hours = max_gap_count * freq_in_minutes / 60

            # Anzahl der "großen" Lücken (z.B. > 6 Stunde)
            large_gaps = (gap_lengths[gap_lengths > (360 / freq_in_minutes)]).count()
        else:
            max_gap_hours = 0
            large_gaps = 0

        results.append({
            "File": filename,
            "Total Rows": total_rows,
            "NaN Count": total_nans,
            "NaN %": f"{nan_pct:.2f}%",
            "Max Gap (Hours)": round(max_gap_hours, 2),
            "Large Gaps (>1h)": large_gaps
        })

    return pd.DataFrame(results)


# --- CONFIGURATION ---
# Hier die exakten Namen deiner Dateien eintragen
target_files = ["TTP-nit.csv", "NF-nit.csv", "SHA-nit.csv"]
# 1. Pfad zum Skript (Additional Code)
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Pfad zum Projekt-Ordner (eine Ebene höher)
project_root = os.path.dirname(script_dir)

# 3. Pfad zum Data-Ordner (im Projekt-Ordner)
data_dir = os.path.join(project_root, "Data")


# Ausführung
report = analyze_gaps(target_files, directory=data_dir, interval="10min")

print("\n" + "=" * 50)
print("LÜCKEN-ANALYSE BERICHT")
print("=" * 50)
print(report.to_string(index=False))
print("=" * 50)