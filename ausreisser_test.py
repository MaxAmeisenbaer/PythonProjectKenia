import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Beispiel: 5 Werte davor und 5 danach betrachten (Fenstergröße = 11, ohne aktuellen Punkt)
def detect_local_outliers_percentage(series, window_size=5, threshold_percent=100, use_median=False):
    outlier_indices = []
    series = series.reset_index(drop=True)  # Für den Fall, dass der Index Datumswerte o.Ä. enthält

    for i in range(window_size, len(series) - window_size):
        window = list(series[i - window_size:i]) + list(series[i + 1:i + 1 + window_size])
        if use_median:
            local_value = np.median(window)
        else:
            local_value = np.mean(window)

        if local_value == 0:
            continue  # Vermeide Division durch 0

        current_value = series[i]
        percent_diff = abs(current_value - local_value) / abs(local_value) * 100

        if percent_diff > threshold_percent:
            outlier_indices.append(i)

    return outlier_indices

def show_outlier_context(series, outlier_indices, window_size=5):
    context_data = []

    for idx in outlier_indices:
        # Stelle sicher, dass der Kontextbereich im gültigen Bereich liegt
        start = max(0, idx - window_size)
        end = min(len(series), idx + window_size + 1)

        snippet = series.iloc[start:end].reset_index(drop=True)
        context_data.append((idx, snippet))

    return context_data

df = pd.read_csv("Data/SHA-nit.csv")



def export_all_contexts_one_sheet_with_date(df, outlier_indices, window_size=5, file_name="ausreisser_kontext_alle_mit_datum.xlsx"):
    rows = []

    for idx in outlier_indices:
        start = max(0, idx - window_size)
        end = min(len(df), idx + window_size + 1)

        for i in range(start, end):
            position = 'Ausreißer' if i == idx else ('Vorher' if i < idx else 'Nachher')
            rows.append({
                'Zeitstempel': df.at[i, 'date'],
                'Wert': df.at[i, 'value'],
                'Position': position,
                'Ausreisser_bei_Zeit': df.at[idx, 'date']  # zum Gruppieren/Filtern in Excel
            })


    df_all = pd.DataFrame(rows)
    df_all.to_excel(file_name, index=False)

# Lokale Ausreißer mit ±50% Schwelle erkennen (gegen Mittelwert)
indices = detect_local_outliers_percentage(df['value'], window_size=5, threshold_percent=100)

# alle Ausreißer in einem Sheet
export_all_contexts_one_sheet_with_date(df, indices, window_size=5)

plt.figure(figsize=(14,5))
plt.plot(df['value'], label='Werte')
plt.scatter(df.index[indices], df['value'].iloc[indices], color='red', label='Lokale Ausreißer (±50%)')
plt.legend()
plt.title('Lokale Ausreißer auf Basis prozentualer Abweichung')
plt.show()