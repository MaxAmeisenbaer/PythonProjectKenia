import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pandas as pd
import os
import seaborn as sns
import glob

def load_csv(filename, directory="Data"):
    data_csv = pd.read_csv(os.path.join(directory, filename))  # Datei einlesen
    data_csv['date'] = pd.to_datetime(data_csv['date'], format='%Y-%m-%d %H:%M:%S')  # Datetime-Umwandlung

    data_csv.set_index('date', inplace=True)  # Setze den Index auf 'date'

    time = data_csv.index.tolist()  # Zeitliste als Vorraussetzung für Resampling
    data = data_csv["value"].interpolate("linear", limit_direction="both").tolist()  # Interpolation
    start = time.index(pd.Timestamp("2015-04-21 11:00:00"))  # Zeit auf Kenia-Startpunkt setzen

    data_csv = data_csv.resample("24h").mean()  # Resampling auf tägliche Mittelwerte


    data_len = len(data_csv)  # Länge des DataFrames nach Resampling



    return time[start:], data[start:], data_csv, data_len

#Ursula dient um einen Output zu generieren
def ursula(filename):
    time, data, data_csv, data_len = load_csv(filename= filename, directory="Data")    #Wichtig um lokale Daten aus load_csv zu holen
    print(data_csv.head(), data_len)


#load_csv(filename="SHA-nit.csv",)
#load_csv(filename= "Mar-stemp15.csv",)
#ursula(filename= "SHA-nit.csv")

def plot_one_data(filename):
    _, _, data_csv, _ = load_csv(filename= filename, )
    plt.figure(figsize=(10,6))
    sns.lineplot(data= data_csv, x= "date", y= "value")

    plt.xlabel('date')
    plt.ylabel('value')

    plt.show()

#plot_one_data(filename= "SHA-nit.csv")

def explore_startpoints(directory, pattern):
    """
    Liest alle CSV-Dateien aus einem Verzeichnis ein, prüft die ersten relevanten Datenpunkte
    und vergleicht die Startzeitpunkte der Dateien.

    :param directory: Verzeichnis mit den CSV-Dateien
    :param pattern: Dateinamensmuster (z. B. 'SHA-*.csv')
    """
    file_pattern = os.path.join(directory, pattern)
    file_list = glob.glob(file_pattern)  # Alle passenden Dateien finden
    print(f"Gefundene Dateien: {file_list}")

    start_dates = {}  # Dictionary für die ersten Zeitpunkte jeder Datei

    for file in file_list:

            df = pd.read_csv(file)
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')

            # Entferne alle Datenpunkte vor dem 01.04.2015 um 10:00 Uhr
            df = df[df["date"] >= "2015-04-01 10:00:00"]

            # Falls der DataFrame nicht leer ist, speichere das erste Datum

            first_date = df.iloc[0]["date"]
            start_dates[os.path.basename(file)] = first_date
            print(f"{os.path.basename(file)} - Erster relevanter Wert: {first_date}")

    # Prüfen, ob alle Startzeitpunkte identisch sind
    unique_dates = set(start_dates.values())

    if len(unique_dates) == 1:
        print("\n✅ Alle Dateien starten zur selben Zeit:", list(unique_dates)[0])
    else:
        print("\n⚠️ Unterschiedliche Startzeitpunkte gefunden:")
        for file, date in start_dates.items():
            print(f"{file}: {date}")




# Aufruf der Funktion für alle relevanten Dateien im Verzeichnis "Data"
#explore_startpoints(directory="Data", pattern="SMF-*.csv")

# 📂 Alle CSV-Dateien im "Data"-Ordner einlesen
csv_files = glob.glob(os.path.join("Data", "*.csv"))

data_list = []

for file in csv_files:
    df = pd.read_csv(file)

    # Sicherstellen, dass die "date"-Spalte als Datum erkannt wird
    df["date"] = pd.to_datetime(df["date"])

    # Start- und Endzeitpunkt für jede Datei bestimmen
    start_date = df["date"].min()
    end_date = df["date"].max()

    # Speichern der Datei-Informationen
    data_list.append({"file_name": os.path.basename(file), "start": start_date, "end": end_date})

# In DataFrame umwandeln
data = pd.DataFrame(data_list)


# 🎨 Gantt-Chart erstellen
fig, ax = plt.subplots(figsize=(15, 28))

# Jede Datei als horizontale Bar hinzufügen
for i, row in data.iterrows():
    ax.barh(row["file_name"], row["end"] - row["start"], left=row["start"], color="skyblue")

# 📅 Achsen formatieren
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=45)
plt.xlabel("Datum")
plt.ylabel("CSV-Dateien")
plt.title("Gantt-Chart: Zeiträume der CSV-Dateien")
plt.grid(True)

# 📊 Anzeigen
plt.show()
