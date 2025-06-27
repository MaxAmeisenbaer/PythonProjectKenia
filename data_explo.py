import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os
import seaborn as sns
import glob

def load_csv(filename, directory="Data"):
    """
    Lädt eine CSV-Datei aus dem angegebenen Verzeichnis, wandelt das Datum in datetime um,
    interpoliert fehlende Werte, resampelt auf Tagesmittelwerte und liefert Zeitreihen-Daten.

    :param filename: Name der CSV-Datei
    :param directory: Verzeichnis, in dem sich die Datei befindet
    :return: (Zeitliste ab Startpunkt, interpolierte Daten ab Startpunkt,
              DataFrame mit täglichen Mittelwerten, Länge des DataFrames)
    """
    data_csv = pd.read_csv(os.path.join(directory, filename))
    data_csv['date'] = pd.to_datetime(data_csv['date'], format='%Y-%m-%d %H:%M:%S')
    data_csv.set_index('date', inplace=True)

    time = data_csv.index.tolist()
    data = data_csv["value"].interpolate("linear", limit_direction="both").tolist()
    start = time.index(pd.Timestamp("2015-04-21 11:00:00"))

    data_csv = data_csv.resample("24h").mean()
    data_len = len(data_csv)

    return time[start:], data[start:], data_csv, data_len


def ursula(filename):
    """
    Testfunktion zur Ausgabe der geladenen CSV-Daten (Kopfzeilen und Länge).

    :param filename: Name der CSV-Datei
    """
    time, data, data_csv, data_len = load_csv(filename=filename, directory="Data")
    print(data_csv.head(), data_len)


def plot_one_data(filename):
    """
    Plottet die Zeitreihe einer einzelnen CSV-Datei mit Seaborn.

    :param filename: Name der CSV-Datei
    """
    _, _, data_csv, _ = load_csv(filename=filename)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data_csv, x="date", y="value")

    plt.xlabel('date')
    plt.ylabel('value')
    plt.title(f'Daten aus {filename}')
    plt.show()


def explore_startpoints(directory, pattern):
    """
    Vergleicht Startzeitpunkte mehrerer CSV-Dateien in einem Verzeichnis
    und prüft auf Konsistenz der Zeiträume.

    :param directory: Verzeichnis mit den CSV-Dateien
    :param pattern: Dateinamensmuster (z. B. 'SHA-*.csv')
    """
    file_pattern = os.path.join(directory, pattern)
    file_list = glob.glob(file_pattern)
    print(f"Gefundene Dateien: {file_list}")

    start_dates = {}

    for file in file_list:
        df = pd.read_csv(file)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
        df = df[df["date"] >= "2015-04-01 10:00:00"]

        if not df.empty:
            first_date = df.iloc[0]["date"]
            start_dates[os.path.basename(file)] = first_date
            print(f"{os.path.basename(file)} - Erster relevanter Wert: {first_date}")

    unique_dates = set(start_dates.values())

    if len(unique_dates) == 1:
        print("\n✅ Alle Dateien starten zur selben Zeit:", list(unique_dates)[0])
    else:
        print("\n⚠️ Unterschiedliche Startzeitpunkte gefunden:")
        for file, date in start_dates.items():
            print(f"{file}: {date}")


def gantt_chart():
    """
    Erstellt ein Gantt-Diagramm, das die Zeiträume (Start bis Ende)
    aller CSV-Dateien im 'Data'-Verzeichnis visualisiert.
    """
    csv_files = glob.glob(os.path.join("Data", "*.csv"))
    data_list = []

    for file in csv_files:
        df = pd.read_csv(file)
        df["date"] = pd.to_datetime(df["date"])
        start_date = df["date"].min()
        end_date = df["date"].max()

        data_list.append({
            "file_name": os.path.basename(file),
            "start": start_date,
            "end": end_date
        })

    data = pd.DataFrame(data_list)

    fig, ax = plt.subplots(figsize=(15, 28))
    for i, row in data.iterrows():
        ax.barh(row["file_name"], row["end"] - row["start"], left=row["start"], color="skyblue")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    plt.xlabel("Datum")
    plt.ylabel("CSV-Dateien")
    plt.title("Gantt-Chart: Zeiträume der CSV-Dateien")
    plt.grid(True)
    plt.show()
