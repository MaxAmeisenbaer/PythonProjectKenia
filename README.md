# Vorhersage von Nitrat-Konzentrationen in kenianischen Hochlandgewässern unter Verwendung von neuronalen Netzwerken

## Projektbeschreibung
Dieses Projekt beschäftigt sich mit der **Vorhersage der Nitrat-Konzentrationen in Hochlandgewässern Kenias**.  
Hierfür werden verschiedene **LSTM-Modelle (Long Short-Term Memory Neural Networks)** entwickelt und evaluiert, um die Eignung dieser Modelle für die Zeitreihenprognose von Wasserqualitätsdaten zu untersuchen.  

Ziele des Projekts:
- Entwicklung unterschiedlicher LSTM-Modelle  
- Prognose der Nitrat-Konzentration auf Basis von unterschiedlich vielen Umweltfaktoren (Wasser-, Wetter-, Boden- und Winddaten)  
- Vergleich und Evaluierung der Modellleistung anhand verschiedener Gütemaße  

---

## Technologien
- **Python 3.9** oder **Python 3.12**  
- Alle weiteren benötigten Pakete sind in der Datei `requirements.txt` aufgelistet.  

---
## Voraussetzungen
- Ordner "Data", im Repository, welches die Ursprungsdaten aller verwendeter Stationen sowie Standortdaten der Stationen beinhaltet
- ausreichende Computerleistung oder besser Zugang zu Compute-Cloud-Architektur für die Berechunng der Modelle

---

## Nutzung
- `hp_training.py` ausführen und beste Parameter-Settings für Modelle ermitteln
-  In `lstm_main.py` `main`-Funktion (Ende des Dokuments9 das gewünschte Szenario eingeben ("benchmark","disch","not_ec"...)
- `lstm_main.py` ausführen
- für Visualisierung `create_figures.py` ausführen

---

## Struktur
- ├── Data/                 # Rohdaten & vorbereitete Datensätze
- ├── models/               # Ergebnisse der trainierten Modelle
- ├── figures/              # Grafiken zu den verschiedenen Modellen nach `create_figures.py`
- ├── Additional Code/      # zusätzliche Skripts (nicht relevant)
- ├── requirements.txt      # Python-Abhängigkeiten
- ├── scenario_config.py    # Konfiguration der verschiedenen Modell-Szenarien für Auswahl der Input-Faktoren (wird durch `lstm_main.py` aufgerufen)
- ├── hp_training.py        # Benchmark-Szenario ausführen, um beste Hyperparameter für Modelle zu bestimmen
- ├── log_transformation.py # Normalisierung und Transformation der Daten in bestmögliche Form 
- ├── lstm_main.py          # Haupt-Skript, welcher ausgeführt wird, um Modelle zu berechnen
- ├── lstm_model.py         # Skript, in dem die Modellstruktur aufgebaut wird (wird durch `lstm_main.py` aufgerufen)
- ├── data_prepro.py        # Vorbereitung der Daten vor Modellberechnung (wird durch `lstm_main.py` aufgerufen)
- └── README.md             # Projektbeschreibung

---

## Autor

- Projekt von Maximilian Mäder
- basierend auf Arbeit von Henri Breuer 
- Unter der Leitung von Aaron Neill
- Daten erhoben von Dr. Suzanne Jacobs