import copy
#Mar-Daten werden in Pipeline zu Air-Daten hinzugefügt
MASTER_STATIONS = ["Air","Cha","Chi","Fin","Fun","KFS","Kur","NF","SHA","TF2","TTP","WSH"]

MASTER_MEASUREMENTS = {
    "Air": ["dir", "ec15", "gust", "par", "prec", "rh", "stemp15", "temp", "vwc15", "wind"],
    "Cha": ["prec"],
    "Chi": ["prec"],
    "Fin": ["prec"],
    "Fun": ["prec"],
    "KFS": ["dir", "gust", "par", "rh", "stemp15", "temp", "vwc15", "wind"],
    "Kur": ["prec"],
    "NF": ["disch", "doc", "elc", "nit", "prec", "tcd", "toc", "tsp", "tur", "wl"],
    "SHA": ["disch", "doc", "elc", "nit", "prec", "tcd", "toc", "tsp", "tur", "wl"],
    "TF2": ["temp","prec"],
    "TTP": ["disch", "doc", "elc", "nit", "prec", "tcd", "toc", "tsp", "tur", "wl"],
    "WSH": ["dir", "ec15", "gust", "par", "prec", "rh", "stemp15", "temp", "vwc15", "wind"]
}

DEFAULT_TARGETS = ["NF_nit","SHA_nit","TTP_nit"]


def get_scenario_config(scenario_type, missing_measurements=None, custom_targets=None, missing_stations=None):
    """
    Generiert eine Konfiguration basierend auf dem Master-Set.
    :param scenario_type: Name/Typ des szenarios
    :param custom_targets: Output-Targets (default: NF_nit,SHA_nit,TTP_nit)
    :param missing_stations: Stationen, die ausgeschlossen werden
    :param missing_measurements: Messvariablen, die ausgeschlossen werden
    """

    stations = copy.deepcopy(MASTER_STATIONS)
    measurements = copy.deepcopy(MASTER_MEASUREMENTS)
    target_features = custom_targets if custom_targets is not None else copy.deepcopy(DEFAULT_TARGETS)
    config_name = scenario_type

    # Stationen ausschließen
    if missing_stations:
        for s in missing_stations:
            if s in stations:
                stations.remove(s)
                if s in measurements:
                    del measurements[s]
    # Messungen ausschließen
    if missing_measurements:
        for station, measures_to_remove in missing_measurements.items():
            if station in measurements:
                current_measures = measurements[station]
                new_measures = [m for m in current_measures if m not in measures_to_remove]
                measurements[station] = new_measures
    # Wenn Station keine Messungen mehr hat
    if not measurements[station]:
        if station in stations:
            stations.remove(station)
        del measurements[station]
    for t in target_features:
        t_station, t_measure = t.split("_")
        # ie Station wurde gelöscht
        if t_station not in stations:
            print(f"[RECOVERY] Target '{t}' benötigt Station '{t_station}'. Füge Station wieder hinzu.")
            stations.append(t_station)
            if t_station not in measurements:
                measurements[t_station] = []

        # Die Station ist da, aber der Messwert wurde gelöscht
        if t_measure not in measurements[t_station]:
            print(f"[RECOVERY] Target '{t}' benötigt Messwert '{t_measure}'. Füge Messwert zu '{t_station}' hinzu.")
            if t_measure not in measurements[t_station]:
                measurements[t_station].append(t_measure)
    stations.sort()
    for s in measurements:
        measurements[s].sort()

    return stations, measurements, target_features, config_name

#In Arbeit - JLUKICHAT

def get_test_code_config():
    """
    Variante, die nur für minimale Tests gemacht ist
    Übrig bleiben: Fin,Fun Prec + Custom_Targets - durch SHA und TTP_tur werden get_scenario_config-features getestet
    :return: stations, measurements, target_features, config_name
    """
    scenario_type = "test_code"
    excluded_measurements = ["disch", "doc", "elc", "prec", "tcd", "toc", "tsp", "tur", "wl"]
    custom_targets = ["NF_nit","SHA_nit","TTP_nit","TTP_tur"]
    excluded_stations = ["Air","Cha","Chi","KFS","Kur","NF","Mar","TF2","TTP","WSH"]

    stations, measurements, target_features, config_name = get_scenario_config(scenario_type= scenario_type,
                                                                               missing_measurements= excluded_measurements,
                                                                               custom_targets= custom_targets,
                                                                               missing_stations= excluded_stations)

    return stations, measurements, target_features, config_name

def get_benchmark_config():
    """
    Excluded Measurements: None
    :return: stations, measurements, target_features, config_name
    """
    scenario_type = "benchmark"
    stations, measurements, target_features, config_name = get_scenario_config(scenario_type)

    return stations, measurements, target_features, config_name


def get_not_atmos_config():
    """
    Excluded Measurements: par,rh
    :return: stations, measurements, target_features, config_name
    """
    scenario_type = "not_atmos"
    excluded_measurements = ["par", "rh"]
    stations, measurements, target_features, config_name = get_scenario_config(scenario_type=scenario_type,
                                                                               missing_measurements=excluded_measurements)

    return stations, measurements, target_features, config_name


def get_not_ec_config():
    """
    Excluded Measurements: ec15,elc
    :return: stations, measurements, target_features, config_name
    """
    scenario_type = "not_ec"
    excluded_measurements = ["ec15", "elc"]
    stations, measurements, target_features, config_name = get_scenario_config(scenario_type=scenario_type,
                                                                               missing_measurements=excluded_measurements)
    return stations, measurements, target_features, config_name


def get_not_prec_config():
    """
    Excluded Measurements: prec
    :return: stations, measurements, target_features, config_name
    """
    scenario_type = "not_prec"
    excluded_measurements = ["prec"]
    stations, measurements, target_features, config_name = get_scenario_config(scenario_type=scenario_type,
                                                                               missing_measurements=excluded_measurements)
    return stations, measurements, target_features, config_name


def get_not_soil_config():
    """
    Excluded Measurements: ec15,stemp15,vwc15
    :return: stations, measurements, target_features, config_name
    """
    scenario_type = "not_soil"
    excluded_measurements = ["ec15", "stemp15", "vwc15"]
    stations, measurements, target_features, config_name = get_scenario_config(scenario_type=scenario_type,
                                                                               missing_measurements=excluded_measurements)
    return stations, measurements, target_features, config_name


def get_not_temp_config():
    """
    Excluded Measurements: tcp,temp,tsp
    :return: stations, measurements, target_features, config_name
    """
    scenario_type = "not_temp"
    excluded_measurements = ["tcp", "temp", "tsp"]
    stations, measurements, target_features, config_name = get_scenario_config(scenario_type=scenario_type,
                                                                               missing_measurements=excluded_measurements)
    return stations, measurements, target_features, config_name


def get_not_water_lvl_config():
    """
    Excluded Measurements: disch,wl
    :return: stations, measurements, target_features, config_name
    """
    scenario_type = "not_water_lvl"
    excluded_measurements = ["disch", "wl"]
    stations, measurements, target_features, config_name = get_scenario_config(scenario_type=scenario_type,
                                                                               missing_measurements=excluded_measurements)
    return stations, measurements, target_features, config_name


def get_not_water_qual_config():
    """
    Excluded Measurements: doc,toc,tur
    :return: stations, measurements, target_features, config_name
    """
    scenario_type = "not_water_qual"
    excluded_measurements = ["doc", "toc", "tur"]
    stations, measurements, target_features, config_name = get_scenario_config(scenario_type=scenario_type,
                                                                               missing_measurements=excluded_measurements)
    return stations, measurements, target_features, config_name


def get_not_wind_config():
    """
    Excluded Measurements: dir,wind,gust
    :return: stations, measurements, target_features, config_name
    """
    scenario_type = "not_wind"
    excluded_measurements = ["dir", "wind", "gust"]
    stations, measurements, target_features, config_name = get_scenario_config(scenario_type=scenario_type,
                                                                               missing_measurements=excluded_measurements)
    return stations, measurements, target_features, config_name