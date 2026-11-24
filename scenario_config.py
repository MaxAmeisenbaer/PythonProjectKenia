import copy
#Mar data is added to Air data in the pipeline
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


def _get_scenario_config(scenario_type, missing_measurements=None, custom_targets=None, missing_stations=None):
    """
    Generates a configuration based on the master set. Is used by more specific scenario-setups

    :param scenario_type: Name/type of the scenario
    :param custom_targets: Output targets (default: DEFAULT_TARGETS)
    :param missing_stations: Stations to be excluded
    :param missing_measurements: Measurement variables to be excluded

    :return: Information relevant for the setup of the dataframe for PreProcessing (stations, measurements, target_feature, config_name)
    """

    stations = copy.deepcopy(MASTER_STATIONS)
    measurements = copy.deepcopy(MASTER_MEASUREMENTS)
    target_features = custom_targets if custom_targets is not None else copy.deepcopy(DEFAULT_TARGETS)
    config_name = scenario_type

    # Exclude stations
    if missing_stations:
        for s in missing_stations:
            if s in stations:
                stations.remove(s)
                if s in measurements:
                    del measurements[s]
    # Exclude measurements
    if missing_measurements:
        if isinstance(missing_measurements, (list, tuple, set)):
            missing_measurements = {s: list(missing_measurements) for s in measurements}
        for station, measures in list(missing_measurements.items()):
            if station not in measurements:
                continue
            measurements[station] = [m for m in measurements[station] if m not in measures]
            if not measurements[station]:
                stations.remove(station)
                del measurements[station]
    for t in target_features:
        t_station, t_measure = t.split("_")
        # The station was removed
        if t_station not in stations:
            print(f"[RECOVERY] Target '{t}' benötigt Station '{t_station}'. Füge Station wieder hinzu.")
            stations.append(t_station)
            if t_station not in measurements:
                measurements[t_station] = []

        # The station is present, but the measurement was removed
        if t_measure not in measurements[t_station]:
            print(f"[RECOVERY] Target '{t}' benötigt Messwert '{t_measure}'. Füge Messwert zu '{t_station}' hinzu.")
            if t_measure not in measurements[t_station]:
                measurements[t_station].append(t_measure)
    stations.sort()
    for s in measurements:
        measurements[s].sort()

    return stations, measurements, target_features, config_name

def _add_disch():
    """
    Function that adds discharge-variables to the DEFAULT_TARGETS

    :return: Custom-target with added discharge-variables and water-level excluded from the measurements
    """

    custom_targets = list(DEFAULT_TARGETS)
    for station, measurements in MASTER_MEASUREMENTS.items():
        if "disch" in measurements:
            custom_targets.append(f"{station}_disch")
    excluded_measurements = ["NF_wl","SHA_wl","TTP_wl"]

    return custom_targets, excluded_measurements


def get_test_code_config():
    """
    Variant intended only for minimal tests
    Remaining: Fin,Fun prec + custom targets - the _get_scenario_config features are tested via SHA and TTP_tur

    :return: Information relevant for the setup of the dataframe for PreProcessing (stations, measurements, target_feature, config_name)
    """
    scenario_type = "test_code"
    excluded_measurements = ["disch", "doc", "elc", "prec", "tcd", "toc", "tsp", "tur", "wl"]
    custom_targets = ["NF_nit","SHA_nit","TTP_nit","TTP_tur"]
    excluded_stations = ["Air","Cha","Chi","KFS","Kur","NF","Mar","TF2","TTP","WSH"]

    stations, measurements, target_features, config_name = _get_scenario_config(scenario_type= scenario_type,
                                                                               missing_measurements= excluded_measurements,
                                                                               custom_targets= custom_targets,
                                                                               missing_stations= excluded_stations)

    return stations, measurements, target_features, config_name

def get_benchmark_config():
    """
    Base-Scenario including all available stations and measurements

    :return: Information relevant for the setup of the dataframe for PreProcessing (stations, measurements, target_feature, config_name)
    """
    scenario_type = "benchmark"
    stations, measurements, target_features, config_name = _get_scenario_config(scenario_type)

    return stations, measurements, target_features, config_name


def get_not_atmos_config():
    """
    Adapted Scenario including all stations but excluding:
    photosynthetically active radiation (par) and Relative Humidity (rh)

    :return: Information relevant for the setup of the dataframe for PreProcessing (stations, measurements, target_feature, config_name)
    """
    scenario_type = "not_atmos"
    excluded_measurements = ["par", "rh"]
    stations, measurements, target_features, config_name = _get_scenario_config(scenario_type=scenario_type,
                                                                               missing_measurements=excluded_measurements)

    return stations, measurements, target_features, config_name


def get_not_ec_config():
    """
    Adapted Scenario including all stations but excluding:
    electric charge in 15cm deep soil (ec15) and electric charge measured in the water (elc)

    :return: Information relevant for the setup of the dataframe for PreProcessing (stations, measurements, target_feature, config_name)
    """
    scenario_type = "not_ec"
    excluded_measurements = ["ec15", "elc"]
    stations, measurements, target_features, config_name = _get_scenario_config(scenario_type=scenario_type,
                                                                               missing_measurements=excluded_measurements)
    return stations, measurements, target_features, config_name


def get_not_prec_config():
    """
    Adapted Scenario including all stations but excluding:
    Precepitation (prec)

    :return: Information relevant for the setup of the dataframe for PreProcessing (stations, measurements, target_feature, config_name)
    """
    scenario_type = "not_prec"
    excluded_measurements = ["prec"]
    stations, measurements, target_features, config_name = _get_scenario_config(scenario_type=scenario_type,
                                                                               missing_measurements=excluded_measurements)
    return stations, measurements, target_features, config_name


def get_not_soil_config():
    """
    Adapted Scenario including all stations but excluding:
    Soil measurements of electricity (ec15), soil temperature (stemp15) and soil moisture (vwc15) in 15cm depth

    :return: Information relevant for the setup of the dataframe for PreProcessing (stations, measurements, target_feature, config_name)
    """
    scenario_type = "not_soil"
    excluded_measurements = ["ec15", "stemp15", "vwc15"]
    stations, measurements, target_features, config_name = _get_scenario_config(scenario_type=scenario_type,
                                                                               missing_measurements=excluded_measurements)
    return stations, measurements, target_features, config_name


def get_not_temp_config():
    """
    Adapted Scenario including all stations but excluding:
    All temperature measurements in the water (tsp,tcd) and in the ait (temp)

    :return: Information relevant for the setup of the dataframe for PreProcessing (stations, measurements, target_feature, config_name)
    """
    scenario_type = "not_temp"
    excluded_measurements = ["tcd", "temp", "tsp"]
    stations, measurements, target_features, config_name = _get_scenario_config(scenario_type=scenario_type,
                                                                               missing_measurements=excluded_measurements)
    return stations, measurements, target_features, config_name


def get_not_water_lvl_config():
    """
    Adapted Scenario including all stations but excluding:
    discharge (disch) and waterlevel (wl) of the river-measurements

    :return: Information relevant for the setup of the dataframe for PreProcessing (stations, measurements, target_feature, config_name)
    """
    scenario_type = "not_water_lvl"
    excluded_measurements = ["disch", "wl"]
    stations, measurements, target_features, config_name = _get_scenario_config(scenario_type=scenario_type,
                                                                               missing_measurements=excluded_measurements)
    return stations, measurements, target_features, config_name


def get_not_water_qual_config():
    """
    Adapted Scenario including all stations but excluding:
    Water-Quality Measurements as dilluded organic carbon (doc), total organic carbon (toc) and turbidity (tur)

    :return: Information relevant for the setup of the dataframe for PreProcessing (stations, measurements, target_feature, config_name)
    """
    scenario_type = "not_water_qual"
    excluded_measurements = ["doc", "toc", "tur"]
    stations, measurements, target_features, config_name = _get_scenario_config(scenario_type=scenario_type,
                                                                               missing_measurements=excluded_measurements)
    return stations, measurements, target_features, config_name


def get_not_wind_config():
    """
    Adapted Scenario including all stations but excluding:
    All wind-related variables as wind speed (wind), gust speed (gust) and wind direction (dir)

    :return: Information relevant for the setup of the dataframe for PreProcessing (stations, measurements, target_feature, config_name)
    """
    scenario_type = "not_wind"
    excluded_measurements = ["dir", "wind", "gust"]
    stations, measurements, target_features, config_name = _get_scenario_config(scenario_type=scenario_type,
                                                                               missing_measurements=excluded_measurements)
    return stations, measurements, target_features, config_name

def get_disch_config():
    """
    Adapted Scenario including all stations and all measurements but changing:
    discharge (disch) as target feature, water level (wl) excluded for data leakage purposes

    :return: Information relevant for the setup of the dataframe for PreProcessing (stations, measurements, target_feature, config_name)
    """
    scenario_type = "disch"
    custom_target, excluded_measurements = _add_disch()
    stations, measurements, target_features, config_name = _get_scenario_config(scenario_type=scenario_type,
                                                                               missing_measurements=excluded_measurements,
                                                                               custom_targets=custom_target)
    return stations, measurements, target_features, config_name