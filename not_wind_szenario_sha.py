def get_not_wind_config():
    """
    Excluded: dir,gust,wind
    :return:
    """
    config_name = "not_wind"
    stations = ["SHA", "WSH", "TTP", "TF2", "NF", "Kur", "KFS", "Fun", "Fin", "Cha", "Chi"]

    measurements = {
        "SHA": ["disch", "doc", "elc", "nit", "tcd", "toc", "tsp", "tur", "wl", "prec"],
        "WSH": ["ec15", "par", "rh", "stemp15", "temp", "vwc15", "prec"],
        "TTP": ["disch", "doc", "elc", "nit", "tcd", "toc", "tsp", "tur", "wl", "prec"],
        "TF2": ["temp", "prec"],
        "NF": ["disch", "doc", "elc", "nit", "tcd", "toc", "tsp", "tur", "wl", "prec"],
        "Kur": ["prec"],
        "KFS": ["par", "rh", "stemp15", "temp", "vwc15"],
        "Fun": ["prec"],
        "Fin": ["prec"],
        "Cha": ["prec"],
        "Chi": ["prec"]

    }
    target_features = ["SHA_nit", "TTP_nit", "NF_nit"]

    return stations, measurements, target_features, config_name