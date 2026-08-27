def get_not_water_lvl_config():
    """
    Excluded: disch,wl
    :return:
    """
    config_name = "not_water_lvl"
    stations = ["SHA", "WSH", "TTP", "TF2", "NF", "Kur", "KFS", "Fun", "Fin", "Cha", "Chi"]

    measurements = {
        "SHA": ["doc", "elc", "nit", "tcd", "toc", "tsp", "tur", "prec"],
        "WSH": ["dir", "ec15", "gust", "par", "rh", "stemp15", "temp", "vwc15", "wind", "prec"],
        "TTP": ["doc", "elc", "tcd", "toc", "tsp", "tur", "prec"],
        "TF2": ["temp", "prec"],
        "NF": ["doc", "elc", "tcd", "toc", "tsp", "tur", "prec"],
        "Kur": ["prec"],
        "KFS": ["dir", "gust", "par", "rh", "stemp15", "temp", "vwc15", "wind"],
        "Fun": ["prec"],
        "Fin": ["prec"],
        "Cha": ["prec"],
        "Chi": ["prec"]

    }
    target_feature = "SHA_nit"

    return stations, measurements, target_feature, config_name