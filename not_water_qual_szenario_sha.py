def get_not_water_qual_config():
    """
    Excluded Parameters: toc,doc,tur
    :return:
    """
    config_name = "not_water_qual"
    stations = ["SHA", "WSH", "TTP", "TF2", "NF", "Kur", "KFS", "Fun", "Fin", "Cha", "Chi"]

    measurements = {
        "SHA": ["disch", "elc", "nit", "tcd", "tsp", "wl", "prec"],
        "WSH": ["dir", "ec15", "gust", "par", "rh", "stemp15", "temp", "vwc15", "wind", "prec"],
        "TTP": ["disch", "elc", "tcd", "tsp", "wl", "prec"],
        "TF2": ["temp", "prec"],
        "NF": ["disch", "elc", "tcd", "tsp", "wl", "prec"],
        "Kur": ["prec"],
        "KFS": ["dir", "gust", "par", "rh", "stemp15", "temp", "vwc15", "wind"],
        "Fun": ["prec"],
        "Fin": ["prec"],
        "Cha": ["prec"],
        "Chi": ["prec"]

    }
    target_feature = "SHA_nit"

    return stations, measurements, target_feature, config_name