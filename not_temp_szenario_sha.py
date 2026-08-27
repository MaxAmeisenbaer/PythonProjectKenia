def get_not_temp_config():
    """
    Excluded: tcd,temp,tsp
    :return:
    """
    config_name = "not_temp"
    stations = ["SHA", "WSH", "TTP", "TF2", "NF", "Kur", "KFS", "Fun", "Fin", "Cha", "Chi"]

    measurements = {
        "SHA": ["disch", "doc", "elc", "nit", "toc", "tur", "wl", "prec"],
        "WSH": ["dir", "ec15", "gust", "par", "rh", "stemp15", "vwc15", "wind", "prec"],
        "TTP": ["disch", "doc", "elc", "toc", "tur", "wl", "prec"],
        "TF2": ["prec"],
        "NF": ["disch", "doc", "elc", "toc", "tur", "wl", "prec"],
        "Kur": ["prec"],
        "KFS": ["dir", "gust", "par", "rh", "stemp15", "vwc15", "wind"],
        "Fun": ["prec"],
        "Fin": ["prec"],
        "Cha": ["prec"],
        "Chi": ["prec"]

    }
    target_feature = "SHA_nit"

    return stations, measurements, target_feature, config_name