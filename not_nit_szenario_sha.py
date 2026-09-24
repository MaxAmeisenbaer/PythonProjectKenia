def get_not_nit_config():
    config_name = "not_nit"
    stations = ["SHA", "WSH", "TTP", "TF2", "NF", "Kur", "KFS", "Fun", "Fin", "Cha", "Chi"]

    measurements = {
        "SHA": ["disch", "doc", "elc", "nit", "tcd", "toc", "tsp", "tur", "wl", "prec"],
        "WSH": ["dir", "ec15", "gust", "par", "rh", "stemp15", "temp", "vwc15", "wind", "prec"],
        "TTP": ["disch", "doc", "elc", "nit", "tcd", "toc", "tsp", "tur", "wl", "prec"],
        "TF2": ["temp", "prec"],
        "NF": ["disch", "doc", "elc", "nit", "tcd", "toc", "tsp", "tur", "wl", "prec"],
        "Kur": ["prec"],
        "KFS": ["dir", "gust", "par", "rh", "stemp15", "temp", "vwc15", "wind"],
        "Fun": ["prec"],
        "Fin": ["prec"],
        "Cha": ["prec"],
        "Chi": ["prec"]

    }
    target_features = ["SHA_nit", "TTP_nit", "NF_nit"]

    return stations, measurements, target_features, config_name