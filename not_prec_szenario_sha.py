def get_not_prec_config():
    config_name = "not_prec"
    stations = ["SHA", "WSH", "TTP", "TF2", "NF", "KFS"]

    measurements = {
        "SHA": ["disch", "doc", "elc", "nit", "tcd", "toc", "tsp", "tur", "wl"],
        "WSH": ["dir", "ec15", "gust", "par", "rh", "stemp15", "temp", "vwc15", "wind"],
        "TTP": ["disch", "doc", "elc", "tcd", "toc", "tsp", "tur", "wl"],
        "TF2": ["temp"],
        "NF": ["disch", "doc", "elc", "tcd", "toc", "tsp", "tur", "wl"],
        "KFS": ["dir", "gust", "par", "rh", "stemp15", "temp", "vwc15", "wind"]

    }
    target_feature = "SHA_nit"

    return stations, measurements, target_feature, config_name