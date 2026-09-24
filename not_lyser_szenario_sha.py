def get_not_lyser_config():
    config_name = "not_lyser"
    stations = ["SHA", "WSH", "TTP", "TF2", "NF", "Kur", "KFS", "Fun", "Fin", "Cha", "Chi"]

    measurements = {
        "SHA": ["disch", "nit", "wl", "prec"],
        "WSH": ["dir", "ec15", "gust", "par", "rh", "stemp15", "temp", "vwc15", "wind", "prec"],
        "TTP": ["disch", "nit", "wl", "prec"],
        "TF2": ["temp", "prec"],
        "NF": ["disch", "nit", "wl", "prec"],
        "Kur": ["prec"],
        "KFS": ["dir", "gust", "par", "rh", "stemp15", "temp", "vwc15", "wind"],
        "Fun": ["prec"],
        "Fin": ["prec"],
        "Cha": ["prec"],
        "Chi": ["prec"]

    }
    target_features = ["SHA_nit", "TTP_nit", "NF_nit"]

    return stations, measurements, target_features, config_name