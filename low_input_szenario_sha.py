def get_low_input_config():
    config_name = "low_input"
    stations = ["SHA", "WSH", "TTP", "TF2", "NF", "Kur", "KFS", "Fun", "Fin", "Cha", "Chi"]

    measurements = {
        "SHA": ["disch", "prec", "nit"],
        "WSH": ["gust","par", "rh","temp","wind","prec"],
        "TTP": ["disch", "prec", "nit"],
        "TF2": ["temp","prec"],
        "NF": ["disch", "prec", "nit"],
        "Kur": ["prec"],
        "KFS": ["gust", "par", "rh","temp","wind"],
        "Fun": ["prec"],
        "Fin": ["prec"],
        "Cha": ["prec"],
        "Chi": ["prec"]
    }
    target_features = ["SHA_nit", "TTP_nit", "NF_nit"]

    return stations, measurements, target_features, config_name