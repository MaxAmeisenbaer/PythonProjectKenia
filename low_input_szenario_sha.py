def get_low_input_config():
    config_name = "low_input"
    stations = ["SHA", "WSH", "TTP", "TF2", "NF", "Kur", "KFS", "Fun", "Fin", "Cha", "Chi"]

    measurements = {
        "SHA": ["disch", "prec"],
        "WSH": ["gust","par", "rh","temp","wind","prec"],
        "TTP": ["disch", "prec"],
        "TF2": ["temp","prec"],
        "NF": ["disch", "prec"],
        "Kur": ["prec"],
        "KFS": ["gust", "par", "rh","temp","wind","prec"],
        "Fun": ["prec"],
        "Fin": ["prec"],
        "Cha": ["prec"],
        "Chi": ["prec"]
    }
    target_feature = "SHA-nit"

    return stations, measurements, target_feature, config_name