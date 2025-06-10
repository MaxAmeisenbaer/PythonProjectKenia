def get_benchmark_config():
    config_name = "benchmark"
    stations = ["SHA", "WSH", "TTP", "TF2", "NF", "Kur", "KFS", "Fun", "Fin", "Cha", "Chi"]

    measurements = {
        "SHA": ["disch", "doc", "elc", "nit", "tcd", "toc", "tsp", "tur", "wl", "prec"],
        "WSH": ["dir", "ec15", "gust", "par", "rh", "stemp15", "temp", "vwc15", "wind", "prec"],
        "TTP": ["disch", "doc", "elc", "nit", "tcd", "toc", "tsp", "tur", "wl", "prec"],
        "TF2": ["temp", "prec"],
        "NF": ["disch", "doc", "elc", "nit", "tcd", "toc", "tsp", "tur", "wl", "prec"],
        "Kur": ["prec"],
        "KFS": ["dir", "gust", "par", "rh", "stemp15", "temp", "vwc15", "wind", "prec"],
        "Fun": ["prec"],
        "Fin": ["prec"],
        "Cha": ["prec"],
        "Chi": ["prec"]

    }
    target_feature = "SHA_nit"

    return stations, measurements, target_feature, config_name