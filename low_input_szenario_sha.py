def get_low_input_config():
    stations = ["SHA", "WSH", "TTP", "TF2", "NF", "Kur", "KFS", "Fun", "Fin", "Cha", "Chi"]

    measurements = {
        "SHA": ["disch", "prec"],
        "WSH": ["prec"],
        "TTP": ["prec"],
        "TF2": ["prec"],
        "NF": ["disch", "prec"],
        "Kur": ["prec"],
        "KFS": ["prec"],
        "Fun": ["prec"],
        "Fin": ["prec"],
        "Cha": ["prec"],
        "Chi": ["prec"]
    }
    target_feature = "SHA-nit"

    return stations, measurements, target_feature