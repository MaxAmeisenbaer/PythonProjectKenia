def get_test_code_config():
    config_name = "test_code"
    stations = ["SHA", "Fun"]

    measurements = {
        "SHA": ["disch", "nit"],
        "TTP": ["nit"],
        "Fun": ["prec"]
    }
    target_features = ["SHA_nit","TTP_nit"]

    return stations, measurements, target_features, config_name