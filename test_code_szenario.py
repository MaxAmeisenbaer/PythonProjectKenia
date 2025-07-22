def get_test_code_config():
    config_name = "test_code"
    stations = ["SHA", "KFS"]

    measurements = {
        "SHA": ["disch", "nit"],
        "KFS": ["prec"]
    }
    target_feature = "SHA_nit"

    return stations, measurements, target_feature, config_name